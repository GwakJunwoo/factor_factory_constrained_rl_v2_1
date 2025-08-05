from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from factor_factory.rlc.callback import SaveBestProgramCallback

import gymnasium as gym
import numpy as np
import pandas as pd

from .compiler import eval_prefix
from .utils import tokens_to_infix, calc_tree_depth, count_entries
from .grammar import ARITY, N_TOKENS
from .cache import get_program_cache
from .signal_generator import generate_signal_realtime, validate_signal_timing
from ..backtest import vector_backtest
from ..backtest.realistic_engine import realistic_backtest

# =====================================================
#  Realistic Trading Environment (미래 정보 누출 방지)
# =====================================================

@dataclass
class RLCConfig:
    max_len: int = 21
    length_penalty: float = 0.0005
    # reward penalties (합리적으로 조정)
    lambda_depth: float = 0.008        # 4배 증가: 복잡도 강력 억제
    # lambda_turnover 제거됨: PnL에 이미 거래비용 반영
    lambda_const1: float = 0.3         # 1/6로 감소: 상수 사용 허용
    lambda_std: float = 0.05           # 1/10로 감소: 변동성 페널티 대폭 완화
    lambda_drawdown: float = 1.5       # 새로운 MDD 페널티 추가
    # trading params (현실적 설정)
    commission: float = 0.0008  # 0.08% (현실적 수수료)
    slippage: float = 0.0015   # 0.15% (현실적 슬리피지)
    leverage: int = 1
    # signal thresholds (적응적 임계값)
    long_threshold: float = 1.5   
    short_threshold: float = -1.5
    # realistic timing parameters
    signal_delay: int = 1       # 신호 생성부터 거래 결정까지 지연
    execution_delay: int = 1    # 거래 결정부터 체결까지 지연
    rebalance_frequency: str = 'D'  # 리밸런싱 빈도
    # speed knobs
    eval_stride: int = 2
    max_eval_bars: int = 20_000
    cache_size: int = 2_048
    # rolling window for z-score (미래 정보 누출 방지)
    rolling_window: int = 252  # 1년 데이터 기준 정규화
    min_signal_periods: int = 50  # 신호 생성 최소 기간
    # validation parameters
    max_signal_change_ratio: float = 0.3  # 최대 신호 변경 비율
    max_correlation_threshold: float = 0.08  # 신호-수익률 상관관계 임계값


class ProgramEnv(gym.Env):
    """현실적 거래 환경을 시뮬레이션하는 강화학습 환경
       - 미래 정보 누출 방지
       - 실제 거래 지연 시간 반영
       - 현실적 거래 비용 및 제약 조건
    """

    def __init__(self, df: pd.DataFrame, cfg: RLCConfig = None):
        super().__init__()
        self.df = df.copy()
        self.cfg = cfg or RLCConfig()
        
        # 환경 설정
        self.action_space = gym.spaces.Discrete(N_TOKENS)
        obs_dim = self.cfg.max_len + 2
        self.observation_space = gym.spaces.Box(-1, 1, shape=(obs_dim,), dtype=np.float32)
        
        # 상태 초기화
        self.tokens: List[int] = []
        self.need = 1
        self.done = False
        self.episode_count = 0
        
        # 캐시 시스템
        self.cache = get_program_cache(self.cfg.cache_size)
        self.cache_hits = 0
        self.total_programs_evaluated = 0
        
        # 검증 통계
        self.validation_failures = 0
        self.total_validations = 0

    # -------------------- reset --------------------
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.tokens: List[int] = []
        self.need = 1
        self.done = False
        self.episode_count += 1
        return self._obs(), {}

    # -------------------- observation --------------
    def _obs(self):
        vec = np.full(self.cfg.max_len + 2, -1.0, dtype=np.float32)
        for i, tok in enumerate(self.tokens[-self.cfg.max_len :]):
            vec[i] = (tok / (N_TOKENS - 1)) * 2 - 1
        vec[-2] = np.tanh(self.need / 8)
        vec[-1] = (len(self.tokens) / self.cfg.max_len) * 2 - 1
        return vec

    # -------------------- token legality -----------
    def _legal(self, tok: int) -> bool:
        rem = self.cfg.max_len - len(self.tokens) - 1
        need_after = self.need - 1 + ARITY[tok]
        return 0 <= need_after <= rem + 1

    # -------------------- step ---------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done; call reset().")

        tok = int(action)
        if not self._legal(tok):
            self.done = True
            return self._obs(), -1.0, True, False, {"invalid": True}

        self.tokens.append(tok)
        self.need = self.need - 1 + ARITY[tok]
        reward = -self.cfg.length_penalty

        terminated = False
        info: Dict = {}

        if self.need == 0:
            try:
                self.total_programs_evaluated += 1
                
                # 캐시된 결과 확인
                cached_result = self.cache.get(self.tokens, self.df)
                if cached_result is not None:
                    self.cache_hits += 1
                    raw = cached_result.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                else:
                    raw = eval_prefix(self.tokens, self.df).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                
                if raw.empty or raw.std(ddof=0) < 1e-6:
                    reward = -1.0; terminated = True; info = {"invalid_signal": True}
                else:
                    # 현실적 신호 생성 (미래 정보 누출 방지)
                    signal = generate_signal_realtime(
                        raw,
                        lookback_window=self.cfg.rolling_window,
                        long_threshold=self.cfg.long_threshold,
                        short_threshold=self.cfg.short_threshold,
                        min_periods=self.cfg.min_signal_periods,
                        rebalance_frequency=self.cfg.rebalance_frequency
                    )
                    
                    # 신호 검증
                    self.total_validations += 1
                    price = self.df["close"].reindex(signal.index)
                    validation_result = validate_signal_timing(self.df, signal, price)
                    
                    if validation_result['has_future_leak']:
                        self.validation_failures += 1
                        print(f"⚠️ 미래 정보 누출 감지: {validation_result['issues']}")
                        reward = -2.0  # 강한 패널티
                        terminated = True
                        info = {"future_leak": True, "validation": validation_result}
                    else:
                        # 현실적 백테스트 실행
                        equity, pnl = realistic_backtest(
                            price, signal,
                            commission=self.cfg.commission,
                            slippage=self.cfg.slippage,
                            leverage=self.cfg.leverage,
                            signal_delay=self.cfg.signal_delay,
                            execution_delay=self.cfg.execution_delay,
                            max_position_change=1.0,  # 유동성 제약
                            impact_factor=0.0002      # 시장 충격
                        )
                        
                        pnl_sum = float(pnl.sum())
                        trades = count_entries(signal.values)
                        depth = calc_tree_depth(self.tokens)
                        
                        # 신호 품질 평가
                        signal_changes = signal.diff().abs().sum()
                        signal_change_ratio = signal_changes / len(signal)
                        
                        # 페널티 계산
                        cnt = Counter(self.tokens)
                        const_ratio = cnt.get(12, 0) / len(self.tokens)
                        std_pen = self.cfg.lambda_std / raw.std(ddof=0)
                        
                        # MDD 페널티 추가
                        cummax = equity.cummax()
                        drawdown = (equity - cummax) / cummax
                        max_drawdown = abs(drawdown.min())
                        mdd_penalty = self.cfg.lambda_drawdown * max_drawdown
                        
                        # 신호 품질 페널티
                        signal_quality_penalty = 0.0
                        if signal_change_ratio > self.cfg.max_signal_change_ratio:
                            signal_quality_penalty += 0.5 * (signal_change_ratio - self.cfg.max_signal_change_ratio)
                        
                        # 최종 보상 계산 (거래횟수 페널티 제거)
                        reward = (
                            pnl_sum                           # 주 보상 (이미 거래비용 반영)
                            - self.cfg.lambda_depth * depth   # 복잡도 페널티 (강화)
                            - self.cfg.lambda_const1 * const_ratio  # 상수 페널티 (완화)
                            - std_pen                         # 변동성 페널티 (완화)
                            - mdd_penalty                     # MDD 페널티 (신규)
                            - signal_quality_penalty         # 신호 품질 페널티
                        )
                        
                        # 통계 출력
                        cache_hit_rate = self.cache_hits / self.total_programs_evaluated if self.total_programs_evaluated > 0 else 0
                        validation_failure_rate = self.validation_failures / self.total_validations if self.total_validations > 0 else 0
                        
                        print(f"[R={reward:.4f}] pnl={pnl_sum:.4f} depth={depth} MDD={max_drawdown:.3f} CONST%={const_ratio:.2f}")
                        print(f"  신호변경률={signal_change_ratio:.1%} 검증실패율={validation_failure_rate:.1%} 캐시적중={cache_hit_rate:.2%}")
                        print(f"  거래횟수={trades} (페널티 제거됨)")
                        print(f"  {tokens_to_infix(self.tokens)}")
                        
                        info = {
                            "pnl": pnl_sum, "depth": depth, "trades": trades, "program": self.tokens.copy(),
                            "cache_hits": self.cache_hits, "total_evaluated": self.total_programs_evaluated,
                            "signal_change_ratio": signal_change_ratio,
                            "validation_failure_rate": validation_failure_rate,
                            "validation_result": validation_result
                        } 
                        terminated = True
            except Exception as e:
                reward = -1.0; info = {"error": str(e), "program": self.tokens.copy()}; terminated = True

        self.done = terminated
        return self._obs(), float(reward), terminated, False, info
