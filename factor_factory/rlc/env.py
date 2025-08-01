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
from ..backtest import vector_backtest

# =====================================================
#  ProgramEnv (z‑score → ±2 진입 / 0 청산 버전)
# =====================================================

@dataclass
class RLCConfig:
    max_len: int = 21
    length_penalty: float = 0.0005
    # reward penalties
    lambda_depth: float = 0.002
    lambda_turnover: float = 0.0005
    lambda_const1: float = 3.0
    lambda_std: float = 1.0      # 미세 진동 신호 패널티
    # trading params
    commission: float = 0.0004
    slippage: float = 0.0010
    leverage: int = 1
    # speed knobs
    eval_stride: int = 2
    max_eval_bars: int = 20_000
    cache_size: int = 2_048


class ProgramEnv(gym.Env):
    """트리 결과를 z‑score 후
       ‑ z ≥ +2  → +1 (Long)
       ‑ z ≤ −2  → −1 (Short)
       ‑ else    →  0 (Flat)
    """

    metadata = {}

    def __init__(self, df: pd.DataFrame, cfg: RLCConfig):
        self.cfg = cfg
        view = df.iloc[-cfg.max_eval_bars :]
        self.df = view.iloc[:: max(1, cfg.eval_stride)].copy()

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(cfg.max_len + 2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_TOKENS)

        self.reset()

    # -------------------- reset --------------------
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.tokens: List[int] = []
        self.need = 1
        self.done = False
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
                raw = eval_prefix(self.tokens, self.df).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                if raw.empty or raw.std(ddof=0) < 1e-6:
                    reward = -1.0; terminated = True; info = {"invalid_signal": True}
                else:
                    # z‑score (전체 구간 기준)
                    z = (raw - raw.mean()) / raw.std(ddof=0)
                    sig = pd.Series(0, index=z.index, dtype=float)
                    sig[z >= 2.0] = 1.0
                    sig[z <= -2.0] = -1.0
                    # backtest on discrete signal
                    price = self.df["close"].reindex(sig.index)
                    equity, pnl = vector_backtest(price, sig, commission=self.cfg.commission, slippage=self.cfg.slippage, leverage=self.cfg.leverage)
                    pnl_sum = float(pnl.sum())
                    trades = count_entries(sig.values)
                    depth = calc_tree_depth(self.tokens)

                    cnt = Counter(self.tokens)
                    const_ratio = cnt.get(12, 0) / len(self.tokens)
                    std_pen = self.cfg.lambda_std / raw.std(ddof=0)

                    reward = (
                        pnl_sum
                        - self.cfg.lambda_depth * depth
                        - self.cfg.lambda_turnover * trades
                        - self.cfg.lambda_const1 * const_ratio
                        - std_pen
                    )

                    print(f"[R={reward:.4f}] pnl={pnl_sum:.4f} depth={depth} trades={trades} CONST%={const_ratio:.2f}\n{tokens_to_infix(self.tokens)}")

                    info = {"pnl": pnl_sum, "depth": depth, "trades": trades, "program": self.tokens.copy()} 
                    terminated = True
            except Exception as e:
                reward = -1.0; info = {"error": str(e), "program": self.tokens.copy()}; terminated = True

        self.done = terminated
        return self._obs(), float(reward), terminated, False, info
