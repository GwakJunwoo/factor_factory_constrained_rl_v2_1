from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import gymnasium as gym
import numpy as np, pandas as pd
from factor_factory.rlc.compiler import calc_tree_depth
from .grammar import N_TOKENS, ARITY
from .compiler import eval_prefix
from ..backtest import vector_backtest
from ..selection import compute_metrics

@dataclass
class RLCConfig:
    max_len: int = 21
    length_penalty: float = 0.001
    lambda_dd: float = 2.0
    lambda_depth: float = 0.1
    lambda_turnover: float = 0.5
    alpha_win: float = 0.25
    commission: float = 0.0004
    slippage: float = 0.0010
    leverage: int = 1
    eval_stride: int = 2
    max_eval_bars: int = 20000
    cache_size: int = 2048

class ProgramEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, df: pd.DataFrame, cfg: RLCConfig):
        self.df_full = df
        self.cfg = cfg
        view = df.iloc[-self.cfg.max_eval_bars:]
        self.df = view.iloc[:: max(1, self.cfg.eval_stride)].copy()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.max_len+2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_TOKENS)
        self._cache: Dict[Tuple[int, ...], Tuple[float, dict]] = {}
        self._cache_order: List[Tuple[int, ...]] = []
        self.reset()

    def _cache_get(self, key: Tuple[int, ...]):
        return self._cache.get(key, None)

    def _cache_put(self, key: Tuple[int, ...], val):
        if key in self._cache: return
        self._cache[key] = val
        self._cache_order.append(key)
        if len(self._cache_order) > self.cfg.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.tokens: List[int] = []
        self.need = 1
        self.done = False
        return self._obs(), {}

    def _obs(self):
        vec = np.full(self.cfg.max_len+2, -1.0, dtype=np.float32)
        for i, tok in enumerate(self.tokens[-self.cfg.max_len:]):
            vec[i] = (tok / (N_TOKENS-1))*2 - 1
        vec[-2] = np.tanh(self.need/8)
        vec[-1] = (len(self.tokens) / self.cfg.max_len)*2 - 1
        return vec

    def _legal(self, tok:int) -> bool:
        remaining = self.cfg.max_len - len(self.tokens) - 1
        need_after = self.need - 1 + ARITY[tok]
        return 0 <= need_after <= remaining + 1

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished")
        tok = int(action)
        reward = 0.0
        if not self._legal(tok):
            self.done = True
            return self._obs(), -1.0, True, False, {"invalid": True}
        self.tokens.append(tok)
        self.need = self.need - 1 + ARITY[tok]
        reward -= self.cfg.length_penalty

        terminated = False; info = {}
        if self.need == 0:
            key = tuple(self.tokens)
            cache_val = self._cache_get(key)
            if cache_val is not None:
                r_fin, info = cache_val
                reward += float(r_fin)
                terminated = True
            else:
                try:
                    sig = eval_prefix(self.tokens, self.df).dropna()
                    if sig.empty or sig.std(ddof=0) == 0:
                        r_fin = -1.0
                        info = {"program": self.tokens.copy(), "empty": True}
                    else:
                        price = self.df["close"].reindex(sig.index)
                        equity, pnl = vector_backtest(price, sig,
                            commission=self.cfg.commission,
                            slippage=self.cfg.slippage,
                            leverage=self.cfg.leverage,
                        )
                        m = compute_metrics(pnl, equity, sig)
                        sh = m.get("sharpe") or -1.0
                        wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
                        mdd = m.get("mdd") or -1.0
                        turnover = m.get("turnover") or 0.0
                        tree_depth = calc_tree_depth(self.tokens)

                        # 보상 계산
                        mdd_term = (mdd if mdd > 0 else -self.cfg.lambda_dd * abs(mdd))
                        r_fin = (
                            sh +
                            self.cfg.alpha_win * wr +
                            mdd_term -
                            self.cfg.lambda_depth * tree_depth -
                            self.cfg.lambda_turnover * turnover
                        )
                        r_fin = float(np.clip(r_fin, -10.0, 10.0))
                        info = {"metrics": m, "program": self.tokens.copy()}
                    self._cache_put(key, (r_fin, info))
                    reward += r_fin
                    terminated = True
                except Exception as e:
                    reward += -1.0
                    info = {"error": str(e), "program": self.tokens.copy()}
                    terminated = True

        self.done = terminated
        if not np.isfinite(reward):
            reward = -1.0
        return self._obs(), float(reward), terminated, False, info
