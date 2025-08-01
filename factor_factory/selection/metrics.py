
from __future__ import annotations
import numpy as np, pandas as pd

def sharpe(pnl: pd.Series, freq:int=365) -> float:
    ann_ret = pnl.mean() * freq
    ann_vol = pnl.std(ddof=0) * (freq ** 0.5)
    return ann_ret / ann_vol if ann_vol != 0 else float("nan")

def max_drawdown(equity: pd.Series) -> float:
    roll = equity.cummax()
    dd = equity/roll - 1.0
    return float(dd.min())

def turnover(signal: pd.Series) -> float:
    return float(signal.diff().abs().sum())

def compute_metrics(pnl: pd.Series, equity: pd.Series, signal: pd.Series, freq:int=365) -> dict:
    cagr = float(equity.iloc[-1] ** (freq / len(equity)) - 1) if len(equity)>0 else float("nan")
    return {
        "cagr": cagr,
        "sharpe": sharpe(pnl, freq=freq),
        "mdd": max_drawdown(equity),
        "turnover": turnover(signal),
    }
