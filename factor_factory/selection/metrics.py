
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

def calmar_ratio(pnl: pd.Series, equity: pd.Series, freq: int = 365) -> float:
    """Calmar 비율 = CAGR / |Max Drawdown|"""
    cagr = float(equity.iloc[-1] ** (freq / len(equity)) - 1) if len(equity) > 0 else 0
    mdd = abs(max_drawdown(equity))
    return cagr / mdd if mdd != 0 else float("nan")

def win_rate(pnl: pd.Series) -> float:
    """승률 계산"""
    positive_days = (pnl > 0).sum()
    total_days = len(pnl[pnl != 0])  # 거래가 있는 날만 계산
    return positive_days / total_days if total_days > 0 else 0

def profit_factor(pnl: pd.Series) -> float:
    """수익 팩터 = 총 이익 / 총 손실"""
    profit = pnl[pnl > 0].sum()
    loss = abs(pnl[pnl < 0].sum())
    return profit / loss if loss != 0 else float("inf")

def max_consecutive_losses(pnl: pd.Series) -> int:
    """최대 연속 손실 일수"""
    losses = (pnl < 0).astype(int)
    max_consec = 0
    current_consec = 0
    
    for loss in losses:
        if loss:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0
    
    return max_consec

def information_ratio(pnl: pd.Series, benchmark_pnl: pd.Series = None) -> float:
    """정보 비율 (벤치마크 대비 초과 수익률의 안정성)"""
    if benchmark_pnl is None:
        benchmark_pnl = pd.Series(0, index=pnl.index)  # 무위험 수익률 0으로 가정
    
    excess_return = pnl - benchmark_pnl.reindex(pnl.index, fill_value=0)
    return excess_return.mean() / excess_return.std() if excess_return.std() != 0 else float("nan")

def compute_metrics(pnl: pd.Series, equity: pd.Series, signal: pd.Series, freq:int=365) -> dict:
    cagr = float(equity.iloc[-1] ** (freq / len(equity)) - 1) if len(equity)>0 else float("nan")
    return {
        "cagr": cagr,
        "sharpe": sharpe(pnl, freq=freq),
        "mdd": max_drawdown(equity),
        "turnover": turnover(signal),
        "calmar": calmar_ratio(pnl, equity, freq),
        "win_rate": win_rate(pnl),
        "profit_factor": profit_factor(pnl),
        "max_consecutive_losses": max_consecutive_losses(pnl),
        "information_ratio": information_ratio(pnl),
        "total_trades": int((signal.diff() != 0).sum()),
        "avg_trade_pnl": float(pnl.mean()) if len(pnl) > 0 else 0,
    }
