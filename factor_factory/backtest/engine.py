
from __future__ import annotations
import pandas as pd, numpy as np

def vector_backtest(price: pd.Series, signal: pd.Series, *, commission:float=0.0004, slippage:float=0.0010, leverage:int=1):
    signal = signal.shift(1).fillna(0).clip(-1,1)
    ret = price.pct_change().fillna(0)
    pnl = signal * ret * leverage
    cost = (signal.diff().abs() * (commission + slippage)).fillna(0)
    pnl_net = pnl - cost
    equity = (1+pnl_net).cumprod()
    return equity, pnl_net
