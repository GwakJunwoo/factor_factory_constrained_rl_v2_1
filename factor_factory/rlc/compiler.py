
from __future__ import annotations
import numpy as np, pandas as pd
from .grammar import TOKENS, OPS, TERMS, ARITY, name

def _sma(x: pd.Series, n:int) -> pd.Series:
    return x.rolling(n).mean()

def _rsi(x: pd.Series, n:int=14) -> pd.Series:
    diff = x.diff()
    up = (diff.clip(lower=0)).rolling(n).mean()
    down = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / (down.replace(0,np.nan))
    rsi = 100 - 100/(1+rs)
    return (rsi.fillna(50)/50 - 1.0).astype(np.float32)  # ~[-1,1]

def _get_terminal(df: pd.DataFrame, tok:int) -> pd.Series:
    if tok==4: return df["close"].astype(np.float32)
    if tok==5: return df["open"].astype(np.float32)
    if tok==6: return df["high"].astype(np.float32)
    if tok==7: return df["low"].astype(np.float32)
    if tok==8: return df["volume"].replace(0,np.nan).ffill().astype(np.float32)
    if tok==9: return _sma(df["close"].astype(np.float32),10).astype(np.float32)
    if tok==10: return _sma(df["close"].astype(np.float32),20).astype(np.float32)
    if tok==11: return _rsi(df["close"].astype(np.float32),14).astype(np.float32)
    if tok==12: return pd.Series(np.float32(1.0), index=df.index)
    raise ValueError(f"Unknown terminal token: {tok}")

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    eps = np.float32(1e-8)
    return a / (b.abs() + eps)

def eval_prefix(tokens: list[int], df: pd.DataFrame) -> pd.Series:
    """Evaluate prefix program to a raw series; raise on invalid."""
    stack = []
    for tok in reversed(tokens):
        if tok in TERMS:
            stack.append(_get_terminal(df, tok))
        elif tok in OPS:
            if len(stack) < 2:
                raise ValueError("Invalid stack underflow")
            x = stack.pop(); y = stack.pop()
            if tok==0: z = x + y
            elif tok==1: z = x - y
            elif tok==2: z = x * y
            elif tok==3: z = _safe_div(x, y)
            else: raise ValueError("Unknown op")
            stack.append(z.astype(np.float32))
        else:
            raise ValueError(f"Unknown token {tok}")
    if len(stack) != 1:
        raise ValueError("Invalid program (stack not singleton)")
    raw = stack[0].replace([np.inf, -np.inf], np.nan).ffill().fillna(0).astype(np.float32)
    # normalize & squash → [-1,1] signal
    std = raw.rolling(64).std().replace(0, np.nan)
    std_fallback = np.float32(raw.std()) if np.isfinite(raw.std()) and raw.std()!=0 else np.float32(1.0)
    std = std.fillna(std_fallback).astype(np.float32)
    z = (raw / (std * np.float32(2.0))).clip(-10, 10).astype(np.float32)
    z = z.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    sig_np = np.tanh(z.values, dtype=np.float32)
    sig_np = np.clip(sig_np, -1.0, 1.0, dtype=np.float32)
    return pd.Series(sig_np, index=raw.index, name="signal")

def calc_tree_depth(tokens: list[str]) -> int:
    """후위표기/전위표기 트리 깊이 계산"""
    def _helper(it):
        token = next(it)
        if token in ["add", "sub", "mul", "div"]:  # 이진 연산자
            l = _helper(it)
            r = _helper(it)
            return 1 + max(l, r)
        else:  # 단항 또는 터미널
            return 1
    return _helper(iter(tokens))
