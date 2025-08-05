
from __future__ import annotations
import numpy as np, pandas as pd
from functools import lru_cache
from .grammar import TOKENS, OPS, TERMS, ARITY, name
from .cache import get_program_cache

def _sma(x: pd.Series, n:int) -> pd.Series:
    return x.rolling(n).mean()

def _ema(x: pd.Series, n:int) -> pd.Series:
    return x.ewm(span=n).mean()

def _rsi(x: pd.Series, n:int=14) -> pd.Series:
    diff = x.diff()
    up = (diff.clip(lower=0)).rolling(n).mean()
    down = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / (down.replace(0,np.nan))
    rsi = 100 - 100/(1+rs)
    return (rsi.fillna(50)/50 - 1.0).astype(np.float32)  # ~[-1,1]

def _bbands(x: pd.Series, n:int=20) -> tuple[pd.Series, pd.Series]:
    sma = x.rolling(n).mean()
    std = x.rolling(n).std()
    upper = sma + 2*std
    lower = sma - 2*std
    return upper, lower

def _macd(x: pd.Series) -> pd.Series:
    ema12 = x.ewm(span=12).mean()
    ema26 = x.ewm(span=26).mean()
    return (ema12 - ema26).astype(np.float32)

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n:int=14) -> pd.Series:
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return (k_percent.fillna(50)/50 - 1.0).astype(np.float32)  # ~[-1,1]

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
    if tok==13: return _sma(df["close"].astype(np.float32),5).astype(np.float32)
    if tok==14: return _ema(df["close"].astype(np.float32),10).astype(np.float32)
    if tok==15: return _ema(df["close"].astype(np.float32),20).astype(np.float32)
    if tok==16: 
        upper, _ = _bbands(df["close"].astype(np.float32),20)
        return upper.astype(np.float32)
    if tok==17: 
        _, lower = _bbands(df["close"].astype(np.float32),20)
        return lower.astype(np.float32)
    if tok==18: return _macd(df["close"].astype(np.float32)).astype(np.float32)
    if tok==19: return _stochastic(df["high"], df["low"], df["close"], 14).astype(np.float32)
    raise ValueError(f"Unknown terminal token: {tok}")

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    eps = np.float32(1e-8)
    return a / (b.abs() + eps)

def eval_prefix(tokens: list[int], df: pd.DataFrame, use_cache: bool = True) -> pd.Series:
    """Evaluate prefix program to a raw series; raise on invalid."""
    
    # 캐시 확인
    if use_cache:
        cache = get_program_cache()
        cached_result = cache.get(tokens, df)
        if cached_result is not None:
            return cached_result
    
    stack = []
    for tok in reversed(tokens):
        if tok in TERMS:
            stack.append(_get_terminal(df, tok))
        elif ARITY[tok] == 1:  # 단항 연산자
            if len(stack) < 1:
                raise ValueError("Invalid stack underflow for unary operator")
            x = stack.pop()
            if tok == 22: z = x.abs()  # ABS
            elif tok == 23: z = np.log(x.abs() + np.float32(1e-8))  # LOG (안전한 로그)
            elif tok == 24: z = x.shift(1).fillna(0)  # LAG1
            else: raise ValueError(f"Unknown unary operator {tok}")
            stack.append(z.astype(np.float32))
        elif ARITY[tok] == 2:  # 이진 연산자
            if len(stack) < 2:
                raise ValueError("Invalid stack underflow for binary operator")
            x = stack.pop(); y = stack.pop()
            if tok==0: z = x + y  # ADD
            elif tok==1: z = x - y  # SUB
            elif tok==2: z = x * y  # MUL
            elif tok==3: z = _safe_div(x, y)  # DIV
            elif tok==20: z = np.maximum(x, y)  # MAX
            elif tok==21: z = np.minimum(x, y)  # MIN
            else: raise ValueError(f"Unknown binary operator {tok}")
            stack.append(z.astype(np.float32))
        else:
            raise ValueError(f"Unknown token {tok}")
    if len(stack) != 1:
        raise ValueError("Invalid program (stack not singleton)")
    raw = stack[0].replace([np.inf, -np.inf], np.nan).ffill().fillna(0).astype(np.float32)
    
    # 롤링 윈도우 정규화 (미래 정보 누출 방지)
    rolling_window = 64  # 기본 윈도우 크기
    rolling_mean = raw.rolling(rolling_window, min_periods=10).mean()
    rolling_std = raw.rolling(rolling_window, min_periods=10).std()
    
    # 초기 구간은 expanding 윈도우 사용
    expanding_mean = raw.expanding(min_periods=1).mean()
    expanding_std = raw.expanding(min_periods=1).std()
    
    # 롤링 값이 없는 초기 구간은 expanding 값으로 대체
    mean_series = rolling_mean.fillna(expanding_mean)
    std_series = rolling_std.fillna(expanding_std).replace(0, np.nan)
    
    # 표준편차가 0인 경우 대체값 사용
    std_fallback = np.float32(raw.std()) if np.isfinite(raw.std()) and raw.std()!=0 else np.float32(1.0)
    std_series = std_series.fillna(std_fallback).astype(np.float32)
    
    # Z-score 계산 후 클리핑
    z = ((raw - mean_series) / (std_series * np.float32(2.0))).clip(-10, 10).astype(np.float32)
    z = z.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    
    # tanh로 [-1,1] 범위로 압축
    sig_np = np.tanh(z.values, dtype=np.float32)
    sig_np = np.clip(sig_np, -1.0, 1.0, dtype=np.float32)
    result = pd.Series(sig_np, index=raw.index, name="signal")
    
    # 캐시에 저장
    if use_cache:
        cache.put(tokens, df, result)
    
    return result
