"""
Fast Backtest Engine with Caching and Optimizations

백테스트 성능을 대폭 향상시키는 최적화된 엔진:
1. 벡터화 연산 최적화
2. 메트릭 계산 캐싱
3. 부분 백테스트 지원
4. NumPy 최적화
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from numba import jit
import logging

from ..rlc.enhanced_cache import get_backtest_cache


@jit(nopython=True, cache=True)
def _fast_backtest_core(
    returns: np.ndarray,
    signals: np.ndarray, 
    commission: float,
    slippage: float,
    leverage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba 최적화된 백테스트 핵심 로직
    
    Returns:
        equity, pnl_net
    """
    n = len(returns)
    pnl = np.zeros(n)
    cost = np.zeros(n)
    equity = np.ones(n)
    
    prev_signal = 0.0
    
    for i in range(n):
        signal = signals[i]
        ret = returns[i]
        
        # PnL 계산
        pnl[i] = signal * ret * leverage
        
        # 거래 비용 계산
        signal_change = abs(signal - prev_signal)
        cost[i] = signal_change * (commission + slippage)
        
        # 순수익
        net_pnl = pnl[i] - cost[i]
        
        # 누적 수익률
        if i == 0:
            equity[i] = 1.0 + net_pnl
        else:
            equity[i] = equity[i-1] * (1.0 + net_pnl)
        
        prev_signal = signal
    
    pnl_net = pnl - cost
    return equity, pnl_net


def fast_vector_backtest(
    price: pd.Series, 
    signal: pd.Series, 
    *,
    commission: float = 0.0008,
    slippage: float = 0.0015,
    leverage: float = 1.0,
    use_cache: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    고속 벡터화 백테스트
    
    주요 최적화:
    1. Numba JIT 컴파일
    2. 결과 캐싱
    3. 벡터화 연산
    4. 메모리 효율성
    """
    
    # 캐시 확인
    if use_cache:
        cache = get_backtest_cache()
        cached_result = cache.get(signal, price, commission, slippage)
        if cached_result is not None:
            return cached_result[0], cached_result[1]  # equity, pnl
    
    # 데이터 전처리
    signal_shifted = signal.shift(1).fillna(0).clip(-1, 1)
    returns = price.pct_change().fillna(0)
    
    # NumPy 배열로 변환 (성능 향상)
    returns_np = returns.values
    signals_np = signal_shifted.values
    
    # 핵심 백테스트 실행 (Numba 최적화)
    equity_np, pnl_net_np = _fast_backtest_core(
        returns_np, signals_np, commission, slippage, leverage
    )
    
    # 결과를 pandas Series로 변환
    equity = pd.Series(equity_np, index=price.index)
    pnl_net = pd.Series(pnl_net_np, index=price.index)
    
    # 캐시에 저장
    if use_cache:
        # 메트릭은 나중에 계산할 때 캐시
        cache.put(signal, price, commission, slippage, equity, pnl_net, {})
    
    return equity, pnl_net


@jit(nopython=True, cache=True)
def _fast_metrics_core(
    equity: np.ndarray,
    pnl: np.ndarray,
    returns: np.ndarray,
    trading_days_per_year: float = 252.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Numba 최적화된 메트릭 계산
    
    Returns:
        cagr, sharpe, mdd, calmar, volatility, total_return
    """
    n = len(equity)
    
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Total Return
    total_return = equity[-1] - 1.0
    
    # CAGR
    years = n / trading_days_per_year
    if years > 0 and equity[-1] > 0:
        cagr = (equity[-1] ** (1.0 / years)) - 1.0
    else:
        cagr = 0.0
    
    # Volatility
    if n > 1:
        volatility = np.std(pnl) * np.sqrt(trading_days_per_year)
    else:
        volatility = 0.0
    
    # Sharpe Ratio
    if volatility > 0:
        sharpe = (np.mean(pnl) * trading_days_per_year) / volatility
    else:
        sharpe = 0.0
    
    # Maximum Drawdown
    peak = equity[0]
    max_dd = 0.0
    
    for i in range(1, n):
        if equity[i] > peak:
            peak = equity[i]
        
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Calmar Ratio
    if max_dd > 0:
        calmar = cagr / max_dd
    else:
        calmar = 0.0
    
    return cagr, sharpe, max_dd, calmar, volatility, total_return


def fast_compute_metrics(
    equity: pd.Series,
    pnl: pd.Series,
    signal: pd.Series,
    price: pd.Series = None,
    trading_days_per_year: float = 252.0,
    use_cache: bool = True
) -> Dict[str, float]:
    """
    고속 메트릭 계산
    
    캐시를 활용하여 동일한 조합의 메트릭 재계산 방지
    """
    
    # 백테스트 캐시에서 메트릭 확인
    if use_cache and price is not None:
        cache = get_backtest_cache()
        # 임시로 commission=0으로 캐시 키 생성 (메트릭용)
        cached_result = cache.get(signal, price, 0.0, 0.0)
        if cached_result is not None and cached_result[2]:  # 메트릭이 있는 경우
            return cached_result[2]
    
    # NumPy 배열로 변환
    equity_np = equity.values
    pnl_np = pnl.values
    
    # 추가 메트릭을 위한 수익률 계산
    if price is not None:
        returns_np = price.pct_change().fillna(0).values
    else:
        returns_np = pnl_np  # PnL을 수익률로 사용
    
    # 핵심 메트릭 계산 (Numba 최적화)
    cagr, sharpe, mdd, calmar, volatility, total_return = _fast_metrics_core(
        equity_np, pnl_np, returns_np, trading_days_per_year
    )
    
    # 추가 메트릭 계산
    metrics = {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'mdd': float(mdd),
        'calmar': float(calmar),
        'volatility': float(volatility),
        'total_return': float(total_return)
    }
    
    # 신호 기반 메트릭
    if len(signal) > 0:
        # Turnover (거래 빈도)
        signal_changes = signal.diff().abs().sum()
        metrics['turnover'] = float(signal_changes / len(signal))
        
        # 포지션 통계
        long_ratio = (signal > 0.1).mean()
        short_ratio = (signal < -0.1).mean()
        neutral_ratio = (abs(signal) <= 0.1).mean()
        
        metrics.update({
            'long_ratio': float(long_ratio),
            'short_ratio': float(short_ratio), 
            'neutral_ratio': float(neutral_ratio)
        })
        
        # Win Rate (양수 PnL 비율)
        if len(pnl) > 0:
            win_rate = (pnl > 0).mean()
            metrics['win_rate'] = float(win_rate)
            
            # Profit Factor
            gains = pnl[pnl > 0].sum()
            losses = abs(pnl[pnl < 0].sum())
            if losses > 0:
                profit_factor = gains / losses
            else:
                profit_factor = float('inf') if gains > 0 else 0
            metrics['profit_factor'] = float(profit_factor)
    
    # 캐시에 메트릭 저장
    if use_cache and price is not None:
        cache = get_backtest_cache()
        cache.put(signal, price, 0.0, 0.0, equity, pnl, metrics)
    
    return metrics


class FastBacktestEngine:
    """
    통합 고속 백테스트 엔진
    
    특징:
    1. 캐시 기반 중복 계산 방지
    2. Numba JIT 최적화
    3. 배치 백테스트 지원
    4. 성능 모니터링
    """
    
    def __init__(self, 
                 commission: float = 0.0008,
                 slippage: float = 0.0015,
                 leverage: float = 1.0,
                 use_cache: bool = True):
        
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.use_cache = use_cache
        
        # 성능 통계
        self.stats = {
            'total_backtests': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        logging.info(f"✅ FastBacktestEngine 초기화 (commission={commission:.4f}, slippage={slippage:.4f})")
    
    def backtest(self, 
                 price: pd.Series, 
                 signal: pd.Series,
                 compute_metrics: bool = True) -> Dict[str, any]:
        """
        단일 백테스트 실행
        
        Returns:
            result dict with equity, pnl, metrics
        """
        start_time = time.time()
        
        # 백테스트 실행
        equity, pnl = fast_vector_backtest(
            price, signal,
            commission=self.commission,
            slippage=self.slippage, 
            leverage=self.leverage,
            use_cache=self.use_cache
        )
        
        result = {
            'equity': equity,
            'pnl': pnl,
            'success': True
        }
        
        # 메트릭 계산
        if compute_metrics:
            metrics = fast_compute_metrics(
                equity, pnl, signal, price,
                use_cache=self.use_cache
            )
            result['metrics'] = metrics
            result['reward'] = metrics['sharpe']  # 기본 보상은 Sharpe Ratio
        
        # 통계 업데이트
        elapsed = time.time() - start_time
        self.stats['total_backtests'] += 1
        self.stats['total_time'] += elapsed
        self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_backtests']
        
        return result
    
    def batch_backtest(self, 
                      price: pd.Series,
                      signals: Dict[str, pd.Series],
                      compute_metrics: bool = True) -> Dict[str, Dict]:
        """
        배치 백테스트 - 여러 신호를 동시 처리
        """
        results = {}
        
        for name, signal in signals.items():
            try:
                result = self.backtest(price, signal, compute_metrics)
                results[name] = result
            except Exception as e:
                logging.error(f"배치 백테스트 실패 ({name}): {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_stats(self) -> Dict[str, any]:
        """엔진 통계 반환"""
        cache_stats = {}
        if self.use_cache:
            cache = get_backtest_cache()
            cache_stats = cache.get_stats()
        
        return {
            **self.stats,
            'cache_stats': cache_stats
        }


# 전역 백테스트 엔진
_global_backtest_engine = None

def get_fast_backtest_engine(**kwargs) -> FastBacktestEngine:
    """전역 고속 백테스트 엔진 반환"""
    global _global_backtest_engine
    if _global_backtest_engine is None:
        _global_backtest_engine = FastBacktestEngine(**kwargs)
    return _global_backtest_engine
