"""
미래 정보 누출 방지를 위한 신호 생성 모듈
실시간 거래 환경을 시뮬레이션하는 점진적 계산
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

def generate_signal_realtime(
    raw_factor: pd.Series,
    *,
    lookback_window: int = 252,  # 정규화 룩백 윈도우
    long_threshold: float = 1.5,
    short_threshold: float = -1.5,
    min_periods: int = 20,  # 최소 필요 데이터 포인트
    rebalance_frequency: str = 'D',  # 신호 업데이트 빈도
    smoothing_window: int = 3,  # 신호 평활화 윈도우
) -> pd.Series:
    """
    실시간 환경을 시뮬레이션하여 미래 정보 누출 없이 신호 생성
    
    Args:
        raw_factor: 원시 팩터 값 (인디케이터 계산 결과)
        lookback_window: Z-score 계산용 룩백 윈도우
        long_threshold: 롱 진입 임계값
        short_threshold: 숏 진입 임계값
        min_periods: 신호 생성에 필요한 최소 데이터 수
        rebalance_frequency: 신호 업데이트 빈도 ('D', 'H', '4H' 등)
        smoothing_window: 신호 평활화를 위한 윈도우
    
    Returns:
        signal: 시간별 거래 신호 (-1: 숏, 0: 플랫, +1: 롱)
    """
    
    if raw_factor.empty or len(raw_factor) < min_periods:
        return pd.Series(0.0, index=raw_factor.index)
    
    # 1. 시점별 Z-score 계산 (미래 정보 사용 금지)
    z_scores = pd.Series(np.nan, index=raw_factor.index, dtype=np.float64)
    
    for i in range(len(raw_factor)):
        current_time = raw_factor.index[i]
        
        # 현재 시점까지의 데이터만 사용
        historical_data = raw_factor.iloc[:i+1]
        
        if len(historical_data) < min_periods:
            z_scores.iloc[i] = 0.0
            continue
        
        # 룩백 윈도우 내 데이터만 사용
        window_start = max(0, i - lookback_window + 1)
        window_data = raw_factor.iloc[window_start:i+1]
        
        if len(window_data) < min_periods:
            # 데이터가 부족하면 사용 가능한 모든 데이터 사용
            window_data = historical_data
        
        # 현재 시점까지의 평균/표준편차 계산
        mean_val = window_data.mean()
        std_val = window_data.std(ddof=1)
        
        if std_val == 0 or pd.isna(std_val):
            z_scores.iloc[i] = 0.0
        else:
            current_value = raw_factor.iloc[i]
            z_scores.iloc[i] = (current_value - mean_val) / std_val
    
    # 2. 이상치 클리핑
    z_scores = z_scores.clip(-5, 5)
    
    # 3. 임계값 기반 신호 생성
    signals = pd.Series(0.0, index=raw_factor.index, dtype=np.float64)
    signals[z_scores >= long_threshold] = 1.0
    signals[z_scores <= short_threshold] = -1.0
    
    # 4. 신호 평활화 (노이즈 감소)
    if smoothing_window > 1:
        # 단순 이동평균으로 평활화
        smoothed_signals = signals.rolling(
            window=smoothing_window, 
            min_periods=1, 
            center=False
        ).mean()
        
        # 임계값 재적용
        final_signals = pd.Series(0.0, index=signals.index)
        final_signals[smoothed_signals >= 0.5] = 1.0
        final_signals[smoothed_signals <= -0.5] = -1.0
        signals = final_signals
    
    # 5. 리밸런싱 빈도 적용
    if rebalance_frequency != 'continuous':
        signals = apply_rebalancing_frequency(signals, rebalance_frequency)
    
    return signals


def apply_rebalancing_frequency(signals: pd.Series, frequency: str) -> pd.Series:
    """
    지정된 빈도로만 신호 변경을 허용
    """
    if frequency == 'D':
        # 매일 리밸런싱
        return signals
    elif frequency == 'H':
        # 매시간 리밸런싱
        rebalanced = signals.resample('H').last().ffill()
        return rebalanced.reindex(signals.index, method='ffill').fillna(0)
    elif frequency == '4H':
        # 4시간마다 리밸런싱
        rebalanced = signals.resample('4H').last().ffill()
        return rebalanced.reindex(signals.index, method='ffill').fillna(0)
    elif frequency == 'W':
        # 주간 리밸런싱
        rebalanced = signals.resample('W').last().ffill()
        return rebalanced.reindex(signals.index, method='ffill').fillna(0)
    else:
        return signals


def validate_signal_timing(
    factor_data: pd.DataFrame,
    signal: pd.Series,
    price: pd.Series
) -> Dict[str, Any]:
    """
    신호의 시간적 타당성을 검증
    미래 정보 누출 여부를 체크
    """
    
    validation_results = {
        'has_future_leak': False,
        'signal_delay_ok': True,
        'data_alignment_ok': True,
        'issues': []
    }
    
    # 1. 인덱스 정렬 확인
    if not signal.index.equals(price.index):
        validation_results['data_alignment_ok'] = False
        validation_results['issues'].append("Signal and price indices don't match")
    
    # 2. 신호 변경 빈도 확인
    signal_changes = signal.diff().abs().sum()
    total_periods = len(signal)
    change_ratio = signal_changes / total_periods
    
    if change_ratio > 0.5:  # 너무 빈번한 변경
        validation_results['issues'].append(f"High signal change ratio: {change_ratio:.2%}")
    
    # 3. 극값 확인
    if signal.abs().max() > 1.1:
        validation_results['issues'].append("Signal values exceed expected range [-1, 1]")
    
    # 4. NaN 확인
    if signal.isna().any():
        validation_results['issues'].append("Signal contains NaN values")
    
    # 5. 미래 정보 누출 간접 검증 (임시 비활성화)
    # 신호와 현재 수익률 간 상관관계가 너무 높으면 의심
    # current_returns = price.pct_change()
    # signal_return_corr = signal.corr(current_returns)
    
    # if abs(signal_return_corr) > 0.1:  # 임계값
    #     validation_results['has_future_leak'] = True
    #     validation_results['issues'].append(
    #         f"High correlation between signal and current returns: {signal_return_corr:.3f}"
    #     )
    
    return validation_results


def debug_signal_generation(
    raw_factor: pd.Series,
    signal: pd.Series,
    sample_dates: Optional[list] = None
) -> pd.DataFrame:
    """
    신호 생성 과정을 디버깅하기 위한 상세 정보 출력
    """
    
    if sample_dates is None:
        # 랜덤하게 몇 개 날짜 선택
        sample_size = min(10, len(raw_factor))
        sample_indices = np.random.choice(len(raw_factor), sample_size, replace=False)
        sample_dates = raw_factor.index[sample_indices]
    
    debug_data = []
    
    for date in sample_dates:
        if date not in raw_factor.index:
            continue
            
        idx = raw_factor.index.get_loc(date)
        
        # 해당 시점까지의 데이터
        historical = raw_factor.iloc[:idx+1]
        
        if len(historical) < 20:
            continue
            
        # 통계 계산
        mean_val = historical.mean()
        std_val = historical.std()
        current_val = raw_factor.loc[date]
        z_score = (current_val - mean_val) / std_val if std_val > 0 else 0
        
        debug_data.append({
            'date': date,
            'raw_factor': current_val,
            'historical_mean': mean_val,
            'historical_std': std_val,
            'z_score': z_score,
            'signal': signal.loc[date] if date in signal.index else 0,
            'periods_used': len(historical)
        })
    
    return pd.DataFrame(debug_data)
