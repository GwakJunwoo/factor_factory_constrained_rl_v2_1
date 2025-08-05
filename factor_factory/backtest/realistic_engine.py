"""
실거래 타당성 검증을 위한 개선된 백테스트 엔진
미래 정보 누출 방지 및 현실적 거래 조건 반영
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def realistic_backtest(
    price: pd.Series, 
    signal: pd.Series, 
    *,
    commission: float = 0.0004,
    slippage: float = 0.0010,
    leverage: int = 1,
    signal_delay: int = 1,  # 신호 발생부터 실제 거래까지 지연
    execution_delay: int = 1,  # 주문부터 체결까지 지연
    max_position_change: float = 1.0,  # 최대 포지션 변경량 (유동성 제약)
    impact_factor: float = 0.0005,  # 대량 거래 시 시장 충격
) -> Tuple[pd.Series, pd.Series]:
    """
    현실적인 거래 조건을 반영한 백테스트
    
    Args:
        price: 가격 시계열 (close price)
        signal: 원시 신호 시계열 (-1 ~ +1)
        commission: 수수료율
        slippage: 슬리피지율
        leverage: 레버리지
        signal_delay: 신호 생성부터 거래 결정까지 지연 (기본 1)
        execution_delay: 거래 결정부터 실제 체결까지 지연 (기본 1)
        max_position_change: 한 번에 변경 가능한 최대 포지션 (유동성 제약)
        impact_factor: 포지션 변경 시 시장 충격 계수
    
    Returns:
        equity: 누적 자산 가치
        pnl: 일일 손익률
    """
    
    # 1. 신호 전처리 및 지연 적용
    raw_signal = signal.fillna(0).clip(-1, 1)
    
    # 신호 발생 지연: t시점 신호는 t+signal_delay에 인식
    delayed_signal = raw_signal.shift(signal_delay).fillna(0)
    
    # 2. 포지션 계산 (유동성 제약 적용)
    target_positions = delayed_signal.copy()
    actual_positions = pd.Series(0.0, index=target_positions.index)
    
    current_pos = 0.0
    for i, target_pos in enumerate(target_positions):
        if i == 0:
            actual_positions.iloc[i] = 0.0  # 초기 포지션 0
            continue
            
        position_change = target_pos - current_pos
        
        # 유동성 제약: 한 번에 변경 가능한 양 제한
        if abs(position_change) > max_position_change:
            position_change = np.sign(position_change) * max_position_change
        
        current_pos += position_change
        current_pos = np.clip(current_pos, -1.0, 1.0)  # 최대 레버리지 제한
        actual_positions.iloc[i] = current_pos
    
    # 3. 거래 체결 지연 적용
    # 실제 포지션은 execution_delay만큼 추가 지연
    executed_positions = actual_positions.shift(execution_delay).fillna(0)
    
    # 4. 수익률 계산 (현실적 타이밍)
    returns = price.pct_change().fillna(0)
    
    # 포지션 변경량 계산
    position_changes = executed_positions.diff().fillna(0)
    
    # 5. 거래 비용 계산
    # 기본 거래 비용
    trading_costs = position_changes.abs() * (commission + slippage)
    
    # 시장 충격 비용 (대량 거래 시 추가 비용)
    market_impact = position_changes.abs() * impact_factor * np.log(1 + position_changes.abs())
    
    # 6. 실제 손익 계산
    # 포지션은 다음 기간 수익률에 영향
    strategy_returns = executed_positions.shift(1).fillna(0) * returns * leverage
    
    # 총 거래 비용
    total_costs = trading_costs + market_impact
    
    # 순 손익
    net_pnl = strategy_returns - total_costs
    
    # 7. 누적 자산 계산
    equity = (1 + net_pnl).cumprod()
    
    return equity, net_pnl


def walk_forward_backtest(
    price: pd.Series,
    signal: pd.Series,
    *,
    commission: float = 0.0004,
    slippage: float = 0.0010,
    leverage: int = 1,
    rebalance_frequency: str = 'D',  # 리밸런싱 빈도
) -> Tuple[pd.Series, pd.Series]:
    """
    워크포워드 방식의 백테스트 (더 현실적)
    매 리밸런싱 시점에서만 포지션 변경 허용
    """
    
    # 리밸런싱 시점 생성
    if rebalance_frequency == 'D':
        rebalance_dates = price.index
    elif rebalance_frequency == 'H':
        rebalance_dates = price.resample('H').last().index
    elif rebalance_frequency == 'W':
        rebalance_dates = price.resample('W').last().index
    else:
        raise ValueError(f"Unsupported rebalance frequency: {rebalance_frequency}")
    
    # 신호를 리밸런싱 시점에만 적용
    rebalanced_signal = signal.reindex(rebalance_dates, method='ffill').fillna(0)
    rebalanced_signal = rebalanced_signal.reindex(price.index, method='ffill').fillna(0)
    
    # 일반 백테스트 실행
    return realistic_backtest(
        price, rebalanced_signal,
        commission=commission,
        slippage=slippage,
        leverage=leverage,
        signal_delay=1,
        execution_delay=1
    )


# 하위 호환성을 위한 기존 함수 유지 (deprecated)
def vector_backtest(price: pd.Series, signal: pd.Series, *, commission:float=0.0004, slippage:float=0.0010, leverage:int=1):
    """
    기존 백테스트 함수 (하위 호환성용)
    ⚠️ 미래 정보 누출 위험 있음. realistic_backtest 사용 권장
    """
    return realistic_backtest(
        price, signal,
        commission=commission,
        slippage=slippage,
        leverage=leverage,
        signal_delay=1,
        execution_delay=0  # 기존 동작 유지
    )
