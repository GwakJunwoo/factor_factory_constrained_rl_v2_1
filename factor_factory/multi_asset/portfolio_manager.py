"""
포트폴리오 관리자

실제 계좌 기반 포지션 관리 및 리밸런싱
종목별 가격 차이를 고려한 포지션 크기 계산
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import warnings


class PortfolioManager:
    """실제 계좌 기반 포지션 관리 클래스"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_pct: float = 0.2,
                 commission_rate: float = 0.0008,
                 slippage_rate: float = 0.0015):
        """
        Args:
            initial_capital: 초기 자본금
            max_position_pct: 종목별 최대 포지션 비율
            commission_rate: 거래 수수료율
            slippage_rate: 슬리피지율
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 포지션 관리
        self.positions: Dict[str, float] = {}  # {symbol: position_value}
        self.shares: Dict[str, float] = {}     # {symbol: share_count}
        self.weights: Dict[str, float] = {}    # {symbol: weight}
        
        # 거래 기록
        self.trade_history = []
        self.portfolio_history = []
        
        # 통계
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        
    def calculate_target_positions(self, 
                                 signals: pd.Series,
                                 prices: pd.Series,
                                 volatilities: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        신호와 가격을 고려한 목표 포지션 계산
        
        Args:
            signals: 정규화된 신호 (symbol → signal_strength)
            prices: 현재 가격 (symbol → price)
            volatilities: 변동성 (symbol → volatility, 선택사항)
            
        Returns:
            target_positions: {symbol: target_value} (달러 기준)
        """
        target_positions = {}
        
        # 유효한 신호만 선택
        valid_signals = signals.dropna()
        valid_signals = valid_signals[valid_signals != 0]
        
        if len(valid_signals) == 0:
            return {symbol: 0.0 for symbol in signals.index}
        
        # 총 신호 강도 계산
        total_signal_strength = valid_signals.abs().sum()
        
        if total_signal_strength == 0:
            return {symbol: 0.0 for symbol in signals.index}
        
        # 변동성 조정 (있는 경우)
        if volatilities is not None:
            # 변동성 역수로 가중치 조정
            vol_weights = 1.0 / (volatilities.fillna(volatilities.mean()) + 1e-8)
            adjusted_signals = valid_signals * vol_weights[valid_signals.index]
        else:
            adjusted_signals = valid_signals
        
        # 포지션 크기 계산
        for symbol in signals.index:
            if symbol not in adjusted_signals.index:
                target_positions[symbol] = 0.0
                continue
                
            signal_strength = adjusted_signals[symbol]
            price = prices[symbol]
            
            if pd.isna(price) or price <= 0:
                target_positions[symbol] = 0.0
                continue
            
            # 신호 강도에 비례한 포지션 비율
            position_ratio = signal_strength / total_signal_strength
            
            # 최대 포지션 제한 적용
            position_ratio = np.clip(position_ratio, -self.max_position_pct, self.max_position_pct)
            
            # 달러 기준 포지션 크기
            target_value = position_ratio * self.current_capital
            target_positions[symbol] = target_value
        
        return target_positions
    
    def rebalance(self, 
                  target_positions: Dict[str, float],
                  current_prices: pd.Series,
                  timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        포트폴리오 리밸런싱 실행
        
        Args:
            target_positions: 목표 포지션 {symbol: target_value}
            current_prices: 현재 가격
            timestamp: 현재 시점
            
        Returns:
            rebalance_result: 리밸런싱 결과 딕셔너리
        """
        trades = []
        total_commission = 0.0
        total_slippage = 0.0
        
        # 현재 포지션 업데이트 (가격 변화 반영)
        self._update_current_positions(current_prices)
        
        for symbol, target_value in target_positions.items():
            current_value = self.positions.get(symbol, 0.0)
            current_price = current_prices.get(symbol, np.nan)
            
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # 거래 필요성 확인
            trade_value = target_value - current_value
            
            # 최소 거래 금액 필터 (거래 비용 고려)
            min_trade_value = self.current_capital * 0.001  # 0.1%
            if abs(trade_value) < min_trade_value:
                continue
            
            # 거래 실행
            trade_result = self._execute_trade(
                symbol, trade_value, current_price, timestamp
            )
            
            if trade_result:
                trades.append(trade_result)
                total_commission += trade_result['commission']
                total_slippage += trade_result['slippage']
        
        # 포트폴리오 상태 업데이트
        self._update_portfolio_state()
        
        # 포트폴리오 기록
        portfolio_state = {
            'timestamp': timestamp,
            'total_value': self.get_total_portfolio_value(current_prices),
            'cash': self.current_capital,
            'positions': self.positions.copy(),
            'weights': self.weights.copy(),
            'num_positions': len([p for p in self.positions.values() if abs(p) > 0])
        }
        self.portfolio_history.append(portfolio_state)
        
        return {
            'trades': trades,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'portfolio_state': portfolio_state
        }
    
    def _execute_trade(self, 
                      symbol: str, 
                      trade_value: float, 
                      price: float, 
                      timestamp: pd.Timestamp) -> Optional[Dict]:
        """개별 거래 실행"""
        
        # 거래 수수료 계산
        commission = abs(trade_value) * self.commission_rate
        
        # 슬리피지 계산
        slippage = abs(trade_value) * self.slippage_rate
        
        # 총 거래 비용
        total_cost = commission + slippage
        
        # 잔고 확인 (매수의 경우)
        if trade_value > 0:  # 매수
            required_cash = trade_value + total_cost
            if self.current_capital < required_cash:
                # 현금 부족 시 가능한 만큼만 매수
                available_cash = self.current_capital * 0.95  # 5% 여유
                trade_value = max(0, available_cash - total_cost)
                if trade_value < abs(trade_value) * 0.1:  # 너무 작으면 취소
                    return None
        
        # 거래 실행
        shares_traded = trade_value / price
        execution_price = price * (1 + np.sign(trade_value) * self.slippage_rate)
        
        # 포지션 업데이트
        self.shares[symbol] = self.shares.get(symbol, 0.0) + shares_traded
        self.positions[symbol] = self.shares[symbol] * price
        
        # 현금 업데이트
        self.current_capital -= (trade_value + total_cost)
        
        # 비용 누계
        self.total_commission_paid += commission
        self.total_slippage_cost += slippage
        
        # 거래 기록
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'trade_value': trade_value,
            'shares_traded': shares_traded,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'slippage': slippage,
            'total_cost': total_cost
        }
        
        self.trade_history.append(trade_record)
        return trade_record
    
    def _update_current_positions(self, current_prices: pd.Series):
        """현재 가격을 반영하여 포지션 가치 업데이트"""
        for symbol in self.shares:
            if symbol in current_prices:
                price = current_prices[symbol]
                if not pd.isna(price) and price > 0:
                    self.positions[symbol] = self.shares[symbol] * price
    
    def _update_portfolio_state(self):
        """포트폴리오 가중치 계산"""
        total_position_value = sum(abs(v) for v in self.positions.values())
        
        if total_position_value > 0:
            self.weights = {
                symbol: position / total_position_value 
                for symbol, position in self.positions.items()
            }
        else:
            self.weights = {}
    
    def get_total_portfolio_value(self, current_prices: pd.Series) -> float:
        """총 포트폴리오 가치 계산"""
        self._update_current_positions(current_prices)
        total_position_value = sum(self.positions.values())
        return self.current_capital + total_position_value
    
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """포트폴리오 통계 계산"""
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # 수익률 계산
        portfolio_df['portfolio_return'] = portfolio_df['total_value'].pct_change()
        
        stats = {
            'total_return': (portfolio_df['total_value'].iloc[-1] / self.initial_capital) - 1,
            'total_value': portfolio_df['total_value'].iloc[-1],
            'max_drawdown': self._calculate_max_drawdown(portfolio_df['total_value']),
            'volatility': portfolio_df['portfolio_return'].std() * np.sqrt(252),  # 연환산
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_df['portfolio_return']),
            'total_trades': len(self.trade_history),
            'total_commission': self.total_commission_paid,
            'total_slippage': self.total_slippage_cost,
            'avg_num_positions': portfolio_df['num_positions'].mean(),
            'current_positions': len([p for p in self.positions.values() if abs(p) > 0])
        }
        
        return stats
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """최대 낙폭 계산"""
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - (risk_free_rate / 252)  # 일간 무위험 수익률
        if returns.std() == 0:
            return 0.0
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def reset(self):
        """포트폴리오 상태 초기화"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.shares = {}
        self.weights = {}
        self.trade_history = []
        self.portfolio_history = []
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
