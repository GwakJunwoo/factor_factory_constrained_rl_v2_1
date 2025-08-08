"""
벡터화 백테스트 엔진

행렬 연산을 활용한 고속 백테스트
다종목 포트폴리오의 성과 평가
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import warnings
from scipy import stats

from .cross_section import CrossSectionNormalizer
from .portfolio_manager import PortfolioManager


class VectorizedBacktest:
    """벡터화된 백테스트 엔진"""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.0008,
                 slippage_rate: float = 0.0015,
                 max_position_pct: float = 0.2,
                 rebalance_frequency: str = 'D'):
        """
        Args:
            initial_capital: 초기 자본금
            commission_rate: 거래 수수료율
            slippage_rate: 슬리피지율
            max_position_pct: 종목별 최대 포지션 비율
            rebalance_frequency: 리밸런싱 빈도 ('D', 'W', 'M')
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_pct = max_position_pct
        self.rebalance_frequency = rebalance_frequency
        
        # 결과 저장
        self.backtest_results = {}
        self.portfolio_history = pd.DataFrame()
        self.trade_history = pd.DataFrame()
        
    def run_backtest(self,
                    factor_matrix: pd.DataFrame,
                    price_matrix: pd.DataFrame,
                    normalizer_method: str = 'z_score',
                    long_threshold: float = 1.0,
                    short_threshold: float = -1.0,
                    market_neutral: bool = True,
                    target_long_ratio: float = 0.3,
                    target_short_ratio: float = 0.3) -> Dict[str, Any]:
        """
        벡터화된 백테스트 실행
        
        Args:
            factor_matrix: [timestamp x symbol] 팩터 값 행렬
            price_matrix: [timestamp x symbol] 가격 행렬
            normalizer_method: 정규화 방법
            long_threshold: 롱 임계값 (임계값 방식 사용시)
            short_threshold: 숏 임계값 (임계값 방식 사용시)
            market_neutral: 마켓 뉴트럴 신호 사용 여부
            target_long_ratio: 롱 포지션 목표 비율 (마켓 뉴트럴시)
            target_short_ratio: 숏 포지션 목표 비율 (마켓 뉴트럴시)
            
        Returns:
            backtest_results: 백테스트 결과 딕셔너리
        """
        print(f"🚀 벡터화 백테스트 시작")
        print(f"  📊 데이터 크기: {factor_matrix.shape}")
        print(f"  📅 기간: {factor_matrix.index[0]} ~ {factor_matrix.index[-1]}")
        print(f"  🎯 정규화 방법: {normalizer_method}")
        print(f"  💰 초기 자본: ${self.initial_capital:,.0f}")
        
        # 1. 크로스 섹션 정규화
        normalizer = CrossSectionNormalizer(method=normalizer_method, min_assets=2)
        normalized_matrix = normalizer.normalize(factor_matrix)
        
        # 2. 신호 생성
        if market_neutral:
            signals_matrix = normalizer.create_market_neutral_signals(
                normalized_matrix, target_long_ratio, target_short_ratio
            )
        else:
            signals_matrix = normalizer.generate_signals(
                normalized_matrix, long_threshold, short_threshold
            )
        
        # 3. 리밸런싱 일정 생성
        rebalance_dates = self._get_rebalance_dates(
            factor_matrix.index, self.rebalance_frequency
        )
        
        # 4. 포트폴리오 시뮬레이션
        portfolio_manager = PortfolioManager(
            initial_capital=self.initial_capital,
            max_position_pct=self.max_position_pct,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        
        portfolio_values = []
        daily_returns = []
        
        print(f"  🔄 포트폴리오 시뮬레이션 시작 ({len(rebalance_dates)}회 리밸런싱)")
        
        for i, timestamp in enumerate(factor_matrix.index):
            current_prices = price_matrix.loc[timestamp]
            
            # 리밸런싱 시점 확인
            if timestamp in rebalance_dates:
                current_signals = signals_matrix.loc[timestamp]
                
                # 변동성 계산 (최근 20일)
                volatilities = self._calculate_volatilities(
                    price_matrix, timestamp, window=20
                )
                
                # 목표 포지션 계산
                target_positions = portfolio_manager.calculate_target_positions(
                    current_signals, current_prices, volatilities
                )
                
                # 리밸런싱 실행
                rebalance_result = portfolio_manager.rebalance(
                    target_positions, current_prices, timestamp
                )
            
            # 포트폴리오 가치 계산
            portfolio_value = portfolio_manager.get_total_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            
            # 일간 수익률 계산
            if i > 0:
                daily_return = (portfolio_value / portfolio_values[i-1]) - 1
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0.0)
            
            # 진행 상황 출력
            if i % 500 == 0:
                print(f"    진행: {i+1}/{len(factor_matrix)} ({(i+1)/len(factor_matrix)*100:.1f}%)")
        
        # 5. 결과 정리
        portfolio_df = pd.DataFrame({
            'timestamp': factor_matrix.index,
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns
        }).set_index('timestamp')
        
        # 6. 성과 지표 계산
        performance_metrics = self._calculate_performance_metrics(
            portfolio_df, signals_matrix, normalizer
        )
        
        # 7. 벤치마크 비교 (균등 가중 포트폴리오)
        benchmark_metrics = self._calculate_benchmark_performance(price_matrix)
        
        # 결과 저장
        self.portfolio_history = portfolio_df
        self.trade_history = pd.DataFrame(portfolio_manager.trade_history)
        
        results = {
            'performance_metrics': performance_metrics,
            'benchmark_metrics': benchmark_metrics,
            'portfolio_history': portfolio_df,
            'signals_matrix': signals_matrix,
            'normalized_matrix': normalized_matrix,
            'trade_history': self.trade_history,
            'portfolio_manager_stats': portfolio_manager.get_portfolio_statistics(),
            'signal_stats': normalizer.get_signal_statistics(signals_matrix)
        }
        
        self.backtest_results = results
        
        print(f"✅ 백테스트 완료!")
        print(f"  📈 총 수익률: {performance_metrics['total_return']:.2%}")
        print(f"  📊 샤프 비율: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"  📉 최대 낙폭: {performance_metrics['max_drawdown']:.2%}")
        
        return results
    
    def _get_rebalance_dates(self, 
                           date_index: pd.DatetimeIndex, 
                           frequency: str) -> List[pd.Timestamp]:
        """리밸런싱 일정 생성"""
        if frequency == 'D':
            return date_index.tolist()
        elif frequency == 'W':
            return date_index[date_index.dayofweek == 0].tolist()  # 월요일
        elif frequency == 'M':
            return date_index[date_index.is_month_end].tolist()
        else:
            return date_index.tolist()
    
    def _calculate_volatilities(self,
                              price_matrix: pd.DataFrame,
                              current_timestamp: pd.Timestamp,
                              window: int = 20) -> pd.Series:
        """변동성 계산"""
        # 현재 시점까지의 데이터
        current_idx = price_matrix.index.get_loc(current_timestamp)
        start_idx = max(0, current_idx - window + 1)
        
        price_window = price_matrix.iloc[start_idx:current_idx+1]
        returns = price_window.pct_change().dropna()
        
        if len(returns) < 2:
            return pd.Series(0.1, index=price_matrix.columns)  # 기본값
        
        volatilities = returns.std() * np.sqrt(252)  # 연환산
        return volatilities.fillna(0.1)
    
    def _calculate_performance_metrics(self,
                                     portfolio_df: pd.DataFrame,
                                     signals_matrix: pd.DataFrame,
                                     normalizer: CrossSectionNormalizer) -> Dict[str, float]:
        """성과 지표 계산"""
        returns = portfolio_df['daily_return']
        portfolio_values = portfolio_df['portfolio_value']
        
        # 기본 수익률 지표
        total_return = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 기타 지표
        win_rate = (returns > 0).mean()
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 신호 관련 지표
        signal_stats = normalizer.get_signal_statistics(signals_matrix)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'signal_turnover': signal_stats.get('signal_turnover', 0),
            'long_ratio': signal_stats.get('long_ratio', 0),
            'short_ratio': signal_stats.get('short_ratio', 0)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """소르티노 비율 계산"""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    def _calculate_benchmark_performance(self, price_matrix: pd.DataFrame) -> Dict[str, float]:
        """벤치마크 (균등 가중) 성과 계산"""
        # 균등 가중 포트폴리오 수익률
        daily_returns = price_matrix.pct_change().mean(axis=1)
        
        # 누적 수익률
        cumulative_returns = (1 + daily_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # 연환산 수익률
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # 변동성
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 샤프 비율
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """백테스트 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if self.portfolio_history.empty:
                print("❌ 백테스트 결과가 없습니다.")
                return
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Multi-Asset Factor Model Backtest Results', fontsize=16)
            
            # 1. 포트폴리오 가치 추이
            axes[0, 0].plot(self.portfolio_history.index, self.portfolio_history['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value')
            axes[0, 0].set_ylabel('Value ($)')
            axes[0, 0].grid(True)
            
            # 2. 일간 수익률
            axes[0, 1].plot(self.portfolio_history.index, self.portfolio_history['daily_return'])
            axes[0, 1].set_title('Daily Returns')
            axes[0, 1].set_ylabel('Return')
            axes[0, 1].grid(True)
            
            # 3. 드로우다운
            portfolio_values = self.portfolio_history['portfolio_value']
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            
            axes[1, 0].fill_between(self.portfolio_history.index, drawdown, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_ylabel('Drawdown')
            axes[1, 0].grid(True)
            
            # 4. 월별 수익률 히트맵
            monthly_returns = self.portfolio_history['daily_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            if len(monthly_returns) > 0:
                axes[1, 1].hist(monthly_returns, bins=20, alpha=0.7)
                axes[1, 1].set_title('Monthly Returns Distribution')
                axes[1, 1].set_xlabel('Monthly Return')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True)
            
            # 5. 거래 통계
            if not self.trade_history.empty:
                trade_pnl = self.trade_history.groupby('timestamp')['trade_value'].sum()
                axes[2, 0].bar(trade_pnl.index, trade_pnl.values, alpha=0.7)
                axes[2, 0].set_title('Daily Trading Volume')
                axes[2, 0].set_ylabel('Trade Value ($)')
                axes[2, 0].grid(True)
            
            # 6. 성과 지표 테이블
            if 'performance_metrics' in self.backtest_results:
                metrics = self.backtest_results['performance_metrics']
                
                metrics_text = f"""
Total Return: {metrics['total_return']:.2%}
Annual Return: {metrics['annualized_return']:.2%}
Volatility: {metrics['volatility']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Win Rate: {metrics['win_rate']:.2%}
                """.strip()
                
                axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                               verticalalignment='center', fontfamily='monospace')
                axes[2, 1].set_title('Performance Metrics')
                axes[2, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 차트 저장: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("❌ matplotlib이 설치되지 않아 차트를 생성할 수 없습니다.")
        except Exception as e:
            print(f"❌ 차트 생성 오류: {e}")
    
    def save_results(self, save_dir: str):
        """백테스트 결과 저장"""
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # CSV 파일들 저장
        self.portfolio_history.to_csv(save_path / 'portfolio_history.csv')
        
        if not self.trade_history.empty:
            self.trade_history.to_csv(save_path / 'trade_history.csv')
        
        if 'signals_matrix' in self.backtest_results:
            self.backtest_results['signals_matrix'].to_csv(save_path / 'signals_matrix.csv')
        
        # 성과 지표 저장
        if 'performance_metrics' in self.backtest_results:
            metrics_df = pd.DataFrame([self.backtest_results['performance_metrics']])
            metrics_df.to_csv(save_path / 'performance_metrics.csv', index=False)
        
        print(f"💾 백테스트 결과 저장 완료: {save_path}")
