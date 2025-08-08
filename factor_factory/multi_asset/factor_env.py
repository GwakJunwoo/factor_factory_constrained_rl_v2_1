"""
다종목 MCTS 환경

크로스 섹션 정규화를 활용한 다종목 팩터 발견 환경
기존 단일 종목 환경을 확장하여 진정한 팩터 모델 구현
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..rlc import RLCConfig
from ..multi_asset import (
    MultiAssetDataManager, 
    CrossSectionNormalizer, 
    PortfolioManager,
    VectorizedBacktest
)


class MultiAssetFactorEnv:
    """다종목 팩터 발견을 위한 강화학습 환경"""
    
    def __init__(self, 
                 symbols: List[str],
                 data_dict: Dict[str, pd.DataFrame],
                 config: RLCConfig,
                 normalizer_method: str = 'z_score',
                 market_neutral: bool = True,
                 target_long_ratio: float = 0.3,
                 target_short_ratio: float = 0.3):
        """
        Args:
            symbols: 종목 리스트
            data_dict: 종목별 데이터 딕셔너리
            config: RLC 설정
            normalizer_method: 크로스 섹션 정규화 방법
            market_neutral: 마켓 뉴트럴 전략 사용 여부
            target_long_ratio: 롱 포지션 목표 비율
            target_short_ratio: 숏 포지션 목표 비율
        """
        self.symbols = symbols
        self.config = config
        self.normalizer_method = normalizer_method
        self.market_neutral = market_neutral
        self.target_long_ratio = target_long_ratio
        self.target_short_ratio = target_short_ratio
        
        # 다종목 데이터 매니저 초기화
        self.data_manager = MultiAssetDataManager(symbols, interval='1h')
        self.data_manager.data_dict = data_dict
        self.data_manager.align_data(method='inner')
        
        # 가격 행렬 준비
        self.price_matrix = self.data_manager.get_price_matrix('close')
        
        # 환경 상태
        self.current_step = 0
        self.max_steps = len(self.price_matrix) - config.eval_stride
        self.episode_stats = []
        
        # 성과 추적
        self.best_performance = -np.inf
        self.episode_count = 0
        
        print(f"🏭 다종목 팩터 환경 초기화")
        print(f"  📊 종목 수: {len(symbols)}")
        print(f"  📅 데이터 기간: {self.price_matrix.index[0]} ~ {self.price_matrix.index[-1]}")
        print(f"  🎯 정규화 방법: {normalizer_method}")
        print(f"  💰 마켓 뉴트럴: {market_neutral}")
        
    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.current_step = 0
        self.episode_count += 1
        
        # 시작 지점 랜덤화 (충분한 과거 데이터 확보)
        min_start = max(252, self.config.rolling_window)  # 최소 1년
        max_start = self.max_steps - 1000  # 최소 1000스텝 남김
        
        if max_start > min_start:
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = min_start
        
        return self._get_observation()
    
    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행"""
        try:
            # 팩터 계산
            factor_matrix = self.data_manager.calculate_factor_matrix(action)
            
            # 평가 구간 설정
            eval_start = self.current_step
            eval_end = min(self.current_step + self.config.eval_stride, len(self.price_matrix))
            
            eval_factor_matrix = factor_matrix.iloc[eval_start:eval_end]
            eval_price_matrix = self.price_matrix.iloc[eval_start:eval_end]
            
            # 백테스트 실행
            backtest = VectorizedBacktest(
                initial_capital=100000,
                commission_rate=self.config.commission,
                slippage_rate=self.config.slippage,
                max_position_pct=0.2,
                rebalance_frequency='D'
            )
            
            results = backtest.run_backtest(
                factor_matrix=eval_factor_matrix,
                price_matrix=eval_price_matrix,
                normalizer_method=self.normalizer_method,
                market_neutral=self.market_neutral,
                target_long_ratio=self.target_long_ratio,
                target_short_ratio=self.target_short_ratio
            )
            
            # 보상 계산
            reward = self._calculate_reward(results)
            
            # 다음 스텝으로 이동
            self.current_step += self.config.eval_stride
            done = self.current_step >= self.max_steps
            
            # 통계 저장
            episode_stat = {
                'step': self.current_step,
                'reward': reward,
                'total_return': results['performance_metrics']['total_return'],
                'sharpe_ratio': results['performance_metrics']['sharpe_ratio'],
                'max_drawdown': results['performance_metrics']['max_drawdown'],
                'signal_turnover': results['performance_metrics']['signal_turnover'],
                'long_ratio': results['performance_metrics']['long_ratio'],
                'short_ratio': results['performance_metrics']['short_ratio']
            }
            self.episode_stats.append(episode_stat)
            
            # 최고 성과 업데이트
            if reward > self.best_performance:
                self.best_performance = reward
            
            info = {
                'performance_metrics': results['performance_metrics'],
                'signal_stats': results['signal_stats'],
                'episode_stat': episode_stat
            }
            
            return self._get_observation(), reward, done, info
            
        except Exception as e:
            # 오류 발생시 페널티
            warning_msg = f"환경 스텝 오류: {str(e)}"
            warnings.warn(warning_msg)
            
            reward = -5.0
            done = True
            info = {'error': str(e)}
            
            return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """현재 관측값 반환"""
        if self.current_step >= len(self.price_matrix):
            # 범위 초과시 마지막 유효한 값 사용
            self.current_step = len(self.price_matrix) - 1
        
        current_timestamp = self.price_matrix.index[self.current_step]
        
        # 모든 종목의 기술적 지표를 concat하여 observation 생성
        observations = []
        
        for symbol in self.symbols:
            symbol_data = self.data_manager.get_symbol_data(symbol)
            
            # 현재 시점까지의 데이터
            current_data = symbol_data.loc[:current_timestamp]
            
            if len(current_data) < 50:  # 최소 데이터 요구사항
                # 데이터 부족시 기본값
                obs = np.zeros(23, dtype=np.float32)
            else:
                # 기존 단일 종목 환경과 동일한 방식으로 observation 생성
                obs = self._extract_single_asset_observation(current_data)
            
            observations.append(obs)
        
        # 모든 종목의 관측값을 concat
        combined_obs = np.concatenate(observations)
        
        # 크기 조정 (기존 MCTS 네트워크와 호환성 유지)
        if len(combined_obs) > 23:
            # 너무 큰 경우 평균값 사용
            combined_obs = combined_obs.reshape(-1, 23).mean(axis=0)
        elif len(combined_obs) < 23:
            # 부족한 경우 패딩
            padding = np.zeros(23 - len(combined_obs), dtype=np.float32)
            combined_obs = np.concatenate([combined_obs, padding])
        
        return combined_obs.astype(np.float32)
    
    def _extract_single_asset_observation(self, data: pd.DataFrame) -> np.ndarray:
        """단일 종목의 관측값 추출 (기존 방식과 동일)"""
        try:
            # 최근 값들 정규화
            close = data['close'].iloc[-1]
            volume = data['volume'].iloc[-1]
            
            # 기술적 지표들 (최근 20일 기준으로 정규화)
            recent_data = data.tail(20)
            
            price_features = [
                (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else 0,  # 20일 수익률
                (data['high'].iloc[-1] / data['close'].iloc[-1] - 1),  # 고가 대비
                (data['low'].iloc[-1] / data['close'].iloc[-1] - 1),   # 저가 대비
                np.log(volume + 1e-8),  # 로그 거래량
            ]
            
            # 이동평균 관련
            sma_5 = recent_data['close'].rolling(5).mean().iloc[-1] if len(recent_data) >= 5 else close
            sma_10 = recent_data['close'].rolling(10).mean().iloc[-1] if len(recent_data) >= 10 else close
            sma_20 = recent_data['close'].rolling(20).mean().iloc[-1] if len(recent_data) >= 20 else close
            
            ma_features = [
                (close / sma_5 - 1) if sma_5 > 0 else 0,
                (close / sma_10 - 1) if sma_10 > 0 else 0,
                (close / sma_20 - 1) if sma_20 > 0 else 0,
                (sma_5 / sma_20 - 1) if sma_20 > 0 else 0,
            ]
            
            # 변동성 관련
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 1 else 0.01
            
            vol_features = [
                volatility,
                returns.iloc[-1] if len(returns) > 0 else 0,  # 최근 수익률
                returns.iloc[-5:].mean() if len(returns) >= 5 else 0,  # 5일 평균 수익률
            ]
            
            # RSI 계산
            rsi = self._calculate_simple_rsi(recent_data['close'])
            
            # 기타 기술적 지표
            tech_features = [
                (rsi - 50) / 50,  # RSI 정규화
                0.0,  # 예비
                0.0,  # 예비
            ]
            
            # 시장 상태 (트렌드, 변동성 체계)
            market_features = [
                np.tanh(price_features[0]),  # 트렌드 강도
                np.tanh(volatility * 10),    # 변동성 수준
                0.0,  # 예비
                0.0,  # 예비
                0.0,  # 예비
                0.0,  # 예비
                0.0,  # 예비
                0.0,  # 예비
                0.0   # 예비
            ]
            
            # 모든 특성 결합
            all_features = price_features + ma_features + vol_features + tech_features + market_features
            
            # 23차원으로 맞춤
            if len(all_features) > 23:
                all_features = all_features[:23]
            elif len(all_features) < 23:
                all_features.extend([0.0] * (23 - len(all_features)))
            
            # 값 정규화 및 클리핑
            obs = np.array(all_features, dtype=np.float32)
            obs = np.clip(obs, -10, 10)
            obs = np.nan_to_num(obs, 0.0)
            
            return obs
            
        except Exception as e:
            # 오류시 기본값 반환
            return np.zeros(23, dtype=np.float32)
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """간단한 RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            diff = prices.diff().dropna()
            gain = diff.where(diff > 0, 0).rolling(period).mean()
            loss = (-diff.where(diff < 0, 0)).rolling(period).mean()
            
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
            
        except:
            return 50.0
    
    def _calculate_reward(self, backtest_results: Dict, signal_stats: Dict = None) -> float:
        """백테스트 결과를 바탕으로 보상 계산"""
        if signal_stats is None:
            signal_stats = backtest_results.get('signal_stats', {})
            
        metrics = backtest_results.get('performance_metrics', {})
        
        # 기본 수익률 보상
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # 다종목 특성 고려 보상 구성
        return_reward = total_return * 10  # 수익률 기반
        risk_reward = sharpe_ratio * 2     # 위험 조정 수익률
        drawdown_penalty = abs(max_drawdown) * 5  # 낙폭 페널티
        
        # 신호 품질 보상
        turnover_penalty = signal_stats.get('signal_turnover', 0) * 2  # 과도한 거래 페널티
        
        # 마켓 뉴트럴 균형 보상
        long_ratio = signal_stats.get('long_ratio', 0)
        short_ratio = signal_stats.get('short_ratio', 0)
        
        if self.market_neutral:
            # 롱숏 균형 보상
            balance_score = 1.0 - abs(long_ratio - short_ratio)
            balance_reward = balance_score * 1.0
        else:
            balance_reward = 0.0
        
        # 총 보상 계산
        total_reward = (
            return_reward + 
            risk_reward - 
            drawdown_penalty - 
            turnover_penalty + 
            balance_reward
        )
        
        # 클리핑
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return float(total_reward)
    
    def get_statistics(self) -> Dict[str, Any]:
        """환경 통계 반환"""
        if not self.episode_stats:
            return {}
        
        stats_df = pd.DataFrame(self.episode_stats)
        
        return {
            'episode_count': self.episode_count,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'best_performance': self.best_performance,
            'avg_reward': stats_df['reward'].mean(),
            'avg_return': stats_df['total_return'].mean(),
            'avg_sharpe': stats_df['sharpe_ratio'].mean(),
            'avg_drawdown': stats_df['max_drawdown'].mean(),
            'avg_turnover': stats_df['signal_turnover'].mean(),
            'symbols': self.symbols,
            'data_shape': self.price_matrix.shape
        }
    
    def evaluate_program(self, tokens: List[int]) -> Dict[str, Any]:
        """
        프로그램(토큰 시퀀스) 평가 - 유효하지 않은 프로그램도 결과 저장
        
        Args:
            tokens: 평가할 프로그램 토큰
            
        Returns:
            평가 결과 딕셔너리
        """
        try:
            # 팩터 계산 시도
            factor_matrix = self.data_manager.calculate_factor_matrix(tokens)
            
            if factor_matrix is None or factor_matrix.shape[0] == 0:
                # 유효하지 않은 프로그램도 결과로 저장
                from ..rlc.utils import tokens_to_infix
                formula = tokens_to_infix(tokens) if tokens else "Empty"
                
                return {
                    'success': False,
                    'reward': -1.0,
                    'error': 'Invalid factor matrix',
                    'tokens': tokens,
                    'formula': formula,
                    'metrics': {
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'volatility': 0.0
                    },
                    'signal_stats': {
                        'signal_turnover': 0.0,
                        'long_ratio': 0.0,
                        'short_ratio': 0.0
                    }
                }
            
            # 백테스트 실행
            backtest = VectorizedBacktest(
                initial_capital=100000,
                commission_rate=self.config.commission,
                slippage_rate=self.config.slippage,
                max_position_pct=0.2,
                rebalance_frequency='D'
            )
            
            results = backtest.run_backtest(
                factor_matrix=factor_matrix,
                price_matrix=self.price_matrix,
                normalizer_method=self.normalizer_method,
                market_neutral=self.market_neutral,
                target_long_ratio=self.target_long_ratio,
                target_short_ratio=self.target_short_ratio
            )
            
            # 성능 지표 추출
            metrics = results['performance_metrics']
            
            # 보상 계산
            reward = self._calculate_reward(results)
            
            # 최고 성능 업데이트
            if reward > self.best_performance:
                self.best_performance = reward
                self.best_strategy = {
                    'program': tokens,
                    'metrics': metrics,
                    'reward': reward
                }
            
            from ..rlc.utils import tokens_to_infix
            formula = tokens_to_infix(tokens) if tokens else "Empty"
            
            return {
                'success': True,
                'reward': reward,
                'metrics': metrics,
                'signal_stats': results.get('signal_stats', {}),
                'tokens': tokens,
                'formula': formula
            }
            
        except Exception as e:
            # 오류 발생시에도 결과 저장
            from ..rlc.utils import tokens_to_infix
            formula = tokens_to_infix(tokens) if tokens else f"Error: {str(e)[:50]}"
            
            return {
                'success': False,
                'reward': -2.0,  # 오류는 더 큰 페널티
                'error': str(e),
                'tokens': tokens,
                'formula': formula,
                'metrics': {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'volatility': 0.0
                },
                'signal_stats': {
                    'signal_turnover': 0.0,
                    'long_ratio': 0.0,
                    'short_ratio': 0.0
                }
            }
    
    def get_best_strategy(self) -> Optional[Dict[str, Any]]:
        """최고 성능 전략 반환"""
        return getattr(self, 'best_strategy', None)
