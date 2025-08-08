"""
다종목 데이터 관리자

여러 종목의 OHLCV 데이터를 로딩하고 시간 동기화하여
팩터 계산을 위한 통합된 데이터 구조 제공
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
from pathlib import Path

from ..data import ParquetCache, DATA_ROOT
from ..rlc.compiler import eval_prefix


class MultiAssetDataManager:
    """다종목 데이터를 동기화하여 관리하는 클래스"""
    
    def __init__(self, symbols: List[str], interval: str = "1h"):
        """
        Args:
            symbols: 종목 리스트 (예: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
            interval: 시간 간격 (예: '1h', '4h', '1d')
        """
        self.symbols = symbols
        self.interval = interval
        self.cache = ParquetCache(DATA_ROOT)
        
        # 데이터 저장소
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        self.common_index: Optional[pd.DatetimeIndex] = None
        
        # 통계 정보
        self.load_stats = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """모든 종목의 데이터를 개별적으로 로딩"""
        print(f"📈 다종목 데이터 로딩 시작: {self.symbols}")
        
        for symbol in self.symbols:
            try:
                df = self.cache.load(symbol, self.interval)
                self.data_dict[symbol] = df
                
                self.load_stats[symbol] = {
                    'shape': df.shape,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'missing_values': df.isnull().sum().sum()
                }
                
                print(f"  ✅ {symbol}: {df.shape[0]}개 데이터, "
                      f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
                
            except Exception as e:
                print(f"  ❌ {symbol} 로딩 실패: {e}")
                self.load_stats[symbol] = {'error': str(e)}
        
        return self.data_dict
    
    def align_data(self, method: str = 'inner') -> pd.DataFrame:
        """
        모든 종목 데이터를 시간 축으로 동기화
        
        Args:
            method: 'inner' (교집합), 'outer' (합집합), 'left' (첫번째 기준)
            
        Returns:
            MultiIndex DataFrame with columns (symbol, ohlcv)
        """
        if not self.data_dict:
            raise ValueError("데이터가 로딩되지 않았습니다. load_data()를 먼저 실행하세요.")
        
        print(f"🔄 데이터 정렬 방식: {method}")
        
        # 공통 시간 인덱스 찾기
        if method == 'inner':
            # 모든 종목에 공통으로 존재하는 시간만 사용
            common_idx = self.data_dict[self.symbols[0]].index
            for symbol in self.symbols[1:]:
                common_idx = common_idx.intersection(self.data_dict[symbol].index)
        elif method == 'outer':
            # 모든 종목의 시간을 합집합으로 사용
            common_idx = self.data_dict[self.symbols[0]].index
            for symbol in self.symbols[1:]:
                common_idx = common_idx.union(self.data_dict[symbol].index)
        else:  # left
            # 첫 번째 종목의 시간 기준 사용
            common_idx = self.data_dict[self.symbols[0]].index
        
        self.common_index = common_idx.sort_values()
        
        # MultiIndex DataFrame 생성
        aligned_dfs = []
        
        for symbol in self.symbols:
            df = self.data_dict[symbol]
            
            # 공통 인덱스로 재인덱싱
            reindexed_df = df.reindex(self.common_index)
            
            # 결측값 처리 (forward fill)
            reindexed_df = reindexed_df.fillna(method='ffill')
            
            # 컬럼명에 심볼 추가
            reindexed_df.columns = pd.MultiIndex.from_product(
                [[symbol], reindexed_df.columns]
            )
            
            aligned_dfs.append(reindexed_df)
        
        # 모든 데이터 결합
        self.aligned_data = pd.concat(aligned_dfs, axis=1)
        
        print(f"  📊 정렬된 데이터 크기: {self.aligned_data.shape}")
        print(f"  📅 공통 기간: {self.common_index[0].strftime('%Y-%m-%d')} ~ "
              f"{self.common_index[-1].strftime('%Y-%m-%d')}")
        
        return self.aligned_data
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """특정 종목의 정렬된 데이터 반환"""
        if self.aligned_data is None:
            raise ValueError("데이터가 정렬되지 않았습니다. align_data()를 먼저 실행하세요.")
        
        return self.aligned_data[symbol]
    
    def get_price_matrix(self, price_type: str = 'close') -> pd.DataFrame:
        """
        모든 종목의 특정 가격을 행렬로 반환
        
        Args:
            price_type: 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            DataFrame with shape [timestamp x symbol]
        """
        if self.aligned_data is None:
            raise ValueError("데이터가 정렬되지 않았습니다. align_data()를 먼저 실행하세요.")
        
        price_matrix = pd.DataFrame(index=self.common_index)
        
        for symbol in self.symbols:
            price_matrix[symbol] = self.aligned_data[symbol][price_type]
        
        return price_matrix
    
    def calculate_factor_matrix(self, program: List[int]) -> pd.DataFrame:
        """
        모든 종목에 동일한 팩터 프로그램을 적용하여 팩터 값 행렬 생성
        
        Args:
            program: 팩터 프로그램 토큰 시퀀스
            
        Returns:
            DataFrame with shape [timestamp x symbol] containing factor values
        """
        if self.aligned_data is None:
            raise ValueError("데이터가 정렬되지 않았습니다. align_data()를 먼저 실행하세요.")
        
        factor_matrix = pd.DataFrame(index=self.common_index)
        
        print(f"🔢 팩터 계산 중: {len(self.symbols)}개 종목")
        
        for symbol in self.symbols:
            try:
                # 종목별 데이터 추출
                symbol_data = self.get_symbol_data(symbol)
                
                # 팩터 계산
                factor_values = eval_prefix(program, symbol_data)
                
                # 결과 저장
                factor_matrix[symbol] = factor_values
                
                print(f"  ✅ {symbol}: 팩터 계산 완료")
                
            except Exception as e:
                print(f"  ❌ {symbol}: 팩터 계산 실패 - {e}")
                factor_matrix[symbol] = np.nan
        
        # NaN 처리
        factor_matrix = factor_matrix.fillna(method='ffill').fillna(0)
        
        return factor_matrix
    
    def get_returns_matrix(self, periods: int = 1) -> pd.DataFrame:
        """
        모든 종목의 수익률 행렬 계산
        
        Args:
            periods: 수익률 계산 기간 (1 = 1기간 후 수익률)
            
        Returns:
            DataFrame with shape [timestamp x symbol] containing returns
        """
        close_prices = self.get_price_matrix('close')
        
        # 기간별 수익률 계산
        returns_matrix = close_prices.pct_change(periods).shift(-periods)
        
        return returns_matrix
    
    def get_statistics(self) -> Dict:
        """데이터 매니저 통계 정보 반환"""
        stats = {
            'symbols': self.symbols,
            'interval': self.interval,
            'load_stats': self.load_stats,
        }
        
        if self.aligned_data is not None:
            stats.update({
                'aligned_shape': self.aligned_data.shape,
                'common_periods': len(self.common_index),
                'start_date': self.common_index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': self.common_index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            })
        
        return stats
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """데이터 품질 검증"""
        validation_results = {}
        
        if self.aligned_data is None:
            return {'error': '데이터가 정렬되지 않음'}
        
        for symbol in self.symbols:
            symbol_data = self.get_symbol_data(symbol)
            
            validation_results[symbol] = {
                'missing_ratio': symbol_data.isnull().sum().sum() / symbol_data.size,
                'zero_volume_ratio': (symbol_data['volume'] == 0).sum() / len(symbol_data),
                'price_consistency': self._check_price_consistency(symbol_data),
                'outlier_detection': self._detect_price_outliers(symbol_data)
            }
        
        return validation_results
    
    def _check_price_consistency(self, df: pd.DataFrame) -> Dict:
        """가격 일관성 검사 (high >= low, close between high/low 등)"""
        issues = []
        
        # High >= Low 검사
        if (df['high'] < df['low']).any():
            issues.append('high < low 발견')
        
        # Close가 High-Low 범위 내에 있는지 검사
        if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
            issues.append('close가 high-low 범위 밖에 있음')
        
        # Open이 High-Low 범위 내에 있는지 검사
        if ((df['open'] > df['high']) | (df['open'] < df['low'])).any():
            issues.append('open이 high-low 범위 밖에 있음')
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues
        }
    
    def _detect_price_outliers(self, df: pd.DataFrame, threshold: float = 5.0) -> Dict:
        """가격 이상치 탐지"""
        close_returns = df['close'].pct_change()
        
        # Z-score 기반 이상치 탐지
        z_scores = np.abs((close_returns - close_returns.mean()) / close_returns.std())
        outliers = z_scores > threshold
        
        return {
            'outlier_count': outliers.sum(),
            'outlier_ratio': outliers.sum() / len(df),
            'max_z_score': z_scores.max() if not z_scores.empty else 0
        }
