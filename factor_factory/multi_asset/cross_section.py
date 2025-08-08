"""
크로스 섹션 정규화 모듈

시점별로 여러 종목의 팩터 값을 정규화하여
롱숏 신호를 생성하는 모듈
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from scipy import stats


class CrossSectionNormalizer:
    """시점별 크로스 섹션 정규화를 수행하는 클래스"""
    
    def __init__(self, method: str = 'z_score', min_assets: int = 2):
        """
        Args:
            method: 정규화 방법 ('z_score', 'rank', 'percentile', 'mad')
            min_assets: 정규화에 필요한 최소 종목 수
        """
        self.method = method
        self.min_assets = min_assets
        self.normalize_stats = {}
        
    def normalize(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        팩터 행렬을 크로스 섹션 정규화
        
        Args:
            factor_matrix: [timestamp x symbol] 팩터 값 행렬
            
        Returns:
            normalized_signals: [timestamp x symbol] 정규화된 신호 행렬
        """
        if self.method == 'z_score':
            return self._z_score_normalize(factor_matrix)
        elif self.method == 'rank':
            return self._rank_normalize(factor_matrix)
        elif self.method == 'percentile':
            return self._percentile_normalize(factor_matrix)
        elif self.method == 'mad':
            return self._mad_normalize(factor_matrix)
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {self.method}")
    
    def _z_score_normalize(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """각 시점별로 종목간 z-score 정규화"""
        print(f"📊 Z-Score 크로스 섹션 정규화 실행")
        
        normalized_matrix = pd.DataFrame(
            index=factor_matrix.index,
            columns=factor_matrix.columns,
            dtype=np.float32
        )
        
        stats_list = []
        
        for timestamp in factor_matrix.index:
            cross_section = factor_matrix.loc[timestamp]
            
            # 유효한 값들만 선택 (NaN 제외)
            valid_values = cross_section.dropna()
            
            if len(valid_values) < self.min_assets:
                # 종목 수가 부족하면 0으로 설정
                normalized_matrix.loc[timestamp] = 0.0
                continue
            
            # 크로스 섹션 통계 계산
            cs_mean = valid_values.mean()
            cs_std = valid_values.std()
            
            if cs_std == 0 or pd.isna(cs_std):
                # 표준편차가 0이면 모든 값이 동일 → 0으로 설정
                normalized_matrix.loc[timestamp] = 0.0
            else:
                # Z-score 계산
                z_scores = (cross_section - cs_mean) / cs_std
                normalized_matrix.loc[timestamp] = z_scores.fillna(0.0)
            
            # 통계 저장
            stats_list.append({
                'timestamp': timestamp,
                'cross_mean': cs_mean,
                'cross_std': cs_std,
                'valid_assets': len(valid_values),
                'max_signal': normalized_matrix.loc[timestamp].max(),
                'min_signal': normalized_matrix.loc[timestamp].min()
            })
        
        # 클리핑 (-3, 3) → 극값 제거
        normalized_matrix = normalized_matrix.clip(-3.0, 3.0)
        
        # 통계 저장
        self.normalize_stats['z_score'] = pd.DataFrame(stats_list)
        
        print(f"  ✅ 정규화 완료: {normalized_matrix.shape}")
        print(f"  📈 신호 범위: [{normalized_matrix.min().min():.3f}, {normalized_matrix.max().max():.3f}]")
        
        return normalized_matrix
    
    def _rank_normalize(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """각 시점별로 종목간 순위 기반 정규화 (0~1 → -1~1)"""
        print(f"📊 Rank 크로스 섹션 정규화 실행")
        
        normalized_matrix = pd.DataFrame(
            index=factor_matrix.index,
            columns=factor_matrix.columns,
            dtype=np.float32
        )
        
        for timestamp in factor_matrix.index:
            cross_section = factor_matrix.loc[timestamp]
            valid_values = cross_section.dropna()
            
            if len(valid_values) < self.min_assets:
                normalized_matrix.loc[timestamp] = 0.0
                continue
            
            # 순위 계산 (1부터 N까지)
            ranks = valid_values.rank(method='dense')
            
            # 0~1 범위로 정규화
            rank_normalized = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else 0.5
            
            # -1~1 범위로 변환
            rank_signals = (rank_normalized * 2) - 1
            
            # 전체 데이터에 할당 (NaN은 0으로)
            normalized_matrix.loc[timestamp] = cross_section.map(
                lambda x: rank_signals.get(x, 0.0) if pd.notna(x) else 0.0
            )
        
        print(f"  ✅ Rank 정규화 완료: {normalized_matrix.shape}")
        return normalized_matrix
    
    def _percentile_normalize(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """각 시점별로 백분위 기반 정규화"""
        print(f"📊 Percentile 크로스 섹션 정규화 실행")
        
        normalized_matrix = pd.DataFrame(
            index=factor_matrix.index,
            columns=factor_matrix.columns,
            dtype=np.float32
        )
        
        for timestamp in factor_matrix.index:
            cross_section = factor_matrix.loc[timestamp]
            valid_values = cross_section.dropna()
            
            if len(valid_values) < self.min_assets:
                normalized_matrix.loc[timestamp] = 0.0
                continue
            
            # 백분위 계산
            percentiles = valid_values.rank(pct=True)
            
            # 정규분포의 역함수를 사용해 z-score로 변환
            z_scores = stats.norm.ppf(percentiles.clip(0.001, 0.999))
            
            # 전체 데이터에 할당
            result = cross_section.copy()
            for symbol in valid_values.index:
                result[symbol] = z_scores[symbol]
            result = result.fillna(0.0)
            
            normalized_matrix.loc[timestamp] = result
        
        # 클리핑
        normalized_matrix = normalized_matrix.clip(-3.0, 3.0)
        
        print(f"  ✅ Percentile 정규화 완료: {normalized_matrix.shape}")
        return normalized_matrix
    
    def _mad_normalize(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """각 시점별로 MAD(Median Absolute Deviation) 기반 정규화"""
        print(f"📊 MAD 크로스 섹션 정규화 실행")
        
        normalized_matrix = pd.DataFrame(
            index=factor_matrix.index,
            columns=factor_matrix.columns,
            dtype=np.float32
        )
        
        for timestamp in factor_matrix.index:
            cross_section = factor_matrix.loc[timestamp]
            valid_values = cross_section.dropna()
            
            if len(valid_values) < self.min_assets:
                normalized_matrix.loc[timestamp] = 0.0
                continue
            
            # 중앙값과 MAD 계산
            median_val = valid_values.median()
            mad_val = np.median(np.abs(valid_values - median_val))
            
            if mad_val == 0:
                normalized_matrix.loc[timestamp] = 0.0
            else:
                # MAD 기반 정규화 (1.4826은 정규분포에서 MAD를 표준편차로 변환하는 상수)
                mad_scores = (cross_section - median_val) / (mad_val * 1.4826)
                normalized_matrix.loc[timestamp] = mad_scores.fillna(0.0)
        
        # 클리핑
        normalized_matrix = normalized_matrix.clip(-3.0, 3.0)
        
        print(f"  ✅ MAD 정규화 완료: {normalized_matrix.shape}")
        return normalized_matrix
    
    def generate_signals(self, 
                        normalized_matrix: pd.DataFrame,
                        long_threshold: float = 1.0,
                        short_threshold: float = -1.0) -> pd.DataFrame:
        """
        정규화된 값을 바탕으로 롱숏 신호 생성
        
        Args:
            normalized_matrix: 정규화된 팩터 행렬
            long_threshold: 롱 진입 임계값
            short_threshold: 숏 진입 임계값
            
        Returns:
            signals: [timestamp x symbol] 신호 행렬 (-1: 숏, 0: 중립, 1: 롱)
        """
        signals = pd.DataFrame(
            index=normalized_matrix.index,
            columns=normalized_matrix.columns,
            dtype=np.float32
        )
        
        # 임계값 기반 신호 생성
        signals[normalized_matrix >= long_threshold] = 1.0   # 롱
        signals[normalized_matrix <= short_threshold] = -1.0  # 숏
        signals[(normalized_matrix > short_threshold) & 
               (normalized_matrix < long_threshold)] = 0.0   # 중립
        
        # NaN 처리
        signals = signals.fillna(0.0)
        
        return signals
    
    def get_signal_statistics(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """신호 통계 정보 반환"""
        long_signals = (signals == 1.0).sum().sum()
        short_signals = (signals == -1.0).sum().sum()
        neutral_signals = (signals == 0.0).sum().sum()
        total_signals = signals.size
        
        stats = {
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'neutral_signals': neutral_signals,
            'long_ratio': long_signals / total_signals,
            'short_ratio': short_signals / total_signals,
            'neutral_ratio': neutral_signals / total_signals,
            'signal_turnover': self._calculate_turnover(signals)
        }
        
        return stats
    
    def _calculate_turnover(self, signals: pd.DataFrame) -> float:
        """신호 변경 빈도 계산 (일일 턴오버율)"""
        changes = signals.diff().abs().sum(axis=1)
        total_positions = signals.abs().sum(axis=1)
        
        # 0으로 나누기 방지
        turnover_rates = changes / (total_positions + 1e-8)
        
        return turnover_rates.mean()
    
    def create_market_neutral_signals(self, 
                                    normalized_matrix: pd.DataFrame,
                                    target_long_ratio: float = 0.3,
                                    target_short_ratio: float = 0.3) -> pd.DataFrame:
        """
        마켓 뉴트럴 신호 생성 (상위 N%는 롱, 하위 N%는 숏)
        
        Args:
            normalized_matrix: 정규화된 팩터 행렬
            target_long_ratio: 롱 포지션 비율
            target_short_ratio: 숏 포지션 비율
            
        Returns:
            signals: 마켓 뉴트럴 신호 행렬
        """
        signals = pd.DataFrame(
            index=normalized_matrix.index,
            columns=normalized_matrix.columns,
            dtype=np.float32
        )
        
        for timestamp in normalized_matrix.index:
            cross_section = normalized_matrix.loc[timestamp].dropna()
            
            if len(cross_section) < self.min_assets:
                signals.loc[timestamp] = 0.0
                continue
            
            # 상위/하위 임계값 계산
            long_threshold = cross_section.quantile(1 - target_long_ratio)
            short_threshold = cross_section.quantile(target_short_ratio)
            
            # 신호 생성
            timestamp_signals = pd.Series(0.0, index=normalized_matrix.columns)
            timestamp_signals[cross_section >= long_threshold] = 1.0
            timestamp_signals[cross_section <= short_threshold] = -1.0
            
            signals.loc[timestamp] = timestamp_signals
        
        signals = signals.fillna(0.0)
        
        print(f"📊 마켓 뉴트럴 신호 생성 완료")
        print(f"  Target Long: {target_long_ratio:.1%}, Short: {target_short_ratio:.1%}")
        
        return signals
