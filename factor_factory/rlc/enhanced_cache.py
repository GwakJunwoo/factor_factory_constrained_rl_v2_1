"""
Enhanced Caching System for Program Evaluation

MCTS와 백테스트에서 동일한 프로그램의 중복 평가를 방지하고
성능을 대폭 향상시키는 캐시 시스템
"""

from __future__ import annotations
import hashlib
import pickle
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging


class FastProgramCache:
    """고속 프로그램 캐시 - 메모리 + 디스크 하이브리드"""
    
    def __init__(self, 
                 memory_size: int = 4096,     # 메모리 캐시 크기
                 disk_cache_dir: str = "cache_disk",  # 디스크 캐시 디렉토리
                 enable_disk: bool = True):    # 디스크 캐시 활성화
        
        self.memory_size = memory_size
        self.enable_disk = enable_disk
        
        # 메모리 캐시
        self._memory_cache = {}
        self._access_order = []
        
        # 디스크 캐시
        if enable_disk:
            self.disk_dir = Path(disk_cache_dir)
            self.disk_dir.mkdir(exist_ok=True)
        
        # 통계
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'total_requests': 0,
            'save_time': 0.0,
            'load_time': 0.0
        }
        
        logging.info(f"✅ FastProgramCache 초기화 (메모리: {memory_size}, 디스크: {enable_disk})")
    
    def _get_cache_key(self, tokens: list[int], data_hash: str) -> str:
        """캐시 키 생성 - 토큰과 데이터 해시 조합"""
        token_str = "_".join(map(str, tokens))
        return f"{token_str}_{data_hash}"
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """데이터 해시 생성 - 빠른 해시 함수 사용"""
        # DataFrame의 핵심 정보로 해시 생성
        key_data = (
            df.shape,
            tuple(df.columns),
            df.index[0] if len(df) > 0 else 0,
            df.index[-1] if len(df) > 0 else 0,
            df.iloc[0].sum() if len(df) > 0 else 0,
            df.iloc[-1].sum() if len(df) > 0 else 0
        )
        return str(hash(key_data))
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """디스크 캐시 파일 경로"""
        # 키를 해시해서 디렉토리 구조 생성 (collision 방지)
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        subdir = key_hash[:2]  # 첫 2글자로 서브디렉토리
        return self.disk_dir / subdir / f"{key_hash}.pkl"
    
    def get(self, tokens: list[int], df: pd.DataFrame) -> Optional[pd.Series]:
        """캐시에서 결과 조회"""
        self.stats['total_requests'] += 1
        
        cache_key = self._get_cache_key(tokens, self._get_data_hash(df))
        
        # 1. 메모리 캐시 확인
        if cache_key in self._memory_cache:
            self.stats['memory_hits'] += 1
            # LRU 업데이트
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._memory_cache[cache_key].copy()
        
        # 2. 디스크 캐시 확인
        if self.enable_disk:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                try:
                    start_time = time.time()
                    with open(disk_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    self.stats['load_time'] += time.time() - start_time
                    self.stats['disk_hits'] += 1
                    
                    # 메모리 캐시에도 저장
                    self._put_memory(cache_key, result)
                    return result.copy()
                    
                except Exception as e:
                    logging.warning(f"디스크 캐시 로드 실패: {e}")
        
        # 3. 캐시 미스
        self.stats['misses'] += 1
        return None
    
    def put(self, tokens: list[int], df: pd.DataFrame, result: pd.Series):
        """캐시에 결과 저장"""
        cache_key = self._get_cache_key(tokens, self._get_data_hash(df))
        
        # 메모리 캐시에 저장
        self._put_memory(cache_key, result)
        
        # 디스크 캐시에 저장 (비동기적으로)
        if self.enable_disk:
            self._put_disk(cache_key, result)
    
    def _put_memory(self, cache_key: str, result: pd.Series):
        """메모리 캐시에 저장"""
        # 캐시 크기 초과 시 LRU 제거
        if len(self._memory_cache) >= self.memory_size and cache_key not in self._memory_cache:
            oldest_key = self._access_order.pop(0)
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = result.copy()
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
    
    def _put_disk(self, cache_key: str, result: pd.Series):
        """디스크 캐시에 저장"""
        try:
            start_time = time.time()
            disk_path = self._get_disk_path(cache_key)
            disk_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(disk_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats['save_time'] += time.time() - start_time
            
        except Exception as e:
            logging.warning(f"디스크 캐시 저장 실패: {e}")
    
    def clear(self):
        """캐시 초기화"""
        self._memory_cache.clear()
        self._access_order.clear()
        
        if self.enable_disk and self.disk_dir.exists():
            import shutil
            shutil.rmtree(self.disk_dir)
            self.disk_dir.mkdir(exist_ok=True)
        
        # 통계 초기화
        for key in self.stats:
            self.stats[key] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        hit_rate = (self.stats['memory_hits'] + self.stats['disk_hits']) / total
        memory_rate = self.stats['memory_hits'] / total
        disk_rate = self.stats['disk_hits'] / total
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_hit_rate': memory_rate,
            'disk_hit_rate': disk_rate,
            'memory_size': len(self._memory_cache),
            'avg_load_time': self.stats['load_time'] / max(1, self.stats['disk_hits']),
            'avg_save_time': self.stats['save_time'] / max(1, self.stats['total_requests'])
        }


class BacktestCache:
    """백테스트 결과 캐시 - 메트릭까지 포함"""
    
    def __init__(self, cache_size: int = 1024):
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _get_key(self, signal_hash: str, price_hash: str, commission: float, slippage: float) -> str:
        """백테스트 캐시 키 생성"""
        return f"{signal_hash}_{price_hash}_{commission}_{slippage}"
    
    def _get_series_hash(self, series: pd.Series) -> str:
        """Series 해시 생성"""
        return str(hash(tuple([
            series.shape[0],
            series.iloc[0] if len(series) > 0 else 0,
            series.iloc[-1] if len(series) > 0 else 0,
            series.sum(),
            series.std()
        ])))
    
    def get(self, signal: pd.Series, price: pd.Series, 
            commission: float, slippage: float) -> Optional[Tuple[pd.Series, pd.Series, Dict]]:
        """백테스트 결과 조회"""
        self.stats['total_requests'] += 1
        
        key = self._get_key(
            self._get_series_hash(signal),
            self._get_series_hash(price),
            commission, slippage
        )
        
        if key in self._cache:
            self.stats['hits'] += 1
            # LRU 업데이트
            self._access_order.remove(key)
            self._access_order.append(key)
            
            cached_data = self._cache[key]
            return (
                cached_data['equity'].copy(),
                cached_data['pnl'].copy(),
                cached_data['metrics'].copy()
            )
        
        self.stats['misses'] += 1
        return None
    
    def put(self, signal: pd.Series, price: pd.Series, 
            commission: float, slippage: float,
            equity: pd.Series, pnl: pd.Series, metrics: Dict):
        """백테스트 결과 저장"""
        key = self._get_key(
            self._get_series_hash(signal),
            self._get_series_hash(price),
            commission, slippage
        )
        
        # 캐시 크기 초과 시 LRU 제거
        if len(self._cache) >= self.cache_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = {
            'equity': equity.copy(),
            'pnl': pnl.copy(),
            'metrics': metrics.copy()
        }
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.stats['total_requests']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }


# 전역 캐시 인스턴스들
_fast_program_cache = None
_backtest_cache = None

def get_fast_program_cache(memory_size: int = 4096, 
                          enable_disk: bool = True) -> FastProgramCache:
    """전역 고속 프로그램 캐시 반환"""
    global _fast_program_cache
    if _fast_program_cache is None:
        _fast_program_cache = FastProgramCache(memory_size, enable_disk=enable_disk)
    return _fast_program_cache

def get_backtest_cache(cache_size: int = 1024) -> BacktestCache:
    """전역 백테스트 캐시 반환"""
    global _backtest_cache
    if _backtest_cache is None:
        _backtest_cache = BacktestCache(cache_size)
    return _backtest_cache

def clear_all_caches():
    """모든 캐시 초기화"""
    global _fast_program_cache, _backtest_cache
    
    if _fast_program_cache:
        _fast_program_cache.clear()
    if _backtest_cache:
        _backtest_cache._cache.clear()
        _backtest_cache._access_order.clear()
    
    logging.info("🧹 모든 캐시가 초기화되었습니다")

def get_cache_statistics() -> Dict[str, Any]:
    """전체 캐시 통계 반환"""
    stats = {}
    
    if _fast_program_cache:
        stats['program_cache'] = _fast_program_cache.get_stats()
    
    if _backtest_cache:
        stats['backtest_cache'] = _backtest_cache.get_stats()
    
    return stats
