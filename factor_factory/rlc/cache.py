from __future__ import annotations
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Tuple

class ProgramCache:
    """프로그램 평가 결과를 캐싱하는 클래스"""
    
    def __init__(self, cache_size: int = 2048):
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
    
    def get_key(self, tokens: list[int], df_hash: str) -> str:
        """캐시 키 생성"""
        return f"{tuple(tokens)}_{df_hash}"
    
    def get_df_hash(self, df: pd.DataFrame) -> str:
        """DataFrame의 해시 생성 (shape와 일부 값 기반)"""
        return f"{df.shape}_{hash(tuple(df.index[:10]))}"
    
    def get(self, tokens: list[int], df: pd.DataFrame) -> pd.Series | None:
        """캐시에서 결과 조회"""
        key = self.get_key(tokens, self.get_df_hash(df))
        if key in self._cache:
            # LRU 업데이트
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()
        return None
    
    def put(self, tokens: list[int], df: pd.DataFrame, result: pd.Series):
        """캐시에 결과 저장"""
        key = self.get_key(tokens, self.get_df_hash(df))
        
        # 캐시 크기 초과 시 LRU 제거
        if len(self._cache) >= self.cache_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = result.copy()
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """캐시 초기화"""
        self._cache.clear()
        self._access_order.clear()

# 전역 캐시 인스턴스
_program_cache = None

def get_program_cache(cache_size: int = 2048) -> ProgramCache:
    """전역 프로그램 캐시 반환 (필요시 생성)"""
    global _program_cache
    if _program_cache is None or _program_cache.cache_size != cache_size:
        _program_cache = ProgramCache(cache_size)
    return _program_cache

def clear_program_cache():
    """프로그램 캐시 초기화"""
    global _program_cache
    if _program_cache is not None:
        _program_cache.clear()
