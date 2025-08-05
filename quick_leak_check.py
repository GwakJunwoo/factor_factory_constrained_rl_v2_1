#!/usr/bin/env python3
"""
빠른 미래 정보 누출 검증 스크립트
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

def quick_leak_check():
    """빠른 미래 정보 누출 체크"""
    print("🔍 Factor Factory v2.1 - 빠른 미래 정보 누출 체크")
    print("=" * 60)
    
    try:
        from factor_factory.rlc.compiler import eval_prefix
        from factor_factory.rlc.signal_generator import generate_signal_realtime, validate_signal_timing
        from factor_factory.data import ParquetCache, DATA_ROOT
        
        # 데이터 로드
        cache = ParquetCache(DATA_ROOT)
        df = cache.get("BTCUSDT", "1h")
        
        if df is None or df.empty:
            print("❌ 데이터 없음")
            return
        
        print(f"📊 데이터: {len(df):,} 포인트 ({df.index[0]} ~ {df.index[-1]})")
        
        # 테스트 프로그램: MACD DIV LAG1(SMA20)
        test_program = [6, 2, 18, 3, 21]
        raw_factor = eval_prefix(test_program, df)
        
        # 기존 방식 (위험)
        old_z = (raw_factor - raw_factor.mean()) / raw_factor.std()
        old_signal = pd.Series(0.0, index=old_z.index)
        old_signal[old_z >= 1.5] = 1.0
        old_signal[old_z <= -1.5] = -1.0
        
        # 새로운 방식 (안전)
        new_signal = generate_signal_realtime(raw_factor, lookback_window=252)
        
        # 검증
        price = df["close"]
        old_val = validate_signal_timing(df, old_signal, price)
        new_val = validate_signal_timing(df, new_signal, price)
        
        print(f"\n📋 검증 결과:")
        print(f"   기존 방식: {'❌ 미래 정보 누출' if old_val['has_future_leak'] else '✅ 안전'}")
        print(f"   개선 방식: {'❌ 미래 정보 누출' if new_val['has_future_leak'] else '✅ 안전'}")
        
        # 신호 통계 비교
        print(f"\n📊 신호 통계:")
        print(f"   기존: Long {(old_signal==1).sum():,}, Flat {(old_signal==0).sum():,}, Short {(old_signal==-1).sum():,}")
        print(f"   개선: Long {(new_signal==1).sum():,}, Flat {(new_signal==0).sum():,}, Short {(new_signal==-1).sum():,}")
        
        # 상관관계 체크
        old_corr = old_signal.corr(price.pct_change())
        new_corr = new_signal.corr(price.pct_change())
        
        print(f"\n🔗 신호-수익률 상관관계:")
        print(f"   기존: {old_corr:.4f} {'⚠️ 의심' if abs(old_corr) > 0.08 else '✅'}")
        print(f"   개선: {new_corr:.4f} {'⚠️ 의심' if abs(new_corr) > 0.08 else '✅'}")
        
        if not new_val['has_future_leak']:
            print(f"\n✅ 미래 정보 누출 방지 성공!")
            print(f"   → 실거래에서 신뢰할 수 있는 성능 기대")
        else:
            print(f"\n❌ 추가 검토 필요")
            print(f"   → 문제: {new_val['issues']}")
            
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    quick_leak_check()
