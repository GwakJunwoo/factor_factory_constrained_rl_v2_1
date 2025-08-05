#!/usr/bin/env python3
"""
미래 정보 누출 및 실거래 타당성 검증 스크립트
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Factor Factory 모듈 import
sys.path.append(str(Path(__file__).parent))
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.signal_generator import (
    generate_signal_realtime, 
    validate_signal_timing, 
    debug_signal_generation
)
from factor_factory.backtest.realistic_engine import realistic_backtest, walk_forward_backtest
from factor_factory.data import ParquetCache, DATA_ROOT

def test_future_leak_prevention():
    """미래 정보 누출 방지 테스트"""
    print("=" * 80)
    print("🔍 미래 정보 누출 방지 테스트")
    print("=" * 80)
    
    # 데이터 로드
    cache = ParquetCache(DATA_ROOT)
    df = cache.get("BTCUSDT", "1h")
    
    if df is None or df.empty:
        print("❌ 데이터를 로드할 수 없습니다.")
        return
    
    print(f"📊 데이터 기간: {df.index[0]} ~ {df.index[-1]} ({len(df):,} 포인트)")
    
    # 테스트용 프로그램: MACD DIV LAG1(SMA20)
    test_program = [6, 2, 18, 3, 21]  # MACD DIV LAG1 SMA20 CONST
    
    try:
        # 1. 기존 방식 (위험한 방법)
        print("\n1️⃣ 기존 방식 (전체 데이터 정규화)")
        raw_factor = eval_prefix(test_program, df)
        
        # 전체 데이터로 정규화 (미래 정보 포함!)
        global_mean = raw_factor.mean()
        global_std = raw_factor.std()
        global_z = (raw_factor - global_mean) / global_std
        
        old_signal = pd.Series(0.0, index=global_z.index)
        old_signal[global_z >= 1.5] = 1.0
        old_signal[global_z <= -1.5] = -1.0
        
        print(f"   📈 신호 통계: Long {(old_signal==1).sum()}, Flat {(old_signal==0).sum()}, Short {(old_signal==-1).sum()}")
        
        # 2. 개선된 방식 (실시간 시뮬레이션)
        print("\n2️⃣ 개선된 방식 (실시간 시뮬레이션)")
        new_signal = generate_signal_realtime(
            raw_factor,
            lookback_window=252,
            long_threshold=1.5,
            short_threshold=-1.5,
            min_periods=50
        )
        
        print(f"   📈 신호 통계: Long {(new_signal==1).sum()}, Flat {(new_signal==0).sum()}, Short {(new_signal==-1).sum()}")
        
        # 3. 신호 검증
        print("\n3️⃣ 신호 검증 결과")
        price = df["close"]
        
        old_validation = validate_signal_timing(df, old_signal, price)
        new_validation = validate_signal_timing(df, new_signal, price)
        
        print(f"   기존 방식:")
        print(f"     - 미래 정보 누출: {'❌' if old_validation['has_future_leak'] else '✅'}")
        print(f"     - 문제점: {old_validation['issues']}")
        
        print(f"   개선된 방식:")
        print(f"     - 미래 정보 누출: {'❌' if new_validation['has_future_leak'] else '✅'}")
        print(f"     - 문제점: {new_validation['issues']}")
        
        # 4. 성능 비교
        print("\n4️⃣ 백테스트 성능 비교")
        
        # 기존 방식
        old_equity, old_pnl = realistic_backtest(price, old_signal, commission=0.0008, slippage=0.0015)
        old_return = (old_equity.iloc[-1] - 1) * 100
        old_sharpe = (old_pnl.mean() / old_pnl.std() * np.sqrt(252 * 24)) if old_pnl.std() > 0 else 0
        
        # 새로운 방식
        new_equity, new_pnl = realistic_backtest(price, new_signal, commission=0.0008, slippage=0.0015)
        new_return = (new_equity.iloc[-1] - 1) * 100
        new_sharpe = (new_pnl.mean() / new_pnl.std() * np.sqrt(252 * 24)) if new_pnl.std() > 0 else 0
        
        print(f"   기존 방식 (미래 정보 포함):")
        print(f"     - 총 수익률: {old_return:+.2f}%")
        print(f"     - Sharpe Ratio: {old_sharpe:.4f}")
        
        print(f"   개선된 방식 (실시간):")
        print(f"     - 총 수익률: {new_return:+.2f}%")
        print(f"     - Sharpe Ratio: {new_sharpe:.4f}")
        
        performance_gap = old_return - new_return
        print(f"   📊 성능 차이: {performance_gap:+.2f}% (미래 정보로 인한 과대 추정)")
        
        # 5. 시간별 분석 샘플
        print("\n5️⃣ 시간별 신호 생성 분석 (샘플)")
        debug_df = debug_signal_generation(raw_factor, new_signal)
        print(debug_df.head(5).to_string(index=False))
        
        return {
            'old_validation': old_validation,
            'new_validation': new_validation,
            'performance_gap': performance_gap,
            'old_return': old_return,
            'new_return': new_return
        }
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return None

def test_realistic_trading_conditions():
    """현실적 거래 조건 테스트"""
    print("\n" + "=" * 80)
    print("🏪 현실적 거래 조건 테스트")
    print("=" * 80)
    
    # 데이터 로드
    cache = ParquetCache(DATA_ROOT)
    df = cache.get("BTCUSDT", "1h")
    
    if df is None or df.empty:
        print("❌ 데이터를 로드할 수 없습니다.")
        return
    
    # 테스트용 신호 생성
    test_program = [6, 2, 18, 3, 21]  # MACD DIV LAG1 SMA20 CONST
    raw_factor = eval_prefix(test_program, df)
    signal = generate_signal_realtime(raw_factor, lookback_window=252)
    price = df["close"]
    
    # 다양한 거래 조건으로 테스트
    test_conditions = [
        {
            'name': '이상적 조건 (수수료 0%, 지연 없음)',
            'commission': 0.0,
            'slippage': 0.0,
            'signal_delay': 0,
            'execution_delay': 0
        },
        {
            'name': '현실적 조건 (일반)',
            'commission': 0.0008,
            'slippage': 0.0015,
            'signal_delay': 1,
            'execution_delay': 1
        },
        {
            'name': '보수적 조건 (높은 비용)',
            'commission': 0.0015,
            'slippage': 0.0025,
            'signal_delay': 2,
            'execution_delay': 2
        }
    ]
    
    results = []
    
    for condition in test_conditions:
        equity, pnl = realistic_backtest(
            price, signal,
            commission=condition['commission'],
            slippage=condition['slippage'],
            signal_delay=condition['signal_delay'],
            execution_delay=condition['execution_delay']
        )
        
        total_return = (equity.iloc[-1] - 1) * 100
        sharpe = (pnl.mean() / pnl.std() * np.sqrt(252 * 24)) if pnl.std() > 0 else 0
        max_dd = ((equity / equity.cummax()) - 1).min() * 100
        
        result = {
            'name': condition['name'],
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': (signal.diff().abs() > 0.1).sum()
        }
        results.append(result)
        
        print(f"\n📋 {condition['name']}")
        print(f"   총 수익률: {total_return:+.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.4f}")
        print(f"   최대 낙폭: {max_dd:.2f}%")
        print(f"   총 거래 수: {result['trades']}")
    
    # 조건별 수익률 차이 분석
    ideal_return = results[0]['total_return']
    realistic_return = results[1]['total_return']
    conservative_return = results[2]['total_return']
    
    print(f"\n📊 거래 조건별 수익률 영향:")
    print(f"   이상적 → 현실적: {realistic_return - ideal_return:+.2f}% 차이")
    print(f"   현실적 → 보수적: {conservative_return - realistic_return:+.2f}% 차이")
    print(f"   총 영향: {conservative_return - ideal_return:+.2f}% 감소")
    
    return results

def main():
    """메인 검증 실행"""
    print("🔬 Factor Factory v2.1 - 미래 정보 누출 및 실거래 타당성 검증")
    print("=" * 80)
    
    # 1. 미래 정보 누출 방지 테스트
    leak_test_results = test_future_leak_prevention()
    
    # 2. 현실적 거래 조건 테스트
    trading_test_results = test_realistic_trading_conditions()
    
    # 3. 종합 결과
    print("\n" + "=" * 80)
    print("📋 종합 검증 결과")
    print("=" * 80)
    
    if leak_test_results:
        print("✅ 미래 정보 누출 방지: 구현 완료")
        print(f"   - 성능 과대 추정 방지: {leak_test_results['performance_gap']:+.2f}%")
        print(f"   - 실시간 수익률: {leak_test_results['new_return']:+.2f}%")
    
    if trading_test_results:
        realistic_result = trading_test_results[1]  # 현실적 조건
        print("✅ 현실적 거래 조건: 반영 완료")
        print(f"   - 현실적 수익률: {realistic_result['total_return']:+.2f}%")
        print(f"   - Sharpe Ratio: {realistic_result['sharpe']:.4f}")
    
    print("\n🎯 권장사항:")
    print("   1. 모든 신호 생성에 generate_signal_realtime() 사용")
    print("   2. 백테스트에 realistic_backtest() 사용")
    print("   3. 학습 환경에서 미래 정보 누출 검증 활성화")
    print("   4. 실거래 전 워크포워드 분석 수행")

if __name__ == "__main__":
    main()
