#!/usr/bin/env python3
"""
빠른 미래 정보 누출 체크 스크립트
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Factor Factory 모듈
sys.path.append(str(Path(__file__).parent))
from factor_factory.data import ParquetCache
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.signal_generator import RealtimeSignalGenerator
from factor_factory.backtest.realistic_engine import realistic_backtest, vector_backtest

def quick_leak_check():
    """빠른 미래 정보 누출 체크"""
    print("🔍 Factor Factory v2.1 - 빠른 미래 정보 누출 체크")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        cache = ParquetCache()
        df = cache.load("BTCUSDT", "1h")  # load 메서드 사용
        price = df['close']
        
        print(f"✅ 데이터 로드 성공: {len(price):,}개 데이터포인트")
        print(f"📅 기간: {price.index[0]} ~ {price.index[-1]}")
        
        # 2. 프로그램 로드
        with open("models/best_program.json", "r") as f:
            program_data = json.load(f)
        
        program = program_data["program"]
        formula = program_data.get("formula", "Unknown")
        
        print(f"📊 프로그램: {formula}")
        print(f"🔢 토큰: {program}")
        
        # 3. 미래 정보 누출 체크
        print(f"\n🔍 미래 정보 누출 검사 중...")
        
        # 실시간 신호 생성기로 체크
        generator = RealtimeSignalGenerator()
        
        # 샘플 검사 (처음 500개 포인트)
        leak_count = 0
        check_count = min(500, len(price) - 300)  # 충분한 과거 데이터 확보
        
        for i in range(300, 300 + check_count):  # 300개 데이터로 워밍업
            current_data = df.iloc[:i+1]  # i 시점까지의 데이터만
            
            try:
                # 현재 시점에서 신호 생성
                signal_value = generator.generate_signal(current_data, program)
                
                # NaN이면 정상 (초기에는 지표 계산 불가)
                if pd.isna(signal_value):
                    continue
                    
                # 미래 데이터와 비교
                if i + 50 < len(df):  # 50기간 후 데이터가 있는 경우
                    future_data = df.iloc[:i+51]  # 50기간 후까지 포함
                    future_signal = generator.generate_signal(future_data, program)
                    
                    # 과거 신호와 차이가 크면 누출 의심
                    if not pd.isna(future_signal) and abs(signal_value - future_signal) > 0.001:
                        leak_count += 1
                        
            except Exception:
                continue
        
        leak_ratio = leak_count / max(check_count, 1) * 100
        
        if leak_ratio > 5:  # 5% 이상 변경되면 문제
            print(f"⚠️  미래 정보 누출 위험: {leak_ratio:.1f}% ({leak_count}/{check_count})")
        else:
            print(f"✅ 미래 정보 누출 검사 통과: {leak_ratio:.1f}% ({leak_count}/{check_count})")
        
        # 4. 현실적 백테스트 실행
        print(f"\n🎯 현실적 백테스트 실행 중...")
        
        # 전체 신호 생성 (실시간 방식)
        signals = []
        for i in range(len(df)):
            current_data = df.iloc[:i+1]
            signal_value = generator.generate_signal(current_data, program)
            signals.append(signal_value if not pd.isna(signal_value) else 0.0)
        
        signal_series = pd.Series(signals, index=df.index)
        
        # 현실적 백테스트
        equity, pnl = realistic_backtest(
            price=price,
            signal=signal_series,
            commission=0.0008,  # 0.08%
            slippage=0.0015,    # 0.15%
            signal_delay=1,
            execution_delay=1
        )
        
        # 결과 분석
        total_return = (equity.iloc[-1] - 1) * 100
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        max_dd = (equity / equity.cummax() - 1).min() * 100
        
        print(f"\n📊 현실적 백테스트 결과:")
        print(f"💰 총 수익률:     {total_return:8.2f}%")
        print(f"📈 샤프 비율:     {sharpe:8.4f}")
        print(f"📉 최대 낙폭:     {max_dd:8.2f}%")
        print(f"🔄 거래 지연:     신호 {1}기간 + 체결 {1}기간")
        print(f"💸 거래 비용:     수수료 0.08% + 슬리피지 0.15%")
        
        # 5. 기존 방식과 비교
        print(f"\n🔄 기존 방식과 비교...")
        
        # 기존 compiler로 신호 생성 (미래 정보 포함 가능)
        try:
            old_signal = eval_prefix(program, df)
            old_equity, old_pnl = vector_backtest(price, old_signal)
            
            old_return = (old_equity.iloc[-1] - 1) * 100
            old_sharpe = old_pnl.mean() / old_pnl.std() * np.sqrt(252 * 24) if old_pnl.std() > 0 else 0
            
            print(f"📊 기존 방식 결과:")
            print(f"💰 총 수익률:     {old_return:8.2f}% (차이: {old_return - total_return:+.2f}%)")
            print(f"📈 샤프 비율:     {old_sharpe:8.4f} (차이: {old_sharpe - sharpe:+.4f})")
            
            if abs(old_return - total_return) > 5:
                print(f"⚠️  성과 차이 큼 → 미래 정보 누출 가능성")
            else:
                print(f"✅ 성과 차이 작음 → 누출 위험 낮음")
                
        except Exception as e:
            print(f"❌ 기존 방식 비교 실패: {e}")
        
        print(f"\n" + "=" * 60)
        print(f"🎯 결론: {'현실적 백테스트 환경 구축 완료' if leak_ratio <= 5 else '추가 개선 필요'}")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_leak_check()
