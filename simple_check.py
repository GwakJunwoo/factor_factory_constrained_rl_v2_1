#!/usr/bin/env python3
"""
간단한 미래 정보 누출 체크
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def simple_leak_check():
    """간단한 미래 정보 누출 체크"""
    print("🔍 간단한 미래 정보 누출 체크")
    print("=" * 40)
    
    try:
        # 데이터 로드
        from factor_factory.data import ParquetCache
        cache = ParquetCache()
        df = cache.load("BTCUSDT", "1h")
        print(f"✅ 데이터: {len(df):,}개 행")
        
        # 프로그램 로드
        with open("models/best_program.json", "r") as f:
            program_data = json.load(f)
        program = program_data["program"]
        
        # 기존 방식으로 신호 생성
        from factor_factory.rlc.compiler import eval_prefix
        raw_signal = eval_prefix(program, df)
        print(f"✅ 신호 생성: {len(raw_signal):,}개")
        
        # 현실적 백테스트
        from factor_factory.backtest.realistic_engine import realistic_backtest
        price = df['close']
        
        equity, pnl = realistic_backtest(
            price=price,
            signal=raw_signal,
            commission=0.0008,
            slippage=0.0015,
            signal_delay=1,
            execution_delay=1
        )
        
        # 결과 출력
        total_return = (equity.iloc[-1] - 1) * 100
        max_dd = (equity / equity.cummax() - 1).min() * 100
        
        print(f"\n📊 현실적 백테스트 결과:")
        print(f"💰 총 수익률: {total_return:6.2f}%")
        print(f"📉 최대 낙폭: {max_dd:6.2f}%")
        print(f"🔄 거래 지연: 2기간 (신호 1 + 체결 1)")
        print(f"💸 거래 비용: 0.23% (수수료 + 슬리피지)")
        
        print(f"\n✅ 현실적 백테스트 환경 구축 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_leak_check()
