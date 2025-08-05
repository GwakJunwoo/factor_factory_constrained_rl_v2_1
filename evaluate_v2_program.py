#!/usr/bin/env python3
"""
새로운 v2 프로그램 빠른 평가 스크립트
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Factor Factory 모듈 import
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics
from factor_factory.data import ParquetCache, DATA_ROOT

# 시각화 모듈 (안전한 import)
try:
    from factor_factory.visualization import create_trading_report
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("⚠️ 시각화 모듈을 사용할 수 없습니다. 차트 생성이 생략됩니다.")

def evaluate_v2_program():
    """새로운 v2 프로그램 평가"""
    
    print("🎯 Factor Factory v2.1 - 새로운 프로그램 평가")
    print("=" * 60)
    
    # 1. 새로운 프로그램 로드
    v2_program_path = Path("best_results_v2/best_program.json")
    if not v2_program_path.exists():
        print(f"❌ 오류: {v2_program_path} 파일을 찾을 수 없습니다.")
        print("먼저 다음 명령으로 새로운 프로그램을 탐색하세요:")
        print("python -m factor_factory.scripts.cli_rlc_infer --model models/ppo_program_v2.zip ...")
        return
    
    with open(v2_program_path) as f:
        v2_data = json.load(f)
    
    tokens = v2_data["tokens"]
    formula = tokens_to_infix(tokens)
    
    print(f"📋 V2 프로그램 정보:")
    print(f"   - 수식: {formula}")
    print(f"   - 토큰 수: {len(tokens)}")
    print(f"   - 깊이: {calc_tree_depth(tokens)}")
    
    # 2. 데이터 로드
    print(f"\n📊 데이터 로딩 중...")
    cache = ParquetCache(DATA_ROOT)
    df = cache.load("BTCUSDT", "1h")
    print(f"   - 데이터 크기: {df.shape}")
    print(f"   - 기간: {df.index[0]} ~ {df.index[-1]}")
    
    # 3. 신호 생성 및 백테스트
    print(f"\n⚡ 백테스트 실행 중...")
    try:
        raw_signal = eval_prefix(tokens, df)
        if raw_signal.empty or raw_signal.std() < 1e-6:
            print("❌ 오류: 유효하지 않은 신호가 생성되었습니다.")
            return
        
        # 신호 정규화 (단순 Z-score)
        signal = (raw_signal - raw_signal.mean()) / raw_signal.std()
        price = df["close"]
        
        # 백테스트 실행
        equity, pnl = vector_backtest(price, signal)
        metrics = compute_metrics(equity, pnl, signal)
        
        print(f"✅ 백테스트 완료!")
        
    except Exception as e:
        print(f"❌ 백테스트 오류: {e}")
        return
    
    # 4. 성과 분석 출력
    print(f"\n📈 V2 프로그램 성과 분석")
    print("-" * 40)
    print(f"🎊 총 수익률:                {((equity.iloc[-1] - 1) * 100):8.2f}%")
    print(f"📊 Sharpe Ratio:            {metrics['sharpe']:8.4f}")
    print(f"📉 Max Drawdown:            {(metrics['mdd'] * 100):8.2f}%")
    print(f"🔢 총 거래 수:                {metrics.get('total_trades', 0):8d}")
    print(f"🔄 회전율:                    {metrics.get('turnover', 0):8.2f}")
    print(f"📊 연간 변동성:              {(pnl.std() * np.sqrt(252 * 24) * 100):8.2f}%")
    
    # 추가 메트릭
    if 'calmar' in metrics:
        print(f"📈 Calmar Ratio:            {metrics['calmar']:8.4f}")
    if 'win_rate' in metrics:
        print(f"🎯 승률:                     {(metrics['win_rate'] * 100):8.1f}%")
    if 'profit_factor' in metrics:
        print(f"💰 Profit Factor:           {metrics['profit_factor']:8.4f}")
    
    # 5. 결과 저장
    output_dir = Path("evaluation_v2")
    output_dir.mkdir(exist_ok=True)
    
    # CSV 저장
    result_df = pd.DataFrame({
        "price": price,
        "signal": signal,
        "equity": equity,
        "pnl": pnl
    })
    result_df.to_csv(output_dir / "results.csv")
    
    # 메트릭 저장
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 프로그램 정보 저장
    program_info = {
        "tokens": tokens,
        "formula": formula,
        "depth": calc_tree_depth(tokens),
        "length": len(tokens)
    }
    with open(output_dir / "program_info.json", "w") as f:
        json.dump(program_info, f, indent=2)
    
    print(f"\n💾 결과 저장 완료: {output_dir}/")
    
    # 6. 차트 생성 (안전하게)
    if CHARTS_AVAILABLE:
        print(f"\n📊 차트 생성 중...")
        try:
            charts_dir = Path("charts_v2")
            charts_dir.mkdir(exist_ok=True)
            
            create_trading_report(
                price=price,
                signal=signal,
                equity=equity,
                pnl=pnl,
                metrics=metrics,
                formula=formula,
                output_dir=str(charts_dir)
            )
            print(f"✅ 차트 생성 완료: {charts_dir}/")
            
        except Exception as e:
            print(f"⚠️ 차트 생성 중 오류: {e}")
            print("   차트 없이 계속 진행합니다.")
    
    print(f"\n🎯 V2 프로그램 평가 완료!")
    print(f"   📁 결과 파일: {output_dir}/")
    if CHARTS_AVAILABLE:
        print(f"   📊 차트 파일: charts_v2/")

if __name__ == "__main__":
    evaluate_v2_program()
