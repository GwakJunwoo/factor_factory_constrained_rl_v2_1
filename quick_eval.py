#!/usr/bin/env python3
"""
기존 최적 프로그램을 개선된 시스템으로 평가하는 간단한 스크립트
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics
from factor_factory.data import ParquetCache, DATA_ROOT

# 시각화 모듈 (선택적 import)
try:
    from factor_factory.visualization import create_trading_report
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ matplotlib이 설치되지 않아 차트 기능을 사용할 수 없습니다.")
    print("   설치: pip install matplotlib")

def evaluate_program(tokens, symbol="BTCUSDT", interval="1h"):
    """프로그램 평가 함수"""
    
    # 프로그램 정보 출력
    infix_formula = tokens_to_infix(tokens)
    tree_depth = calc_tree_depth(tokens)
    
    print("=" * 80)
    print("📊 PROGRAM EVALUATION REPORT")
    print("=" * 80)
    print(f"🔢 토큰 수: {len(tokens)}")
    print(f"📏 트리 깊이: {tree_depth}")
    print(f"🧮 수식: {infix_formula}")
    print(f"🔢 토큰 리스트: {tokens}")
    print("=" * 80)

    # 데이터 로드
    try:
        df = ParquetCache(DATA_ROOT).load(symbol, interval)
        print(f"📈 데이터: {symbol}_{interval}")
        print(f"📅 기간: {df.index[0]} ~ {df.index[-1]} ({len(df):,} bars)")
    except FileNotFoundError:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {symbol}_{interval}")
        return None
    
    # 시그널 생성
    try:
        sig = eval_prefix(tokens, df).dropna().rename("signal")
        price = df["close"].reindex(sig.index)
        
        if sig.empty:
            print("⚠️ 시그널이 비어 있습니다.")
            return None
            
        print(f"📊 시그널 통계:")
        print(f"   - 범위: [{sig.min():.4f}, {sig.max():.4f}]")
        print(f"   - 평균: {sig.mean():.4f}")
        print(f"   - 표준편차: {sig.std():.4f}")
        print(f"   - 롱 신호 비율: {(sig > 0.1).mean():.2%}")
        print(f"   - 숏 신호 비율: {(sig < -0.1).mean():.2%}")
        print(f"   - 플랫 비율: {(abs(sig) <= 0.1).mean():.2%}")
        
    except Exception as e:
        print(f"❌ 시그널 생성 오류: {e}")
        return None

    # 백테스트
    try:
        print("\n🔄 백테스트 실행 중...")
        equity, pnl = vector_backtest(price, sig)
        metrics = compute_metrics(pnl, equity, sig)
        
        # 성능 메트릭 출력
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE METRICS")
        print("=" * 80)
        
        print(f"💰 CAGR (연평균 성장률):        {metrics['cagr']:8.2%}")
        print(f"📈 Sharpe Ratio:              {metrics['sharpe']:8.4f}")
        print(f"📉 Max Drawdown:              {metrics['mdd']:8.2%}")
        print(f"🔄 Turnover:                  {metrics['turnover']:8.2f}")
        print(f"🏆 Calmar Ratio:              {metrics['calmar']:8.4f}")
        print(f"🎯 Win Rate:                  {metrics['win_rate']:8.2%}")
        print(f"💎 Profit Factor:             {metrics['profit_factor']:8.4f}")
        print(f"📊 Information Ratio:         {metrics['information_ratio']:8.4f}")
        print(f"📉 Max Consecutive Losses:    {metrics['max_consecutive_losses']:8d} days")
        print(f"🔢 Total Trades:              {metrics['total_trades']:8d}")
        print(f"💵 Avg Trade PnL:             {metrics['avg_trade_pnl']:8.6f}")
        
        # 최종 수익률
        total_return = (equity.iloc[-1] - 1) * 100
        print(f"🎊 Total Return:              {total_return:8.2f}%")
        print("=" * 80)
        
        return {
            "formula": infix_formula,
            "metrics": metrics,
            "equity": equity,
            "signal": sig,
            "price": price
        }
        
    except Exception as e:
        print(f"❌ 백테스트 오류: {e}")
        return None

def main():
    """메인 함수"""
    
    # 기존 최적 프로그램 로드
    try:
        with open('models/best_program.json', 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "tokens" in data:
                tokens = data["tokens"]
            else:
                tokens = data  # 구버전 형식
        
        print("🔍 기존 최적 프로그램을 개선된 시스템으로 평가합니다...")
        result = evaluate_program(tokens)
        
        if result:
            print("\n✅ 평가 완료!")
            
            # 차트 생성 (matplotlib이 사용 가능한 경우)
            if VISUALIZATION_AVAILABLE:
                try:
                    print("\n📊 트레이딩 차트 생성 중...")
                    create_trading_report(
                        price=result["price"],
                        signal=result["signal"], 
                        equity=result["equity"],
                        pnl=result["equity"].pct_change().fillna(0),
                        metrics=result["metrics"],
                        formula=result["formula"],
                        output_dir="quick_eval_charts"
                    )
                    print("📊 차트가 'quick_eval_charts' 폴더에 저장되었습니다!")
                except Exception as e:
                    print(f"⚠️ 차트 생성 중 오류: {e}")
            else:
                print("\n📊 차트를 생성하려면 matplotlib을 설치하세요:")
                print("   pip install matplotlib")
        else:
            print("\n❌ 평가 실패!")
            
    except FileNotFoundError:
        print("❌ models/best_program.json 파일을 찾을 수 없습니다.")
        print("   먼저 학습을 실행하거나 기존 프로그램 파일을 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
