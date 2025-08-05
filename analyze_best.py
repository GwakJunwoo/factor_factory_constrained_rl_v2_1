#!/usr/bin/env python3
"""
현재 최적 프로그램 (MACD DIV LAG1(SMA20))에 대한 상세 분석
"""

import os
# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# FutureWarning 억제
warnings.filterwarnings('ignore', category=FutureWarning)

from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics
from factor_factory.data import ParquetCache, DATA_ROOT

# 시각화 모듈 (선택적 import)
try:
    from factor_factory.visualization import create_trading_report
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("📊 차트 기능을 사용하려면 matplotlib을 설치하세요: pip install matplotlib")

def analyze_best_program():
    """최적 프로그램 상세 분석"""
    
    # 프로그램 정보
    tokens = [3, 18, 24, 10]  # DIV MACD LAG1 SMA20
    formula = tokens_to_infix(tokens)
    
    print("=" * 80)
    print("🏆 BEST PROGRAM DETAILED ANALYSIS")
    print("=" * 80)
    print(f"🧮 수식: {formula}")
    print(f"🔢 토큰: {tokens}")
    print(f"📏 트리 깊이: {calc_tree_depth(tokens)}")
    print(f"📝 설명: MACD를 1일 지연된 SMA20으로 나눈 값")
    print("=" * 80)
    
    # 데이터 로드
    try:
        df = ParquetCache(DATA_ROOT).load("BTCUSDT", "1h")
        print(f"📈 데이터: BTCUSDT_1h")
        print(f"📅 기간: {df.index[0]} ~ {df.index[-1]} ({len(df):,} bars)")
    except FileNotFoundError:
        print("❌ BTCUSDT_1h 데이터를 찾을 수 없습니다.")
        return None
    
    # 시그널 생성
    try:
        print("\n🔄 시그널 생성 중...")
        sig = eval_prefix(tokens, df).dropna().rename("signal")
        price = df["close"].reindex(sig.index)
        
        print(f"📊 시그널 통계:")
        print(f"   - 범위: [{sig.min():.4f}, {sig.max():.4f}]")
        print(f"   - 평균: {sig.mean():.4f}")
        print(f"   - 표준편차: {sig.std():.4f}")
        print(f"   - 0이 아닌 신호: {(sig.abs() > 0.01).sum():,}/{len(sig):,} ({(sig.abs() > 0.01).mean():.2%})")
        
        # 강한 신호 분석 (절댓값이 0.5 이상)
        strong_signals = sig.abs() > 0.5
        if strong_signals.any():
            strong_long = (sig > 0.5).sum()
            strong_short = (sig < -0.5).sum()
            print(f"   - 강한 롱 신호: {strong_long:,} ({strong_long/len(sig):.2%})")
            print(f"   - 강한 숏 신호: {strong_short:,} ({strong_short/len(sig):.2%})")
        
    except Exception as e:
        print(f"❌ 시그널 생성 오류: {e}")
        return None
    
    # 백테스트
    try:
        print(f"\n🔄 백테스트 실행 중...")
        equity, pnl = vector_backtest(price, sig)
        metrics = compute_metrics(pnl, equity, sig)
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE METRICS")
        print("=" * 80)
        
        # 주요 메트릭
        print(f"💰 CAGR (연평균 성장률):        {metrics['cagr']:8.2%}")
        print(f"📈 Sharpe Ratio:              {metrics['sharpe']:8.4f}")
        print(f"📉 Max Drawdown:              {metrics['mdd']:8.2%}")
        print(f"🏆 Calmar Ratio:              {metrics['calmar']:8.4f}")
        print(f"🎯 Win Rate:                  {metrics['win_rate']:8.2%}")
        print(f"💎 Profit Factor:             {metrics['profit_factor']:8.4f}")
        
        # 거래 통계
        print(f"\n📈 TRADING STATISTICS")
        print(f"🔢 Total Trades:              {metrics['total_trades']:8d}")
        print(f"🔄 Turnover:                  {metrics['turnover']:8.2f}")
        print(f"💵 Avg Trade PnL:             {metrics['avg_trade_pnl']:8.6f}")
        print(f"📉 Max Consecutive Losses:    {metrics['max_consecutive_losses']:8d} days")
        print(f"📊 Information Ratio:         {metrics['information_ratio']:8.4f}")
        
        # 수익률 분석
        total_return = (equity.iloc[-1] - 1) * 100
        annual_vol = pnl.std() * np.sqrt(252 * 24) * 100  # 시간당 → 연간
        
        print(f"\n💰 RETURN ANALYSIS")
        print(f"🎊 Total Return:              {total_return:8.2f}%")
        print(f"📊 Annual Volatility:         {annual_vol:8.2f}%")
        print(f"🏦 Risk-Free Rate (가정):      {0:8.1f}%")
        
        # 월별/연도별 통계
        monthly_returns = pnl.resample('ME').sum() * 100  # 'M' deprecated, use 'ME'
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        
        print(f"\n📅 PERIOD ANALYSIS")
        print(f"📈 Positive Months:           {positive_months:8d}/{total_months} ({positive_months/total_months:.1%})")
        print(f"📊 Best Month:                {monthly_returns.max():8.2f}%")
        print(f"📉 Worst Month:               {monthly_returns.min():8.2f}%")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 백테스트 오류: {e}")
        return None
    
    # 차트 생성
    if CHARTS_AVAILABLE:
        try:
            print(f"\n📊 상세 차트 생성 중...")
            create_trading_report(
                price=price,
                signal=sig,
                equity=equity,
                pnl=pnl,
                metrics=metrics,
                formula=formula,
                output_dir="best_program_analysis"
            )
            
            # 추가 분석을 위한 데이터 저장
            analysis_data = pd.DataFrame({
                'price': price,
                'signal': sig,
                'equity': equity,
                'pnl': pnl,
                'cumulative_pnl': pnl.cumsum()
            })
            
            analysis_data.to_csv("best_program_analysis/detailed_data.csv")
            print(f"📁 상세 데이터 저장: best_program_analysis/detailed_data.csv")
            
        except Exception as e:
            print(f"⚠️ 차트 생성 중 오류: {e}")
    else:
        print(f"\n📊 차트를 생성하려면 다음을 실행하세요:")
        print(f"   pip install matplotlib")
    
    # 전략 해석
    print(f"\n" + "=" * 80)
    print("🧠 STRATEGY INTERPRETATION")
    print("=" * 80)
    print(f"이 전략은 MACD를 1일 지연된 SMA20으로 나눈 비율을 사용합니다.")
    print(f"- MACD: 단기(12일) EMA - 장기(26일) EMA")
    print(f"- LAG1(SMA20): 1일 전의 20일 단순이동평균")
    print(f"- 해석: MACD 모멘텀이 과거 추세 대비 얼마나 강한지 측정")
    print(f"- 양수: 상승 모멘텀이 과거 추세보다 강함 → 롱 신호")
    print(f"- 음수: 하락 모멘텀이 과거 추세보다 강함 → 숏 신호")
    print("=" * 80)
    
    return {
        'formula': formula,
        'tokens': tokens,
        'metrics': metrics,
        'signal_stats': {
            'range': [sig.min(), sig.max()],
            'mean': sig.mean(),
            'std': sig.std()
        }
    }

if __name__ == "__main__":
    result = analyze_best_program()
    if result:
        print(f"\n✅ 분석 완료! 상세 결과는 'best_program_analysis' 폴더를 확인하세요.")
    else:
        print(f"\n❌ 분석 실패!")
