#!/usr/bin/env python3
"""
V3 프로그램 성과 평가 스크립트
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.signal_generator import generate_signal_realtime
from factor_factory.backtest.realistic_engine import realistic_backtest
from factor_factory.rlc.utils import tokens_to_infix
import matplotlib.pyplot as plt

def load_v3_program():
    """V3 모델의 최고 성과 프로그램 로드"""
    
    with open('rlc_out/best_program.json', 'r') as f:
        data = json.load(f)
    
    tokens = data['tokens']
    reward = data.get('reward', 'Unknown')
    
    print(f"📊 V3 최고 성과 프로그램")
    print(f"보상: {reward}")
    print(f"토큰: {tokens}")
    print(f"수식: {tokens_to_infix(tokens)}")
    print("-" * 60)
    
    return tokens

def evaluate_v3_program():
    """V3 프로그램 전체 성과 평가"""
    
    print("🎯 V3 프로그램 성과 평가 시작")
    print("=" * 60)
    
    # 1. 프로그램 로드
    tokens = load_v3_program()
    
    # 2. 데이터 로드
    print("📈 데이터 로딩...")
    df = pd.read_parquet('data_cache/BTCUSDT_1h.parquet')
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"데이터 크기: {len(df):,} rows")
    
    # 3. 신호 생성
    print("\n🔧 신호 생성 중...")
    try:
        raw_signal = eval_prefix(tokens, df)
        print(f"Raw 신호 생성 완료: {len(raw_signal):,} points")
        
        # 현실적 신호 변환
        signal = generate_signal_realtime(
            raw_signal,
            lookback_window=252,
            long_threshold=1.5,
            short_threshold=-1.5,
            min_periods=50,
            rebalance_frequency='D'
        )
        print(f"현실적 신호 변환 완료: {len(signal):,} points")
        
    except Exception as e:
        print(f"❌ 신호 생성 실패: {e}")
        return None
    
    # 4. 백테스트 실행
    print("\n💰 백테스트 실행 중...")
    try:
        price = df["close"].reindex(signal.index)
        
        equity, pnl = realistic_backtest(
            price, signal,
            commission=0.0008,      # 0.08%
            slippage=0.0015,        # 0.15%
            leverage=1,
            signal_delay=1,
            execution_delay=1,
            max_position_change=1.0,
            impact_factor=0.0002
        )
        
        print(f"백테스트 완료: {len(equity):,} periods")
        
    except Exception as e:
        print(f"❌ 백테스트 실패: {e}")
        return None
    
    # 5. 성과 분석
    print("\n📊 성과 분석")
    print("-" * 40)
    
    # 기본 지표
    total_return = (equity.iloc[-1] - 1) * 100
    annual_return = total_return * (365.25 * 24 / len(pnl))
    
    # 위험 지표
    volatility = pnl.std() * np.sqrt(365.25 * 24) * 100
    
    # MDD 계산
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = abs(drawdown.min()) * 100
    
    # 거래 지표
    trades = (signal.diff() != 0).sum()
    win_trades = (pnl > 0).sum()
    lose_trades = (pnl <= 0).sum()
    win_rate = win_trades / len(pnl) * 100 if len(pnl) > 0 else 0
    
    # Sharpe ratio
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(365.25 * 24)
    
    # 결과 출력
    print(f"총 수익률:     {total_return:8.2f}%")
    print(f"연간 수익률:   {annual_return:8.2f}%")
    print(f"변동성:       {volatility:8.2f}%")
    print(f"최대 낙폭:     {max_drawdown:8.2f}%")
    print(f"샤프 비율:     {sharpe:8.3f}")
    print(f"승률:         {win_rate:8.1f}%")
    print(f"총 거래횟수:   {trades:8d}")
    
    # 6. 비교 분석 (Buy & Hold와 비교)
    print("\n🔄 Buy & Hold 대비 분석")
    print("-" * 40)
    
    bnh_return = (price.iloc[-1] / price.iloc[0] - 1) * 100
    bnh_annual = bnh_return * (365.25 * 24 / len(price))
    
    price_changes = price.pct_change().dropna()
    bnh_volatility = price_changes.std() * np.sqrt(365.25 * 24) * 100
    bnh_sharpe = price_changes.mean() / (price_changes.std() + 1e-8) * np.sqrt(365.25 * 24)
    
    # Buy & Hold MDD
    bnh_cummax = price.cummax()
    bnh_drawdown = (price - bnh_cummax) / bnh_cummax
    bnh_max_drawdown = abs(bnh_drawdown.min()) * 100
    
    print(f"{'지표':15s} {'V3 전략':>10s} {'Buy&Hold':>10s} {'차이':>10s}")
    print("-" * 50)
    print(f"{'총 수익률':15s} {total_return:9.2f}% {bnh_return:9.2f}% {total_return-bnh_return:+9.2f}%")
    print(f"{'연간 수익률':15s} {annual_return:9.2f}% {bnh_annual:9.2f}% {annual_return-bnh_annual:+9.2f}%")
    print(f"{'변동성':15s} {volatility:9.2f}% {bnh_volatility:9.2f}% {volatility-bnh_volatility:+9.2f}%")
    print(f"{'최대 낙폭':15s} {max_drawdown:9.2f}% {bnh_max_drawdown:9.2f}% {max_drawdown-bnh_max_drawdown:+9.2f}%")
    print(f"{'샤프 비율':15s} {sharpe:9.3f} {bnh_sharpe:9.3f} {sharpe-bnh_sharpe:+9.3f}")
    
    # 7. 월별 수익률 분석
    print("\n📅 월별 수익률 분석")
    print("-" * 40)
    
    # PnL을 월별로 그룹화
    pnl_monthly = pnl.resample('M').sum() * 100
    positive_months = (pnl_monthly > 0).sum()
    total_months = len(pnl_monthly)
    
    print(f"수익 월:       {positive_months:8d}")
    print(f"손실 월:       {total_months - positive_months:8d}")
    print(f"월 승률:       {positive_months/total_months*100:8.1f}%")
    print(f"최고 월 수익:   {pnl_monthly.max():8.2f}%")
    print(f"최악 월 손실:   {pnl_monthly.min():8.2f}%")
    
    # 8. 시각화 저장을 위한 데이터 반환
    return {
        'equity': equity,
        'pnl': pnl,
        'signal': signal,
        'price': price,
        'tokens': tokens,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades': trades
        }
    }

def create_v3_report():
    """V3 성과 리포트 생성"""
    
    print("\n🎯 V3 성과 리포트 생성 중...")
    
    result = evaluate_v3_program()
    if result is None:
        print("❌ 평가 실패로 리포트 생성 불가")
        return
    
    # 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 누적 수익률
    ax1 = axes[0, 0]
    ax1.plot(result['equity'].index, (result['equity'] - 1) * 100, 'b-', linewidth=2, label='V3 Strategy')
    
    # Buy & Hold 비교
    bnh_equity = result['price'] / result['price'].iloc[0]
    ax1.plot(result['price'].index, (bnh_equity - 1) * 100, 'gray', alpha=0.7, label='Buy & Hold')
    
    ax1.set_title('누적 수익률 비교', fontsize=14, fontweight='bold')
    ax1.set_ylabel('수익률 (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 일별 PnL
    ax2 = axes[0, 1]
    colors = ['red' if x < 0 else 'green' for x in result['pnl']]
    ax2.bar(range(len(result['pnl'][-100:])), result['pnl'][-100:] * 100, 
            color=colors[-100:], alpha=0.7, width=1.0)
    ax2.set_title('최근 100일 일별 PnL', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PnL (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 신호 분포
    ax3 = axes[1, 0]
    ax3.hist(result['signal'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('신호 분포', fontsize=14, fontweight='bold')
    ax3.set_xlabel('신호 값')
    ax3.set_ylabel('빈도')
    ax3.grid(True, alpha=0.3)
    
    # 4. 드로우다운
    ax4 = axes[1, 1]
    cummax = result['equity'].cummax()
    drawdown = (result['equity'] - cummax) / cummax * 100
    ax4.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax4.plot(drawdown.index, drawdown, 'red', linewidth=1)
    ax4.set_title('드로우다운', fontsize=14, fontweight='bold')
    ax4.set_ylabel('드로우다운 (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    report_path = 'output/v3_performance_report.png'
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"✅ 성과 리포트 저장: {report_path}")
    
    plt.show()
    
    # 메트릭 저장
    metrics_path = 'output/v3_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(result['metrics'], f, indent=2)
    print(f"✅ 메트릭 저장: {metrics_path}")

if __name__ == "__main__":
    create_v3_report()
