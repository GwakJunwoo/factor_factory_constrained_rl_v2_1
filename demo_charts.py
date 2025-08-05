#!/usr/bin/env python3
"""
시각화 기능 데모 - 샘플 데이터로 차트 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 시각화 모듈 테스트
try:
    from factor_factory.visualization import create_trading_report, plot_trading_analysis
    print("✅ 시각화 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 시각화 모듈 로드 실패: {e}")
    print("matplotlib 설치 필요: pip install matplotlib")
    exit(1)

def create_sample_data(n_days=1000):
    """샘플 트레이딩 데이터 생성"""
    
    # 날짜 인덱스 생성
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # 가격 데이터 (랜덤워크 + 트렌드)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)  # 일일 수익률
    price = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    
    # 트레이딩 시그널 생성 (모멘텀 기반)
    sma_short = price.rolling(10).mean()
    sma_long = price.rolling(30).mean()
    
    # 단순 이동평균 크로스오버 전략
    raw_signal = (sma_short / sma_long - 1) * 10  # 정규화
    signal = np.tanh(raw_signal).fillna(0)  # [-1, 1] 범위로 압축
    
    # 이산 신호로 변환
    discrete_signal = pd.Series(0, index=dates)
    discrete_signal[signal > 0.3] = 1    # 롱
    discrete_signal[signal < -0.3] = -1  # 숏
    
    # 백테스트 (간단한 벡터화)
    position = discrete_signal.shift(1).fillna(0)
    daily_returns = price.pct_change().fillna(0)
    strategy_returns = position * daily_returns
    
    # 수수료 적용
    turnover = position.diff().abs().fillna(0)
    costs = turnover * 0.001  # 0.1% 수수료
    net_returns = strategy_returns - costs
    
    # 누적 수익률
    equity = (1 + net_returns).cumprod()
    
    return price, signal, equity, net_returns

def demo_visualization():
    """시각화 데모 실행"""
    
    print("📊 샘플 데이터로 시각화 데모를 실행합니다...")
    
    # 샘플 데이터 생성
    price, signal, equity, pnl = create_sample_data()
    
    # 간단한 메트릭 계산
    total_return = (equity.iloc[-1] - 1) * 100
    volatility = pnl.std() * np.sqrt(252) * 100
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
    max_dd = ((equity / equity.cummax()) - 1).min() * 100
    
    metrics = {
        'cagr': (equity.iloc[-1] ** (252/len(equity)) - 1),
        'sharpe': sharpe,
        'mdd': max_dd / 100,
        'turnover': signal.diff().abs().sum(),
        'calmar': (equity.iloc[-1] ** (252/len(equity)) - 1) / abs(max_dd/100) if max_dd != 0 else 0,
        'win_rate': (pnl > 0).mean(),
        'profit_factor': pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else float('inf'),
        'information_ratio': sharpe,
        'max_consecutive_losses': 5,  # 샘플값
        'total_trades': int((signal.diff() != 0).sum()),
        'avg_trade_pnl': pnl.mean()
    }
    
    print(f"📈 샘플 데이터 통계:")
    print(f"   - 기간: {len(price)} 일")
    print(f"   - 총 수익률: {total_return:.2f}%")
    print(f"   - 샤프 비율: {sharpe:.2f}")
    print(f"   - 최대 낙폭: {max_dd:.2f}%")
    
    # 단일 차트 테스트
    print("\n📊 트레이딩 분석 차트 생성 중...")
    try:
        plot_trading_analysis(
            price=price,
            signal=signal,
            equity=equity,
            pnl=pnl,
            title="Sample Trading Strategy Demo",
            save_path="demo_chart.png"
        )
        print("✅ 단일 차트 생성 성공: demo_chart.png")
    except Exception as e:
        print(f"❌ 단일 차트 생성 실패: {e}")
    
    # 완전한 리포트 생성
    print("\n📊 완전한 트레이딩 리포트 생성 중...")
    try:
        create_trading_report(
            price=price,
            signal=signal,
            equity=equity,
            pnl=pnl,
            metrics=metrics,
            formula="(SMA10 / SMA30 - 1) * 10",  # 샘플 수식
            output_dir="demo_charts"
        )
        print("✅ 완전한 리포트 생성 성공!")
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")

if __name__ == "__main__":
    demo_visualization()
