#!/usr/bin/env python3
"""
트레이딩 시그널과 성과를 시각화하는 모듈
"""

import os
# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

# 경고 메시지 억제 (FutureWarning for 'M' deprecated)
warnings.filterwarnings('ignore', category=FutureWarning)

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_trading_analysis(
    price: pd.Series,
    signal: pd.Series,
    equity: pd.Series,
    pnl: pd.Series,
    title: str = "Trading Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    트레이딩 분석 차트를 생성합니다.
    
    Args:
        price: 가격 시리즈
        signal: 시그널 시리즈 (-1: 숏, 0: 플랫, 1: 롱)
        equity: 누적 수익률 시리즈
        pnl: 일일 손익 시리즈
        title: 차트 제목
        save_path: 저장 경로 (None이면 화면에 표시)
        figsize: 차트 크기
    """
    
    # 시그널 변화점 감지 (진입/청산 지점)
    signal_diff = signal.diff().fillna(0)
    
    # 진입 신호
    long_entries = signal_diff > 0.5  # 롱 진입
    short_entries = signal_diff < -0.5  # 숏 진입
    
    # 청산 신호
    exits = (signal_diff.abs() > 0.5) & (signal.shift(1).abs() > 0.1) & (signal.abs() < 0.1)
    
    # 서브플롯 생성
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 가격 차트 + 매수/매도 신호
    ax1 = axes[0]
    ax1.plot(price.index, price.values, 'k-', linewidth=1, label='Price', alpha=0.8)
    
    # 매수 신호 (녹색 삼각형 위쪽)
    long_points = price[long_entries]
    if not long_points.empty:
        ax1.scatter(long_points.index, long_points.values, 
                   marker='^', color='green', s=100, label='Long Entry', zorder=5)
    
    # 매도 신호 (빨간색 삼각형 아래쪽)
    short_points = price[short_entries]
    if not short_points.empty:
        ax1.scatter(short_points.index, short_points.values, 
                   marker='v', color='red', s=100, label='Short Entry', zorder=5)
    
    # 청산 신호 (주황색 X)
    exit_points = price[exits]
    if not exit_points.empty:
        ax1.scatter(exit_points.index, exit_points.values, 
                   marker='x', color='orange', s=80, label='Exit', zorder=5)
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Price Chart with Trading Signals', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 누적 수익률 (Equity Curve)
    ax2 = axes[1]
    equity_pct = (equity - 1) * 100  # 백분율로 변환
    ax2.plot(equity.index, equity_pct.values, 'b-', linewidth=2, label='Cumulative Return')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 최대 낙폭 영역 표시
    cummax = equity_pct.cummax()
    drawdown = equity_pct - cummax
    ax2.fill_between(equity.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_title('Cumulative Return & Drawdown', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 일일 손익 (Daily PnL)
    ax3 = axes[2]
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in pnl.values]
    ax3.bar(pnl.index, pnl.values * 100, color=colors, alpha=0.7, width=0.8)
    ax3.axhline(y=0, color='black', linewidth=1)
    
    ax3.set_ylabel('Daily PnL (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Daily Profit & Loss', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # X축 날짜 포맷팅
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 차트 저장: {save_path}")
    else:
        plt.show()

def plot_signal_distribution(signal: pd.Series, save_path: Optional[str] = None) -> None:
    """시그널 분포를 히스토그램으로 표시"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 시그널 히스토그램
    ax1.hist(signal.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(signal.mean(), color='red', linestyle='--', label=f'Mean: {signal.mean():.4f}')
    ax1.set_xlabel('Signal Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Signal Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 시그널 시계열
    ax2.plot(signal.index, signal.values, linewidth=1, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Long Threshold')
    ax2.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5, label='Short Threshold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal Value')
    ax2.set_title('Signal Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 시그널 분포 차트 저장: {save_path}")
    else:
        plt.show()

def plot_monthly_returns(pnl: pd.Series, save_path: Optional[str] = None) -> None:
    """월별 수익률 히트맵"""
    
    # 월별 수익률 계산 (FutureWarning 방지)
    monthly_returns = pnl.resample('ME').sum() * 100  # 백분율
    
    # 연도와 월 추출
    years = monthly_returns.index.year.unique()
    months = range(1, 13)
    
    # 히트맵 데이터 생성
    heatmap_data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        year_data = monthly_returns[monthly_returns.index.year == year]
        for date, value in year_data.items():
            month = date.month - 1  # date가 인덱스이므로 date.month 사용
            heatmap_data[i, month] = value
    
    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.4)))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    # 축 설정
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    
    # 값 표시
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    
    # 컬러바
    cbar = plt.colorbar(im)
    cbar.set_label('Monthly Return (%)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 월별 수익률 히트맵 저장: {save_path}")
    else:
        plt.show()

def create_trading_report(
    price: pd.Series,
    signal: pd.Series, 
    equity: pd.Series,
    pnl: pd.Series,
    metrics: dict,
    formula: str,
    output_dir: str = "charts"
) -> None:
    """완전한 트레이딩 리포트 생성"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 메인 트레이딩 분석 차트
    title = f"Trading Analysis - {formula}"
    plot_trading_analysis(
        price, signal, equity, pnl,
        title=title,
        save_path=str(output_path / "trading_analysis.png")
    )
    
    # 시그널 분포 차트
    plot_signal_distribution(
        signal,
        save_path=str(output_path / "signal_distribution.png")
    )
    
    # 월별 수익률 히트맵
    plot_monthly_returns(
        pnl,
        save_path=str(output_path / "monthly_returns.png")
    )
    
    # 성과 요약 텍스트 파일
    summary_text = f"""
트레이딩 전략 성과 리포트
{'='*50}

전략 수식: {formula}

주요 성과 지표:
- CAGR: {metrics['cagr']:.2%}
- Sharpe Ratio: {metrics['sharpe']:.4f}
- Max Drawdown: {metrics['mdd']:.2%}
- Calmar Ratio: {metrics['calmar']:.4f}
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.4f}

거래 통계:
- 총 거래 수: {metrics['total_trades']}
- 회전율: {metrics['turnover']:.2f}
- 평균 거래 손익: {metrics['avg_trade_pnl']:.6f}
- 최대 연속 손실: {metrics['max_consecutive_losses']} days

시그널 통계:
- 시그널 범위: [{signal.min():.4f}, {signal.max():.4f}]
- 평균: {signal.mean():.4f}
- 표준편차: {signal.std():.4f}
- 롱 신호 비율: {(signal > 0.1).mean():.2%}
- 숏 신호 비율: {(signal < -0.1).mean():.2%}
- 플랫 비율: {(abs(signal) <= 0.1).mean():.2%}

최종 수익률: {((equity.iloc[-1] - 1) * 100):.2f}%
"""
    
    with open(output_path / "performance_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"📊 완전한 트레이딩 리포트 생성 완료: {output_path}")
    print(f"   - trading_analysis.png: 메인 분석 차트")
    print(f"   - signal_distribution.png: 시그널 분포")
    print(f"   - monthly_returns.png: 월별 수익률 히트맵")
    print(f"   - performance_summary.txt: 성과 요약")
