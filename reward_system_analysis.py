#!/usr/bin/env python3
"""
강화학습 보상 시스템 상세 분석 및 시각화
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_reward_components():
    """보상 시스템 구성 요소 분석"""
    
    print("🏆 FACTOR FACTORY 강화학습 보상 시스템 분석")
    print("=" * 60)
    
    # 실제 설정값
    config = {
        'lambda_depth': 0.002,
        'lambda_turnover': 0.0005, 
        'lambda_const1': 2.0,
        'lambda_std': 0.5,
        'length_penalty': 0.0005,
        'commission': 0.0008,
        'slippage': 0.0015,
        'signal_delay': 1,
        'execution_delay': 1
    }
    
    print("📊 현재 설정값:")
    for key, value in config.items():
        print(f"  {key:20}: {value}")
    
    # 가상 시나리오 분석
    scenarios = [
        {"name": "Simple Strategy", "pnl": 0.05, "depth": 2, "trades": 20, "const_ratio": 0.1, "std": 2.0},
        {"name": "Complex Strategy", "pnl": 0.08, "depth": 5, "trades": 100, "const_ratio": 0.0, "std": 3.0},
        {"name": "Overfit Strategy", "pnl": 0.12, "depth": 8, "trades": 500, "const_ratio": 0.0, "std": 1.0},
        {"name": "Constant Strategy", "pnl": 0.02, "depth": 1, "trades": 5, "const_ratio": 0.8, "std": 0.5},
    ]
    
    print(f"\n🎯 시나리오별 보상 분석:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'PnL':>8} {'Final':>8} {'Depth':>6} {'Trade':>6} {'Const':>6} {'Std':>6}")
    print("-" * 80)
    
    for scenario in scenarios:
        pnl = scenario["pnl"]
        depth_pen = config['lambda_depth'] * scenario["depth"]
        turnover_pen = config['lambda_turnover'] * scenario["trades"] 
        const_pen = config['lambda_const1'] * scenario["const_ratio"]
        std_pen = config['lambda_std'] / scenario["std"]
        
        final_reward = pnl - depth_pen - turnover_pen - const_pen - std_pen
        
        print(f"{scenario['name']:<20} {pnl:>7.4f} {final_reward:>7.4f} "
              f"{-depth_pen:>6.4f} {-turnover_pen:>6.4f} {-const_pen:>6.4f} {-std_pen:>6.4f}")
    
    return scenarios, config

def plot_reward_sensitivity():
    """보상 함수의 파라미터 민감도 분석"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('보상 함수 파라미터 민감도 분석', fontsize=16, fontweight='bold')
    
    # 기본 시나리오
    base_pnl = 0.05
    base_depth = 3
    base_trades = 50
    base_const = 0.2
    base_std = 2.0
    
    # 1. 깊이 페널티 영향
    ax1 = axes[0, 0]
    depths = np.arange(1, 10)
    lambda_depths = [0.001, 0.002, 0.005, 0.01]
    
    for lambda_d in lambda_depths:
        rewards = [base_pnl - lambda_d * d for d in depths]
        ax1.plot(depths, rewards, label=f'λ_depth={lambda_d}', marker='o')
    
    ax1.set_xlabel('Tree Depth')
    ax1.set_ylabel('Reward')
    ax1.set_title('깊이 페널티 영향')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 거래횟수 페널티 영향  
    ax2 = axes[0, 1]
    trades = np.arange(10, 200, 10)
    lambda_turnovers = [0.0001, 0.0005, 0.001, 0.002]
    
    for lambda_t in lambda_turnovers:
        rewards = [base_pnl - lambda_t * t for t in trades]
        ax2.plot(trades, rewards, label=f'λ_turnover={lambda_t}', marker='s')
    
    ax2.set_xlabel('Number of Trades')
    ax2.set_ylabel('Reward') 
    ax2.set_title('거래횟수 페널티 영향')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 상수 사용 페널티
    ax3 = axes[1, 0]
    const_ratios = np.linspace(0, 1, 20)
    lambda_consts = [0.5, 1.0, 2.0, 5.0]
    
    for lambda_c in lambda_consts:
        rewards = [base_pnl - lambda_c * c for c in const_ratios]
        ax3.plot(const_ratios, rewards, label=f'λ_const={lambda_c}', marker='^')
    
    ax3.set_xlabel('Constant Usage Ratio')
    ax3.set_ylabel('Reward')
    ax3.set_title('상수 사용 페널티 영향')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 변동성 페널티
    ax4 = axes[1, 1]
    stds = np.linspace(0.5, 5.0, 20)
    lambda_stds = [0.1, 0.5, 1.0, 2.0]
    
    for lambda_s in lambda_stds:
        rewards = [base_pnl - lambda_s / s for s in stds]
        ax4.plot(stds, rewards, label=f'λ_std={lambda_s}', marker='d')
    
    ax4.set_xlabel('Signal Standard Deviation')
    ax4.set_ylabel('Reward')
    ax4.set_title('변동성 페널티 영향')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_dir = Path("reward_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "reward_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"📊 민감도 분석 차트 저장: {output_dir / 'reward_sensitivity.png'}")
    
    return fig

def plot_realistic_trading_timeline():
    """현실적 거래 타이밍 시각화"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 시간축 생성
    times = np.arange(0, 10)
    price_data = 100 + np.cumsum(np.random.randn(10) * 0.5)
    
    # 타임라인 시각화
    ax.plot(times, price_data, 'k-', linewidth=2, label='Price', alpha=0.7)
    
    # 신호 생성 시점 (t=3)
    signal_time = 3
    ax.axvline(signal_time, color='blue', linestyle='--', alpha=0.7, label='Signal Generated')
    ax.text(signal_time + 0.1, price_data[signal_time] + 1, 'Signal\nGenerated\n(uses data t≤3)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # 거래 결정 시점 (t=4, signal_delay=1)
    decision_time = signal_time + 1
    ax.axvline(decision_time, color='orange', linestyle='--', alpha=0.7, label='Trading Decision')
    ax.text(decision_time + 0.1, price_data[decision_time] - 1, 'Trading\nDecision\n(+1 delay)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # 실제 체결 시점 (t=5, execution_delay=1)
    execution_time = decision_time + 1
    ax.axvline(execution_time, color='red', linestyle='--', alpha=0.7, label='Order Execution')
    ax.text(execution_time + 0.1, price_data[execution_time] + 1, 'Order\nExecution\n(+1 delay)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # 수익률 실현 시점 (t=6)
    pnl_time = execution_time + 1
    ax.axvline(pnl_time, color='green', linestyle='--', alpha=0.7, label='PnL Realization')
    ax.text(pnl_time + 0.1, price_data[pnl_time] - 1, 'PnL\nRealization\n(position × return)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # 미래 정보 누출 방지 영역 표시
    ax.axvspan(signal_time + 0.01, 10, alpha=0.1, color='red', 
               label='Future Data\n(NOT USED)')
    
    ax.set_xlabel('Time Periods', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('현실적 거래 타이밍 및 미래 정보 누출 방지', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 정보 상자 추가
    info_text = """
현실적 거래 조건:
• Signal Delay: 1 period
• Execution Delay: 1 period  
• Commission: 0.08%
• Slippage: 0.15%
• Market Impact: 0.02%
• Total Delay: 3 periods
    """
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke"))
    
    plt.tight_layout()
    
    # 저장
    output_dir = Path("reward_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "realistic_trading_timeline.png", dpi=300, bbox_inches='tight')
    print(f"📊 거래 타이밍 차트 저장: {output_dir / 'realistic_trading_timeline.png'}")
    
    return fig

def generate_learning_progress_simulation():
    """학습 진행 과정 시뮬레이션"""
    
    # 가상의 학습 데이터 생성
    episodes = np.arange(1, 1001)
    
    # 초기에는 낮고 점진적으로 증가하는 보상
    base_reward = -0.5 + 0.8 * (1 - np.exp(-episodes / 200))
    noise = np.random.normal(0, 0.2, len(episodes))
    rewards = base_reward + noise
    
    # 이동평균으로 부드러운 트렌드 생성
    window = 50
    reward_ma = pd.Series(rewards).rolling(window).mean()
    
    # 성공률 (유효한 프로그램 비율)
    success_rate = 0.1 + 0.8 * (1 - np.exp(-episodes / 150))
    success_noise = np.random.normal(0, 0.05, len(episodes))
    success_rate = np.clip(success_rate + success_noise, 0, 1)
    
    # 캐시 적중률
    cache_hit_rate = 0.2 + 0.6 * (1 - np.exp(-episodes / 100))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('강화학습 진행 과정 시뮬레이션', fontsize=16, fontweight='bold')
    
    # 1. 보상 변화
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Reward')
    ax1.plot(episodes, reward_ma, color='red', linewidth=2, label=f'{window}-Episode MA')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('보상 진화')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 성공률 변화
    ax2 = axes[0, 1]
    ax2.plot(episodes, success_rate * 100, color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('유효한 프로그램 생성 비율')
    ax2.grid(True, alpha=0.3)
    
    # 3. 캐시 적중률
    ax3 = axes[1, 0]
    ax3.plot(episodes, cache_hit_rate * 100, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cache Hit Rate (%)')
    ax3.set_title('캐시 적중률 (탐색 효율성)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 페널티 구성 요소 변화
    ax4 = axes[1, 1]
    depth_penalty = 0.005 * np.exp(-episodes / 300)  # 깊이 페널티 감소
    turnover_penalty = 0.003 * np.exp(-episodes / 250)  # 거래 페널티 감소
    const_penalty = 0.4 * np.exp(-episodes / 180)  # 상수 페널티 감소
    
    ax4.plot(episodes, depth_penalty, label='Depth Penalty', alpha=0.8)
    ax4.plot(episodes, turnover_penalty, label='Turnover Penalty', alpha=0.8)
    ax4.plot(episodes, const_penalty, label='Constant Penalty', alpha=0.8)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Penalty')
    ax4.set_title('페널티 구성 요소 변화')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_dir = Path("reward_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "learning_progress.png", dpi=300, bbox_inches='tight')
    print(f"📊 학습 진행 차트 저장: {output_dir / 'learning_progress.png'}")
    
    return fig

if __name__ == "__main__":
    print("🚀 Factor Factory 강화학습 보상 시스템 상세 분석 시작\n")
    
    # 1. 보상 구성 요소 분석
    scenarios, config = analyze_reward_components()
    
    # 2. 파라미터 민감도 분석
    print(f"\n📈 보상 함수 민감도 분석 중...")
    plot_reward_sensitivity()
    
    # 3. 현실적 거래 타이밍 시각화
    print(f"\n⏰ 현실적 거래 타이밍 분석 중...")
    plot_realistic_trading_timeline()
    
    # 4. 학습 진행 과정 시뮬레이션
    print(f"\n🎯 학습 진행 과정 시뮬레이션 중...")
    generate_learning_progress_simulation()
    
    print(f"\n✅ 모든 분석 완료!")
    print(f"📁 결과 파일들이 'reward_analysis/' 폴더에 저장되었습니다.")
    print(f"   - reward_sensitivity.png: 파라미터 민감도 분석")
    print(f"   - realistic_trading_timeline.png: 현실적 거래 타이밍")  
    print(f"   - learning_progress.png: 학습 진행 과정 시뮬레이션")
