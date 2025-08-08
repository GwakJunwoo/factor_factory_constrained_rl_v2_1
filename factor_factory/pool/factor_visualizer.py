#!/usr/bin/env python3
"""
Factor Visualizer - 팩터 시각화 및 비교 분석

기능:
1. 팩터 성능 비교 차트
2. 팩터 상관관계 분석
3. 리스크-리턴 산점도
4. 시계열 성과 비교
5. 팩터 특성 히트맵
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from .factor_pool import FactorPool, FactorRecord

plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FactorVisualizer:
    """팩터 시각화 클래스"""
    
    def __init__(self, factor_pool: FactorPool):
        self.pool = factor_pool
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_performance_comparison(
        self,
        factor_ids: List[str] = None,
        n_top: int = 10,
        figsize: tuple = (15, 10),
        save_path: str = None
    ):
        """팩터 성능 비교 차트"""
        
        if factor_ids is None:
            factors = self.pool.get_top_factors(n_top)
        else:
            factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if not factors:
            print("⚠️ 비교할 팩터가 없습니다")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('🏆 Factor Performance Comparison', fontsize=16, fontweight='bold')
        
        # 데이터 준비
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'volatility']
        metric_labels = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor', 'Volatility (%)']
        
        factor_names = [f.name for f in factors]
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//3, i%3]
            
            values = [getattr(f, metric) for f in factors]
            
            # 백분율 변환
            if metric in ['total_return', 'max_drawdown', 'win_rate', 'volatility']:
                values = [v * 100 for v in values]
            
            bars = ax.bar(range(len(factors)), values, color=self.colors[:len(factors)])
            
            # 값 표시
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(label, fontweight='bold')
            ax.set_xticks(range(len(factors)))
            ax.set_xticklabels([f.factor_id[:8] for f in factors], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 기준선 추가
            if metric == 'sharpe_ratio':
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Good (1.0)')
            elif metric == 'win_rate':
                ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 성능 비교 차트 저장: {save_path}")
        
        plt.show()
    
    def plot_risk_return_scatter(
        self,
        factor_ids: List[str] = None,
        n_top: int = 20,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """리스크-리턴 산점도"""
        
        if factor_ids is None:
            factors = self.pool.get_top_factors(n_top)
        else:
            factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if not factors:
            print("⚠️ 분석할 팩터가 없습니다")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 데이터 준비
        returns = [f.total_return * 100 for f in factors]
        risks = [f.volatility * 100 for f in factors]
        sharpes = [f.sharpe_ratio for f in factors]
        drawdowns = [f.max_drawdown * 100 for f in factors]
        
        # 샤프 비율에 따른 색상
        scatter = ax.scatter(risks, returns, c=sharpes, s=[100 + dd*5 for dd in drawdowns], 
                           alpha=0.7, cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # 컬러바
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio', fontweight='bold')
        
        # 팩터 ID 라벨
        for i, factor in enumerate(factors):
            ax.annotate(factor.factor_id[:8], (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 효율적 프론티어 가이드라인
        efficient_risks = np.linspace(min(risks), max(risks), 100)
        efficient_returns = []
        
        for risk in efficient_risks:
            # 해당 리스크 레벨에서 최대 수익률 찾기
            candidates = [(r, ret) for r, ret in zip(risks, returns) if abs(r - risk) < 2]
            if candidates:
                efficient_returns.append(max(candidates, key=lambda x: x[1])[1])
            else:
                efficient_returns.append(0)
        
        ax.plot(efficient_risks, efficient_returns, 'r--', alpha=0.5, label='Efficient Frontier (Guide)')
        
        ax.set_xlabel('Risk (Volatility %)', fontweight='bold')
        ax.set_ylabel('Return (%)', fontweight='bold')
        ax.set_title('🎯 Risk-Return Scatter Plot\n(Size: Max Drawdown, Color: Sharpe Ratio)', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 사분면 구분선
        mean_return = np.mean(returns)
        mean_risk = np.mean(risks)
        ax.axhline(y=mean_return, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=mean_risk, color='gray', linestyle='-', alpha=0.3)
        
        # 사분면 라벨
        ax.text(max(risks)*0.9, max(returns)*0.9, 'High Risk\nHigh Return', 
               ha='center', va='center', fontweight='bold', alpha=0.6)
        ax.text(min(risks)*1.1, max(returns)*0.9, 'Low Risk\nHigh Return', 
               ha='center', va='center', fontweight='bold', alpha=0.6, color='green')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 리스크-리턴 산점도 저장: {save_path}")
        
        plt.show()
    
    def plot_equity_curves(
        self,
        factor_ids: List[str],
        figsize: tuple = (15, 8),
        save_path: str = None
    ):
        """팩터 수익률 곡선 비교"""
        
        factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if not factors:
            print("⚠️ 비교할 팩터가 없습니다")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 수익률 곡선
        for i, factor in enumerate(factors):
            timeseries = self.pool.get_factor_timeseries(factor.factor_id)
            
            if 'equity' in timeseries:
                equity = timeseries['equity']
                ax1.plot(equity.index, equity.values, 
                        label=f'{factor.factor_id[:8]} (R:{factor.total_return:.1%})',
                        color=self.colors[i % len(self.colors)], linewidth=2)
                
                # 드로우다운 계산
                cummax = equity.cummax()
                drawdown = (equity - cummax) / cummax
                ax2.fill_between(drawdown.index, drawdown.values, 0, 
                               alpha=0.3, color=self.colors[i % len(self.colors)])
        
        ax1.set_title('📈 Factor Equity Curves Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Cumulative Return', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('📉 Drawdown Comparison', fontweight='bold')
        ax2.set_ylabel('Drawdown', fontweight='bold')
        ax2.set_xlabel('Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 수익률 곡선 비교 저장: {save_path}")
        
        plt.show()
    
    def plot_factor_characteristics_heatmap(
        self,
        factor_ids: List[str] = None,
        n_top: int = 15,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """팩터 특성 히트맵"""
        
        if factor_ids is None:
            factors = self.pool.get_top_factors(n_top)
        else:
            factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if not factors:
            print("⚠️ 분석할 팩터가 없습니다")
            return
        
        # 데이터 준비
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                  'profit_factor', 'volatility', 'depth', 'length']
        
        data = []
        factor_names = []
        
        for factor in factors:
            row = []
            for metric in metrics:
                value = getattr(factor, metric)
                # 정규화를 위해 백분율 변환
                if metric in ['total_return', 'max_drawdown', 'win_rate', 'volatility']:
                    value *= 100
                row.append(value)
            data.append(row)
            factor_names.append(factor.factor_id[:8])
        
        # DataFrame 생성
        df = pd.DataFrame(data, columns=metrics, index=factor_names)
        
        # 정규화 (0-1 스케일)
        df_normalized = (df - df.min()) / (df.max() - df.min())
        
        # max_drawdown은 낮을수록 좋으므로 역정규화
        df_normalized['max_drawdown'] = 1 - df_normalized['max_drawdown']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 히트맵 생성
        sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Normalized Score (0=Worst, 1=Best)'}, ax=ax)
        
        ax.set_title('🔥 Factor Characteristics Heatmap\n(Normalized Scores)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Factors', fontweight='bold')
        
        # 컬럼 라벨 회전
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 팩터 특성 히트맵 저장: {save_path}")
        
        plt.show()
    
    def plot_factor_correlation_matrix(
        self,
        factor_ids: List[str],
        figsize: tuple = (10, 8),
        save_path: str = None
    ):
        """팩터 간 상관관계 매트릭스"""
        
        factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if len(factors) < 2:
            print("⚠️ 상관관계 분석을 위해 최소 2개 팩터가 필요합니다")
            return
        
        # 각 팩터의 PnL 시계열 수집
        pnl_data = {}
        
        for factor in factors:
            timeseries = self.pool.get_factor_timeseries(factor.factor_id)
            if 'pnl' in timeseries:
                pnl_data[factor.factor_id[:8]] = timeseries['pnl']
        
        if len(pnl_data) < 2:
            print("⚠️ PnL 데이터가 부족합니다")
            return
        
        # DataFrame 생성 및 상관관계 계산
        df = pd.DataFrame(pnl_data)
        correlation_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 히트맵 생성
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdBu_r', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('🔗 Factor Correlation Matrix\n(PnL Correlations)', 
                    fontweight='bold', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 상관관계 매트릭스 저장: {save_path}")
        
        plt.show()
    
    def create_factor_report(
        self,
        factor_ids: List[str] = None,
        n_top: int = 10,
        save_dir: str = "factor_reports"
    ):
        """종합 팩터 분석 리포트 생성"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("📊 Factor Analysis Report 생성 중...")
        
        # 1. 성능 비교
        self.plot_performance_comparison(
            factor_ids, n_top, 
            save_path=f"{save_dir}/01_performance_comparison.png"
        )
        
        # 2. 리스크-리턴 산점도
        self.plot_risk_return_scatter(
            factor_ids, n_top, 
            save_path=f"{save_dir}/02_risk_return_scatter.png"
        )
        
        # 3. 특성 히트맵
        self.plot_factor_characteristics_heatmap(
            factor_ids, n_top, 
            save_path=f"{save_dir}/03_characteristics_heatmap.png"
        )
        
        # 4. 상위 5개 팩터의 수익률 곡선
        if factor_ids is None:
            top_factors = self.pool.get_top_factors(5)
            top_ids = [f.factor_id for f in top_factors]
        else:
            top_ids = factor_ids[:5]
        
        if top_ids:
            self.plot_equity_curves(
                top_ids, 
                save_path=f"{save_dir}/04_equity_curves.png"
            )
        
        # 5. 상관관계 매트릭스 (상위 8개)
        if factor_ids is None:
            top_factors = self.pool.get_top_factors(8)
            corr_ids = [f.factor_id for f in top_factors]
        else:
            corr_ids = factor_ids[:8]
        
        if len(corr_ids) >= 2:
            self.plot_factor_correlation_matrix(
                corr_ids, 
                save_path=f"{save_dir}/05_correlation_matrix.png"
            )
        
        print(f"✅ Factor Analysis Report 완료: {save_dir}")

if __name__ == "__main__":
    # 테스트
    from factor_pool import FactorPool
    
    pool = FactorPool("test_factor_pool")
    visualizer = FactorVisualizer(pool)
    
    # 샘플 리포트 생성
    visualizer.create_factor_report(n_top=5, save_dir="sample_factor_reports")
