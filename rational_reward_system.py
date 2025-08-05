#!/usr/bin/env python3
"""
합리적으로 개선된 보상 시스템
- 이중 페널티 제거 (PnL에 이미 거래비용 반영됨)
- 실용적이고 균형잡힌 보상 구조
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RationalRLCConfig:
    """합리적으로 개선된 RL 환경 설정"""
    max_len: int = 21
    
    # === 핵심 페널티 (간소화) ===
    lambda_complexity: float = 0.005    # 복잡도 페널티 (depth + length)
    lambda_overfitting: float = 1.0     # 과적합 방지 (MDD 기반)
    lambda_instability: float = 0.3     # 불안정성 페널티 (변동성 기반)
    
    # === 품질 보너스 ===
    alpha_consistency: float = 0.2      # 일관성 보너스
    alpha_efficiency: float = 0.1       # 효율성 보너스 (승률 기반)
    
    # === 보상 스케일링 ===
    reward_scale: float = 10.0          # 주 보상 스케일링
    penalty_cap: float = 2.0            # 페널티 상한
    
    # === 정규화 설정 ===
    normalize_rewards: bool = True
    reward_window: int = 200

class RationalRewardSystem:
    """합리적으로 설계된 보상 시스템"""
    
    def __init__(self, config: RationalRLCConfig):
        self.config = config
        self.episode_count = 0
        self.reward_history = []
        
    def calculate_reward(
        self,
        pnl: pd.Series,
        equity: pd.Series, 
        signal: pd.Series,
        tokens: List[int],
        depth: int
    ) -> Dict:
        """메인 보상 계산 함수"""
        
        self.episode_count += 1
        
        # 1. 주 보상: PnL (이미 거래비용 반영됨)
        net_pnl = float(pnl.sum())
        main_reward = net_pnl * self.config.reward_scale
        
        # 2. 복잡도 페널티 (구조적 단순함 격려)
        complexity_penalty = self._calculate_complexity_penalty(tokens, depth)
        
        # 3. 과적합 페널티 (MDD 기반)
        overfitting_penalty = self._calculate_overfitting_penalty(equity)
        
        # 4. 불안정성 페널티 (변동성 기반)
        instability_penalty = self._calculate_instability_penalty(pnl)
        
        # 5. 품질 보너스
        consistency_bonus = self._calculate_consistency_bonus(pnl)
        efficiency_bonus = self._calculate_efficiency_bonus(pnl)
        
        # 구성 요소
        components = {
            'main_reward': main_reward,
            'complexity_penalty': -complexity_penalty,
            'overfitting_penalty': -overfitting_penalty, 
            'instability_penalty': -instability_penalty,
            'consistency_bonus': consistency_bonus,
            'efficiency_bonus': efficiency_bonus
        }
        
        # 총 보상 계산
        total_penalty = complexity_penalty + overfitting_penalty + instability_penalty
        total_bonus = consistency_bonus + efficiency_bonus
        
        # 페널티 캡 적용 (과도한 페널티 방지)
        capped_penalty = min(total_penalty, self.config.penalty_cap)
        
        raw_reward = main_reward - capped_penalty + total_bonus
        
        # 보상 정규화
        if self.config.normalize_rewards:
            final_reward = self._normalize_reward(raw_reward)
        else:
            final_reward = raw_reward
        
        # 히스토리 업데이트
        self.reward_history.append(final_reward)
        
        return {
            'total_reward': final_reward,
            'raw_reward': raw_reward,
            'components': components,
            'episode': self.episode_count,
            'penalty_capped': total_penalty > self.config.penalty_cap
        }
    
    def _calculate_complexity_penalty(self, tokens: List[int], depth: int) -> float:
        """복잡도 페널티: 깊이 + 길이"""
        
        # 깊이 페널티 (지수적 증가)
        depth_penalty = (depth / 10) ** 1.5
        
        # 길이 페널티 (선형 증가)
        length_penalty = len(tokens) / 100
        
        complexity = depth_penalty + length_penalty
        return self.config.lambda_complexity * complexity
    
    def _calculate_overfitting_penalty(self, equity: pd.Series) -> float:
        """과적합 페널티: MDD 기반"""
        
        try:
            # 최대 낙폭 계산
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax
            max_drawdown = abs(drawdown.min())
            
            # MDD가 클수록 과적합 가능성 높음
            # 20% 이상 MDD는 강하게 페널티
            if max_drawdown > 0.2:
                overfitting = (max_drawdown - 0.2) * 2.0 + 0.2
            else:
                overfitting = max_drawdown
                
            return self.config.lambda_overfitting * overfitting
            
        except:
            return 0.0
    
    def _calculate_instability_penalty(self, pnl: pd.Series) -> float:
        """불안정성 페널티: 변동성 기반"""
        
        try:
            # 일일 PnL 변동성
            pnl_volatility = pnl.std()
            
            # 변동성이 평균 수익률 대비 너무 클 때 페널티
            mean_pnl = abs(pnl.mean())
            if mean_pnl > 1e-8:
                volatility_ratio = pnl_volatility / mean_pnl
                # 변동성/수익률 비율이 5배 이상이면 불안정
                instability = max(0, volatility_ratio - 5.0) / 10.0
            else:
                # 수익률이 0에 가까우면 변동성 자체가 문제
                instability = pnl_volatility * 100
                
            return self.config.lambda_instability * instability
            
        except:
            return 0.0
    
    def _calculate_consistency_bonus(self, pnl: pd.Series) -> float:
        """일관성 보너스: 안정적 수익 패턴 격려"""
        
        try:
            if len(pnl) < 10:
                return 0.0
                
            # 수익률의 부호 일관성
            positive_periods = (pnl > 0).sum()
            total_periods = len(pnl)
            
            # 승률이 50%에서 멀어질수록 일관성 있음
            win_rate = positive_periods / total_periods
            consistency_score = abs(win_rate - 0.5)  # 0~0.5
            
            # 추가: 연속 손실 기간 확인
            consecutive_losses = self._max_consecutive_losses(pnl)
            if consecutive_losses > 10:  # 10일 연속 손실은 페널티
                consistency_score *= 0.5
                
            return self.config.alpha_consistency * consistency_score
            
        except:
            return 0.0
    
    def _calculate_efficiency_bonus(self, pnl: pd.Series) -> float:
        """효율성 보너스: 높은 승률 + 수익/손실 비율"""
        
        try:
            if len(pnl) < 5:
                return 0.0
                
            positive_pnl = pnl[pnl > 0]
            negative_pnl = pnl[pnl < 0]
            
            if len(positive_pnl) == 0 or len(negative_pnl) == 0:
                return 0.0
                
            # 평균 수익 대 평균 손실 비율
            avg_profit = positive_pnl.mean()
            avg_loss = abs(negative_pnl.mean())
            
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
            
            # 승률
            win_rate = len(positive_pnl) / len(pnl)
            
            # 효율성 = 승률 * 수익손실비율
            efficiency = win_rate * min(profit_loss_ratio, 3.0)  # 비율 캡
            
            return self.config.alpha_efficiency * efficiency
            
        except:
            return 0.0
    
    def _max_consecutive_losses(self, pnl: pd.Series) -> int:
        """최대 연속 손실 기간 계산"""
        
        consecutive = 0
        max_consecutive = 0
        
        for p in pnl:
            if p <= 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive
    
    def _normalize_reward(self, raw_reward: float) -> float:
        """보상 정규화"""
        
        self.reward_history.append(raw_reward)
        
        # 윈도우 크기 유지
        if len(self.reward_history) > self.config.reward_window:
            self.reward_history.pop(0)
        
        # 충분한 히스토리가 있으면 Z-score 정규화
        if len(self.reward_history) > 20:
            history = np.array(self.reward_history[-self.config.reward_window:])
            mean = history.mean()
            std = history.std() + 1e-8
            
            normalized = (raw_reward - mean) / std
            
            # 클리핑 및 스케일링
            clipped = np.clip(normalized, -3.0, 3.0)
            final = np.tanh(clipped)  # [-1, 1] 범위
            
            return final
        else:
            # 초기에는 단순 스케일링
            return np.tanh(raw_reward)

# 간소화된 환경 설정 클래스
@dataclass 
class SimplifiedRLCConfig:
    """즉시 적용 가능한 간소화된 설정"""
    max_len: int = 21
    
    # === 단순화된 페널티 (기존 대비) ===
    lambda_depth: float = 0.008       # 기존 0.002 → 0.008 (4배 증가)
    lambda_const1: float = 0.3        # 기존 2.0 → 0.3 (상수 사용 허용)  
    lambda_std: float = 0.05          # 기존 0.5 → 0.05 (변동성 페널티 대폭 완화)
    
    # === 제거된 페널티 ===
    # lambda_turnover: 제거됨 (PnL에 이미 거래비용 반영)
    
    # === 새로운 안정성 페널티 ===
    lambda_drawdown: float = 1.5      # MDD 페널티 추가
    
    # 기타 설정
    commission: float = 0.0008
    slippage: float = 0.0015
    leverage: int = 1
    long_threshold: float = 1.5
    short_threshold: float = -1.5

def demonstrate_rational_rewards():
    """합리적 보상 시스템 시연"""
    
    print("🎯 합리적 보상 시스템 시연")
    print("=" * 50)
    
    # 설정
    config = RationalRLCConfig()
    reward_system = RationalRewardSystem(config)
    
    # 가상 시나리오
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    scenarios = [
        # (이름, PnL 특성, 토큰 특성)
        ("균형잡힌 전략", "stable_profit", [1,2,3,4,5], 3),
        ("고수익 고위험", "high_vol", [1,2,3,4,5,6,7,8,9], 5), 
        ("과도하게 복잡", "normal", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 8),
        ("상수 의존", "stable_profit", [12,12,12,12], 1),
        ("불안정한 전략", "unstable", [1,2,3,4,5], 3)
    ]
    
    print(f"{'전략':15s} {'총보상':>8s} {'주보상':>8s} {'복잡도':>8s} {'과적합':>8s} {'불안정':>8s}")
    print("-" * 70)
    
    for name, pnl_type, tokens, depth in scenarios:
        # PnL 생성
        if pnl_type == "stable_profit":
            pnl = pd.Series(np.random.normal(0.0005, 0.002, 1000), index=dates)
        elif pnl_type == "high_vol": 
            pnl = pd.Series(np.random.normal(0.001, 0.01, 1000), index=dates)
        elif pnl_type == "unstable":
            pnl = pd.Series(np.random.normal(0, 0.005, 1000), index=dates)
            pnl.iloc[100:200] = -0.02  # 큰 손실 구간
        else:  # normal
            pnl = pd.Series(np.random.normal(0.0003, 0.003, 1000), index=dates)
        
        equity = (1 + pnl).cumprod()
        signal = pd.Series(np.random.normal(0, 1, 1000), index=dates)
        
        # 보상 계산
        result = reward_system.calculate_reward(pnl, equity, signal, tokens, depth)
        
        comp = result['components']
        print(f"{name:15s} {result['total_reward']:8.4f} {comp['main_reward']:8.4f} "
              f"{comp['complexity_penalty']:8.4f} {comp['overfitting_penalty']:8.4f} {comp['instability_penalty']:8.4f}")
        
        # 첫 번째 시나리오는 상세 출력
        if name == "균형잡힌 전략":
            print(f"\n📊 상세 분석: {name}")
            print(f"   - 일관성 보너스: {comp['consistency_bonus']:8.4f}")
            print(f"   - 효율성 보너스: {comp['efficiency_bonus']:8.4f}")
            print(f"   - 페널티 캡 적용: {'Yes' if result['penalty_capped'] else 'No'}")
            print(f"   - Raw 보상: {result['raw_reward']:8.4f}")
            print()

def compare_old_vs_new():
    """기존 vs 새로운 보상 시스템 비교"""
    
    print("\n🔄 기존 vs 새로운 보상 시스템 비교")
    print("=" * 50)
    
    print("기존 시스템의 문제점:")
    print("❌ 거래횟수 이중 페널티 (PnL에 이미 반영됨)")
    print("❌ 과도한 상수 사용 억제 (lambda_const1=2.0)")  
    print("❌ 부적절한 변동성 페널티 (낮은 변동성에 큰 페널티)")
    print("❌ 복잡도 페널티 부족 (lambda_depth=0.002)")
    
    print("\n새로운 시스템의 개선점:")
    print("✅ 거래횟수 페널티 제거 (이중 페널티 해결)")
    print("✅ 상수 사용 허용 (lambda_const1=0.3)")
    print("✅ 합리적 변동성 처리 (안정성 vs 불안정성 구분)")
    print("✅ 강화된 복잡도 억제 (lambda_complexity=0.005)")
    print("✅ MDD 기반 과적합 방지")
    print("✅ 일관성/효율성 보너스 추가")
    print("✅ 페널티 캡으로 과도한 억제 방지")
    
    print("\n권장 적용 순서:")
    print("1️⃣ 즉시 적용: lambda_turnover 제거, 가중치 조정")
    print("2️⃣ 단기 적용: MDD 페널티 추가")  
    print("3️⃣ 중기 적용: 품질 보너스 시스템")
    print("4️⃣ 장기 적용: 완전한 합리적 보상 시스템")

if __name__ == "__main__":
    demonstrate_rational_rewards()
    compare_old_vs_new()
