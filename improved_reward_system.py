#!/usr/bin/env python3
"""
개선된 보상 시스템 구현 및 테스트
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
class ImprovedRLCConfig:
    """개선된 RL 환경 설정"""
    max_len: int = 21
    
    # === 개선된 페널티 가중치 ===
    # 기존 → 개선
    lambda_depth: float = 0.01        # 0.002 → 0.01 (5배 증가: 복잡도 강력 억제)
    lambda_turnover: float = 0.002    # 0.0005 → 0.002 (4배 증가: 과거래 억제)
    lambda_const1: float = 0.5        # 2.0 → 0.5 (1/4로 감소: 상수 사용 허용)
    lambda_std: float = 0.1           # 0.5 → 0.1 (1/5로 감소: 변동성 페널티 완화)
    
    # === 새로운 페널티 추가 ===
    lambda_drawdown: float = 2.0      # MDD 페널티
    lambda_consistency: float = 0.5   # 수익률 일관성 페널티
    lambda_skewness: float = 0.1      # 편향성 페널티
    
    # === 보상 시스템 타입 ===
    reward_type: str = 'improved_basic'  # 'basic', 'improved_basic', 'sharpe', 'multi_objective'
    
    # === 동적 보상 설정 ===
    adaptive_weights: bool = True
    performance_window: int = 100
    reward_normalization: bool = True
    reward_clipping: float = 3.0

class RewardSystem:
    """개선된 보상 시스템"""
    
    def __init__(self, config: ImprovedRLCConfig):
        self.config = config
        self.episode_count = 0
        self.reward_history = []
        self.performance_buffer = []
        
    def calculate_reward(
        self,
        pnl: pd.Series,
        equity: pd.Series,
        signal: pd.Series,
        tokens: List[int],
        depth: int,
        trades: int
    ) -> Dict:
        """통합 보상 계산 함수"""
        
        self.episode_count += 1
        
        if self.config.reward_type == 'basic':
            reward, components = self._basic_reward(pnl, signal, tokens, depth, trades)
        elif self.config.reward_type == 'improved_basic':
            reward, components = self._improved_basic_reward(pnl, equity, signal, tokens, depth, trades)
        elif self.config.reward_type == 'sharpe':
            reward, components = self._sharpe_based_reward(pnl, equity, signal, tokens, depth, trades)
        elif self.config.reward_type == 'multi_objective':
            reward, components = self._multi_objective_reward(pnl, equity, signal, tokens, depth, trades)
        else:
            raise ValueError(f"Unknown reward type: {self.config.reward_type}")
        
        # 적응적 가중치 적용
        if self.config.adaptive_weights:
            reward = self._apply_adaptive_weights(reward, components)
        
        # 보상 정규화 및 클리핑
        if self.config.reward_normalization:
            reward = self._normalize_reward(reward)
        
        # 히스토리 업데이트
        self.reward_history.append(reward)
        
        return {
            'total_reward': reward,
            'components': components,
            'episode': self.episode_count
        }
    
    def _basic_reward(self, pnl, signal, tokens, depth, trades):
        """기존 보상 시스템"""
        pnl_sum = float(pnl.sum())
        
        # 기존 페널티
        from collections import Counter
        cnt = Counter(tokens)
        const_ratio = cnt.get(12, 0) / len(tokens)
        std_pen = self.config.lambda_std / (signal.std() + 1e-8)
        
        components = {
            'pnl': pnl_sum,
            'depth_penalty': -self.config.lambda_depth * depth,
            'turnover_penalty': -self.config.lambda_turnover * trades,
            'const_penalty': -self.config.lambda_const1 * const_ratio,
            'std_penalty': -std_pen
        }
        
        total = sum(components.values())
        return total, components
    
    def _improved_basic_reward(self, pnl, equity, signal, tokens, depth, trades):
        """개선된 기본 보상 시스템"""
        pnl_sum = float(pnl.sum())
        
        # 기존 페널티 (개선된 가중치)
        from collections import Counter
        cnt = Counter(tokens)
        const_ratio = cnt.get(12, 0) / len(tokens)
        std_pen = self.config.lambda_std / (signal.std() + 1e-8)
        
        # 새로운 페널티들
        max_drawdown = abs((equity / equity.cummax() - 1).min())
        drawdown_penalty = self.config.lambda_drawdown * max_drawdown
        
        # 수익률 일관성 (변동 계수의 역수)
        pnl_cv = abs(pnl.mean()) / (pnl.std() + 1e-8)
        consistency_bonus = self.config.lambda_consistency * min(pnl_cv, 2.0)
        
        # 편향성 페널티 (극단적 skewness 억제)
        try:
            skewness = abs(pnl.skew())
            skewness_penalty = self.config.lambda_skewness * max(0, skewness - 1.0)
        except:
            skewness_penalty = 0.0
        
        components = {
            'pnl': pnl_sum,
            'depth_penalty': -self.config.lambda_depth * depth,
            'turnover_penalty': -self.config.lambda_turnover * trades,
            'const_penalty': -self.config.lambda_const1 * const_ratio,
            'std_penalty': -std_pen,
            'drawdown_penalty': -drawdown_penalty,
            'consistency_bonus': consistency_bonus,
            'skewness_penalty': -skewness_penalty
        }
        
        total = sum(components.values())
        return total, components
    
    def _sharpe_based_reward(self, pnl, equity, signal, tokens, depth, trades):
        """Sharpe Ratio 기반 보상"""
        
        # Sharpe Ratio 계산
        sharpe = pnl.mean() / (pnl.std() + 1e-8)
        sharpe_scaled = np.tanh(sharpe * 2.0)  # [-1, 1] 범위로 스케일링
        
        # 구조적 페널티 (경량화)
        structure_penalty = (
            self.config.lambda_depth * 0.5 * depth +
            self.config.lambda_turnover * 0.5 * trades +
            self.config.lambda_const1 * 0.5 * (tokens.count(12) / len(tokens))
        )
        
        components = {
            'sharpe_ratio': sharpe_scaled * 5.0,  # 주 보상
            'structure_penalty': -structure_penalty
        }
        
        total = sum(components.values())
        return total, components
    
    def _multi_objective_reward(self, pnl, equity, signal, tokens, depth, trades):
        """다목적 최적화 보상"""
        
        # 1. 수익성 점수 (30%)
        total_return = (equity.iloc[-1] - 1) * 100
        return_score = np.tanh(total_return / 10)  # ±10% 기준
        
        # 2. 위험성 점수 (25%)
        volatility = pnl.std() * np.sqrt(252 * 24) * 100
        risk_score = max(0, 1 - volatility / 20)  # 20% 변동성 기준
        
        # 3. 안정성 점수 (25%)
        max_dd = abs((equity / equity.cummax() - 1).min())
        stability_score = max(0, 1 - max_dd / 0.2)  # 20% MDD 기준
        
        # 4. 효율성 점수 (20%)
        win_rate = (pnl > 0).sum() / len(pnl)
        efficiency_score = win_rate * 2 - 1  # [0,1] → [-1,1]
        
        # 가중 평균
        multi_score = (
            0.3 * return_score +
            0.25 * risk_score +
            0.25 * stability_score +
            0.2 * efficiency_score
        )
        
        # 구조적 페널티 (경량화)
        structure_penalty = (
            self.config.lambda_depth * 0.3 * depth +
            self.config.lambda_turnover * 0.3 * trades
        )
        
        components = {
            'multi_objective': multi_score * 3.0,  # 주 보상
            'return_component': return_score * 0.3 * 3.0,
            'risk_component': risk_score * 0.25 * 3.0,
            'stability_component': stability_score * 0.25 * 3.0,
            'efficiency_component': efficiency_score * 0.2 * 3.0,
            'structure_penalty': -structure_penalty
        }
        
        total = sum(components.values())
        return total, components
    
    def _apply_adaptive_weights(self, reward, components):
        """적응적 가중치 적용"""
        
        # 초기 단계 (0-1000): 탐색 격려
        if self.episode_count <= 1000:
            exploration_bonus = 0.1
            return reward + exploration_bonus
        
        # 중간 단계 (1000-5000): 균형
        elif self.episode_count <= 5000:
            return reward
        
        # 후기 단계 (5000+): 수렴 격려
        else:
            # 성과가 정체되면 탐색 격려
            if len(self.reward_history) >= 100:
                recent_rewards = self.reward_history[-100:]
                if np.std(recent_rewards) < 0.1:  # 성과 정체
                    exploration_bonus = 0.05
                    return reward + exploration_bonus
            
            return reward
    
    def _normalize_reward(self, reward):
        """보상 정규화 및 클리핑"""
        
        self.reward_history.append(reward)
        
        # 충분한 히스토리가 있으면 Z-score 정규화
        if len(self.reward_history) > 50:
            history = np.array(self.reward_history[-self.config.performance_window:])
            mean = history.mean()
            std = history.std() + 1e-8
            normalized = (reward - mean) / std
        else:
            normalized = reward
        
        # 클리핑
        clipped = np.clip(normalized, -self.config.reward_clipping, self.config.reward_clipping)
        
        # 탄젠트 스케일링 [-1, 1]
        final_reward = np.tanh(clipped)
        
        return final_reward

# 시연 함수
def demonstrate_reward_systems():
    """다양한 보상 시스템 시연"""
    
    print("🎯 개선된 보상 시스템 시연")
    print("=" * 50)
    
    # 가상 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    # 시나리오 1: 고수익 고위험 전략
    pnl1 = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)
    equity1 = (1 + pnl1).cumprod()
    signal1 = pd.Series(np.random.normal(0, 1, 1000), index=dates)
    
    # 시나리오 2: 저수익 저위험 전략  
    pnl2 = pd.Series(np.random.normal(0.0005, 0.005, 1000), index=dates)
    equity2 = (1 + pnl2).cumprod()
    signal2 = pd.Series(np.random.normal(0, 0.5, 1000), index=dates)
    
    # 시나리오 3: 불안정한 전략
    pnl3 = pd.Series(np.random.normal(0, 0.03, 1000), index=dates) 
    pnl3.iloc[100:150] = -0.05  # 큰 손실 구간
    equity3 = (1 + pnl3).cumprod()
    signal3 = pd.Series(np.random.normal(0, 2, 1000), index=dates)
    
    scenarios = [
        ("고수익 고위험", pnl1, equity1, signal1, [1,2,3,4,5], 3, 50),
        ("저수익 저위험", pnl2, equity2, signal2, [1,2,12,12], 2, 10),
        ("불안정한 전략", pnl3, equity3, signal3, [1,2,3,4,5,6,7,8], 5, 100)
    ]
    
    reward_types = ['basic', 'improved_basic', 'sharpe', 'multi_objective']
    
    for reward_type in reward_types:
        print(f"\n📊 {reward_type.upper()} 보상 시스템")
        print("-" * 30)
        
        config = ImprovedRLCConfig(reward_type=reward_type)
        reward_system = RewardSystem(config)
        
        for name, pnl, equity, signal, tokens, depth, trades in scenarios:
            result = reward_system.calculate_reward(pnl, equity, signal, tokens, depth, trades)
            
            print(f"{name:12s}: {result['total_reward']:8.4f}")
            
            # 구성 요소 출력 (improved_basic만)
            if reward_type == 'improved_basic':
                for comp, value in result['components'].items():
                    print(f"  {comp:18s}: {value:8.4f}")
                print()

if __name__ == "__main__":
    demonstrate_reward_systems()
