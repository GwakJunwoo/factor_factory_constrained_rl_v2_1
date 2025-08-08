#!/usr/bin/env python3
"""
MCTS용 Factor Environment

기존 ProgramEnv를 MCTS에 맞게 조정:
- 스텝별 실행 대신 완성된 프로그램의 일괄 평가
- 신경망 학습용 데이터 수집
- 기존 PPO 환경과 호환되는 평가 방식
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..rlc.env import ProgramEnv, RLCConfig
from ..rlc.grammar import ARITY
from ..rlc.utils import tokens_to_infix
from ..rlc.compiler import eval_prefix
from ..rlc.signal_generator import generate_signal_realtime
from ..backtest.realistic_engine import realistic_backtest
from ..data import ParquetCache


class MCTSFactorEnv:
    """MCTS용 팩터 환경"""
    
    def __init__(self, df: pd.DataFrame, config: RLCConfig):
        """
        Args:
            df: 가격 데이터
            config: RL 환경 설정
        """
        self.df = df
        self.cfg = config
        
        # 기존 환경의 캐시와 통계 시스템 활용
        self.base_env = ProgramEnv(df, config)
        
        # 평가 통계
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.evaluation_times = []
        
        print(f"✅ MCTS Factor Environment 초기화")
        print(f"  데이터 크기: {len(df)} 행")
        print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
    
    def evaluate_program(self, tokens: List[int]) -> Dict:
        """
        완성된 프로그램 평가
        
        Args:
            tokens: 완성된 토큰 시퀀스
        
        Returns:
            평가 결과 딕셔너리
        """
        self.total_evaluations += 1
        
        try:
            # 기존 환경의 평가 로직 재사용
            result = self._evaluate_with_base_env(tokens)
            
            if result['success']:
                self.successful_evaluations += 1
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'reward': -1.0,
                'error': str(e),
                'tokens': tokens
            }
    
    def _evaluate_with_base_env(self, tokens: List[int]) -> Dict:
        """기존 환경을 사용한 평가"""
        
        # 환경 리셋
        self.base_env.reset()
        
        # 토큰 시퀀스 단계별 실행
        reward = 0.0
        info = {}
        
        for i, token in enumerate(tokens):
            obs, step_reward, done, truncated, step_info = self.base_env.step(token)
            reward += step_reward
            info.update(step_info)
            
            if done:
                break
        
        # 성공 여부 판단
        success = (
            done and 
            not info.get('invalid', False) and
            not info.get('error', False) and
            not info.get('future_leak', False)
        )
        
        # 복잡성 페널티 추가
        if success:
            complexity_penalty = self._calculate_complexity_penalty(tokens)
            reward = float(reward) - complexity_penalty
        
        return {
            'success': success,
            'reward': float(reward),
            'info': info,
            'tokens': tokens,
            'formula': tokens_to_infix(tokens) if success else None
        }
    
    def get_legal_actions(self, tokens: List[int], need: int) -> List[int]:
        """현재 상태에서 가능한 액션들"""
        
        legal_actions = []
        max_len = min(self.cfg.max_len, 20)  # 최대 깊이 20으로 확장
        
        for token in range(25):  # 0~24
            # 길이 제한
            if len(tokens) >= max_len:
                continue
            
            # 새로운 need 계산
            new_need = need - 1 + ARITY[token]
            
            # need 조건 확인
            if new_need > max_len - len(tokens) - 1:
                continue
            if new_need < 0:
                continue
            
            # LAG1 토큰(12) 과다 사용 방지 (4개까지 허용)
            if token == 12:  # LAG1
                lag_count = tokens.count(12)
                if lag_count >= 4:  # 4개 이상 사용했으면 제외
                    continue
            
            # 연속 같은 토큰 사용 방지
            if len(tokens) >= 2 and tokens[-1] == tokens[-2] == token:
                continue
            
            legal_actions.append(token)
        
        return legal_actions
    
    def _calculate_complexity_penalty(self, tokens: List[int]) -> float:
        """복잡성 페널티 계산"""
        
        # 기본 길이 페널티 (매우 가벼움)
        length_penalty = len(tokens) * 0.005
        
        # 반복 패턴 페널티 (같은 토큰 연속 사용)
        repetition_penalty = 0.0
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        # 3번 이상 연속 사용시 페널티
        if max_consecutive >= 3:
            repetition_penalty = (max_consecutive - 2) * 0.2
        
        # LAG 과다 사용 페널티 (토큰 12가 LAG1)
        lag_count = tokens.count(12)  # LAG1 토큰
        lag_penalty = max(0, (lag_count - 4) * 0.08)  # 4개 초과시 가벼운 페널티
        
        # 깊이 페널티 (18 이상시만 적용, 매우 가벼움)
        depth_penalty = max(0, (len(tokens) - 18) * 0.02)
        
        total_penalty = length_penalty + repetition_penalty + lag_penalty + depth_penalty
        
        return total_penalty
    
    def is_terminal(self, tokens: List[int], need: int) -> bool:
        """터미널 상태 여부"""
        return need == 0
    
    def state_to_observation(self, tokens: List[int], need: int) -> np.ndarray:
        """상태를 신경망 입력 형식으로 변환"""
        
        obs = np.zeros(23, dtype=np.float32)
        
        # 토큰 히스토그램
        if tokens:
            for tok in tokens:
                if 0 <= tok < 23:
                    obs[tok] += 1
        
        # 정규화
        obs = obs / max(1, len(tokens))
        
        return obs
    
    def generate_training_data(
        self, 
        action_sequences: List[List[int]], 
        rewards: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MCTS 탐색 결과로부터 신경망 학습 데이터 생성
        
        Args:
            action_sequences: 액션 시퀀스들
            rewards: 각 시퀀스의 보상
        
        Returns:
            states: 상태 배치 [N, obs_dim]
            policy_targets: 정책 타겟 [N, action_dim] 
            value_targets: 가치 타겟 [N]
        """
        states = []
        policy_targets = []
        value_targets = []
        
        for seq, reward in zip(action_sequences, rewards):
            # 시퀀스의 각 중간 상태에서 학습 데이터 생성
            need = 1
            
            for i, action in enumerate(seq):
                current_tokens = seq[:i]
                
                # 상태 표현
                state = self.state_to_observation(current_tokens, need)
                states.append(state)
                
                # 정책 타겟 (실제 선택된 액션에 확률 1)
                policy_target = np.zeros(25)
                policy_target[action] = 1.0
                policy_targets.append(policy_target)
                
                # 가치 타겟 (최종 보상)
                value_targets.append(reward)
                
                # need 업데이트
                need = need - 1 + ARITY[action]
                
                if need == 0:
                    break
        
        return (
            np.array(states, dtype=np.float32),
            np.array(policy_targets, dtype=np.float32),
            np.array(value_targets, dtype=np.float32)
        )
    
    def get_statistics(self) -> Dict:
        """환경 통계 반환"""
        
        success_rate = (
            self.successful_evaluations / self.total_evaluations 
            if self.total_evaluations > 0 else 0
        )
        
        avg_eval_time = (
            np.mean(self.evaluation_times) 
            if self.evaluation_times else 0
        )
        
        base_stats = {}
        if hasattr(self.base_env, 'total_programs_evaluated'):
            base_stats = {
                'cache_hits': getattr(self.base_env, 'cache_hits', 0),
                'total_programs': getattr(self.base_env, 'total_programs_evaluated', 0),
                'validation_failures': getattr(self.base_env, 'validation_failures', 0),
                'total_validations': getattr(self.base_env, 'total_validations', 0)
            }
        
        return {
            'total_evaluations': self.total_evaluations,
            'successful_evaluations': self.successful_evaluations,
            'success_rate': success_rate,
            'avg_evaluation_time': avg_eval_time,
            **base_stats
        }
    
    def reset_statistics(self):
        """통계 초기화"""
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.evaluation_times = []
        
        # 기존 환경 통계도 초기화
        if hasattr(self.base_env, 'cache_hits'):
            self.base_env.cache_hits = 0
            self.base_env.total_programs_evaluated = 0
            self.base_env.validation_failures = 0
            self.base_env.total_validations = 0


class MCTSDataCollector:
    """MCTS 학습 데이터 수집기"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.states = []
        self.policy_targets = []
        self.value_targets = []
        self.rewards = []
        
    def add_episode(
        self, 
        states: np.ndarray, 
        policies: np.ndarray, 
        values: np.ndarray, 
        final_reward: float
    ):
        """에피소드 데이터 추가"""
        
        # 가치 타겟을 최종 보상으로 설정 (단순화)
        episode_values = np.full(len(states), final_reward)
        
        self.states.extend(states)
        self.policy_targets.extend(policies)
        self.value_targets.extend(episode_values)
        self.rewards.append(final_reward)
        
        # 메모리 관리
        if len(self.states) > self.max_samples:
            excess = len(self.states) - self.max_samples
            self.states = self.states[excess:]
            self.policy_targets = self.policy_targets[excess:]
            self.value_targets = self.value_targets[excess:]
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """랜덤 배치 샘플링"""
        
        if len(self.states) < batch_size:
            return None, None, None
        
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch_states = np.array([self.states[i] for i in indices])
        batch_policies = np.array([self.policy_targets[i] for i in indices])
        batch_values = np.array([self.value_targets[i] for i in indices])
        
        return batch_states, batch_policies, batch_values
    
    def clear(self):
        """데이터 초기화"""
        self.states.clear()
        self.policy_targets.clear()
        self.value_targets.clear()
        self.rewards.clear()
    
    def get_stats(self) -> Dict:
        """수집 통계"""
        return {
            'total_samples': len(self.states),
            'episodes': len(self.rewards),
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'max_reward': np.max(self.rewards) if self.rewards else 0,
            'min_reward': np.min(self.rewards) if self.rewards else 0
        }


# 테스트
if __name__ == "__main__":
    print("🏭 MCTS Factor Environment 테스트")
    
    # 더미 데이터 생성
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # 환경 생성
    config = RLCConfig()
    env = MCTSFactorEnv(df, config)
    
    # 프로그램 평가 테스트
    test_tokens = [1, 2, 3, 12, 4]  # 간단한 프로그램
    result = env.evaluate_program(test_tokens)
    
    print(f"평가 결과: {result}")
    print(f"환경 통계: {env.get_statistics()}")
    
    # 데이터 수집기 테스트
    collector = MCTSDataCollector()
    
    # 더미 에피소드 추가
    states = np.random.randn(5, 23)
    policies = np.random.rand(5, 25)
    values = np.random.randn(5)
    reward = 0.5
    
    collector.add_episode(states, policies, values, reward)
    print(f"수집기 통계: {collector.get_stats()}")
    
    print("✅ 테스트 완료!")
