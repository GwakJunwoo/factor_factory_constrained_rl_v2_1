#!/usr/bin/env python3
"""
MCTS 탐색 알고리즘

AlphaZero 스타일 Monte Carlo Tree Search:
1. Selection: UCB1 + 정책 확률로 리프 노드까지 탐색
2. Expansion: 신경망으로 정책과 가치 예측하여 노드 확장
3. Evaluation: 터미널 노드면 실제 평가, 아니면 신경망 가치 사용
4. Backup: 리프에서 루트까지 가치 정보 백프로파게이션
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import time
import math

from .mcts_node import MCTSNode, MCTSStats
from .neural_network import PolicyValueNetwork


class MCTSSearch:
    """Monte Carlo Tree Search 탐색 엔진"""
    
    def __init__(
        self,
        network: PolicyValueNetwork,
        c_puct: float = 1.0,
        num_simulations: int = 800,
        evaluation_fn: Optional[Callable] = None,
        add_noise: bool = True,
        noise_alpha: float = 0.3,
        noise_epsilon: float = 0.25
    ):
        """
        Args:
            network: 정책-가치 신경망
            c_puct: UCB 탐색 상수
            num_simulations: 시뮬레이션 횟수
            evaluation_fn: 터미널 노드 평가 함수
            add_noise: 루트에서 디리클레 노이즈 추가 여부
            noise_alpha: 디리클레 노이즈 알파 파라미터
            noise_epsilon: 노이즈 혼합 비율
        """
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluation_fn = evaluation_fn
        self.add_noise = add_noise
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon
        
        # 통계
        self.stats = MCTSStats()
        
        # 캐시 (동일한 상태에 대한 평가 결과)
        self.evaluation_cache: Dict[tuple, float] = {}
        
        print(f"✅ MCTS Search 초기화 (simulations: {num_simulations}, c_puct: {c_puct})")
    
    def search(self, root_state: List[int], root_need: int) -> Tuple[np.ndarray, MCTSNode]:
        """
        MCTS 탐색 실행
        
        Args:
            root_state: 루트 상태 (토큰 시퀀스)
            root_need: 루트에서 필요한 토큰 수
        
        Returns:
            action_probs: 액션 확률 분포 [25]
            root_node: 탐색이 완료된 루트 노드
        """
        # 루트 노드 생성
        root = MCTSNode(tokens=root_state, need=root_need)
        
        # 루트 노드 확장
        if not root.is_terminal:
            obs = self._state_to_observation(root_state, root_need)
            policy_probs, value = self.network.predict(obs)
            
            # 디리클레 노이즈 추가 (탐색 다양성 증진)
            if self.add_noise:
                noise = np.random.dirichlet([self.noise_alpha] * 25)
                policy_probs = (1 - self.noise_epsilon) * policy_probs + self.noise_epsilon * noise
            
            root.expand(policy_probs, value)
        
        # 통계 초기화
        self.stats.reset()
        
        # 시뮬레이션 실행
        for i in range(self.num_simulations):
            self._simulate(root)
            
            if (i + 1) % 100 == 0:
                print(f"  시뮬레이션 진행: {i+1}/{self.num_simulations}")
        
        # 액션 확률 분포 계산
        action_probs = root.get_action_probs(temperature=1.0)
        
        print(f"✅ MCTS 탐색 완료: {self.stats}")
        
        return action_probs, root
    
    def _simulate(self, root: MCTSNode):
        """단일 시뮬레이션 실행"""
        
        path = []
        node = root
        
        # 1. Selection: 리프 노드까지 탐색
        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.c_puct)
            path.append(node)
        
        # 통계 업데이트
        self.stats.total_simulations += 1
        self.stats.average_depth = (self.stats.average_depth * (self.stats.total_simulations - 1) + len(path)) / self.stats.total_simulations
        self.stats.max_depth = max(self.stats.max_depth, len(path))
        
        # 2. Evaluation
        if node.is_terminal:
            # 터미널 노드: 실제 평가
            value = self._evaluate_terminal(node)
            self.stats.terminal_reaches += 1
        else:
            # 3. Expansion: 비터미널 노드 확장 후 평가
            obs = self._state_to_observation(node.tokens, node.need)
            policy_probs, value = self.network.predict(obs)
            node.expand(policy_probs, value)
            self.stats.total_expansions += 1
        
        # 4. Backup: 가치 정보 백프로파게이션
        node.backup(value)
    
    def _evaluate_terminal(self, node: MCTSNode) -> float:
        """터미널 노드 평가"""
        
        if self.evaluation_fn is None:
            # 기본 평가: 프로그램 길이 페널티
            return -len(node.tokens) * 0.01
        
        # 캐시 확인
        cache_key = tuple(node.tokens)
        if cache_key in self.evaluation_cache:
            self.stats.cache_hits += 1
            return self.evaluation_cache[cache_key]
        
        # 실제 평가 (백테스트)
        try:
            reward = self.evaluation_fn(node.tokens)
            self.evaluation_cache[cache_key] = reward
            return reward
        except Exception as e:
            print(f"⚠️ 평가 오류: {e}")
            return -1.0
    
    def _state_to_observation(self, tokens: List[int], need: int) -> np.ndarray:
        """상태를 관측 벡터로 변환 (기존 환경과 동일한 형식)"""
        
        # 기존 ProgramEnv._obs() 로직 사용
        obs = np.zeros(23, dtype=np.float32)
        
        # 토큰 히스토그램 (0~24 → 첫 25차원 중 23개 사용)
        if tokens:
            for tok in tokens:
                if 0 <= tok < 23:  # 23차원 제한
                    obs[tok] += 1
        
        # 정규화 (최대 길이 21 기준)
        obs = obs / 21.0
        
        return obs
    
    def get_best_action(self, root: MCTSNode) -> int:
        """가장 좋은 액션 선택 (방문 횟수 기준)"""
        if not root.children:
            return 0  # 기본값
        
        return max(root.children.keys(), 
                  key=lambda a: root.children[a].visit_count)
    
    def get_principal_variation(self, root: MCTSNode, max_depth: int = 10) -> List[int]:
        """
        주요 변화 (가장 좋은 경로) 반환
        
        Args:
            root: 루트 노드
            max_depth: 최대 깊이
        
        Returns:
            최고 방문 횟수 경로의 액션 시퀀스
        """
        pv = []
        node = root
        
        for _ in range(max_depth):
            if not node.children:
                break
            
            best_child = node.get_best_child()
            if best_child is None:
                break
            
            pv.append(best_child.action)
            node = best_child
        
        return pv
    
    def clear_cache(self):
        """평가 캐시 초기화"""
        self.evaluation_cache.clear()
        print("🗑️ MCTS 캐시 초기화됨")


class AdaptiveMCTS(MCTSSearch):
    """적응적 MCTS (시뮬레이션 횟수를 동적으로 조절)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_simulations = 200
        self.max_simulations = 1600
        self.confidence_threshold = 0.1
    
    def search(self, root_state: List[int], root_need: int) -> Tuple[np.ndarray, MCTSNode]:
        """적응적 탐색: 확신이 충분할 때까지 시뮬레이션"""
        
        root = MCTSNode(tokens=root_state, need=root_need)
        
        if not root.is_terminal:
            obs = self._state_to_observation(root_state, root_need)
            policy_probs, value = self.network.predict(obs)
            root.expand(policy_probs, value)
        
        # 최소 시뮬레이션 실행
        for i in range(self.min_simulations):
            self._simulate(root)
        
        # 확신도 기반 추가 시뮬레이션
        for i in range(self.min_simulations, self.max_simulations, 50):
            # 현재 최고 액션의 확신도 계산
            if root.children:
                visit_counts = [child.visit_count for child in root.children.values()]
                total_visits = sum(visit_counts)
                
                if total_visits > 0:
                    max_visits = max(visit_counts)
                    confidence = max_visits / total_visits
                    
                    # 충분히 확신하면 조기 종료
                    if confidence > (1 - self.confidence_threshold):
                        print(f"  조기 종료 (확신도: {confidence:.3f}, 시뮬레이션: {i})")
                        break
            
            # 추가 시뮬레이션
            for _ in range(50):
                self._simulate(root)
        
        action_probs = root.get_action_probs(temperature=1.0)
        return action_probs, root


# 테스트 및 시연
if __name__ == "__main__":
    print("🔍 MCTS Search 테스트")
    
    from .neural_network import PolicyValueNetwork
    
    # 네트워크와 MCTS 생성
    network = PolicyValueNetwork()
    mcts = MCTSSearch(network, num_simulations=100)  # 테스트용으로 적은 횟수
    
    # 더미 평가 함수
    def dummy_eval(tokens):
        return np.random.uniform(-1, 1)
    
    mcts.evaluation_fn = dummy_eval
    
    # 탐색 테스트
    print("탐색 테스트...")
    root_state = [1, 2]  # 초기 토큰
    root_need = 2        # 2개 더 필요
    
    action_probs, root_node = mcts.search(root_state, root_need)
    
    print(f"액션 확률 분포: {action_probs}")
    print(f"최고 액션: {mcts.get_best_action(root_node)}")
    print(f"주요 변화: {mcts.get_principal_variation(root_node, 5)}")
    print(f"루트 노드 정보:\n{root_node}")
    
    print("✅ 테스트 완료!")
