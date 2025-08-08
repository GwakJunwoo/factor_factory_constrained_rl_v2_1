#!/usr/bin/env python3
"""
MCTS 노드 클래스 - AlphaZero 스타일 트리 탐색을 위한 노드 구조
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class MCTSNode:
    """
    Monte Carlo Tree Search 노드
    
    AlphaZero 스타일로 구현:
    - 각 노드는 게임 상태(토큰 시퀀스)를 나타냄
    - UCB1 + 신경망 정책으로 자식 노드 선택
    - 백프로파게이션으로 가치 정보 업데이트
    """
    
    def __init__(
        self, 
        tokens: List[int], 
        need: int, 
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,
        prior_prob: float = 0.0
    ):
        """
        Args:
            tokens: 현재까지의 토큰 시퀀스
            need: 필요한 추가 토큰 수 (0이면 완성된 프로그램)
            parent: 부모 노드
            action: 부모에서 이 노드로 오는 액션
            prior_prob: 신경망에서 예측한 이 액션의 확률
        """
        # 상태 정보
        self.tokens = tokens.copy()
        self.need = need
        self.is_terminal = (need == 0)
        
        # 트리 구조
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.action = action  # 부모에서 이 노드로 오는 액션
        
        # MCTS 통계
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
        # 캐시된 결과
        self.is_expanded = False
        self.reward_cache: Optional[float] = None
        self.legal_actions_cache: Optional[List[int]] = None
        
        # 신경망 예측 결과
        self.policy_probs: Optional[np.ndarray] = None
        self.value_estimate: Optional[float] = None
    
    @property
    def is_root(self) -> bool:
        """루트 노드 여부"""
        return self.parent is None
    
    @property 
    def is_leaf(self) -> bool:
        """리프 노드 여부 (확장되지 않은 노드)"""
        return not self.is_expanded
    
    @property
    def q_value(self) -> float:
        """평균 가치 (Q값)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_legal_actions(self) -> List[int]:
        """현재 상태에서 가능한 액션들 반환"""
        if self.legal_actions_cache is not None:
            return self.legal_actions_cache
        
        if self.is_terminal:
            self.legal_actions_cache = []
            return []
        
        # 기존 RL 환경의 _legal 로직 사용
        from ..rlc.grammar import ARITY
        
        legal_actions = []
        for token in range(25):  # 0~24 토큰 (기존대로 복원)
            # 길이 제한
            if len(self.tokens) >= 21:  # max_len
                continue
            
            # 새로운 need 계산
            new_need = self.need - 1 + ARITY[token]
            
            # need가 너무 커지면 불가능
            if new_need > 21 - len(self.tokens) - 1:
                continue
            
            # need가 음수가 되면 불가능  
            if new_need < 0:
                continue
                
            legal_actions.append(token)
        
        self.legal_actions_cache = legal_actions
        return legal_actions
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        UCB1 + 정책 확률을 사용한 자식 노드 선택
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if not self.children:
            raise ValueError("No children to select from")
        
        best_action = None
        best_ucb = -float('inf')
        
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            # Q 값 (평균 가치)
            q_value = child.q_value
            
            # U 값 (탐색 보너스)
            u_value = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)
            
            # UCB 값
            ucb_value = q_value + u_value
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_action = action
        
        return self.children[best_action]
    
    def expand(self, policy_probs: np.ndarray, value_estimate: float):
        """
        노드 확장 - 신경망 정책으로 자식 노드들 생성
        
        Args:
            policy_probs: 각 액션의 확률 분포 (25차원)
            value_estimate: 현재 상태의 가치 추정
        """
        self.policy_probs = policy_probs
        self.value_estimate = value_estimate
        self.is_expanded = True
        
        legal_actions = self.get_legal_actions()
        
        if not legal_actions:
            return  # 터미널 노드
        
        # 합법적인 액션들에 대해서만 자식 노드 생성
        for action in legal_actions:
            from ..rlc.grammar import ARITY
            
            new_tokens = self.tokens + [action]
            new_need = self.need - 1 + ARITY[action]
            prior_prob = policy_probs[action]
            
            child_node = MCTSNode(
                tokens=new_tokens,
                need=new_need, 
                parent=self,
                action=action,
                prior_prob=prior_prob
            )
            
            self.children[action] = child_node
    
    def backup(self, value: float):
        """
        백프로파게이션 - 리프에서 루트까지 가치 정보 업데이트
        
        Args:
            value: 리프 노드에서 얻은 가치
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 부모 관점에서는 값이 반대 (zero-sum game 아니지만 일관성을 위해)
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        방문 횟수 기반 액션 확률 분포 반환
        
        Args:
            temperature: 탐색 vs 활용 조절 (0에 가까우면 greedy)
        
        Returns:
            25차원 확률 분포
        """
        probs = np.zeros(25)
        
        if not self.children:
            return probs
        
        if temperature == 0:
            # Greedy 선택
            best_action = max(self.children.keys(), 
                            key=lambda a: self.children[a].visit_count)
            probs[best_action] = 1.0
        else:
            # 볼츠만 분포 - 실제 자식 노드들의 방문 횟수만 사용
            for action, child in self.children.items():
                probs[action] = child.visit_count
            
            if probs.sum() == 0:
                # 균등 분포
                legal_actions = self.get_legal_actions()
                for action in legal_actions:
                    probs[action] = 1.0 / len(legal_actions)
            else:
                # 온도 조절된 방문 횟수
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum()
        
        return probs
    
    def get_best_child(self) -> Optional['MCTSNode']:
        """가장 많이 방문된 자식 노드 반환"""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.visit_count)
    
    def get_path_to_root(self) -> List[int]:
        """루트에서 현재 노드까지의 액션 경로"""
        path = []
        node = self
        
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        
        return list(reversed(path))
    
    def __str__(self) -> str:
        """노드 정보 문자열"""
        return (f"MCTSNode(tokens={self.tokens}, need={self.need}, "
                f"visits={self.visit_count}, q={self.q_value:.3f}, "
                f"children={len(self.children)})")
    
    def tree_stats(self, depth: int = 0, max_depth: int = 3) -> str:
        """트리 통계 정보 (디버깅용)"""
        indent = "  " * depth
        stats = f"{indent}{self}\n"
        
        if depth < max_depth and self.children:
            for action, child in sorted(self.children.items()):
                stats += f"{indent}  Action {action}:\n"
                stats += child.tree_stats(depth + 1, max_depth)
        
        return stats


class MCTSStats:
    """MCTS 탐색 통계"""
    
    def __init__(self):
        self.total_simulations = 0
        self.total_expansions = 0
        self.average_depth = 0.0
        self.max_depth = 0
        self.cache_hits = 0
        self.terminal_reaches = 0
    
    def reset(self):
        """통계 초기화"""
        self.total_simulations = 0
        self.total_expansions = 0
        self.average_depth = 0.0
        self.max_depth = 0
        self.cache_hits = 0
        self.terminal_reaches = 0
    
    def __str__(self) -> str:
        return (f"MCTS Stats: sims={self.total_simulations}, "
                f"expansions={self.total_expansions}, "
                f"avg_depth={self.average_depth:.1f}, "
                f"max_depth={self.max_depth}, "
                f"cache_hits={self.cache_hits}, "
                f"terminals={self.terminal_reaches}")
