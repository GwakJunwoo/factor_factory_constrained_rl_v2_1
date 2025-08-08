#!/usr/bin/env python3
"""
정책-가치 신경망 (Policy-Value Network)

AlphaZero 스타일로 하나의 네트워크에서 정책과 가치를 동시에 예측:
- 입력: 현재 토큰 시퀀스 상태 (23차원 observation)  
- 출력: 정책 확률 분포 (25차원) + 상태 가치 (1차원)
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# PyTorch 조건부 import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. 기본 신경망 모드로 실행됩니다.")
    print("PyTorch 설치: pip install torch")


class DummyPolicyValueNetwork:
    """PyTorch가 없을 때 사용하는 더미 신경망"""
    
    def __init__(self, *args, **kwargs):
        print("⚠️ 더미 신경망 모드: 랜덤 출력을 생성합니다.")
        self.input_dim = kwargs.get('input_dim', 23)
        self.action_dim = kwargs.get('action_dim', 25)
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """랜덤 정책과 가치 반환"""
        policy_probs = np.random.rand(self.action_dim)
        policy_probs = policy_probs / policy_probs.sum()
        value = np.random.uniform(-1, 1)
        return policy_probs, value
    
    def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """배치 랜덤 예측"""
        batch_size = observations.shape[0]
        policies = np.random.rand(batch_size, self.action_dim)
        policies = policies / policies.sum(axis=1, keepdims=True)
        values = np.random.uniform(-1, 1, batch_size)
        return policies, values
    
    def eval(self):
        """평가 모드 (더미)"""
        pass
    
    def train(self):
        """학습 모드 (더미)"""
        pass


class DummyNetworkTrainer:
    """더미 네트워크 트레이너"""
    
    def __init__(self, *args, **kwargs):
        self.network = args[0] if args else DummyPolicyValueNetwork()
        print("⚠️ 더미 트레이너 모드: 실제 학습은 수행되지 않습니다.")
    
    def train_step(self, *args, **kwargs):
        return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    
    def evaluate(self, *args, **kwargs):
        return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    
    def save_model(self, filepath: str):
        print(f"⚠️ 더미 모드: 모델 저장 건너뜀 ({filepath})")
    
    def load_model(self, filepath: str):
        print(f"⚠️ 더미 모드: 모델 로드 건너뜀 ({filepath})")


if not TORCH_AVAILABLE:
    # PyTorch가 없으면 더미 클래스 사용
    PolicyValueNetwork = DummyPolicyValueNetwork
    NetworkTrainer = DummyNetworkTrainer
    
else:
    # PyTorch가 있을 때의 실제 구현
    class PolicyValueNetwork(nn.Module):
        """정책-가치 신경망"""
        
        def __init__(
            self,
            input_dim: int = 23,
            hidden_dims: List[int] = [256, 256, 128],
            action_dim: int = 25,
            dropout_rate: float = 0.1
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.action_dim = action_dim
            
            # 공통 특징 추출 네트워크
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            self.feature_extractor = nn.Sequential(*layers)
            
            # 정책 헤드
            self.policy_head = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            
            # 가치 헤드
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(), 
                nn.Linear(64, 1),
                nn.Tanh()
            )
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.feature_extractor(x)
            policy_logits = self.policy_head(features)
            value = self.value_head(features)
            return policy_logits, value
        
        def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
            self.eval()
            with torch.no_grad():
                x = torch.FloatTensor(observation).unsqueeze(0)
                policy_logits, value = self.forward(x)
                policy_probs = F.softmax(policy_logits, dim=1)
                return policy_probs[0].numpy(), value[0].item()
        
        def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self.eval()
            with torch.no_grad():
                x = torch.FloatTensor(observations)
                policy_logits, values = self.forward(x)
                policy_probs = F.softmax(policy_logits, dim=1)
                return policy_probs.numpy(), values.squeeze().numpy()
    
    
    class NetworkTrainer:
        """신경망 학습 관리 클래스"""
        
        def __init__(self, network: PolicyValueNetwork, learning_rate: float = 1e-3, 
                     weight_decay: float = 1e-4, device: str = 'auto'):
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.network = network.to(self.device)
            self.criterion = nn.MSELoss()  # 간단한 손실 함수
            self.optimizer = torch.optim.Adam(
                network.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            
        def train_step(self, states: np.ndarray, policy_targets: np.ndarray, 
                       value_targets: np.ndarray) -> Dict[str, float]:
            self.network.train()
            
            states = torch.FloatTensor(states).to(self.device)
            policy_targets = torch.FloatTensor(policy_targets).to(self.device)
            value_targets = torch.FloatTensor(value_targets).to(self.device)
            
            policy_logits, value_pred = self.network(states)
            
            # 간단한 손실 계산
            policy_loss = -torch.mean(torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1))
            value_loss = self.criterion(value_pred.squeeze(), value_targets)
            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item()
            }
        
        def evaluate(self, states: np.ndarray, policy_targets: np.ndarray, 
                     value_targets: np.ndarray) -> Dict[str, float]:
            self.network.eval()
            with torch.no_grad():
                states = torch.FloatTensor(states).to(self.device)
                policy_targets = torch.FloatTensor(policy_targets).to(self.device)
                value_targets = torch.FloatTensor(value_targets).to(self.device)
                
                policy_logits, value_pred = self.network(states)
                
                policy_loss = -torch.mean(torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1))
                value_loss = self.criterion(value_pred.squeeze(), value_targets)
                
                return {
                    'total_loss': (policy_loss + value_loss).item(),
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item()
                }
        
        def save_model(self, filepath: str):
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, filepath)
            print(f"✅ 모델 저장: {filepath}")
        
        def load_model(self, filepath: str):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 모델 로드: {filepath}")


# 테스트 코드
if __name__ == "__main__":
    print("🧠 Policy-Value Network 테스트")
    
    # 네트워크 생성
    network = PolicyValueNetwork()
    trainer = NetworkTrainer(network)
    
    # 더미 데이터로 테스트
    batch_size = 32
    states = np.random.randn(batch_size, 23)
    policy_targets = np.random.rand(batch_size, 25)
    policy_targets = policy_targets / policy_targets.sum(axis=1, keepdims=True)
    value_targets = np.random.uniform(-1, 1, batch_size)
    
    # 학습 테스트
    print("학습 테스트...")
    loss_dict = trainer.train_step(states, policy_targets, value_targets)
    print(f"손실: {loss_dict}")
    
    # 예측 테스트
    print("예측 테스트...")
    single_state = states[0]
    policy_probs, value = network.predict(single_state)
    print(f"정책 확률 분포 형태: {policy_probs.shape}")
    print(f"정책 확률 합: {policy_probs.sum():.3f}")
    print(f"예측 가치: {value:.3f}")
    
    print("✅ 테스트 완료!")
