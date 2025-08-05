# 🎯 Factor Factory 보상 시스템 개선 가이드

## 📋 현재 보상 구조 분석

### 기존 보상 함수
```python
reward = (
    pnl_sum                    # 주 보상: 백테스트 수익률
    - lambda_depth * depth     # 페널티: 복잡도
    - lambda_turnover * trades # 페널티: 거래횟수
    - lambda_const1 * const_ratio # 페널티: 상수 사용
    - lambda_std / signal_std  # 페널티: 변동성 부족
    - signal_quality_penalty   # 페널티: 신호 품질
)
```

### 현재 설정값 문제점
```python
lambda_depth = 0.002      # 너무 작음 → 과복잡화 방지 효과 미미
lambda_turnover = 0.0005  # 너무 작음 → 과거래 억제 효과 부족
lambda_const1 = 2.0       # 너무 큼 → 상수 사용을 과도하게 억제
lambda_std = 0.5          # 부적절 → 변동성이 작을수록 큰 페널티
```

---

## 🚀 보상 시스템 개선 방안

### 1️⃣ **위험 조정 수익률 기반 보상**

#### A. Sharpe Ratio 기반 보상
```python
def sharpe_based_reward(pnl, risk_free_rate=0.0):
    """샤프 비율을 주 보상으로 사용"""
    returns = pnl
    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / (returns.std() + 1e-8)
    
    # 스케일링: [-2, 2] 범위로 조정
    return np.tanh(sharpe * 2.0)

# 장점: 위험 대비 수익률 최적화
# 단점: 변동성이 낮은 전략에 과도한 보상
```

#### B. Calmar Ratio 기반 보상
```python
def calmar_based_reward(pnl, equity):
    """칼마 비율 기반 보상 (MDD 고려)"""
    annual_return = (equity.iloc[-1] - 1) * (252 * 24 / len(pnl))
    
    # 최대 낙폭 계산
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = abs(drawdown.min())
    
    if max_drawdown < 1e-6:
        return 10.0  # 무손실 전략에 높은 보상
    
    calmar = annual_return / max_drawdown
    return np.tanh(calmar * 0.5)  # 스케일링
```

### 2️⃣ **다목적 최적화 보상**

#### A. 가중 평균 방식
```python
def multi_objective_reward(pnl, equity, signal, trades):
    """여러 목적함수의 가중 평균"""
    
    # 1. 수익성 (30%)
    total_return = (equity.iloc[-1] - 1) * 100
    return_score = np.tanh(total_return / 10)  # ±10% 기준
    
    # 2. 위험성 (25%)
    volatility = pnl.std() * np.sqrt(252 * 24) * 100
    risk_score = max(0, 1 - volatility / 20)  # 20% 변동성 기준
    
    # 3. 안정성 (25%)
    max_dd = abs((equity / equity.cummax() - 1).min())
    stability_score = max(0, 1 - max_dd / 0.2)  # 20% MDD 기준
    
    # 4. 효율성 (20%)
    win_rate = (pnl > 0).sum() / len(pnl)
    efficiency_score = win_rate * 2 - 1  # [0,1] → [-1,1]
    
    return (0.3 * return_score + 
            0.25 * risk_score + 
            0.25 * stability_score + 
            0.2 * efficiency_score)
```

#### B. 파레토 최적화 방식
```python
def pareto_reward(pnl, equity, signal):
    """파레토 프론티어 기반 보상"""
    
    # 목적함수 정의
    objectives = {
        'return': (equity.iloc[-1] - 1) * 100,
        'sharpe': pnl.mean() / (pnl.std() + 1e-8),
        'max_dd': -abs((equity / equity.cummax() - 1).min()),
        'win_rate': (pnl > 0).sum() / len(pnl)
    }
    
    # 정규화
    normalized = {}
    for key, value in objectives.items():
        if key == 'return':
            normalized[key] = np.tanh(value / 5)  # ±5% 기준
        elif key == 'sharpe':
            normalized[key] = np.tanh(value * 2)
        elif key == 'max_dd':
            normalized[key] = np.tanh(value * 10)  # -10% → -1
        elif key == 'win_rate':
            normalized[key] = value * 2 - 1
    
    # 가중 기하평균 (모든 목표가 균형)
    weights = [0.3, 0.3, 0.2, 0.2]
    values = list(normalized.values())
    
    # 음수 처리를 위한 오프셋
    offset_values = [v + 2 for v in values]  # [-1,1] → [1,3]
    geo_mean = np.prod([v**w for v, w in zip(offset_values, weights)])
    
    return geo_mean - 2  # 다시 [-1,1] 범위로
```

### 3️⃣ **동적 보상 시스템**

#### A. 적응적 페널티 가중치
```python
class AdaptiveRewardSystem:
    def __init__(self):
        self.episode_count = 0
        self.performance_history = []
        
    def get_reward(self, pnl, equity, signal, tokens, depth, trades):
        """에피소드 진행에 따라 보상 가중치 조정"""
        
        base_reward = pnl.sum()
        
        # 초기: 탐색 격려 (낮은 페널티)
        if self.episode_count < 1000:
            alpha_complexity = 0.001   # 복잡도 페널티 약함
            alpha_turnover = 0.0001    # 거래 페널티 약함
            alpha_diversity = -0.01    # 다양성 보너스
            
        # 중기: 균형 (중간 페널티)
        elif self.episode_count < 5000:
            alpha_complexity = 0.005
            alpha_turnover = 0.001
            alpha_diversity = 0.0
            
        # 후기: 수렴 (강한 페널티)
        else:
            alpha_complexity = 0.01    # 복잡도 강하게 억제
            alpha_turnover = 0.005     # 과거래 강하게 억제
            alpha_diversity = 0.005    # 과적합 방지
        
        reward = (base_reward 
                 - alpha_complexity * depth
                 - alpha_turnover * trades
                 + alpha_diversity * self._diversity_bonus(tokens))
        
        self.episode_count += 1
        return reward
```

#### B. 상대적 순위 보상
```python
class RankingRewardSystem:
    def __init__(self, buffer_size=1000):
        self.performance_buffer = []
        self.buffer_size = buffer_size
        
    def get_reward(self, pnl, equity, signal):
        """최근 성과 대비 상대적 순위로 보상 계산"""
        
        current_performance = {
            'total_return': equity.iloc[-1] - 1,
            'sharpe': pnl.mean() / (pnl.std() + 1e-8),
            'max_dd': abs((equity / equity.cummax() - 1).min())
        }
        
        # 버퍼에 추가
        self.performance_buffer.append(current_performance)
        if len(self.performance_buffer) > self.buffer_size:
            self.performance_buffer.pop(0)
        
        if len(self.performance_buffer) < 50:
            return 0.0  # 충분한 비교군이 없으면 중립
        
        # 각 지표별 상대 순위 계산
        ranks = {}
        for metric in current_performance:
            values = [p[metric] for p in self.performance_buffer]
            current_value = current_performance[metric]
            
            if metric == 'max_dd':  # 낮을수록 좋음
                rank = sum(1 for v in values if v > current_value)
            else:  # 높을수록 좋음
                rank = sum(1 for v in values if v < current_value)
            
            ranks[metric] = rank / len(values)  # [0,1] 정규화
        
        # 가중 평균 순위
        final_rank = (0.4 * ranks['total_return'] + 
                     0.4 * ranks['sharpe'] + 
                     0.2 * ranks['max_dd'])
        
        return (final_rank - 0.5) * 4  # [0,1] → [-2,2]
```

### 4️⃣ **실용적 개선 방안**

#### A. 즉시 적용 가능한 보상 조정
```python
# 현재 env.py에서 수정 가능한 부분
@dataclass
class ImprovedRLCConfig:
    # 기존 설정
    max_len: int = 21
    
    # 개선된 페널티 가중치
    lambda_depth: float = 0.01        # 5배 증가: 복잡도 강력 억제
    lambda_turnover: float = 0.002    # 4배 증가: 과거래 억제
    lambda_const1: float = 0.5        # 1/4로 감소: 상수 사용 허용
    lambda_std: float = 0.1           # 1/5로 감소: 변동성 페널티 완화
    
    # 새로운 페널티
    lambda_drawdown: float = 2.0      # MDD 페널티 추가
    lambda_consistency: float = 0.5   # 일관성 페널티 추가
    lambda_skewness: float = 0.1      # 편향성 페널티 추가
    
    # 동적 가중치 활성화
    adaptive_weights: bool = True
    performance_window: int = 100     # 성과 비교 윈도우
```

#### B. 보상 정규화 및 클리핑
```python
def normalized_reward(raw_reward, history_window=1000):
    """보상의 정규화 및 클리핑"""
    
    # 역사적 보상 추적
    if not hasattr(normalized_reward, 'reward_history'):
        normalized_reward.reward_history = []
    
    normalized_reward.reward_history.append(raw_reward)
    if len(normalized_reward.reward_history) > history_window:
        normalized_reward.reward_history.pop(0)
    
    # Z-score 정규화
    if len(normalized_reward.reward_history) > 10:
        history = np.array(normalized_reward.reward_history)
        mean = history.mean()
        std = history.std() + 1e-8
        normalized = (raw_reward - mean) / std
    else:
        normalized = raw_reward
    
    # 클리핑 [-3, 3]
    clipped = np.clip(normalized, -3.0, 3.0)
    
    # 탄젠트 스케일링 [-1, 1]
    final_reward = np.tanh(clipped)
    
    return final_reward
```

### 5️⃣ **실험적 보상 시스템**

#### A. 커리큘럼 학습 보상
```python
class CurriculumReward:
    def __init__(self):
        self.phase = 'exploration'  # exploration → exploitation → refinement
        self.episode_count = 0
        
    def get_reward(self, pnl, equity, signal, tokens):
        self.episode_count += 1
        
        # Phase 1: 탐색 (0-2000 에피소드)
        if self.episode_count <= 2000:
            self.phase = 'exploration'
            return self._exploration_reward(pnl, tokens)
            
        # Phase 2: 활용 (2000-8000 에피소드)
        elif self.episode_count <= 8000:
            self.phase = 'exploitation'
            return self._exploitation_reward(pnl, equity, signal)
            
        # Phase 3: 정제 (8000+ 에피소드)
        else:
            self.phase = 'refinement'
            return self._refinement_reward(pnl, equity, signal, tokens)
    
    def _exploration_reward(self, pnl, tokens):
        """탐색 단계: 다양성 격려"""
        base = pnl.sum()
        diversity_bonus = len(set(tokens)) * 0.1  # 토큰 다양성 보너스
        return base + diversity_bonus
    
    def _exploitation_reward(self, pnl, equity, signal):
        """활용 단계: 수익성 중심"""
        return pnl.sum() * 2.0  # 수익률에 가중치
    
    def _refinement_reward(self, pnl, equity, signal, tokens):
        """정제 단계: 안정성 + 단순함"""
        base = pnl.sum()
        stability = -abs((equity / equity.cummax() - 1).min())
        simplicity = -len(tokens) * 0.01
        return base + stability + simplicity
```

#### B. 강화학습 메타 보상
```python
def meta_reward(pnl, equity, signal, episode_reward_history):
    """메타 학습 기반 보상 조정"""
    
    # 기본 보상
    base_reward = pnl.sum()
    
    # 최근 100 에피소드 성과 트렌드
    if len(episode_reward_history) >= 100:
        recent_trend = np.polyfit(range(100), episode_reward_history[-100:], 1)[0]
        
        # 성과 향상 중이면 보너스
        improvement_bonus = max(0, recent_trend * 10)
        
        # 성과 정체 시 탐색 격려
        if abs(recent_trend) < 0.01:
            exploration_bonus = 0.1
        else:
            exploration_bonus = 0.0
            
        return base_reward + improvement_bonus + exploration_bonus
    
    return base_reward
```

---

## 🎯 권장 구현 순서

### 1단계: 즉시 개선 (Low Risk)
```python
# env.py에서 현재 가중치만 조정
lambda_depth: float = 0.01        # 현재 0.002 → 0.01
lambda_turnover: float = 0.002    # 현재 0.0005 → 0.002  
lambda_const1: float = 0.5        # 현재 2.0 → 0.5
```

### 2단계: 위험 조정 보상 추가 (Medium Risk)
```python
# Sharpe ratio 기반 보상으로 변경
reward = sharpe_ratio * 2.0 - penalties
```

### 3단계: 다목적 보상 도입 (High Risk)
```python
# 다목적 최적화 보상 시스템 구현
reward = multi_objective_reward(pnl, equity, signal, trades)
```

### 4단계: 동적 시스템 구현 (Experimental)
```python
# 적응적 또는 커리큘럼 보상 시스템
reward_system = AdaptiveRewardSystem()
reward = reward_system.get_reward(...)
```

각 단계별로 실험하면서 성능 변화를 관찰하고, 가장 효과적인 조합을 찾아 점진적으로 개선하는 것을 권장합니다! 🚀
