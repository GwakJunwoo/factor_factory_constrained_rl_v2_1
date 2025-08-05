# 🤖 Factor Factory 강화학습 시스템 완전 가이드

## 📋 목차
1. [학습 환경 개요](#학습-환경-개요)
2. [에이전트와 액션 스페이스](#에이전트와-액션-스페이스)
3. [상태 표현 (Observation)](#상태-표현-observation)
4. [보상 시스템 (Reward System)](#보상-시스템-reward-system)
5. [페널티 시스템](#페널티-시스템)
6. [학습 과정](#학습-과정)
7. [미래 정보 누출 방지](#미래-정보-누출-방지)

---

## 🏗️ 학습 환경 개요

### 환경 타입: **Token Sequence Construction Environment**
- **목표**: 트레이딩 시그널을 생성하는 최적의 수식(토큰 시퀀스)을 발견
- **방법**: 강화학습 에이전트가 순차적으로 토큰을 선택해 완전한 수식을 구성
- **평가**: 구성된 수식의 백테스트 성과로 보상 계산

```
예시 학습 과정:
Agent → [SMA20] → [DIV] → [MACD] → [Complete Program] → Backtest → Reward
```

---

## 🎯 에이전트와 액션 스페이스

### 액션 스페이스 (25개 토큰)
```python
TOKENS = {
    # 기술지표 (Technical Indicators)
    0: "OPEN", 1: "HIGH", 2: "LOW", 3: "CLOSE", 4: "VOLUME",
    5: "SMA5", 6: "SMA10", 7: "SMA20", 8: "EMA10", 9: "EMA20",
    10: "BBANDS", 11: "MACD", 12: "STOCH",
    
    # 수학 연산자 (Mathematical Operators)  
    13: "ADD", 14: "SUB", 15: "MUL", 16: "DIV",
    17: "ABS", 18: "LOG", 19: "LAG1",
    
    # 비교/선택 연산자 (Comparison Operators)
    20: "MAX", 21: "MIN",
    
    # 상수 (Constants)
    22: "CONST1", 23: "CONST2", 24: "CONST3"
}
```

### 에이전트 행동 방식
1. **순차적 토큰 선택**: 한 번에 하나의 토큰을 선택
2. **구문 규칙 준수**: Prefix 표기법으로 유효한 수식만 생성 가능
3. **길이 제한**: 최대 21개 토큰까지 허용

---

## 📊 상태 표현 (Observation)

### 상태 벡터 구조 (23차원)
```python
observation = [
    token_1_normalized,    # 첫 번째 토큰 (정규화된 값)
    token_2_normalized,    # 두 번째 토큰
    ...,
    token_21_normalized,   # 21번째 토큰 (최대 길이)
    need_normalized,       # 현재 필요한 피연산자 수 (정규화)
    progress_normalized    # 프로그램 완성도 (0~1)
]
```

### 정규화 방식
- **토큰 값**: `(tok / (N_TOKENS - 1)) * 2 - 1` → [-1, 1] 범위
- **Need 값**: `np.tanh(need / 8)` → 부드러운 정규화
- **진행도**: `(length / max_len) * 2 - 1` → [-1, 1] 범위

---

## 🏆 보상 시스템 (Reward System)

### 메인 보상 구조
```python
최종_보상 = 백테스트_수익률 - 깊이_페널티 - 거래횟수_페널티 - 상수사용_페널티 - 변동성_페널티 - 신호품질_페널티
```

### 1. **백테스트 수익률 (Main Reward)**
```python
pnl_sum = 현실적_백테스트_총수익률
# 현실적 조건 반영:
# - 거래 지연 (신호 생성 → 결정 → 체결: 총 2 기간)
# - 수수료: 0.08%
# - 슬리피지: 0.15%  
# - 시장 충격: 대량 거래 시 추가 비용
```

### 2. **현실적 백테스트 과정**
```python
def realistic_backtest_flow():
    t시점_가격_데이터 = 과거_데이터_only  # 미래 정보 차단
    t시점_신호 = 수식(t시점_가격_데이터)
    
    t+1시점_거래결정 = 신호_인식_및_결정  # 1기간 지연
    t+2시점_실제체결 = 주문_체결        # 1기간 지연
    
    수익률 = t+2시점_포지션 × t+3시점_가격변화
    거래비용 = 포지션변경량 × (수수료 + 슬리피지 + 시장충격)
    
    return 순수익률
```

---

## ⚖️ 페널티 시스템

### 1. **구조적 페널티 (Structural Penalties)**
```python
# 🌳 깊이 페널티 (Tree Depth Penalty)
depth_penalty = lambda_depth × tree_depth
# 목적: 과도하게 복잡한 수식 방지
# 설정: lambda_depth = 0.002

# 📏 길이 페널티 (Length Penalty) 
length_penalty = length_penalty × num_tokens
# 목적: 불필요하게 긴 수식 방지  
# 설정: length_penalty = 0.0005 (토큰당)
```

### 2. **거래 효율성 페널티**
```python
# 🔄 회전율 페널티 (Turnover Penalty)
turnover_penalty = lambda_turnover × num_trades
# 목적: 과도한 거래 방지 (현실적 거래비용 고려)
# 설정: lambda_turnover = 0.0005

# 📊 변동성 페널티 (Volatility Penalty)
volatility_penalty = lambda_std / signal_std
# 목적: 너무 변동성이 작은 신호 방지
# 설정: lambda_std = 0.5
```

### 3. **신호 품질 페널티**
```python
# 🎯 신호 변경 페널티
signal_quality_penalty = 0.5 × (change_ratio - max_change_ratio)
# 목적: 과도하게 자주 변하는 신호 방지
# 현실적 거래 가능성 고려

# 📈 상수 사용 페널티  
const_penalty = lambda_const1 × const_ratio
# 목적: 상수만 사용하는 단순한 전략 방지
# 설정: lambda_const1 = 2.0
```

### 4. **미래 정보 누출 페널티**
```python
# 🚨 Future Leak Penalty (강력한 패널티)
if has_future_leak:
    reward = -2.0  # 즉시 에피소드 종료
    
# 검증 항목:
# - 시그널 생성 시점 vs 사용 데이터 시점
# - 정규화 과정에서 미래 통계 사용 여부
# - 롤링 윈도우 준수 여부
```

---

## 🔄 학습 과정 (Learning Process)

### Phase 1: 토큰 시퀀스 구성
```
1. 환경 초기화: reset() → 빈 토큰 리스트, need=1
2. 에이전트 액션: 토큰 선택 (0~24 중 하나)
3. 유효성 검증: 구문 규칙 및 길이 제한 확인
4. 상태 업데이트: 토큰 추가, need 값 조정
5. 중간 보상: -length_penalty (각 토큰당)
6. 완성 여부: need == 0이면 평가, 아니면 반복
```

### Phase 2: 프로그램 평가 및 보상
```
1. 캐시 확인: 동일한 프로그램 이전 평가 결과 확인
2. 신호 생성: eval_prefix() → 원시 신호
3. 현실적 정규화: 롤링 윈도우 기반 Z-score
4. 미래 누출 검증: validate_signal_timing()
5. 백테스트 실행: realistic_backtest()
6. 보상 계산: 수익률 - 페널티들
7. 에피소드 종료: terminated=True
```

### Phase 3: 정책 업데이트 (PPO)
```
1. 경험 수집: 여러 에피소드의 (상태, 액션, 보상) 수집
2. 어드밴티지 계산: A(s,a) = Q(s,a) - V(s)
3. 정책 업데이트: PPO loss 최소화
4. 가치 함수 업데이트: Value function 학습
5. 클리핑: 과도한 정책 변화 방지
```

---

## 🛡️ 미래 정보 누출 방지

### 1. **롤링 윈도우 정규화**
```python
def rolling_zscore_realtime(raw_signal, lookback_window=252):
    """시점별로 과거 데이터만 사용한 정규화"""
    normalized = pd.Series(index=raw_signal.index, dtype=float)
    
    for i in range(len(raw_signal)):
        if i < lookback_window:
            # 초기 기간: 사용 가능한 모든 과거 데이터 사용
            window_data = raw_signal.iloc[:i+1]
        else:
            # 정상 기간: 고정된 lookback window 사용
            window_data = raw_signal.iloc[i-lookback_window+1:i+1]
        
        if len(window_data) > 1:
            mean = window_data.mean()
            std = window_data.std()
            if std > 1e-8:
                normalized.iloc[i] = (raw_signal.iloc[i] - mean) / std
    
    return normalized
```

### 2. **신호 타이밍 검증**
```python
def validate_signal_timing(df, signal, price):
    """신호와 가격 데이터의 시간적 정합성 검증"""
    issues = []
    
    # 1. 신호가 해당 시점 이후 데이터를 사용했는지 확인
    for i, sig_time in enumerate(signal.index):
        if i == 0: continue
        
        # 신호 생성에 사용된 데이터가 신호 시점 이전인지 확인
        available_data = df[df.index < sig_time]
        if len(available_data) < i:
            issues.append(f"Future data used at {sig_time}")
    
    # 2. 정규화 과정 검증
    # 3. 지연 시간 준수 확인
    
    return {
        'has_future_leak': len(issues) > 0,
        'issues': issues
    }
```

### 3. **현실적 거래 타이밍**
```
시점 T: 가격 데이터 관측
시점 T: 신호 계산 (T 시점까지의 과거 데이터만 사용)
시점 T+1: 신호 인식 및 거래 결정 (signal_delay)
시점 T+2: 실제 주문 체결 (execution_delay)  
시점 T+3: 포지션 보유 상태에서 수익률 실현
```

---

## 📈 학습 성과 모니터링

### 실시간 로그 출력
```
[R=0.0234] pnl=0.0256 depth=3 trades=45 CONST%=0.15
  신호변경률=2.3% 검증실패율=0.0% 캐시적중=67.5%
  (DIV (MACD) (LAG1 (SMA20)))
```

### 주요 지표 설명
- **R (Reward)**: 최종 보상값 (높을수록 좋음)
- **pnl**: 백테스트 순손익 (수수료/슬리피지 반영)
- **depth**: 수식 트리 깊이 (낮을수록 단순함)
- **trades**: 총 거래 횟수
- **CONST%**: 상수 토큰 사용 비율
- **신호변경률**: 신호가 변하는 빈도 (낮을수록 안정적)
- **검증실패율**: 미래 정보 누출 검출 비율
- **캐시적중**: 이전 계산 결과 재사용 비율 (성능 최적화)

---

## 🎯 학습 목표와 수렴

### 에이전트가 학습하는 것
1. **유효한 구문 구성**: 문법적으로 올바른 수식 생성
2. **수익성 최적화**: 백테스트 수익률 극대화  
3. **복잡도 관리**: 과도한 복잡성 vs 성능의 균형
4. **현실적 제약 준수**: 거래비용, 지연시간 고려
5. **신호 품질**: 안정적이고 실행 가능한 신호 생성

### 성공적인 수렴 지표
- **보상 증가**: 에피소드별 평균 보상 상승
- **유효한 프로그램 비율 증가**: invalid 액션 감소
- **캐시 적중률 상승**: 효율적인 탐색 전략 학습
- **미래 누출 검출률 감소**: 올바른 시간적 순서 학습

이러한 종합적인 보상 시스템을 통해 에이전트는 현실적이고 수익성 있는 트레이딩 전략을 점진적으로 학습하게 됩니다! 🚀
