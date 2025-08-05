# Factor Factory – Constrained RL (Tree Assembly) v2.1 (Enhanced)

### 🆕 새로운 기능들

- **확장된 토큰 세트**: 25개 토큰 (기존 13개 → 25개)
  - 새로운 기술적 지표: SMA5, EMA10/20, 볼린저 밴드, MACD, Stochastic
  - 새로운 연산자: MAX, MIN, ABS, LOG, LAG1
- **개선된 정규화**: 롤링 윈도우 기반 Z-score (미래 정보 누출 방지)
- **적응적 임계값**: ±2.0 → ±1.5로 완화하여 더 많은 거래 기회 창출
- **LRU 캐싱**: 반복 계산 방지로 성능 향상
- **강화된 메트릭**: Calmar 비율, 승률, 수익 팩터 등 추가
- **보상 함수 최적화**: 페널티 파라미터 조정으로 학습 안정성 향상

### 📊 토큰 목록

#### 터미널 토큰 (Arity = 0)
```
CLOSE, OPEN, HIGH, LOW, VOLUME
SMA5, SMA10, SMA20, EMA10, EMA20
BBANDS_UPPER, BBANDS_LOWER, MACD, RSI14, STOCH
CONST1
```

#### 이진 연산자 (Arity = 2)
```
ADD, SUB, MUL, DIV, MAX, MIN
```

#### 단항 연산자 (Arity = 1)
```
ABS, LOG, LAG1
```

### 🚀 빠른 시작

```bash
# 의존성 설치
pip install -r factor_factory/requirements.txt

# 개선된 설정으로 훈련
python -m factor_factory.scripts.cli_rlc_train \
  --symbol BTCUSDT --interval 1h --timesteps 150000 \
  --save models/ppo_program.zip \
  --eval_stride 2 --max_eval_bars 20000 \
  --long_threshold 1.5 --short_threshold -1.5 \
  --rolling_window 252

# 최적 프로그램 탐색
python -m factor_factory.scripts.cli_rlc_infer \
  --model models/ppo_program.zip \
  --symbol BTCUSDT --interval 1h --tries 512 \
  --outdir rlc_out --eval_stride 2 --max_eval_bars 20000

# 성능 평가
python -m factor_factory.scripts.cli_rlc_eval \
  --program rlc_out/best_program.json \
  --symbol BTCUSDT --interval 1h --outdir rlc_out
```

### 🔧 주요 개선사항

#### 1. 미래 정보 누출 방지
- **롤링 윈도우 정규화**: 과거 252일 기준으로 Z-score 계산
- **Expanding 윈도우**: 초기 구간에서는 사용 가능한 모든 데이터 활용

#### 2. 성능 최적화
- **LRU 캐시**: 동일한 프로그램 재계산 방지
- **벡터화 연산**: NumPy 기반 고속 처리
- **메모리 효율성**: 서브샘플링과 데이터 절단

#### 3. 학습 안정성
- **보상 함수 밸런싱**: 페널티 파라미터 최적화
- **적응적 임계값**: 더 유연한 거래 신호 생성
- **강건한 오류 처리**: NaN/무한대 값 안전 처리

### 📈 새로운 평가 메트릭

```python
{
  "cagr": -0.036,           # 연평균 성장률
  "sharpe": -0.524,         # 샤프 비율
  "mdd": -0.625,            # 최대 낙폭
  "turnover": 829.16,       # 회전율
  "calmar": -0.058,         # 칼마 비율
  "win_rate": 0.45,         # 승률
  "profit_factor": 0.85,    # 수익 팩터
  "max_consecutive_losses": 12,  # 최대 연속 손실 일수
  "information_ratio": -0.52,    # 정보 비율
  "total_trades": 415,      # 총 거래 수
  "avg_trade_pnl": -0.001   # 평균 거래 수익률
}
```

### 🧪 테스트

```bash
# 새로운 기능들 테스트
python test_new_tokens.py
```

테스트 결과:
- ✅ 25개 토큰 모두 정상 작동
- ✅ 캐시 성능 향상 확인
- ✅ 정규화 범위 [-1, 1] 유지
- ✅ 새로운 연산자들 작동 확인

### 🔧 설정 파라미터

#### RLCConfig 주요 설정
```python
@dataclass
class RLCConfig:
    max_len: int = 21                # 최대 프로그램 길이
    long_threshold: float = 1.5      # 롱 진입 임계값
    short_threshold: float = -1.5    # 숏 진입 임계값
    rolling_window: int = 252        # 정규화 윈도우 크기
    lambda_const1: float = 2.0       # 상수 사용 페널티 (완화)
    lambda_std: float = 0.5          # 변동성 페널티 (완화)
    eval_stride: int = 2             # 평가 간격
    max_eval_bars: int = 20_000      # 최대 평가 바 수
```

### 📁 프로젝트 구조

```
factor_factory/
├── rlc/
│   ├── env.py           # 강화학습 환경 (개선된 보상함수)
│   ├── grammar.py       # 확장된 토큰 정의 (25개)
│   ├── compiler.py      # 개선된 컴파일러 (캐싱, 정규화)
│   ├── cache.py         # LRU 캐시 시스템 (신규)
│   ├── utils.py         # 유틸리티 함수들
│   └── callback.py      # 학습 콜백
├── backtest/
│   └── engine.py        # 벡터화된 백테스팅
├── selection/
│   └── metrics.py       # 확장된 성능 메트릭
├── scripts/
│   ├── cli_rlc_train.py # 개선된 훈련 스크립트
│   ├── cli_rlc_infer.py # 개선된 추론 스크립트
│   └── cli_rlc_eval.py  # 평가 스크립트
└── data/
    └── __init__.py      # 데이터 로딩
```

### 🎯 다음 단계 개발 제안

1. **멀티 타임프레임 지원**: 여러 시간대 데이터 동시 활용
2. **동적 포지션 사이징**: 시장 상황에 따른 자동 조절
3. **앙상블 모델**: 여러 최적 프로그램 조합
4. **실시간 트레이딩**: 라이브 데이터 연동
5. **고급 리스크 관리**: VaR, 상관관계 기반 제약

### 📊 성능 비교

| 항목 | v2.0 | v2.1 Enhanced |
|------|------|---------------|
| 토큰 수 | 13개 | 25개 (+92%) |
| 캐시 적중률 | N/A | ~60-80% |
| 정규화 방식 | 전체 기간 | 롤링 윈도우 |
| 임계값 | ±2.0 | ±1.5 |
| 메트릭 수 | 4개 | 11개 |
| 미래 정보 누출 | 있음 | 없음 |

이 개선된 버전은 더욱 현실적이고 robust한 트레이딩 시스템을 제공합니다.
