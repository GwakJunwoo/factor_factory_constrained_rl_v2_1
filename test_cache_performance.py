#!/usr/bin/env python3
"""
캐시 효율성 및 백테스트 성능 테스트 스크립트

향상된 캐시 시스템과 고속 백테스트 엔진의 성능을 테스트하고
기존 시스템과 비교 분석
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.enhanced_cache import get_fast_program_cache, clear_all_caches, get_cache_statistics
from factor_factory.backtest.fast_engine import fast_vector_backtest, get_fast_backtest_engine
from factor_factory.backtest.engine import vector_backtest
from factor_factory.selection.metrics import compute_metrics


def generate_test_programs():
    """테스트용 프로그램들 생성"""
    return [
        [4, 9, 1],              # CLOSE + SMA10
        [4, 10, 2],             # CLOSE * SMA20  
        [4, 10, 1, 11, 2],      # (CLOSE - SMA20) / RSI
        [6, 7, 0, 4, 1],        # (HIGH + LOW - CLOSE)
        [4, 24, 1],             # CLOSE - LAG1(CLOSE)
        [4, 9, 1, 10, 1, 20],   # ABS((CLOSE + SMA10) - SMA20)
        [4, 6, 7, 0, 2, 11, 3], # CLOSE * (HIGH + LOW) / RSI
        [18, 11, 1],            # MACD + RSI
        [4, 16, 1, 17, 1],      # CLOSE - BBANDS_UPPER + BBANDS_LOWER
        [4, 13, 2, 14, 3]       # CLOSE * SMA5 / EMA10
    ]


def test_cache_performance():
    """캐시 성능 테스트"""
    print("🧪 캐시 성능 테스트 시작")
    print("=" * 60)
    
    # 데이터 로드
    cache = ParquetCache(DATA_ROOT)
    df = cache.load("BTCUSDT", "1h")
    df = df.tail(2000)  # 최근 2000개 데이터
    
    print(f"데이터 크기: {df.shape}")
    
    # 테스트 프로그램들
    test_programs = generate_test_programs()
    
    # 캐시 초기화
    clear_all_caches()
    
    # 첫 번째 실행 (캐시 미스)
    print("\n1️⃣ 첫 번째 실행 (캐시 미스)")
    start_time = time.time()
    
    for i, program in enumerate(test_programs):
        try:
            result = eval_prefix(program, df, use_fast_cache=True)
            print(f"  프로그램 {i+1}: {len(result)} 포인트")
        except Exception as e:
            print(f"  프로그램 {i+1}: 실패 ({e})")
    
    first_run_time = time.time() - start_time
    print(f"총 시간: {first_run_time:.2f}초")
    
    # 두 번째 실행 (캐시 히트)
    print("\n2️⃣ 두 번째 실행 (캐시 히트)")
    start_time = time.time()
    
    for i, program in enumerate(test_programs):
        try:
            result = eval_prefix(program, df, use_fast_cache=True)
            print(f"  프로그램 {i+1}: {len(result)} 포인트")
        except Exception as e:
            print(f"  프로그램 {i+1}: 실패 ({e})")
    
    second_run_time = time.time() - start_time
    print(f"총 시간: {second_run_time:.2f}초")
    
    # 성능 향상 계산
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"\n🚀 성능 향상: {speedup:.1f}배")
    
    # 캐시 통계
    cache_stats = get_cache_statistics()
    if cache_stats and 'program_cache' in cache_stats:
        stats = cache_stats['program_cache']
        print(f"📊 캐시 통계:")
        print(f"  - 전체 요청: {stats.get('total_requests', 0)}")
        print(f"  - 메모리 히트: {stats.get('memory_hits', 0)}")
        print(f"  - 디스크 히트: {stats.get('disk_hits', 0)}")
        print(f"  - 히트율: {stats.get('hit_rate', 0)*100:.1f}%")
    
    return {
        'first_run_time': first_run_time,
        'second_run_time': second_run_time,
        'speedup': speedup,
        'cache_stats': cache_stats
    }


def test_backtest_performance():
    """백테스트 성능 테스트"""
    print("\n🏃 백테스트 성능 테스트 시작")
    print("=" * 60)
    
    # 데이터 준비
    cache = ParquetCache(DATA_ROOT)
    df = cache.load("BTCUSDT", "1h")
    df = df.tail(5000)  # 최근 5000개 데이터
    
    price = df['close']
    
    # 테스트 신호들 생성 (다양한 패턴)
    signals = []
    for i in range(10):
        # 랜덤 신호 생성
        np.random.seed(i)
        signal = pd.Series(
            np.random.randn(len(price)) * 0.5,
            index=price.index
        ).clip(-1, 1)
        signals.append(signal)
    
    print(f"테스트 신호: {len(signals)}개")
    print(f"데이터 크기: {len(price)} 포인트")
    
    # 기존 백테스트 엔진 테스트
    print("\n1️⃣ 기존 백테스트 엔진")
    start_time = time.time()
    
    old_results = []
    for i, signal in enumerate(signals):
        try:
            equity, pnl = vector_backtest(price, signal)
            old_results.append((equity, pnl))
            print(f"  신호 {i+1}: 완료")
        except Exception as e:
            print(f"  신호 {i+1}: 실패 ({e})")
    
    old_time = time.time() - start_time
    print(f"총 시간: {old_time:.2f}초")
    
    # 고속 백테스트 엔진 테스트
    print("\n2️⃣ 고속 백테스트 엔진")
    start_time = time.time()
    
    fast_results = []
    for i, signal in enumerate(signals):
        try:
            equity, pnl = fast_vector_backtest(price, signal, use_cache=True)
            fast_results.append((equity, pnl))
            print(f"  신호 {i+1}: 완료")
        except Exception as e:
            print(f"  신호 {i+1}: 실패 ({e})")
    
    fast_time = time.time() - start_time
    print(f"총 시간: {fast_time:.2f}초")
    
    # 성능 비교
    speedup = old_time / fast_time if fast_time > 0 else float('inf')
    print(f"\n🚀 백테스트 성능 향상: {speedup:.1f}배")
    
    # 결과 일치성 확인
    if old_results and fast_results:
        print("\n📊 결과 정확성 검증:")
        for i, ((old_eq, old_pnl), (fast_eq, fast_pnl)) in enumerate(zip(old_results, fast_results)):
            # 최종 값 비교
            old_final = old_eq.iloc[-1] if len(old_eq) > 0 else 0
            fast_final = fast_eq.iloc[-1] if len(fast_eq) > 0 else 0
            diff = abs(old_final - fast_final)
            
            print(f"  신호 {i+1}: 기존={old_final:.6f}, 고속={fast_final:.6f}, 차이={diff:.8f}")
    
    # 백테스트 엔진 통계
    engine = get_fast_backtest_engine()
    engine_stats = engine.get_stats()
    print(f"\n📈 백테스트 엔진 통계:")
    print(f"  - 총 백테스트: {engine_stats.get('total_backtests', 0)}")
    print(f"  - 평균 시간: {engine_stats.get('avg_time', 0):.4f}초")
    
    return {
        'old_time': old_time,
        'fast_time': fast_time,
        'speedup': speedup,
        'engine_stats': engine_stats
    }


def test_mcts_integration():
    """MCTS 환경에서 통합 성능 테스트"""
    print("\n🌲 MCTS 통합 성능 테스트")
    print("=" * 60)
    
    try:
        from factor_factory.mcts import MCTSFactorEnv
        from factor_factory.rlc import RLCConfig
        
        # 데이터 준비
        cache = ParquetCache(DATA_ROOT)
        df = cache.load("BTCUSDT", "1h")
        df = df.tail(1000)  # 작은 데이터셋
        
        # MCTS 환경 생성
        config = RLCConfig()
        env = MCTSFactorEnv(df, config)
        
        # 테스트 프로그램들
        test_programs = generate_test_programs()[:5]  # 5개만 테스트
        
        print(f"테스트 프로그램: {len(test_programs)}개")
        
        # 성능 측정
        start_time = time.time()
        results = []
        
        for i, program in enumerate(test_programs):
            try:
                result = env.evaluate_program(program)
                results.append(result)
                
                if result['success']:
                    print(f"  프로그램 {i+1}: 성공 (보상={result['reward']:.4f})")
                else:
                    print(f"  프로그램 {i+1}: 실패 ({result.get('error', 'unknown')})")
                    
            except Exception as e:
                print(f"  프로그램 {i+1}: 오류 ({e})")
        
        total_time = time.time() - start_time
        print(f"총 평가 시간: {total_time:.2f}초")
        print(f"평균 시간/프로그램: {total_time/len(test_programs):.3f}초")
        
        # 환경 통계
        env_stats = env.get_statistics()
        print(f"\n📊 MCTS 환경 통계:")
        print(f"  - 총 평가: {env_stats.get('total_evaluations', 0)}")
        print(f"  - 성공률: {env_stats.get('success_rate', 0)*100:.1f}%")
        print(f"  - 캐시 히트율: {env_stats.get('cache_hit_rate', 0)*100:.1f}%")
        print(f"  - 평균 평가 시간: {env_stats.get('avg_eval_time', 0):.3f}초")
        
        if 'enhanced_cache_stats' in env_stats:
            cache_stats = env_stats['enhanced_cache_stats']
            print(f"  - 향상된 캐시 히트율: {cache_stats.get('hit_rate', 0)*100:.1f}%")
        
        return {
            'total_time': total_time,
            'avg_time_per_program': total_time / len(test_programs),
            'env_stats': env_stats,
            'success_count': sum(1 for r in results if r.get('success', False))
        }
        
    except ImportError as e:
        print(f"❌ MCTS 모듈 import 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ MCTS 테스트 실패: {e}")
        return None


def main():
    """메인 테스트 실행"""
    print("🔬 캐시 효율성 및 백테스트 성능 테스트")
    print("=" * 80)
    
    results = {}
    
    # 1. 캐시 성능 테스트
    try:
        results['cache'] = test_cache_performance()
    except Exception as e:
        print(f"❌ 캐시 테스트 실패: {e}")
        results['cache'] = None
    
    # 2. 백테스트 성능 테스트  
    try:
        results['backtest'] = test_backtest_performance()
    except Exception as e:
        print(f"❌ 백테스트 테스트 실패: {e}")
        results['backtest'] = None
    
    # 3. MCTS 통합 테스트
    try:
        results['mcts'] = test_mcts_integration()
    except Exception as e:
        print(f"❌ MCTS 테스트 실패: {e}")
        results['mcts'] = None
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📋 테스트 결과 요약")
    print("=" * 80)
    
    if results.get('cache'):
        cache_results = results['cache']
        print(f"🧠 캐시 성능:")
        print(f"  - 속도 향상: {cache_results['speedup']:.1f}배")
        if cache_results.get('cache_stats', {}).get('program_cache'):
            cache_stats = cache_results['cache_stats']['program_cache']
            print(f"  - 히트율: {cache_stats.get('hit_rate', 0)*100:.1f}%")
    
    if results.get('backtest'):
        bt_results = results['backtest']
        print(f"🏃 백테스트 성능:")
        print(f"  - 속도 향상: {bt_results['speedup']:.1f}배")
        print(f"  - 기존 시간: {bt_results['old_time']:.2f}초")
        print(f"  - 고속 시간: {bt_results['fast_time']:.2f}초")
    
    if results.get('mcts'):
        mcts_results = results['mcts']
        print(f"🌲 MCTS 통합:")
        print(f"  - 평균 평가 시간: {mcts_results['avg_time_per_program']:.3f}초")
        print(f"  - 성공한 프로그램: {mcts_results['success_count']}개")
    
    # 결과 저장
    output_file = Path("cache_performance_test_results.json")
    try:
        with open(output_file, 'w') as f:
            # datetime 객체 등을 처리하기 위해 기본 직렬화
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 결과 저장: {output_file}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
    
    print("\n✅ 모든 테스트 완료!")


if __name__ == "__main__":
    main()
