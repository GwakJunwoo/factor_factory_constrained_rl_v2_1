#!/usr/bin/env python3
"""
새로운 토큰들과 개선된 기능들을 테스트하는 스크립트
"""

import pandas as pd
import numpy as np
from factor_factory.rlc.grammar import TOKENS, N_TOKENS, TOKEN_NAMES
from factor_factory.rlc.compiler import eval_prefix, _get_terminal
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.data import ParquetCache, DATA_ROOT

def create_test_data():
    """테스트용 데이터 생성"""
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')  # 'H' → 'h'로 변경
    np.random.seed(42)
    
    # 가격 데이터 생성 (랜덤워크 + 트렌드)
    returns = np.random.normal(0.0002, 0.02, len(dates))  # 평균 수익률과 변동성
    close = pd.Series(100 * np.cumprod(1 + returns), index=dates)  # Series로 변환
    
    # OHLC 생성
    high = close * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    open_prices = close.shift(1).fillna(close.iloc[0])  # close[0] → close.iloc[0]
    volume = pd.Series(np.random.lognormal(10, 1, len(dates)), index=dates)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df

def test_new_tokens():
    """새로운 토큰들 테스트"""
    print("=== 새로운 토큰 테스트 ===")
    print(f"총 토큰 수: {N_TOKENS}")
    print("\n토큰 목록:")
    for tok_id, (name, arity) in TOKENS.items():
        print(f"  {tok_id:2d}: {name:12s} (arity={arity})")
    
    # 테스트 데이터 생성
    df = create_test_data()
    
    print("\n=== 터미널 토큰 테스트 ===")
    terminal_tokens = [tok for tok, (name, arity) in TOKENS.items() if arity == 0]
    
    for tok in terminal_tokens:
        try:
            result = _get_terminal(df, tok)
            name = TOKEN_NAMES[tok]
            print(f"  {name:15s}: {result.shape}, mean={result.mean():.4f}, std={result.std():.4f}")
        except Exception as e:
            print(f"  {TOKEN_NAMES[tok]:15s}: ERROR - {e}")

def test_program_compilation():
    """프로그램 컴파일 테스트"""
    print("\n=== 프로그램 컴파일 테스트 ===")
    df = create_test_data()
    
    # 간단한 테스트 프로그램들
    test_programs = [
        [2, 4, 9],          # MUL CLOSE SMA10 -> (CLOSE * SMA10)
        [0, 6, 7],          # ADD HIGH LOW -> (HIGH + LOW)
        [22, 1, 4, 10],     # ABS SUB CLOSE SMA20 -> ABS(CLOSE - SMA20)
        [20, 4, 5],         # MAX CLOSE OPEN -> MAX(CLOSE, OPEN)
        [24, 4],            # LAG1 CLOSE -> LAG1(CLOSE)
    ]
    
    for i, tokens in enumerate(test_programs):
        try:
            result = eval_prefix(tokens, df)
            infix = tokens_to_infix(tokens)
            depth = calc_tree_depth(tokens)
            print(f"  프로그램 {i+1}: {infix}")
            print(f"    토큰: {tokens}")
            print(f"    깊이: {depth}")
            print(f"    결과: shape={result.shape}, mean={result.mean():.4f}, std={result.std():.4f}")
            print()
        except Exception as e:
            print(f"  프로그램 {i+1}: ERROR - {e}")
            print(f"    토큰: {tokens}")
            print()

def test_cache_performance():
    """캐시 성능 테스트"""
    print("\n=== 캐시 성능 테스트 ===")
    from factor_factory.rlc.cache import get_program_cache, clear_program_cache
    import time
    
    # 캐시 초기화
    clear_program_cache()
    cache = get_program_cache()
    
    df = create_test_data()
    test_program = [2, 4, 9]  # CLOSE * SMA10
    
    # 첫 번째 실행 (캐시 미스)
    start_time = time.time()
    result1 = eval_prefix(test_program, df, use_cache=True)
    first_time = time.time() - start_time
    
    # 두 번째 실행 (캐시 히트)
    start_time = time.time()
    result2 = eval_prefix(test_program, df, use_cache=True)
    second_time = time.time() - start_time
    
    print(f"  첫 번째 실행 (캐시 미스): {first_time:.4f}초")
    print(f"  두 번째 실행 (캐시 히트): {second_time:.4f}초")
    print(f"  속도 향상: {first_time/second_time:.1f}배")
    print(f"  결과 일치: {result1.equals(result2)}")

def test_improved_normalization():
    """개선된 정규화 테스트"""
    print("\n=== 개선된 정규화 테스트 ===")
    df = create_test_data()
    
    # 간단한 테스트 프로그램
    tokens = [1, 4, 10]  # CLOSE - SMA20
    result = eval_prefix(tokens, df)
    
    print(f"  정규화된 시그널 범위: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  시그널 평균: {result.mean():.4f}")
    print(f"  시그널 표준편차: {result.std():.4f}")
    print(f"  NaN 값 개수: {result.isna().sum()}")
    print(f"  무한대 값 개수: {np.isinf(result).sum()}")

def test_new_operators():
    """새로운 연산자들 테스트"""
    print("\n=== 새로운 연산자 테스트 ===")
    df = create_test_data()
    
    # 단항 연산자 테스트
    unary_tests = [
        ([22, 4], "ABS(CLOSE)"),
        ([23, 4], "LOG(CLOSE)"),
        ([24, 4], "LAG1(CLOSE)"),
    ]
    
    for tokens, description in unary_tests:
        try:
            result = eval_prefix(tokens, df)
            print(f"  {description:15s}: OK, mean={result.mean():.4f}, std={result.std():.4f}")
        except Exception as e:
            print(f"  {description:15s}: ERROR - {e}")
    
    # 이진 연산자 테스트
    binary_tests = [
        ([20, 4, 5], "MAX(CLOSE, OPEN)"),
        ([21, 6, 7], "MIN(HIGH, LOW)"),
    ]
    
    for tokens, description in binary_tests:
        try:
            result = eval_prefix(tokens, df)
            print(f"  {description:15s}: OK, mean={result.mean():.4f}, std={result.std():.4f}")
        except Exception as e:
            print(f"  {description:15s}: ERROR - {e}")

def test_best_program():
    """기존 최적 프로그램 테스트"""
    print("\n=== 기존 최적 프로그램 테스트 ===")
    
    # best_program.json에서 토큰 로드
    try:
        import json
        with open('models/best_program.json', 'r') as f:
            data = json.load(f)
            tokens = data['tokens']
        
        df = create_test_data()
        
        result = eval_prefix(tokens, df)
        infix = tokens_to_infix(tokens)
        depth = calc_tree_depth(tokens)
        
        print(f"  수식: {infix}")
        print(f"  토큰: {tokens}")
        print(f"  깊이: {depth}")
        print(f"  결과: shape={result.shape}, mean={result.mean():.4f}, std={result.std():.4f}")
        
    except FileNotFoundError:
        print("  models/best_program.json 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_new_tokens()
    test_program_compilation()
    test_cache_performance()
    test_improved_normalization()
    test_new_operators()
    test_best_program()
    print("\n=== 테스트 완료 ===")
