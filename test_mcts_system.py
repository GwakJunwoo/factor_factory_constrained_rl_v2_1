#!/usr/bin/env python3
"""
MCTS 시스템 통합 테스트

AlphaZero-style MCTS 시스템의 모든 컴포넌트를 테스트하고 시연
"""

import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import RLCConfig
from factor_factory.mcts import (
    PolicyValueNetwork,
    MCTSSearch, 
    MCTSFactorEnv,
    AlphaZeroTrainer
)
from factor_factory.pool import FactorPool


def test_neural_network():
    """신경망 테스트"""
    print("🧠 Policy-Value Network 테스트")
    
    network = PolicyValueNetwork()
    
    # 단일 예측 테스트
    obs = np.random.randn(23)
    policy_probs, value = network.predict(obs)
    
    print(f"  입력 크기: {obs.shape}")
    print(f"  정책 출력: {policy_probs.shape}, 합: {policy_probs.sum():.3f}")
    print(f"  가치 출력: {value:.3f}")
    
    # 배치 예측 테스트
    batch_obs = np.random.randn(32, 23)
    batch_policies, batch_values = network.predict_batch(batch_obs)
    
    print(f"  배치 입력: {batch_obs.shape}")
    print(f"  배치 정책: {batch_policies.shape}")
    print(f"  배치 가치: {batch_values.shape}")
    
    print("✅ 신경망 테스트 완료\n")


def test_mcts_search():
    """MCTS 탐색 테스트"""
    print("🔍 MCTS Search 테스트")
    
    network = PolicyValueNetwork()
    
    # 더미 평가 함수
    def dummy_eval(tokens):
        return np.random.uniform(-0.5, 1.0) - len(tokens) * 0.01
    
    mcts = MCTSSearch(
        network=network,
        num_simulations=100,  # 테스트용 적은 수
        evaluation_fn=dummy_eval
    )
    
    # 탐색 실행
    start_time = time.time()
    action_probs, root_node = mcts.search(root_state=[], root_need=1)
    search_time = time.time() - start_time
    
    print(f"  탐색 시간: {search_time:.2f}초")
    print(f"  루트 방문 횟수: {root_node.visit_count}")
    print(f"  자식 노드 수: {len(root_node.children)}")
    print(f"  최고 액션: {mcts.get_best_action(root_node)}")
    print(f"  주요 변화: {mcts.get_principal_variation(root_node, 3)}")
    
    print("✅ MCTS 탐색 테스트 완료\n")


def test_mcts_environment():
    """MCTS 환경 테스트"""
    print("🏭 MCTS Environment 테스트")
    
    # 더미 데이터 생성
    dates = pd.date_range('2024-01-01', periods=500, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 102,
        'low': np.random.randn(500).cumsum() + 98,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    config = RLCConfig()
    env = MCTSFactorEnv(df, config)
    
    # 프로그램 평가 테스트
    test_programs = [
        [1, 2, 12],           # SMA(CLOSE, 20)
        [1, 3, 4, 12],        # (CLOSE + HIGH) / 2
        [1, 2, 5, 12, 6]      # Complex program
    ]
    
    for i, program in enumerate(test_programs):
        result = env.evaluate_program(program)
        print(f"  프로그램 {i+1}: {program}")
        print(f"    성공: {result['success']}")
        print(f"    보상: {result['reward']:.4f}")
        if result.get('formula'):
            print(f"    공식: {result['formula']}")
    
    stats = env.get_statistics()
    print(f"  환경 통계: {stats}")
    
    print("✅ MCTS 환경 테스트 완료\n")


def test_alphazero_trainer():
    """AlphaZero 트레이너 테스트"""
    print("🤖 AlphaZero Trainer 테스트")
    
    # 더미 데이터 생성
    dates = pd.date_range('2024-01-01', periods=200, freq='H')  # 작은 데이터셋
    df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    config = RLCConfig()
    env = MCTSFactorEnv(df, config)
    network = PolicyValueNetwork()
    factor_pool = FactorPool("test_mcts_factor_pool")
    
    # 테스트용 간소화된 트레이너
    trainer = AlphaZeroTrainer(
        env=env,
        network=network,
        factor_pool=factor_pool,
        episodes_per_iteration=5,    # 적은 에피소드
        mcts_simulations=50,         # 적은 시뮬레이션
        training_epochs=2,           # 적은 에포크
        evaluation_episodes=3,       # 적은 평가
        evaluation_interval=1,       # 매번 평가
        save_interval=2,             # 자주 저장
        checkpoint_dir="test_mcts_checkpoints"
    )
    
    print(f"  테스트용 설정:")
    print(f"    에피소드/반복: {trainer.episodes_per_iteration}")
    print(f"    MCTS 시뮬레이션: {trainer.mcts.num_simulations}")
    print(f"    학습 에포크: {trainer.training_epochs}")
    
    # 짧은 학습 실행
    start_time = time.time()
    trainer.train(num_iterations=2)  # 2번만 반복
    training_time = time.time() - start_time
    
    print(f"  학습 시간: {training_time:.1f}초")
    print(f"  발견된 팩터: {len(trainer.discovered_factors)}개")
    print(f"  최고 성능: {trainer.best_performance:.4f}")
    
    # Factor Pool 통계
    pool_stats = factor_pool.get_statistics()
    print(f"  Factor Pool: {pool_stats}")
    
    print("✅ AlphaZero 트레이너 테스트 완료\n")


def integration_test():
    """통합 테스트"""
    print("🔗 MCTS 시스템 통합 테스트")
    
    try:
        # 실제 데이터로 테스트 (가능한 경우)
        try:
            cache = ParquetCache(DATA_ROOT)
            df = cache.get("BTCUSDT", "1h")
            df = df.tail(1000)  # 최근 1000개 데이터만 사용
            print(f"  실제 데이터 사용: {df.shape}")
        except:
            # 더미 데이터 사용
            dates = pd.date_range('2024-01-01', periods=1000, freq='H')
            df = pd.DataFrame({
                'open': np.random.randn(1000).cumsum() + 100,
                'high': np.random.randn(1000).cumsum() + 102, 
                'low': np.random.randn(1000).cumsum() + 98,
                'close': np.random.randn(1000).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 1000)
            }, index=dates)
            print(f"  더미 데이터 사용: {df.shape}")
        
        # 전체 파이프라인 테스트
        config = RLCConfig()
        env = MCTSFactorEnv(df, config)
        network = PolicyValueNetwork()
        
        # MCTS로 프로그램 생성
        mcts = MCTSSearch(
            network=network,
            num_simulations=200,
            evaluation_fn=lambda tokens: env.evaluate_program(tokens)['reward']
        )
        
        print("  MCTS 탐색으로 프로그램 생성 중...")
        action_probs, root_node = mcts.search([], 1)
        
        # 여러 프로그램 생성 및 평가
        programs_found = 0
        total_attempts = 5
        
        for attempt in range(total_attempts):
            try:
                # 프로그램 생성
                program = []
                node = root_node
                
                for _ in range(21):  # 최대 길이
                    if not node.children or node.is_terminal:
                        break
                    
                    # 가장 방문 많은 자식 선택
                    best_child = node.get_best_child()
                    if best_child is None:
                        break
                    
                    program.append(best_child.action)
                    node = best_child
                
                if program:
                    result = env.evaluate_program(program)
                    if result['success']:
                        programs_found += 1
                        print(f"    프로그램 {programs_found}: 보상={result['reward']:.4f}")
                        if result.get('formula'):
                            print(f"      공식: {result['formula']}")
            
            except Exception as e:
                print(f"    시도 {attempt} 실패: {e}")
                continue
        
        success_rate = programs_found / total_attempts
        print(f"  성공률: {programs_found}/{total_attempts} ({success_rate:.1%})")
        
        env_stats = env.get_statistics()
        print(f"  최종 환경 통계: {env_stats}")
        
        print("✅ 통합 테스트 완료\n")
        
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}\n")


def main():
    print("🚀 MCTS 시스템 종합 테스트 시작\n")
    print("="*60)
    
    # 개별 컴포넌트 테스트
    test_neural_network()
    test_mcts_search() 
    test_mcts_environment()
    test_alphazero_trainer()
    
    # 통합 테스트
    integration_test()
    
    print("="*60)
    print("🎉 모든 테스트 완료!")
    
    print("\n📚 MCTS 시스템 사용법:")
    print("1. 학습: python -m factor_factory.scripts.cli_mcts_train --symbol BTCUSDT --iterations 50")
    print("2. 추론: python -m factor_factory.scripts.cli_mcts_infer --model best_model.pt --symbol BTCUSDT --num-searches 20")
    print("3. PPO와 비교: python -m factor_factory.scripts.cli_mcts_infer --compare-ppo ppo_results.json")
    
    print("\n🔄 PPO vs MCTS 비교:")
    print("- PPO: 연속적 정책 개선, 빠른 학습, 안정적")
    print("- MCTS: 트리 탐색 기반, 정확한 평가, 더 나은 장기 계획")
    print("- 두 방법을 병렬로 실행하여 최고의 팩터 발견 가능!")


if __name__ == "__main__":
    main()
