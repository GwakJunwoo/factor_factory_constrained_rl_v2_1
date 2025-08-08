#!/usr/bin/env python3
"""
간단한 MCTS 예제 및 사용법 데모
"""

import numpy as np
import pandas as pd
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_mcts_demo():
    """간단한 MCTS 데모"""
    print("🤖 AlphaZero-style MCTS 시스템 데모")
    print("="*50)
    
    try:
        # Import 테스트
        print("📦 모듈 import 중...")
        from factor_factory.mcts import PolicyValueNetwork, MCTSSearch
        from factor_factory.rlc import RLCConfig
        print("✅ 모듈 import 성공!")
        
        # 신경망 생성
        print("\n🧠 Policy-Value Network 생성...")
        network = PolicyValueNetwork(
            input_dim=23,
            hidden_dims=[128, 64],  # 작은 네트워크
            action_dim=25
        )
        print(f"✅ 네트워크 생성 완료: {network}")
        
        # 간단한 예측 테스트
        print("\n🔮 신경망 예측 테스트...")
        test_obs = np.random.randn(23)
        policy_probs, value = network.predict(test_obs)
        print(f"  정책 확률: 형태={policy_probs.shape}, 합={policy_probs.sum():.3f}")
        print(f"  상태 가치: {value:.3f}")
        print("✅ 예측 테스트 완료!")
        
        # MCTS 탐색 테스트
        print("\n🔍 MCTS 탐색 테스트...")
        
        def simple_eval(tokens):
            """간단한 평가 함수"""
            if len(tokens) == 0:
                return 0.0
            return np.random.uniform(-0.5, 1.0) - len(tokens) * 0.01
        
        mcts = MCTSSearch(
            network=network,
            num_simulations=50,  # 빠른 테스트용
            evaluation_fn=simple_eval
        )
        
        print("  MCTS 탐색 실행 중...")
        action_probs, root_node = mcts.search(root_state=[], root_need=1)
        
        print(f"  탐색 완료!")
        print(f"  루트 방문: {root_node.visit_count}")
        print(f"  자식 수: {len(root_node.children)}")
        print(f"  최고 액션: {np.argmax(action_probs)}")
        print("✅ MCTS 탐색 테스트 완료!")
        
        print("\n🎉 모든 기본 테스트 성공!")
        
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        print("필요한 패키지를 설치해주세요:")
        print("pip install torch pandas numpy")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def show_usage():
    """사용법 안내"""
    print("\n📚 MCTS 시스템 사용법:")
    print("-" * 40)
    
    print("1️⃣ MCTS 모델 학습:")
    print("   python -m factor_factory.scripts.cli_mcts_train \\")
    print("     --symbol BTCUSDT \\")
    print("     --iterations 50 \\")
    print("     --episodes-per-iter 100 \\")
    print("     --mcts-simulations 800")
    
    print("\n2️⃣ 학습된 모델로 추론:")
    print("   python -m factor_factory.scripts.cli_mcts_infer \\")
    print("     --model mcts_results/best_model.pt \\")
    print("     --symbol BTCUSDT \\")
    print("     --num-searches 20")
    
    print("\n3️⃣ PPO와 병렬 실행:")
    print("   # 터미널 1: PPO 학습")
    print("   python -m factor_factory.scripts.cli_rlc_train --symbol BTCUSDT --timesteps 1000 --save models_v4/ppo_model")
    print("")
    print("   # 터미널 2: MCTS 학습") 
    print("   python -m factor_factory.scripts.cli_mcts_train --symbol BTCUSDT --iterations 50 --save-dir mcts_v1")
    
    print("\n4️⃣ 결과 비교:")
    print("   python -m factor_factory.scripts.cli_mcts_infer \\")
    print("     --model mcts_v1/best_model.pt \\")
    print("     --symbol BTCUSDT \\")
    print("     --compare-ppo models_v4/ppo_results.json")


def show_features():
    """MCTS 시스템 특징"""
    print("\n🆚 PPO vs MCTS 비교:")
    print("-" * 40)
    
    print("📊 PPO (Proximal Policy Optimization):")
    print("  ✅ 빠른 학습 속도")
    print("  ✅ 안정적인 수렴")
    print("  ✅ 연속적 정책 개선")
    print("  ⚠️ 지역 최적화에 갇힐 수 있음")
    
    print("\n🌲 MCTS (Monte Carlo Tree Search):")
    print("  ✅ 전역 탐색 능력")
    print("  ✅ 더 정확한 가치 평가")
    print("  ✅ 장기 계획 수립")
    print("  ⚠️ 느린 탐색 속도")
    
    print("\n🔄 병렬 사용의 장점:")
    print("  🎯 PPO: 빠른 탐색으로 좋은 시작점 발견")
    print("  🎯 MCTS: 정밀한 탐색으로 최적 솔루션 발견")
    print("  🎯 Factor Pool: 두 방법의 결과를 모두 저장하여 최고의 팩터 수집")


if __name__ == "__main__":
    simple_mcts_demo()
    show_usage()
    show_features()
    
    print(f"\n🚀 시스템 준비 완료!")
    print(f"이제 기존 PPO와 함께 MCTS를 사용할 수 있습니다!")
