#!/usr/bin/env python3
"""MCTS 시스템 import 테스트"""

print("🔧 MCTS 시스템 import 테스트 시작...")

try:
    # 개별 모듈 import 테스트
    print("1. MCTSNode import...")
    from factor_factory.mcts.mcts_node import MCTSNode
    print("   ✅ MCTSNode 성공")
    
    print("2. PolicyValueNetwork import...")
    from factor_factory.mcts.neural_network import PolicyValueNetwork, NetworkTrainer
    print("   ✅ PolicyValueNetwork 성공")
    
    print("3. MCTSSearch import...")
    from factor_factory.mcts.mcts_search import MCTSSearch
    print("   ✅ MCTSSearch 성공")
    
    print("4. MCTSFactorEnv import...")
    from factor_factory.mcts.mcts_env import MCTSFactorEnv
    print("   ✅ MCTSFactorEnv 성공")
    
    print("5. AlphaZeroTrainer import...")
    from factor_factory.mcts.alphazero_trainer import AlphaZeroTrainer
    print("   ✅ AlphaZeroTrainer 성공")
    
    print("\n6. 패키지 전체 import...")
    from factor_factory.mcts import (
        MCTSNode, PolicyValueNetwork, NetworkTrainer,
        MCTSSearch, MCTSFactorEnv, AlphaZeroTrainer
    )
    print("   ✅ 패키지 전체 import 성공")
    
    print("\n🎉 모든 MCTS 컴포넌트 import 성공!")
    print("✅ AlphaZero-style MCTS 시스템이 준비되었습니다!")
    
except Exception as e:
    print(f"❌ Import 오류: {e}")
    import traceback
    traceback.print_exc()
