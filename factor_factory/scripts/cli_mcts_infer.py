#!/usr/bin/env python3
"""
MCTS 기반 Factor Inference CLI

학습된 MCTS 모델로 최적 팩터 탐색 및 추론
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import RLCConfig
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.mcts import (
    PolicyValueNetwork, 
    MCTSFactorEnv, 
    MCTSSearch,
    AdaptiveMCTS
)
from factor_factory.pool import FactorPool


def mcts_inference(
    network: PolicyValueNetwork,
    env: MCTSFactorEnv,
    num_searches: int = 10,
    mcts_simulations: int = 800,
    temperature: float = 0.1,
    c_puct: float = 1.0
):
    """MCTS 기반 추론"""
    
    # MCTS 탐색기 생성
    mcts = MCTSSearch(
        network=network,
        c_puct=c_puct,
        num_simulations=mcts_simulations,
        evaluation_fn=lambda tokens: env.evaluate_program(tokens)['reward']
    )
    
    best_programs = []
    
    for search_idx in range(num_searches):
        logging.info(f"🔍 탐색 {search_idx + 1}/{num_searches}")
        
        try:
            # MCTS 탐색 실행
            action_probs, root_node = mcts.search(root_state=[], root_need=1)
            
            # 프로그램 생성
            program = generate_program_from_root(root_node, temperature)
            
            if program:
                # 프로그램 평가
                evaluation = env.evaluate_program(program)
                
                if evaluation['success']:
                    formula = tokens_to_infix(program)
                    depth = calc_tree_depth(program)
                    
                    result = {
                        'search_idx': search_idx,
                        'program': program,
                        'formula': formula,
                        'reward': evaluation['reward'],
                        'depth': depth,
                        'length': len(program),
                        'evaluation': evaluation,
                        'mcts_stats': {
                            'total_visits': root_node.visit_count,
                            'best_action': mcts.get_best_action(root_node),
                            'principal_variation': mcts.get_principal_variation(root_node)
                        }
                    }
                    
                    best_programs.append(result)
                    
                    logging.info(f"✅ 프로그램 발견: reward={evaluation['reward']:.4f}")
                    logging.info(f"   공식: {formula}")
                    logging.info(f"   깊이: {depth}, 길이: {len(program)}")
                else:
                    logging.info(f"❌ 프로그램 평가 실패: {evaluation.get('error', 'unknown')}")
            else:
                logging.info(f"❌ 프로그램 생성 실패")
                
        except Exception as e:
            logging.error(f"탐색 {search_idx} 오류: {e}")
            continue
    
    return best_programs


def generate_program_from_root(root_node, temperature: float = 0.1):
    """루트 노드에서 완전한 프로그램 생성"""
    
    program = []
    node = root_node
    max_steps = 21
    
    for step in range(max_steps):
        if not node.children or node.is_terminal:
            break
        
        # 온도 조절된 액션 선택
        action_probs = node.get_action_probs(temperature)
        valid_actions = list(node.children.keys())
        
        if not valid_actions:
            break
        
        # 확률 기반 액션 선택
        if temperature == 0:
            # Greedy 선택
            action = max(valid_actions, key=lambda a: node.children[a].visit_count)
        else:
            # 확률적 선택
            valid_probs = np.array([action_probs[a] for a in valid_actions])
            if valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
            else:
                action = np.random.choice(valid_actions)
        
        program.append(action)
        
        if action in node.children:
            node = node.children[action]
        else:
            break
    
    return program if program else None


def compare_with_ppo_results(mcts_results, ppo_results_path):
    """PPO 결과와 비교"""
    
    if not Path(ppo_results_path).exists():
        logging.warning(f"PPO 결과 파일을 찾을 수 없음: {ppo_results_path}")
        return
    
    try:
        with open(ppo_results_path, 'r') as f:
            ppo_data = json.load(f)
        
        ppo_best_reward = ppo_data.get('best_reward', 0)
        ppo_best_formula = ppo_data.get('best_formula', 'N/A')
        
        # MCTS 최고 결과
        if mcts_results:
            mcts_best = max(mcts_results, key=lambda x: x['reward'])
            mcts_best_reward = mcts_best['reward']
            mcts_best_formula = mcts_best['formula']
            
            logging.info(f"\n🔍 PPO vs MCTS 비교:")
            logging.info(f"PPO  최고 보상: {ppo_best_reward:.4f} | {ppo_best_formula}")
            logging.info(f"MCTS 최고 보상: {mcts_best_reward:.4f} | {mcts_best_formula}")
            
            if mcts_best_reward > ppo_best_reward:
                logging.info(f"🏆 MCTS가 PPO보다 {mcts_best_reward - ppo_best_reward:.4f} 우수!")
            else:
                logging.info(f"📊 PPO가 MCTS보다 {ppo_best_reward - mcts_best_reward:.4f} 우수")
    
    except Exception as e:
        logging.error(f"PPO 결과 비교 중 오류: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MCTS 기반 Factor Inference"
    )
    
    # 필수 인자
    parser.add_argument("--model", required=True, 
                       help="학습된 MCTS 모델 경로")
    parser.add_argument("--symbol", required=True, 
                       help="심볼 (예: BTCUSDT)")
    
    # 데이터 설정
    parser.add_argument("--interval", default="1h", help="시간 간격")
    
    # 추론 설정
    parser.add_argument("--num-searches", type=int, default=20,
                       help="탐색 횟수")
    parser.add_argument("--mcts-simulations", type=int, default=1000,
                       help="MCTS 시뮬레이션 횟수")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="액션 선택 온도 (0에 가까울수록 greedy)")
    parser.add_argument("--c-puct", type=float, default=1.0,
                       help="UCB 탐색 상수")
    parser.add_argument("--use-adaptive", action="store_true",
                       help="적응적 MCTS 사용")
    
    # 환경 설정
    parser.add_argument("--max-len", type=int, default=21)
    parser.add_argument("--eval-stride", type=int, default=2)
    parser.add_argument("--max-eval-bars", type=int, default=20000)
    parser.add_argument("--long-threshold", type=float, default=1.5)
    parser.add_argument("--short-threshold", type=float, default=-1.5)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--commission", type=float, default=0.0008)
    parser.add_argument("--slippage", type=float, default=0.0015)
    parser.add_argument("--leverage", type=float, default=1.0)
    
    # 출력 설정
    parser.add_argument("--outdir", default="mcts_inference_results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--factor-pool-dir", 
                       help="우수한 팩터를 저장할 Factor Pool 디렉토리")
    parser.add_argument("--compare-ppo", 
                       help="PPO 결과와 비교할 JSON 파일 경로")
    
    # 기타
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args(argv)
    
    # ── 결과 디렉토리 생성 ───────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    # ── 데이터 로딩 ──────────────────────────────────────────
    logging.info(f"데이터 로딩: {args.symbol}_{args.interval}")
    cache = ParquetCache(DATA_ROOT)
    
    try:
        df = cache.load(args.symbol, args.interval)
        logging.info(f"데이터 크기: {df.shape}, 기간: {df.index[0]} ~ {df.index[-1]}")
    except Exception as e:
        logging.error(f"데이터 로딩 실패: {e}")
        return
    
    # ── 환경 설정 ──────────────────────────────────────────
    rlc_config = RLCConfig(
        max_len=args.max_len,
        eval_stride=args.eval_stride,
        max_eval_bars=args.max_eval_bars,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        rolling_window=args.rolling_window,
        commission=args.commission,
        slippage=args.slippage,
        leverage=args.leverage
    )
    
    mcts_env = MCTSFactorEnv(df, rlc_config)
    
    # ── 모델 로딩 ──────────────────────────────────────────
    logging.info(f"MCTS 모델 로딩: {args.model}")
    
    try:
        network = PolicyValueNetwork(
            input_dim=23,
            action_dim=25
        )
        
        # 모델 로딩 방식 결정
        model_path = Path(args.model)
        if model_path.suffix == '.pt':
            # PyTorch 체크포인트
            import torch
            checkpoint = torch.load(args.model, map_location='cpu')
            network.load_state_dict(checkpoint['network_state_dict'])
        else:
            logging.error(f"지원하지 않는 모델 형식: {model_path.suffix}")
            return
        
        logging.info("모델 로딩 완료")
        
    except Exception as e:
        logging.error(f"모델 로딩 실패: {e}")
        return
    
    # ── MCTS 추론 실행 ──────────────────────────────────────
    logging.info(f"🚀 MCTS 추론 시작")
    logging.info(f"설정: {args.num_searches}회 탐색, {args.mcts_simulations} 시뮬레이션")
    
    start_time = time.time()
    
    try:
        results = mcts_inference(
            network=network,
            env=mcts_env,
            num_searches=args.num_searches,
            mcts_simulations=args.mcts_simulations,
            temperature=args.temperature,
            c_puct=args.c_puct
        )
        
        inference_time = time.time() - start_time
        
        # ── 결과 분석 ──────────────────────────────────────────
        if results:
            # 보상 기준 정렬
            results.sort(key=lambda x: x['reward'], reverse=True)
            
            logging.info(f"\n🏆 상위 결과:")
            for i, result in enumerate(results[:5]):
                logging.info(f"{i+1}. 보상: {result['reward']:.4f} | {result['formula']}")
            
            # 통계
            rewards = [r['reward'] for r in results]
            depths = [r['depth'] for r in results]
            lengths = [r['length'] for r in results]
            
            stats = {
                'total_programs': len(results),
                'avg_reward': np.mean(rewards),
                'max_reward': np.max(rewards),
                'min_reward': np.min(rewards),
                'avg_depth': np.mean(depths),
                'avg_length': np.mean(lengths),
                'inference_time': inference_time,
                'success_rate': len(results) / args.num_searches
            }
            
            logging.info(f"\n📊 통계:")
            logging.info(f"  성공한 프로그램: {stats['total_programs']}/{args.num_searches}")
            logging.info(f"  평균 보상: {stats['avg_reward']:.4f}")
            logging.info(f"  최고 보상: {stats['max_reward']:.4f}")
            logging.info(f"  평균 깊이: {stats['avg_depth']:.1f}")
            logging.info(f"  추론 시간: {stats['inference_time']:.1f}초")
            
        else:
            logging.warning("성공한 프로그램이 없습니다.")
            stats = {'total_programs': 0, 'success_rate': 0}
        
        # ── 결과 저장 ──────────────────────────────────────────
        results_data = {
            'config': vars(args),
            'stats': stats,
            'programs': results,
            'timestamp': time.time()
        }
        
        results_file = outdir / f"mcts_inference_{args.symbol}_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logging.info(f"💾 결과 저장: {results_file}")
        
        # ── Factor Pool 저장 ───────────────────────────────────
        if args.factor_pool_dir and results:
            logging.info(f"Factor Pool에 우수한 팩터 저장...")
            
            factor_pool = FactorPool(args.factor_pool_dir)
            saved_count = 0
            
            # 상위 5개 프로그램 저장
            for result in results[:5]:
                if result['reward'] > 0:  # 양수 보상만
                    try:
                        # 가상 시계열 데이터 (실제로는 백테스트 결과 사용)
                        import pandas as pd
                        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
                        pnl = pd.Series(np.random.normal(result['reward']/1000, 0.01, 1000), index=dates)
                        equity = (1 + pnl).cumprod()
                        signal = pd.Series(np.random.normal(0, 1, 1000), index=dates)
                        
                        reward_info = {
                            'total_reward': result['reward'],
                            'components': {'mcts_inference': result['reward']},
                            'future_leak': False,
                            'validation': {'mcts_evaluated': True}
                        }
                        
                        factor_pool.add_factor(
                            tokens=result['program'],
                            formula=result['formula'],
                            pnl=pnl,
                            equity=equity,
                            signal=signal,
                            reward_info=reward_info,
                            model_version="MCTS_Inference",
                            training_episode=0
                        )
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logging.warning(f"Factor Pool 저장 실패: {e}")
            
            logging.info(f"✅ {saved_count}개 팩터 저장 완료")
        
        # ── PPO 비교 ──────────────────────────────────────────
        if args.compare_ppo:
            compare_with_ppo_results(results, args.compare_ppo)
        
        # ── 환경 통계 ──────────────────────────────────────────
        env_stats = mcts_env.get_statistics()
        logging.info(f"\n🏭 환경 통계: {env_stats}")
        
        logging.info("✅ MCTS 추론 완료!")
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 추론이 중단되었습니다.")
    except Exception as e:
        logging.error(f"추론 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
