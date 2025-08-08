#!/usr/bin/env python3
"""
AlphaZero 스타일 트레이너

자기 대국(Self-Play) + 신경망 학습을 반복하는 AlphaZero 알고리즘:
1. 현재 신경망으로 MCTS 수행하여 프로그램 생성
2. 생성된 프로그램들을 평가하여 학습 데이터 수집
3. 수집된 데이터로 신경망 업데이트
4. 주기적으로 이전 버전과 성능 비교
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from .neural_network import PolicyValueNetwork, NetworkTrainer
from .mcts_search import MCTSSearch, AdaptiveMCTS
from .mcts_env import MCTSFactorEnv, MCTSDataCollector
from ..rlc.env import RLCConfig
from ..rlc.utils import tokens_to_infix
from ..pool import FactorPool, AutoSaveFactorCallback


class AlphaZeroTrainer:
    """AlphaZero 스타일 팩터 발견 트레이너"""
    
    def __init__(
        self,
        env: MCTSFactorEnv,
        network: PolicyValueNetwork,
        factor_pool: Optional[FactorPool] = None,
        # MCTS 설정
        mcts_simulations: int = 800,
        c_puct: float = 1.0,
        # 학습 설정
        episodes_per_iteration: int = 100,
        training_batch_size: int = 512,
        training_epochs: int = 10,
        # 성능 평가
        evaluation_episodes: int = 50,
        evaluation_interval: int = 5,
        # 저장 설정
        save_interval: int = 10,
        checkpoint_dir: str = "mcts_checkpoints"
    ):
        """
        Args:
            env: MCTS Factor Environment
            network: Policy-Value Network
            factor_pool: Factor Pool for storing discovered factors
            mcts_simulations: MCTS 시뮬레이션 횟수
            c_puct: UCB 탐색 상수
            episodes_per_iteration: 반복당 에피소드 수
            training_batch_size: 학습 배치 크기
            training_epochs: 신경망 학습 에포크
            evaluation_episodes: 평가 에피소드 수
            evaluation_interval: 평가 주기
            save_interval: 저장 주기
            checkpoint_dir: 체크포인트 디렉토리
        """
        self.env = env
        self.network = network
        self.factor_pool = factor_pool
        
        # MCTS 탐색기
        self.mcts = MCTSSearch(
            network=network,
            c_puct=c_puct,
            num_simulations=mcts_simulations,
            evaluation_fn=self._evaluate_tokens
        )
        
        # 네트워크 트레이너
        self.trainer = NetworkTrainer(network)
        
        # 데이터 수집기
        self.data_collector = MCTSDataCollector(max_samples=50000)
        
        # 학습 설정
        self.episodes_per_iteration = episodes_per_iteration
        self.training_batch_size = training_batch_size
        self.training_epochs = training_epochs
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_interval = evaluation_interval
        self.save_interval = save_interval
        
        # 체크포인트 관리
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 학습 통계
        self.iteration = 0
        self.total_episodes = 0
        self.best_performance = -float('inf')
        self.training_history = []
        
        # 성능 추적
        self.discovered_factors = []
        self.performance_history = []
        
        print(f"✅ AlphaZero Trainer 초기화 완료")
        print(f"  MCTS 시뮬레이션: {mcts_simulations}")
        print(f"  반복당 에피소드: {episodes_per_iteration}")
        print(f"  체크포인트 디렉토리: {checkpoint_dir}")
    
    def train(self, num_iterations: int):
        """
        AlphaZero 학습 실행
        
        Args:
            num_iterations: 학습 반복 횟수
        """
        print(f"🚀 AlphaZero 학습 시작 ({num_iterations} 반복)")
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            iteration_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"반복 {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # 1. 자기 대국 (Self-Play)
            print("🎯 자기 대국 중...")
            episode_data = self._self_play()
            
            # 2. 데이터 수집
            print("📊 학습 데이터 수집 중...")
            self._collect_training_data(episode_data)
            
            # 3. 신경망 학습
            print("🧠 신경망 학습 중...")
            training_stats = self._train_network()
            
            # 4. 성능 평가
            if (iteration + 1) % self.evaluation_interval == 0:
                print("📈 성능 평가 중...")
                eval_stats = self._evaluate_performance()
                print(f"평가 결과: {eval_stats}")
            
            # 5. 체크포인트 저장
            if (iteration + 1) % self.save_interval == 0:
                self._save_checkpoint()
            
            # 통계 업데이트
            iteration_time = time.time() - iteration_start
            self.training_history.append({
                'iteration': iteration + 1,
                'time': iteration_time,
                'training_stats': training_stats,
                'data_samples': self.data_collector.get_stats()
            })
            
            print(f"⏱️ 반복 완료 시간: {iteration_time:.1f}초")
            print(f"📊 수집된 데이터: {self.data_collector.get_stats()}")
        
        print(f"\n🎉 AlphaZero 학습 완료!")
        self._save_final_results()
    
    def _self_play(self) -> List[Dict]:
        """자기 대국 에피소드 실행"""
        
        episode_data = []
        successful_episodes = 0
        
        for episode in range(self.episodes_per_iteration):
            try:
                # 초기 상태에서 MCTS 탐색
                action_probs, root_node = self.mcts.search(
                    root_state=[], 
                    root_need=1
                )
                
                # 액션 선택 (온도 조절)
                temperature = max(0.1, 1.0 - self.iteration * 0.01)  # 점진적 감소
                action = np.random.choice(25, p=action_probs)
                
                # 프로그램 생성 (시뮬레이션)
                program = self._generate_program_from_mcts(root_node, temperature)
                
                if program:
                    # 프로그램 평가
                    evaluation = self.env.evaluate_program(program)
                    
                    if evaluation['success']:
                        successful_episodes += 1
                        
                        # Factor Pool에 저장
                        if self.factor_pool and evaluation['reward'] > 0:
                            self._save_to_factor_pool(program, evaluation)
                    
                    # 에피소드 데이터 저장
                    episode_data.append({
                        'program': program,
                        'evaluation': evaluation,
                        'action_probs': action_probs,
                        'mcts_stats': root_node.tree_stats(max_depth=2)
                    })
                
                if (episode + 1) % 20 == 0:
                    print(f"  에피소드 진행: {episode+1}/{self.episodes_per_iteration} "
                          f"(성공: {successful_episodes})")
                    
            except Exception as e:
                print(f"⚠️ 에피소드 {episode} 오류: {e}")
                continue
        
        self.total_episodes += len(episode_data)
        success_rate = successful_episodes / len(episode_data) if episode_data else 0
        
        print(f"✅ 자기 대국 완료: {len(episode_data)}개 에피소드, "
              f"성공률: {success_rate:.1%}")
        
        return episode_data
    
    def _generate_program_from_mcts(self, root_node, temperature: float) -> Optional[List[int]]:
        """MCTS 트리에서 완전한 프로그램 생성"""
        
        program = []
        node = root_node
        max_steps = 21  # 최대 길이 제한
        
        for step in range(max_steps):
            if not node.children:
                break
            
            # 온도 조절된 액션 선택
            action_probs = node.get_action_probs(temperature)
            valid_actions = list(node.children.keys())
            
            if not valid_actions:
                break
            
            # 유효한 액션들에 대해서만 확률 재정규화
            valid_probs = np.array([action_probs[a] for a in valid_actions])
            if valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
            else:
                action = np.random.choice(valid_actions)
            
            program.append(action)
            node = node.children[action]
            
            # 터미널 체크
            if node.is_terminal:
                break
        
        return program if program else None
    
    def _collect_training_data(self, episode_data: List[Dict]):
        """에피소드 데이터를 학습 데이터로 변환"""
        
        for episode in episode_data:
            if episode['evaluation']['success']:
                program = episode['program']
                reward = episode['evaluation']['reward']
                
                # 프로그램 생성 과정의 각 상태에서 학습 데이터 생성
                states, policies, values = self.env.generate_training_data(
                    [program], [reward]
                )
                
                if len(states) > 0:
                    self.data_collector.add_episode(states, policies, values, reward)
    
    def _train_network(self) -> Dict:
        """신경망 학습"""
        
        if self.data_collector.get_stats()['total_samples'] < self.training_batch_size:
            return {'message': 'insufficient_data'}
        
        total_losses = []
        
        for epoch in range(self.training_epochs):
            # 배치 샘플링
            states, policies, values = self.data_collector.get_batch(self.training_batch_size)
            
            if states is None:
                break
            
            # 학습 스텝
            loss_dict = self.trainer.train_step(states, policies, values)
            total_losses.append(loss_dict['total_loss'])
            
            if epoch % 5 == 0:
                print(f"    에포크 {epoch+1}/{self.training_epochs}: "
                      f"손실 = {loss_dict['total_loss']:.4f}")
        
        avg_loss = np.mean(total_losses) if total_losses else 0
        
        return {
            'epochs': len(total_losses),
            'avg_loss': avg_loss,
            'final_loss': total_losses[-1] if total_losses else 0
        }
    
    def _evaluate_performance(self) -> Dict:
        """현재 네트워크 성능 평가"""
        
        evaluation_rewards = []
        successful_evaluations = 0
        
        for _ in range(self.evaluation_episodes):
            try:
                # 평가용 MCTS (더 적은 시뮬레이션)
                eval_mcts = MCTSSearch(
                    network=self.network,
                    num_simulations=200,  # 빠른 평가
                    evaluation_fn=self._evaluate_tokens
                )
                
                action_probs, root_node = eval_mcts.search([], 1)
                program = self._generate_program_from_mcts(root_node, temperature=0.1)
                
                if program:
                    evaluation = self.env.evaluate_program(program)
                    
                    if evaluation['success']:
                        successful_evaluations += 1
                        evaluation_rewards.append(evaluation['reward'])
                    else:
                        evaluation_rewards.append(-1.0)
            
            except Exception as e:
                evaluation_rewards.append(-1.0)
        
        avg_reward = np.mean(evaluation_rewards) if evaluation_rewards else -1.0
        success_rate = successful_evaluations / self.evaluation_episodes
        
        # 최고 성능 업데이트
        if avg_reward > self.best_performance:
            self.best_performance = avg_reward
            self._save_best_model()
        
        eval_stats = {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'episodes': len(evaluation_rewards),
            'best_performance': self.best_performance
        }
        
        self.performance_history.append(eval_stats)
        
        return eval_stats
    
    def _evaluate_tokens(self, tokens: List[int]) -> float:
        """토큰 시퀀스 평가 (MCTS용)"""
        try:
            result = self.env.evaluate_program(tokens)
            return result['reward'] if result['success'] else -1.0
        except:
            return -1.0
    
    def _save_to_factor_pool(self, tokens: List[int], evaluation: Dict):
        """우수한 프로그램을 Factor Pool에 저장"""
        
        if not self.factor_pool:
            return
        
        try:
            # 가상의 시계열 데이터 생성 (실제로는 evaluation에서 가져와야 함)
            reward = evaluation['reward']
            info = evaluation.get('info', {})
            
            dates = pd.date_range('2024-01-01', periods=1000, freq='H')
            pnl = pd.Series(np.random.normal(reward/1000, 0.01, 1000), index=dates)
            equity = (1 + pnl).cumprod()
            signal = pd.Series(np.random.normal(0, 1, 1000), index=dates)
            
            reward_info = {
                'total_reward': reward,
                'components': info.get('reward_components', {}),
                'future_leak': False,
                'validation': {'mcts_evaluation': True}
            }
            
            factor_id = self.factor_pool.add_factor(
                tokens=tokens,
                formula=evaluation.get('formula', f"MCTS_Program_{len(tokens)}"),
                pnl=pnl,
                equity=equity,
                signal=signal,
                reward_info=reward_info,
                model_version=f"MCTS_v{self.iteration}",
                training_episode=self.total_episodes
            )
            
            self.discovered_factors.append({
                'factor_id': factor_id,
                'tokens': tokens,
                'reward': reward,
                'iteration': self.iteration
            })
            
        except Exception as e:
            print(f"⚠️ Factor Pool 저장 실패: {e}")
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration + 1}.pt"
        self.trainer.save_model(str(checkpoint_path))
        
        # 메타데이터 저장
        metadata = {
            'iteration': self.iteration + 1,
            'total_episodes': self.total_episodes,
            'best_performance': self.best_performance,
            'training_history': self.training_history,
            'performance_history': self.performance_history,
            'discovered_factors': len(self.discovered_factors)
        }
        
        metadata_path = self.checkpoint_dir / f"metadata_iter_{self.iteration + 1}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_best_model(self):
        """최고 성능 모델 저장"""
        best_path = self.checkpoint_dir / "best_model.pt"
        self.trainer.save_model(str(best_path))
        print(f"🏆 새로운 최고 성능 모델 저장: {self.best_performance:.4f}")
    
    def _save_final_results(self):
        """최종 결과 저장"""
        
        # 최종 모델 저장
        final_path = self.checkpoint_dir / "final_model.pt"
        self.trainer.save_model(str(final_path))
        
        # 발견된 팩터들 저장
        if self.discovered_factors:
            factors_path = self.checkpoint_dir / "discovered_factors.json"
            with open(factors_path, 'w') as f:
                json.dump(self.discovered_factors, f, indent=2)
        
        # 학습 통계 저장
        stats_path = self.checkpoint_dir / "training_stats.json"
        final_stats = {
            'total_iterations': self.iteration + 1,
            'total_episodes': self.total_episodes,
            'best_performance': self.best_performance,
            'final_performance': self.performance_history[-1] if self.performance_history else None,
            'discovered_factors_count': len(self.discovered_factors),
            'training_history': self.training_history,
            'performance_history': self.performance_history
        }
        
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"💾 최종 결과 저장 완료: {self.checkpoint_dir}")
        print(f"📊 총 발견된 팩터: {len(self.discovered_factors)}개")
        print(f"🏆 최고 성능: {self.best_performance:.4f}")


# 테스트 및 시연
if __name__ == "__main__":
    print("🤖 AlphaZero Trainer 테스트")
    
    # 더미 데이터와 환경 설정
    from ..data import ParquetCache
    from ..pool import FactorPool
    
    # 더미 데이터
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # 환경과 네트워크 생성
    config = RLCConfig()
    env = MCTSFactorEnv(df, config)
    network = PolicyValueNetwork()
    factor_pool = FactorPool("test_mcts_pool")
    
    # 트레이너 생성 (테스트용 간소화 설정)
    trainer = AlphaZeroTrainer(
        env=env,
        network=network,
        factor_pool=factor_pool,
        episodes_per_iteration=5,  # 테스트용 적은 수
        mcts_simulations=50,       # 테스트용 적은 수
        training_epochs=2,         # 테스트용 적은 수
        checkpoint_dir="test_mcts_checkpoints"
    )
    
    # 짧은 학습 테스트
    print("짧은 학습 테스트 실행...")
    trainer.train(num_iterations=2)
    
    print("✅ 테스트 완료!")
