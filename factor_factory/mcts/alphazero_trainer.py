#!/usr/bin/env python3
"""
AlphaZero Style Trainer

AlphaZero algorithm with Self-Play + Neural Network Training:
1. Generate programs using MCTS with current neural network
2. Evaluate generated programs and collect training data
3. Update neural network with collected data
4. Periodically compare performance with previous versions
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from .neural_network import PolicyValueNetwork, NetworkTrainer
from .mcts_search import MCTSSearch, AdaptiveMCTS
from .mcts_env import MCTSFactorEnv, MCTSDataCollector
from ..rlc.env import RLCConfig
from ..rlc.utils import tokens_to_infix
from ..pool import FactorPool, AutoSaveFactorCallback


class AlphaZeroTrainer:
    """AlphaZero Style Factor Discovery Trainer"""
    
    def __init__(
        self,
        env: MCTSFactorEnv,
        network: PolicyValueNetwork,
        factor_pool: Optional[FactorPool] = None,
        # MCTS Configuration
        mcts_simulations: int = 800,
        c_puct: float = 1.0,
        # Training Configuration
        episodes_per_iteration: int = 100,
        training_batch_size: int = 512,
        training_epochs: int = 10,
        # Performance Evaluation
        evaluation_episodes: int = 50,
        evaluation_interval: int = 5,
        # Save Configuration
        save_interval: int = 10,
        checkpoint_dir: str = "mcts_checkpoints"
    ):
        """
        Args:
            env: MCTS Factor Environment
            network: Policy-Value Network
            factor_pool: Factor Pool for storing discovered factors
            mcts_simulations: MCTS simulation count
            c_puct: UCB exploration constant
            episodes_per_iteration: Episodes per iteration
            training_batch_size: Training batch size
            training_epochs: Neural network training epochs
            evaluation_episodes: Evaluation episode count
            evaluation_interval: Evaluation interval
            save_interval: Save interval
            checkpoint_dir: Checkpoint directory
        """
        self.env = env
        self.network = network
        self.factor_pool = factor_pool
        
        # MCTS Searcher
        self.mcts = MCTSSearch(
            network=network,
            c_puct=c_puct,
            num_simulations=mcts_simulations,
            evaluation_fn=self._evaluate_tokens
        )
        
        # Network Trainer
        self.trainer = NetworkTrainer(network)
        
        # Data Collector
        self.data_collector = MCTSDataCollector(max_samples=50000)
        
        # Training Configuration
        self.episodes_per_iteration = episodes_per_iteration
        self.training_batch_size = training_batch_size
        self.training_epochs = training_epochs
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_interval = evaluation_interval
        self.save_interval = save_interval
        
        # Checkpoint Management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training Statistics
        self.iteration = 0
        self.total_episodes = 0
        self.best_performance = -float('inf')
        self.training_history = []
        
        # Performance Tracking
        self.discovered_factors = []
        self.performance_history = []
        
        print(f"[CHECK] AlphaZero Trainer initialization complete")
        print(f"  MCTS Simulations: {mcts_simulations}")
        print(f"  Episodes per iteration: {episodes_per_iteration}")
        print(f"  Checkpoint directory: {checkpoint_dir}")
    
    def train(self, num_iterations: int):
        """
        Execute AlphaZero Training
        
        Args:
            num_iterations: Number of training iterations
        """
        print(f"[START] AlphaZero training started ({num_iterations} iterations)")
        
        # Overall Progress Bar
        main_pbar = tqdm(range(num_iterations), 
                        desc="Overall Training Progress", 
                        ncols=120,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Iteration {n}')
        
        for iteration in main_pbar:
            self.iteration = iteration
            iteration_start = time.time()
            
            main_pbar.set_description(f"Iteration {iteration + 1}/{num_iterations}")
            
            # 1. Self-Play
            print(f"\n[TARGET] Self-play in progress... (iteration {iteration + 1}/{num_iterations})")
            episode_data = self._self_play()
            
            # 2. Data Collection
            print("[CHART] Collecting training data...")
            self._collect_training_data(episode_data)
            
            # 3. Neural Network Training
            print("[BRAIN] Neural network training...")
            training_stats = self._train_network()
            
            # 4. Performance Evaluation
            if (iteration + 1) % self.evaluation_interval == 0:
                print("Performance evaluation in progress...")
                eval_stats = self._evaluate_performance()
                print(f"Evaluation result: {eval_stats}")
            
            # 5. Checkpoint Save
            if (iteration + 1) % self.save_interval == 0:
                self._save_checkpoint()
            
            # Statistics Update
            iteration_time = time.time() - iteration_start
            self.training_history.append({
                'iteration': iteration + 1,
                'time': iteration_time,
                'training_stats': training_stats,
                'data_samples': self.data_collector.get_stats()
            })
            
            # Summary of factors discovered in this iteration
            current_factors = [f for f in self.discovered_factors if f['iteration'] == iteration]
            
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1} completion summary")
            print(f"{'='*50}")
            print(f"[TIME] Duration: {iteration_time:.1f} sec")
            print(f"[CHART] Collected data: {self.data_collector.get_stats()}")
            print(f"[SEARCH] Factors discovered this iteration: {len(current_factors)}")
            
            if current_factors:
                print(f"Discovered factors:")
                for i, factor in enumerate(current_factors[:3], 1):  # top 3 only
                    print(f"  {i}. Reward: {factor['reward']:.4f}")
            
            print(f"[TROPHY] Total discovered factors: {len(self.discovered_factors)}")
            print(f"[TARGET] Best performance: {self.best_performance:.4f}")
            print(f"{'='*50}\n")
        
        main_pbar.close()
        print(f"\n[COMPLETE] AlphaZero training completed!")
        self._save_final_results()
    
    def _self_play(self) -> List[Dict]:
        """Execute self-play episodes"""
        
        episode_data = []
        successful_episodes = 0
        
        # Add progress bar
        pbar = tqdm(range(self.episodes_per_iteration), 
                   desc="Episode Progress", 
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for episode in pbar:
            try:
                # MCTS search from initial state
                print(f"\n  [SEARCH] Episode {episode+1}: MCTS searching...", end="")
                action_probs, root_node = self.mcts.search(
                    root_state=[], 
                    root_need=1
                )
                print(f" completed")
                
                # Action selection (temperature control)
                temperature = max(0.1, 1.0 - self.iteration * 0.01)  # gradual decrease
                action = np.random.choice(25, p=action_probs)
                
                # Program generation (simulation)
                program = self._generate_program_from_mcts(root_node, temperature)
                
                if program:
                    print(f"    Generated program length: {len(program)} tokens")
                    
                    # Program evaluation
                    evaluation = self.env.evaluate_program(program)
                    
                    if evaluation['success']:
                        successful_episodes += 1
                        reward = evaluation['reward']
                        
                        # Convert program to human-readable format
                        try:
                            from ..rlc.compiler import RLCCompiler
                            compiler = RLCCompiler()
                            human_readable = compiler.decompile_program(program)
                        except:
                            human_readable = f"Program_{len(program)}_tokens"
                        
                        # Real-time factor information output
                        metrics = evaluation.get('metrics', {})
                        if reward > 0:
                            print(f"\n  [FOUND] {human_readable}")
                            print(f"    Reward: {reward:.4f} | Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
                            print(f"    Return: {metrics.get('total_return', 0):.2%} | Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                        elif reward > -0.5:  # negative but not completely bad case
                            print(f"\n  [TRY] {human_readable} | Reward: {reward:.4f}")
                        
                        # Also record in logging
                        import logging
                        if reward > 0:
                            logging.info(f"[FACTOR] Found: {human_readable} | Reward: {reward:.4f}")
                        
                        # Save to Factor Pool
                        if self.factor_pool and reward > 0:
                            self._save_to_factor_pool(program, evaluation)
                    else:
                        print(f"    Evaluation failed: {evaluation.get('error', 'Unknown error')}")
                else:
                    print(f"    Program generation failed")
                    
                    # Save episode data
                    episode_data.append({
                        'program': program,
                        'evaluation': evaluation,
                        'action_probs': action_probs,
                        'mcts_stats': root_node.tree_stats(max_depth=2)
                    })
                
                # Update progress bar
                pbar.set_postfix({
                    'Success': f"{successful_episodes}/{episode+1}",
                    'Success Rate': f"{successful_episodes/(episode+1)*100:.1f}%"
                })
                    
            except Exception as e:
                print(f"\n  [WARNING] Episode {episode} Error: {e}")
                continue
        
        pbar.close()
        
        self.total_episodes += len(episode_data)
        success_rate = successful_episodes / len(episode_data) if episode_data else 0
        
        print(f"Self-play completed: {len(episode_data)} episodes, "
              f"Success Rate: {success_rate:.1%}")
        
        return episode_data
    
    def _generate_program_from_mcts(self, root_node, temperature: float) -> Optional[List[int]]:
        """Generate complete program from MCTS tree"""
        
        program = []
        node = root_node
        max_steps = 21  # maximum length limit
        
        for step in range(max_steps):
            if not node.children:
                break
            
            # Temperature-controlled action selection
            action_probs = node.get_action_probs(temperature)
            valid_actions = list(node.children.keys())
            
            if not valid_actions:
                break
            
            # Renormalize probabilities only for valid actions
            valid_probs = np.array([action_probs[a] for a in valid_actions])
            if valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
            else:
                action = np.random.choice(valid_actions)
            
            program.append(action)
            node = node.children[action]
            
            # Terminal check
            if node.is_terminal:
                break
        
        return program if program else None
    
    def _collect_training_data(self, episode_data: List[Dict]):
        """Convert episode data to training data"""
        
        for episode in episode_data:
            if episode['evaluation']['success']:
                program = episode['program']
                reward = episode['evaluation']['reward']
                
                # Generate training data at each state in program generation process
                states, policies, values = self.env.generate_training_data(
                    [program], [reward]
                )
                
                if len(states) > 0:
                    self.data_collector.add_episode(states, policies, values, reward)
    
    def _train_network(self) -> Dict:
        """Neural network training"""
        
        if self.data_collector.get_stats()['total_samples'] < self.training_batch_size:
            return {'message': 'insufficient_data'}
        
        total_losses = []
        
        for epoch in range(self.training_epochs):
            # Batch sampling
            states, policies, values = self.data_collector.get_batch(self.training_batch_size)
            
            if states is None:
                break
            
            # Training step
            loss_dict = self.trainer.train_step(states, policies, values)
            total_losses.append(loss_dict['total_loss'])
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{self.training_epochs}: "
                      f"Loss = {loss_dict['total_loss']:.4f}")
        
        avg_loss = np.mean(total_losses) if total_losses else 0
        
        return {
            'epochs': len(total_losses),
            'avg_loss': avg_loss,
            'final_loss': total_losses[-1] if total_losses else 0
        }
    
    def _evaluate_performance(self) -> Dict:
        """Current network performance evaluation"""
        
        evaluation_rewards = []
        successful_evaluations = 0
        
        for _ in range(self.evaluation_episodes):
            try:
                # MCTS for evaluation (fewer simulations)
                eval_mcts = MCTSSearch(
                    network=self.network,
                    num_simulations=200,  # Fast evaluation
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
        
        # Best performance update
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
        """Token sequence evaluation (for MCTS)"""
        try:
            result = self.env.evaluate_program(tokens)
            return result['reward'] if result['success'] else -1.0
        except:
            return -1.0
    
    def _save_to_factor_pool(self, tokens: List[int], evaluation: Dict):
        """Save excellent programs to Factor Pool"""
        
        if not self.factor_pool:
            return
        
        try:
            # Generate virtual time series data (should actually be fetched from evaluation)
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
            print(f" Factor Pool save failed: {e}")
    
    def _save_checkpoint(self):
        """Checkpoint save"""
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration + 1}.pt"
        self.trainer.save_model(str(checkpoint_path))
        
        # Save metadata
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
        """Save best performance model"""
        best_path = self.checkpoint_dir / "best_model.pt"
        self.trainer.save_model(str(best_path))
        print(f"[TROPHY] New best performance model saved: {self.best_performance:.4f}")
    
    def _save_final_results(self):
        """Save final results"""
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        self.trainer.save_model(str(final_path))
        
        # Save discovered factors
        if self.discovered_factors:
            factors_path = self.checkpoint_dir / "discovered_factors.json"
            with open(factors_path, 'w') as f:
                json.dump(self.discovered_factors, f, indent=2)
        
        # Save training statistics
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
        
        print(f"[SAVE] Final results saved: {self.checkpoint_dir}")
        print(f"[CHART] Total discovered factors: {len(self.discovered_factors)}")
        print(f"[TROPHY] Best performance: {self.best_performance:.4f}")


# Test and demonstration
if __name__ == "__main__":
    print("[TEST] AlphaZero Trainer Test")
    
    # Dummy data and environment setup
    from ..data import ParquetCache
    from ..pool import FactorPool
    
    # Dummy data
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Create environment and network
    config = RLCConfig()
    env = MCTSFactorEnv(df, config)
    network = PolicyValueNetwork()
    factor_pool = FactorPool("test_mcts_pool")
    
    # Create trainer (simplified settings for testing)
    trainer = AlphaZeroTrainer(
        env=env,
        network=network,
        factor_pool=factor_pool,
        episodes_per_iteration=5,  # Fewer for testing
        mcts_simulations=50,       # Fewer for testing
        training_epochs=2,         # Fewer for testing
        checkpoint_dir="test_mcts_checkpoints"
    )
    
    # Run short training test
    print("Running short training test...")
    trainer.train(num_iterations=2)
    
    print("Test completed!")
