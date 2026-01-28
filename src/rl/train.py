"""
Training loop cho RL agents
Theo phong cách Maxim Lapan's Deep RL Hands-On
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import time
from collections import deque

from .agents import PolicyStrategy, SACAgent
from .buffer import ReplayBuffer, EpisodeBuffer


@dataclass
class TrainConfig:
    """Configuration cho training"""
    # Training duration
    max_episodes: int = 1000
    max_steps_per_episode: int = 3000
    max_total_steps: int = 500000
    
    # Evaluation
    eval_frequency: int = 50  # Evaluate every N episodes
    eval_episodes: int = 5
    
    # Logging
    log_frequency: int = 1  # Log every N episodes
    print_frequency: int = 10
    
    # Saving
    save_frequency: int = 100
    save_path: str = 'data/models'
    
    # Early stopping
    target_reward: float = None  # Stop if avg reward > target
    patience: int = 50  # Episodes without improvement
    
    # Misc
    seed: int = None
    render_training: bool = False


@dataclass
class TrainMetrics:
    """Container cho training metrics"""
    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    
    # Running averages
    avg_reward_100: float = 0.0
    avg_length_100: float = 0.0
    success_rate_100: float = 0.0
    
    # Agent metrics
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    alpha: float = 0.0
    q_value: float = 0.0
    
    # Timing
    episode_time: float = 0.0
    fps: float = 0.0


class Trainer:
    """
    Training loop manager.
    Hỗ trợ logging, evaluation, early stopping.
    """
    
    def __init__(
        self,
        env,  # LabyrinthEnv
        agent: PolicyStrategy,
        config: TrainConfig = None
    ):
        self.env = env
        self.agent = agent
        self.config = config or TrainConfig()
        
        # Metrics history
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.successes: deque = deque(maxlen=100)
        self.metrics_history: List[TrainMetrics] = []
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0
        
        # Callbacks
        self._on_episode_end_callbacks: List[Callable] = []
        self._on_step_callbacks: List[Callable] = []
    
    def train(
        self,
        callback: Callable[[TrainMetrics], None] = None
    ) -> List[TrainMetrics]:
        """
        Main training loop.
        
        Args:
            callback: Function called after each episode with metrics
        
        Returns:
            List of training metrics
        """
        if self.config.seed is not None:
            self.env.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        print(f"Starting training for {self.config.max_episodes} episodes...")
        print(f"Max steps per episode: {self.config.max_steps_per_episode}")
        print("-" * 60)
        
        while (self.current_episode < self.config.max_episodes and 
               self.total_steps < self.config.max_total_steps):
            
            # Run episode
            metrics = self._run_episode()
            
            # Update history
            self.episode_rewards.append(metrics.episode_reward)
            self.episode_lengths.append(metrics.episode_length)
            self.successes.append(float(metrics.success))
            self.metrics_history.append(metrics)
            
            # Calculate running averages
            metrics.avg_reward_100 = np.mean(self.episode_rewards)
            metrics.avg_length_100 = np.mean(self.episode_lengths)
            metrics.success_rate_100 = np.mean(self.successes)
            
            # Logging
            if self.current_episode % self.config.print_frequency == 0:
                self._print_progress(metrics)
            
            # Callback
            if callback:
                callback(metrics)
            
            for cb in self._on_episode_end_callbacks:
                cb(metrics)
            
            # Evaluation
            if (self.config.eval_frequency > 0 and 
                self.current_episode % self.config.eval_frequency == 0):
                eval_metrics = self.evaluate(self.config.eval_episodes)
                print(f"  [Eval] Avg reward: {eval_metrics['avg_reward']:.2f}, "
                      f"Success rate: {eval_metrics['success_rate']:.2%}")
            
            # Save checkpoint
            if (self.config.save_frequency > 0 and 
                self.current_episode % self.config.save_frequency == 0):
                self._save_checkpoint()
            
            # Early stopping
            if self._check_early_stopping(metrics):
                print(f"\nEarly stopping at episode {self.current_episode}")
                break
            
            self.current_episode += 1
        
        print("\nTraining completed!")
        print(f"Final avg reward (100 ep): {np.mean(self.episode_rewards):.2f}")
        print(f"Final success rate: {np.mean(self.successes):.2%}")
        
        return self.metrics_history
    
    def _run_episode(self) -> TrainMetrics:
        """Run single episode"""
        start_time = time.time()
        
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        agent_metrics = {}
        
        while not done and episode_steps < self.config.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition (if agent supports it)
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(obs, action, reward, next_obs, done)
            
            # Update agent
            if hasattr(self.agent, 'update'):
                agent_metrics = self.agent.update()
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Step callbacks
            for cb in self._on_step_callbacks:
                cb(obs, action, reward, next_obs, done)
        
        episode_time = time.time() - start_time
        
        return TrainMetrics(
            episode=self.current_episode,
            total_steps=self.total_steps,
            episode_reward=episode_reward,
            episode_length=episode_steps,
            success=info.get('success', False),
            actor_loss=agent_metrics.get('actor_loss', 0.0),
            critic_loss=agent_metrics.get('critic_loss', 0.0),
            alpha=agent_metrics.get('alpha', 0.0),
            q_value=agent_metrics.get('q_value', 0.0),
            episode_time=episode_time,
            fps=episode_steps / episode_time if episode_time > 0 else 0
        )
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate agent với deterministic policy.
        """
        rewards = []
        lengths = []
        successes = []
        
        if hasattr(self.agent, 'eval_mode'):
            self.agent.eval_mode()
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(float(info.get('success', False)))
        
        if hasattr(self.agent, 'train_mode'):
            self.agent.train_mode()
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'success_rate': np.mean(successes),
            'rewards': rewards
        }
    
    def _print_progress(self, metrics: TrainMetrics) -> None:
        """Print training progress"""
        print(f"Episode {metrics.episode:4d} | "
              f"Steps: {metrics.total_steps:7d} | "
              f"Reward: {metrics.episode_reward:8.2f} | "
              f"Avg100: {metrics.avg_reward_100:8.2f} | "
              f"Success: {metrics.success_rate_100:.1%} | "
              f"FPS: {metrics.fps:.0f}")
    
    def _save_checkpoint(self) -> None:
        """Save agent checkpoint"""
        if hasattr(self.agent, 'save'):
            filepath = f"{self.config.save_path}/sac_episode_{self.current_episode}.pt"
            try:
                self.agent.save(filepath)
                print(f"  Saved checkpoint: {filepath}")
            except Exception as e:
                print(f"  Failed to save checkpoint: {e}")
    
    def _check_early_stopping(self, metrics: TrainMetrics) -> bool:
        """Check early stopping conditions"""
        # Target reward reached
        if (self.config.target_reward is not None and 
            metrics.avg_reward_100 >= self.config.target_reward):
            print(f"Target reward {self.config.target_reward} reached!")
            return True
        
        # No improvement
        if metrics.avg_reward_100 > self.best_avg_reward:
            self.best_avg_reward = metrics.avg_reward_100
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        if self.episodes_without_improvement >= self.config.patience:
            return True
        
        return False
    
    def on_episode_end(self, callback: Callable[[TrainMetrics], None]) -> None:
        """Register episode end callback"""
        self._on_episode_end_callbacks.append(callback)
    
    def on_step(self, callback: Callable) -> None:
        """Register step callback"""
        self._on_step_callbacks.append(callback)


def create_sac_trainer(
    env,
    state_dim: int = None,
    action_dim: int = None,
    train_config: TrainConfig = None,
    device: str = 'cpu'
) -> Trainer:
    """
    Factory function tạo SAC trainer.
    """
    from .agents.sac import SACAgent, SACConfig
    
    if state_dim is None:
        state_dim = env.observation_dim
    if action_dim is None:
        action_dim = env.action_dim
    
    sac_config = SACConfig()
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=sac_config,
        device=device
    )
    
    return Trainer(env, agent, train_config)
