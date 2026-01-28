"""
Base Policy classes - Strategy pattern cho RL agents
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class PolicyStrategy(ABC):
    """
    Abstract base class cho policies.
    Strategy pattern - có thể swap policies dễ dàng.
    """
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Observation vector
            deterministic: If True, return deterministic action (for evaluation)
        
        Returns:
            Action vector
        """
        pass
    
    def reset(self) -> None:
        """Reset policy state (if any)"""
        pass
    
    def update(self, *args, **kwargs) -> dict:
        """
        Update policy parameters.
        Returns dict with training metrics.
        """
        return {}
    
    def save(self, filepath: str) -> None:
        """Save policy to file"""
        pass
    
    def load(self, filepath: str) -> None:
        """Load policy from file"""
        pass


class RandomPolicy(PolicyStrategy):
    """
    Random policy - sample uniform random actions.
    Useful for initial exploration và baseline.
    """
    
    def __init__(
        self, 
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0
    ):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """Return random action"""
        return np.random.uniform(
            self.action_low, 
            self.action_high, 
            size=self.action_dim
        ).astype(np.float32)


class EpsilonGreedyWrapper(PolicyStrategy):
    """
    Epsilon-greedy wrapper cho bất kỳ policy nào.
    Với probability epsilon, chọn random action.
    """
    
    def __init__(
        self,
        policy: PolicyStrategy,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        action_dim: int = 2
    ):
        self.policy = policy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim
        self._random_policy = RandomPolicy(action_dim)
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        if deterministic:
            return self.policy.select_action(state, deterministic=True)
        
        if np.random.random() < self.epsilon:
            return self._random_policy.select_action(state)
        else:
            return self.policy.select_action(state, deterministic=False)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset(self) -> None:
        self.policy.reset()
    
    def update(self, *args, **kwargs) -> dict:
        return self.policy.update(*args, **kwargs)


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    Generates temporally correlated noise.
    
    Used in DDPG for continuous action spaces.
    """
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self) -> None:
        """Reset noise state"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.astype(np.float32)


class GaussianNoise:
    """
    Simple Gaussian noise for exploration.
    Used in TD3.
    """
    
    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.1,
        clip: float = 0.5
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.clip = clip
    
    def sample(self) -> np.ndarray:
        """Sample noise"""
        noise = np.random.randn(self.action_dim) * self.sigma
        return np.clip(noise, -self.clip, self.clip).astype(np.float32)
    
    def reset(self) -> None:
        pass
