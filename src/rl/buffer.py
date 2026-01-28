"""
Replay Buffer - Experience storage cho off-policy RL
Theo phong cách Maxim Lapan's Deep RL Hands-On
"""
from __future__ import annotations
from typing import Tuple, List, Optional, NamedTuple
from collections import deque
import numpy as np
import random


class Experience(NamedTuple):
    """Single experience tuple"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Standard Replay Buffer cho off-policy algorithms (DQN, DDPG, SAC, TD3).
    
    Features:
    - Fixed capacity với FIFO eviction
    - Uniform random sampling
    - Support numpy batch sampling
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add experience to buffer"""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch.
        
        Returns:
            (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.float32)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    Samples experiences with probability proportional to their TD error.
    
    Reference: Schaul et al., 2015 - "Prioritized Experience Replay"
    """
    
    def __init__(
        self, 
        capacity: int = 100000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer: List[Optional[Experience]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        self.max_priority = 1.0
        self.min_priority = 1e-6
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience with max priority"""
        exp = Experience(state, action, reward, next_state, done)
        
        self.buffer[self.position] = exp
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Gather experiences
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.float32)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class EpisodeBuffer:
    """
    Buffer để store toàn bộ episode (cho on-policy hoặc analysis).
    """
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self._current_episode: List[Experience] = []
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add step to current episode"""
        exp = Experience(state, action, reward, next_state, done)
        self._current_episode.append(exp)
        
        if done:
            self.episodes.append(self._current_episode.copy())
            self._current_episode.clear()
    
    def get_episode(self, idx: int) -> List[Experience]:
        """Get episode by index"""
        return self.episodes[idx]
    
    def get_last_episode(self) -> List[Experience]:
        """Get most recent completed episode"""
        if self.episodes:
            return self.episodes[-1]
        return []
    
    def sample_episodes(self, n: int) -> List[List[Experience]]:
        """Sample n random episodes"""
        n = min(n, len(self.episodes))
        return random.sample(list(self.episodes), n)
    
    def get_episode_returns(self) -> List[float]:
        """Get list of episode returns"""
        returns = []
        for episode in self.episodes:
            ep_return = sum(exp.reward for exp in episode)
            returns.append(ep_return)
        return returns
    
    def get_episode_lengths(self) -> List[int]:
        """Get list of episode lengths"""
        return [len(ep) for ep in self.episodes]
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def clear(self) -> None:
        self.episodes.clear()
        self._current_episode.clear()
