"""
SAC (Soft Actor-Critic) Agent
State-of-the-art algorithm cho continuous action spaces.

Reference: Haarnoja et al., 2018 - "Soft Actor-Critic: Off-Policy Maximum Entropy 
Deep Reinforcement Learning with a Stochastic Actor"

Theo phong cách Maxim Lapan's Deep RL Hands-On
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import PolicyStrategy
from ..buffer import ReplayBuffer


@dataclass
class SACConfig:
    """Configuration cho SAC agent"""
    # Network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    
    # SAC hyperparameters
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Soft update coefficient
    alpha: float = 0.2            # Initial entropy coefficient
    auto_alpha: bool = True       # Auto-tune alpha
    target_entropy: float = None  # Target entropy (None = -action_dim)
    
    # Training
    batch_size: int = 256
    buffer_size: int = 100000
    warmup_steps: int = 1000      # Random actions before training
    update_every: int = 1         # Update frequency
    num_updates: int = 1          # Updates per step
    
    # Action
    action_scale: float = 1.0
    action_bias: float = 0.0


if TORCH_AVAILABLE:
    
    class MLP(nn.Module):
        """Multi-layer perceptron"""
        def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            hidden_dim: int = 256,
            num_hidden: int = 2,
            activation: nn.Module = nn.ReLU
        ):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for _ in range(num_hidden):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    activation()
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    
    class GaussianActor(nn.Module):
        """
        Gaussian policy network.
        Outputs mean and log_std for action distribution.
        """
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            num_hidden: int = 2,
            action_scale: float = 1.0,
            action_bias: float = 0.0
        ):
            super().__init__()
            
            self.action_dim = action_dim
            self.action_scale = action_scale
            self.action_bias = action_bias
            
            # Shared layers
            layers = []
            prev_dim = state_dim
            for _ in range(num_hidden):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = hidden_dim
            self.shared = nn.Sequential(*layers)
            
            # Mean and log_std heads
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Returns mean and log_std"""
            features = self.shared(state)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mean, log_std
        
        def sample(
            self, 
            state: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Sample action and compute log probability.
            Uses reparameterization trick.
            """
            mean, log_std = self.forward(state)
            std = log_std.exp()
            
            if deterministic:
                action = mean
            else:
                normal = Normal(mean, std)
                # Reparameterization trick
                action = normal.rsample()
            
            # Apply tanh squashing
            action_tanh = torch.tanh(action)
            
            # Scale to action space
            scaled_action = action_tanh * self.action_scale + self.action_bias
            
            # Log probability with tanh correction
            log_prob = normal.log_prob(action)
            # Enforcing Action Bounds (Appendix C of SAC paper)
            log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return scaled_action, log_prob
        
        def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Get action for numpy state (inference)"""
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action, _ = self.sample(state_t, deterministic)
                return action.cpu().numpy().squeeze(0)
    
    
    class TwinQCritic(nn.Module):
        """
        Twin Q-networks (Q1, Q2) for SAC.
        Reduces overestimation bias.
        """
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            num_hidden: int = 2
        ):
            super().__init__()
            
            input_dim = state_dim + action_dim
            
            self.q1 = MLP(input_dim, 1, hidden_dim, num_hidden)
            self.q2 = MLP(input_dim, 1, hidden_dim, num_hidden)
        
        def forward(
            self, 
            state: torch.Tensor, 
            action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Returns Q1(s,a) and Q2(s,a)"""
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)
        
        def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Returns only Q1 (for policy gradient)"""
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa)


class SACAgent(PolicyStrategy):
    """
    Soft Actor-Critic agent.
    
    Features:
    - Maximum entropy RL
    - Twin Q-networks
    - Automatic temperature tuning
    - Soft target updates
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig = None,
        device: str = 'cpu'
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SACAgent")
        
        self.config = config or SACConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Actor
        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            num_hidden=self.config.num_hidden_layers,
            action_scale=self.config.action_scale,
            action_bias=self.config.action_bias
        ).to(self.device)
        
        # Critics
        self.critic = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            num_hidden=self.config.num_hidden_layers
        ).to(self.device)
        
        self.critic_target = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            num_hidden=self.config.num_hidden_layers
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy coefficient (alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.config.alpha
        self.target_entropy = (
            self.config.target_entropy 
            if self.config.target_entropy is not None 
            else -action_dim
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.critic_lr
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], 
            lr=self.config.alpha_lr
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=self.config.buffer_size)
        
        # Training state
        self.total_steps = 0
        self.updates = 0
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action given state"""
        # Warmup: random actions
        if self.total_steps < self.config.warmup_steps and not deterministic:
            return np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)
        
        return self.actor.get_action(state, deterministic)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer"""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update actor and critics.
        Returns training metrics.
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return {}
        
        if self.total_steps < self.config.warmup_steps:
            return {}
        
        metrics = {}
        
        for _ in range(self.config.num_updates):
            # Sample batch
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.config.batch_size
            )
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # Update critics
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_value = rewards + (1 - dones) * self.config.gamma * target_q
            
            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update actor
            new_actions, log_probs = self.actor.sample(states)
            q1_new = self.critic.q1_forward(states, new_actions)
            actor_loss = (self.alpha * log_probs - q1_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha (temperature)
            if self.config.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
                metrics['alpha_loss'] = alpha_loss.item()
            
            # Soft update target networks
            self._soft_update()
            
            self.updates += 1
            
            metrics.update({
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha,
                'q_value': current_q1.mean().item()
            })
        
        return metrics
    
    def _soft_update(self) -> None:
        """Soft update target networks"""
        for param, target_param in zip(
            self.critic.parameters(), 
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save(self, filepath: str) -> None:
        """Save agent to file"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'updates': self.updates
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)
        
        self.alpha = self.log_alpha.exp().item()
    
    def reset(self) -> None:
        """Reset agent state (không reset weights)"""
        pass
    
    def train_mode(self) -> None:
        """Set networks to train mode"""
        self.actor.train()
        self.critic.train()
    
    def eval_mode(self) -> None:
        """Set networks to eval mode"""
        self.actor.eval()
        self.critic.eval()
