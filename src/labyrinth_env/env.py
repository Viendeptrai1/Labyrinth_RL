"""
LabyrinthEnv - Gym-style Environment API
Main interface cho RL training và game play
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import numpy as np

from .core.world import World, BoardConfig
from .core.entity import EntityType
from .core.components import (
    TransformComponent, PhysicsComponent, InventoryComponent, TriggerComponent
)
from .core.events import EventBus, GameEvent
from .core.state_machine import StateMachine, GameState, create_game_state_machine
from .core.commands import CommandQueue, SetTiltCommand, InputMapper
from .level_spec import LevelSpec, LevelLoader, get_builtin_levels


@dataclass
class EnvConfig:
    """Configuration cho environment"""
    # Physics
    dt: float = 1.0 / 60.0  # Timestep (60 FPS)
    max_steps: int = 3000   # Max steps per episode (~50 seconds at 60 FPS)
    
    # Observation
    obs_include_velocity: bool = True
    obs_include_tilt: bool = True
    obs_include_distances: bool = True
    obs_normalize: bool = True
    
    # Action
    action_type: str = 'continuous'  # 'continuous', 'discrete', 'discrete_8dir'
    max_tilt: float = 0.15  # Max tilt angle (radians)
    discrete_tilt_steps: int = 5  # For discrete: -2, -1, 0, 1, 2
    
    # Hybrid movement settings
    steps_per_move: int = 15  # Số physics steps mỗi lần di chuyển
    
    # Rewards
    goal_reward: float = 1000.0
    hole_penalty: float = -100.0
    time_penalty: float = -0.1
    coin_reward_scale: float = 1.0
    distance_reward_scale: float = 0.0  # Dense reward (0 = off)


# Direction mapping cho discrete_8dir
DIRECTION_MAP = {
    0: 'none',
    1: 'up',
    2: 'upright',
    3: 'right',
    4: 'downright',
    5: 'down',
    6: 'downleft',
    7: 'left',
    8: 'upleft'
}


class LabyrinthEnv:
    """
    Labyrinth 3D Environment - Gym-compatible API.
    
    Observation space (continuous):
        - ball_pos (x, z): 2
        - ball_vel (vx, vz): 2 (if include_velocity)
        - board_tilt (pitch, roll): 2 (if include_tilt)
        - goal_direction (dx, dz): 2
        - goal_distance: 1
        - nearest_hole_direction (dx, dz): 2 (if include_distances)
        - nearest_hole_distance: 1 (if include_distances)
        - inventory (num_keys, num_coins, score): 3
        Total: 15 features (default config)
    
    Action space (continuous):
        - target_tilt: (pitch, roll) in [-max_tilt, max_tilt]
    """
    
    def __init__(self, config: EnvConfig = None, levels_dir: str = None):
        self.config = config or EnvConfig()
        
        # Core components
        self.world = World(BoardConfig(max_tilt=self.config.max_tilt))
        self.event_bus = EventBus.get_instance()
        self.state_machine = create_game_state_machine()
        self.command_queue = CommandQueue()
        self.input_mapper = InputMapper(max_tilt=self.config.max_tilt)
        
        # Level management
        self.level_loader = LevelLoader(levels_dir)
        self._register_builtin_levels()
        
        self.current_level: Optional[LevelSpec] = None
        self.current_level_idx: int = 0
        
        # Episode state
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._done: bool = False
        self._info: Dict = {}
        
        # Goal tracking cho dense reward
        self._prev_goal_distance: float = 0.0
        
        # Random state
        self._np_random: Optional[np.random.Generator] = None
        self.seed()
    
    def _register_builtin_levels(self) -> None:
        """Đăng ký built-in levels"""
        for spec in get_builtin_levels():
            self.level_loader.register_level(spec)
        self.level_loader.set_curriculum([s.id for s in get_builtin_levels()])
    
    def seed(self, seed: int = None) -> List[int]:
        """Set random seed"""
        self._np_random = np.random.default_rng(seed)
        return [seed]
    
    # ==================== Gym API ====================
    
    def reset(
        self, 
        level_id: str = None,
        seed: int = None,
        options: Dict = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment và bắt đầu episode mới.
        
        Args:
            level_id: ID của level, None = level hiện tại hoặc level đầu
            seed: Random seed
            options: Additional options
        
        Returns:
            observation, info
        """
        if seed is not None:
            self.seed(seed)
        
        # Determine level
        if level_id is not None:
            self.current_level = self.level_loader.get_level(level_id)
            if self.current_level is None:
                raise ValueError(f"Unknown level: {level_id}")
        elif self.current_level is None:
            curriculum = self.level_loader.get_curriculum()
            if curriculum:
                self.current_level = self.level_loader.get_level(curriculum[0])
            else:
                raise ValueError("No levels available")
        
        # Spawn level
        self.level_loader.spawn_from_spec(self.current_level, self.world)
        
        # Reset state
        self._step_count = 0
        self._episode_reward = 0.0
        self._done = False
        self._info = {'level_id': self.current_level.id}
        
        # Reset tilt
        self.world.set_board_tilt(0.0, 0.0)
        self.input_mapper.reset_tilt()
        
        # Calculate initial goal distance
        self._prev_goal_distance = self._get_goal_distance()
        
        # State machine
        self.state_machine.reset(GameState.LOADING)
        self.state_machine.transition_to(GameState.READY, force=True)
        self.state_machine.transition_to(GameState.PLAYING, force=True)
        
        # Publish event
        self.event_bus.publish(
            GameEvent.EPISODE_STARTED,
            level_id=self.current_level.id
        )
        
        return self._get_observation(), self._info.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Thực hiện 1 step với action.
        
        Args:
            action: [pitch, roll] target tilt trong [-1, 1], 
                    sẽ được scale lên [-max_tilt, max_tilt]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._done:
            return self._get_observation(), 0.0, True, False, self._info.copy()
        
        # Process action
        if self.config.action_type == 'continuous':
            pitch = float(action[0]) * self.config.max_tilt
            roll = float(action[1]) * self.config.max_tilt
        else:
            # Discrete action: 0-4 for pitch, 5-9 for roll (example)
            # Hoặc 0-24 cho 5x5 grid
            pitch_idx = int(action) // self.config.discrete_tilt_steps
            roll_idx = int(action) % self.config.discrete_tilt_steps
            pitch = (pitch_idx - self.config.discrete_tilt_steps // 2) * self.config.max_tilt / 2
            roll = (roll_idx - self.config.discrete_tilt_steps // 2) * self.config.max_tilt / 2
        
        # Apply tilt
        self.world.set_board_tilt(pitch, roll)
        
        # Physics step
        step_info = self.world.step(self.config.dt)
        self._step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(step_info)
        self._episode_reward += reward
        
        # Check termination
        terminated = self._check_terminated(step_info)
        truncated = self._step_count >= self.config.max_steps
        
        self._done = terminated or truncated
        
        # Update info
        self._info.update({
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'triggers': step_info['triggers'],
            'collections': step_info['collections'],
        })
        
        if self._done:
            self._info['final_distance'] = self._get_goal_distance()
            self._info['success'] = any(
                t['type'] == 'goal' for t in step_info['triggers']
            )
            
            # State machine
            if self._info['success']:
                self.state_machine.transition_to(GameState.SUCCESS, force=True)
            else:
                self.state_machine.transition_to(GameState.FAILED, force=True)
            
            self.event_bus.publish(
                GameEvent.EPISODE_ENDED,
                reward=self._episode_reward,
                steps=self._step_count,
                success=self._info['success']
            )
        
        return self._get_observation(), reward, terminated, truncated, self._info.copy()
    
    def render(self, mode: str = 'state') -> Any:
        """
        Render environment.
        
        Args:
            mode: 'state' returns dict, 'human' (not implemented)
        """
        if mode == 'state':
            return self.world.get_state_snapshot()
        return None
    
    def close(self) -> None:
        """Cleanup"""
        self.world.clear()
        self.event_bus.clear_subscribers()
    
    # ==================== Observation ====================
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        obs = []
        
        ball = self.world.ball
        if ball is None:
            return np.zeros(self.observation_dim, dtype=np.float32)
        
        ball_transform = ball.get_component(TransformComponent)
        ball_physics = ball.get_component(PhysicsComponent)
        ball_inventory = ball.get_component(InventoryComponent)
        
        # Ball position (normalized to board size)
        ball_pos = ball_transform.position[[0, 2]]  # x, z
        if self.config.obs_normalize:
            ball_pos = ball_pos / np.array([
                self.world.board_config.width / 2,
                self.world.board_config.height / 2
            ])
        obs.extend(ball_pos)
        
        # Ball velocity
        if self.config.obs_include_velocity:
            vel = ball_physics.velocity[[0, 2]]
            if self.config.obs_normalize:
                vel = vel / 5.0  # Normalize to reasonable range
            obs.extend(vel)
        
        # Board tilt
        if self.config.obs_include_tilt:
            pitch, roll = self.world.board_tilt
            if self.config.obs_normalize:
                pitch = pitch / self.config.max_tilt
                roll = roll / self.config.max_tilt
            obs.extend([pitch, roll])
        
        # Goal direction and distance
        goal_entities = self.world.get_entities_by_type(EntityType.GOAL)
        if goal_entities:
            goal_transform = goal_entities[0].get_component(TransformComponent)
            goal_pos = goal_transform.position[[0, 2]]
            direction = goal_pos - ball_transform.position[[0, 2]]
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance  # Normalize direction
            
            if self.config.obs_normalize:
                distance = distance / (self.world.board_config.width / 2)
            
            obs.extend(direction)
            obs.append(distance)
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        # Nearest hole (if include_distances)
        if self.config.obs_include_distances:
            holes = self.world.get_entities_by_type(EntityType.HOLE)
            if holes:
                min_dist = float('inf')
                nearest_dir = np.zeros(2)
                ball_pos_2d = ball_transform.position[[0, 2]]
                
                for hole in holes:
                    if not hole.active:
                        continue
                    hole_transform = hole.get_component(TransformComponent)
                    hole_pos = hole_transform.position[[0, 2]]
                    dist = np.linalg.norm(hole_pos - ball_pos_2d)
                    if dist < min_dist:
                        min_dist = dist
                        if dist > 0:
                            nearest_dir = (hole_pos - ball_pos_2d) / dist
                
                if self.config.obs_normalize:
                    min_dist = min_dist / (self.world.board_config.width / 2)
                
                obs.extend(nearest_dir)
                obs.append(min(min_dist, 2.0))  # Clip max distance
            else:
                obs.extend([0.0, 0.0, 2.0])  # No holes
        
        # Inventory
        if ball_inventory:
            obs.append(len(ball_inventory.keys))
            obs.append(ball_inventory.coins_collected / 10.0)  # Normalize
            obs.append(ball_inventory.total_score / 1000.0)  # Normalize
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def observation_dim(self) -> int:
        """Dimension của observation vector"""
        dim = 2  # ball pos
        if self.config.obs_include_velocity:
            dim += 2
        if self.config.obs_include_tilt:
            dim += 2
        dim += 3  # goal direction + distance
        if self.config.obs_include_distances:
            dim += 3  # nearest hole
        dim += 3  # inventory
        return dim
    
    @property
    def action_dim(self) -> int:
        """Dimension của action"""
        if self.config.action_type == 'continuous':
            return 2  # pitch, roll
        else:
            return self.config.discrete_tilt_steps ** 2
    
    # ==================== Reward ====================
    
    def _calculate_reward(self, step_info: Dict) -> float:
        """Calculate reward cho step"""
        reward = 0.0
        
        # Time penalty
        reward += self.config.time_penalty
        
        # Triggers
        for trigger in step_info['triggers']:
            if trigger['type'] == 'goal':
                reward += self.config.goal_reward
            elif trigger['type'] == 'hole':
                reward += self.config.hole_penalty
        
        # Collections
        for collect in step_info['collections']:
            if collect['type'] == 'coin':
                reward += collect['value'] * self.config.coin_reward_scale
            elif collect['type'] == 'key':
                reward += 50.0  # Bonus for key
        
        # Dense reward: closer to goal
        if self.config.distance_reward_scale > 0:
            current_dist = self._get_goal_distance()
            dist_delta = self._prev_goal_distance - current_dist
            reward += dist_delta * self.config.distance_reward_scale
            self._prev_goal_distance = current_dist
        
        return reward
    
    def _get_goal_distance(self) -> float:
        """Calculate distance từ ball đến goal"""
        ball = self.world.ball
        goals = self.world.get_entities_by_type(EntityType.GOAL)
        
        if ball is None or not goals:
            return 0.0
        
        ball_pos = ball.get_component(TransformComponent).position[[0, 2]]
        goal_pos = goals[0].get_component(TransformComponent).position[[0, 2]]
        
        return np.linalg.norm(goal_pos - ball_pos)
    
    def _check_terminated(self, step_info: Dict) -> bool:
        """Check if episode should terminate"""
        for trigger in step_info['triggers']:
            if trigger['type'] == 'goal':
                return True
            elif trigger['type'] == 'hole':
                return True
        return False
    
    # ==================== Level Management ====================
    
    def set_level(self, level_id: str) -> bool:
        """Set level by ID"""
        spec = self.level_loader.get_level(level_id)
        if spec is not None:
            self.current_level = spec
            return True
        return False
    
    def next_level(self) -> bool:
        """Advance to next level in curriculum"""
        curriculum = self.level_loader.get_curriculum()
        if not curriculum:
            return False
        
        current_idx = -1
        if self.current_level:
            try:
                current_idx = curriculum.index(self.current_level.id)
            except ValueError:
                pass
        
        next_idx = current_idx + 1
        if next_idx < len(curriculum):
            self.current_level = self.level_loader.get_level(curriculum[next_idx])
            self.current_level_idx = next_idx
            return True
        
        return False
    
    def get_available_levels(self) -> List[str]:
        """Get list of available level IDs"""
        return self.level_loader.get_curriculum()
    
    # ==================== Utilities ====================
    
    def get_state(self) -> Dict:
        """Get full state for visualization"""
        return self.world.get_state_snapshot()
    
    def apply_human_input(self, key: str, dt: float) -> None:
        """Process keyboard input (for human play)"""
        cmd = self.input_mapper.key_to_tilt_delta(key, dt)
        if cmd:
            self.command_queue.push(cmd)
            self.command_queue.process(self.world)
    
    # ==================== Hybrid Movement ====================
    
    def move_direction(self, direction: str, num_steps: int = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Di chuyển ball theo hướng với momentum.
        
        Args:
            direction: 'up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'none'
            num_steps: Số physics steps để chạy (default từ config)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._done:
            return self._get_observation(), 0.0, True, False, self._info.copy()
        
        num_steps = num_steps or self.config.steps_per_move
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # Apply force
        self.world.apply_ball_force(direction)
        
        # Run multiple physics steps
        for _ in range(num_steps):
            step_info = self.world.step_with_friction(self.config.dt)
            self._step_count += 1
            
            # Calculate reward
            reward = self._calculate_reward(step_info)
            total_reward += reward
            self._episode_reward += reward
            
            # Check termination
            terminated = self._check_terminated(step_info)
            truncated = self._step_count >= self.config.max_steps
            
            if terminated or truncated:
                break
        
        self._done = terminated or truncated
        
        # Update info
        self._info.update({
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'triggers': step_info.get('triggers', []),
            'collections': step_info.get('collections', []),
        })
        
        if self._done:
            self._info['final_distance'] = self._get_goal_distance()
            self._info['success'] = any(
                t['type'] == 'goal' for t in step_info.get('triggers', [])
            )
            
            # State machine
            if self._info.get('success'):
                self.state_machine.transition_to(GameState.SUCCESS, force=True)
            else:
                self.state_machine.transition_to(GameState.FAILED, force=True)
            
            self.event_bus.publish(
                GameEvent.EPISODE_ENDED,
                reward=self._episode_reward,
                steps=self._step_count,
                success=self._info.get('success', False)
            )
        
        return self._get_observation(), total_reward, terminated, truncated, self._info.copy()
    
    def step_discrete_8dir(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step với action là 1 trong 9 hướng (0-8).
        0 = none, 1 = up, 2 = upright, 3 = right, ..., 8 = upleft
        """
        direction = DIRECTION_MAP.get(action, 'none')
        return self.move_direction(direction)
