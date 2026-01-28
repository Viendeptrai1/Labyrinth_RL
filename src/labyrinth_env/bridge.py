"""
ZeppelinBridge - Bridge giữa PySpark backend và AngularJS frontend
Sử dụng z.angularBind() để sync state
"""
from __future__ import annotations
from typing import Any, Dict, Callable, Optional
import json
import numpy as np

from .env import LabyrinthEnv, EnvConfig
from .core.events import EventBus, GameEvent


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder hỗ trợ numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class ZeppelinBridge:
    """
    Bridge class để kết nối LabyrinthEnv với Zeppelin AngularJS.
    
    Usage trong Zeppelin notebook:
        ```python
        %pyspark
        from src.labyrinth_env import ZeppelinBridge
        
        bridge = ZeppelinBridge(z)  # z là Zeppelin context
        bridge.init_game('level_01_tutorial')
        bridge.bind_all()  # Bind tất cả state sang Angular
        ```
    
    Trong %angular:
        ```html
        <div ng-controller="LabyrinthCtrl">
            Ball position: {{state.ball.position}}
            Score: {{state.ball.inventory.total_score}}
        </div>
        ```
    """
    
    def __init__(self, zeppelin_context: Any, config: EnvConfig = None):
        """
        Args:
            zeppelin_context: Zeppelin context (z) từ notebook
            config: Environment configuration
        """
        self.z = zeppelin_context
        self.env = LabyrinthEnv(config=config)
        self.event_bus = EventBus.get_instance()
        
        # Callback registrations
        self._state_callbacks: list = []
        self._event_callbacks: Dict[GameEvent, list] = {}
        
        # Metrics tracking
        self._episode_count = 0
        self._total_steps = 0
        self._metrics_history: list = []
        
        # Subscribe to events
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers cho auto-binding"""
        self.event_bus.subscribe(GameEvent.STATE_UPDATED, self._on_state_updated)
        self.event_bus.subscribe(GameEvent.EPISODE_ENDED, self._on_episode_ended)
        self.event_bus.subscribe(GameEvent.COIN_COLLECTED, self._on_coin_collected)
        self.event_bus.subscribe(GameEvent.KEY_COLLECTED, self._on_key_collected)
        self.event_bus.subscribe(GameEvent.GOAL_REACHED, self._on_goal_reached)
        self.event_bus.subscribe(GameEvent.BALL_FELL, self._on_ball_fell)
    
    # ==================== Initialization ====================
    
    def init_game(self, level_id: str = None, seed: int = None) -> Dict:
        """
        Khởi tạo game với level.
        Returns initial state.
        """
        obs, info = self.env.reset(level_id=level_id, seed=seed)
        self._episode_count += 1
        
        # Bind initial state
        self.bind_all()
        
        return {
            'observation': obs.tolist(),
            'info': info,
            'state': self.get_state()
        }
    
    def reset(self, level_id: str = None, seed: int = None) -> Dict:
        """Alias cho init_game"""
        return self.init_game(level_id, seed)
    
    # ==================== Game Control ====================
    
    def step(self, action: list) -> Dict:
        """
        Thực hiện 1 step với action.
        Auto-bind state sau mỗi step.
        
        Args:
            action: [pitch, roll] trong [-1, 1]
        
        Returns:
            dict với observation, reward, done, info, state
        """
        action_np = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        
        self._total_steps += 1
        
        # Auto-bind state
        self.bind_all()
        
        return {
            'observation': obs.tolist(),
            'reward': float(reward),
            'terminated': terminated,
            'truncated': truncated,
            'done': terminated or truncated,
            'info': info,
            'state': self.get_state()
        }
    
    def apply_tilt(self, pitch: float, roll: float) -> Dict:
        """
        Apply tilt trực tiếp (cho human play).
        Pitch/roll trong [-1, 1].
        """
        return self.step([pitch, roll])
    
    def key_input(self, key: str, dt: float = 0.016) -> Dict:
        """
        Xử lý keyboard input (old tilt-based).
        
        Args:
            key: 'w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'
            dt: Delta time
        """
        self.env.apply_human_input(key, dt)
        
        # Get current tilt as action
        pitch, roll = self.env.world.board_tilt
        action = [
            pitch / self.env.config.max_tilt,
            roll / self.env.config.max_tilt
        ]
        
        return self.step(action)
    
    # ==================== Hybrid Movement ====================
    
    def move(self, direction: str, num_steps: int = None) -> Dict:
        """
        Di chuyển ball theo hướng với momentum (hybrid movement).
        Ball sẽ được đẩy và có quán tính.
        
        Args:
            direction: 'up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'none'
            num_steps: Số physics steps (default = 15)
        
        Returns:
            dict với observation, reward, done, info, state
        """
        num_steps = num_steps or self.env.config.steps_per_move
        
        obs, reward, terminated, truncated, info = self.env.move_direction(direction, num_steps)
        
        self._total_steps += num_steps
        
        # Auto-bind state
        self.bind_all()
        
        return {
            'observation': obs.tolist(),
            'reward': float(reward),
            'terminated': terminated,
            'truncated': truncated,
            'done': terminated or truncated,
            'info': info,
            'state': self.get_state()
        }
    
    def move_up(self, num_steps: int = None) -> Dict:
        """Di chuyển lên (W)"""
        return self.move('up', num_steps)
    
    def move_down(self, num_steps: int = None) -> Dict:
        """Di chuyển xuống (S)"""
        return self.move('down', num_steps)
    
    def move_left(self, num_steps: int = None) -> Dict:
        """Di chuyển trái (A)"""
        return self.move('left', num_steps)
    
    def move_right(self, num_steps: int = None) -> Dict:
        """Di chuyển phải (D)"""
        return self.move('right', num_steps)
    
    def move_upleft(self, num_steps: int = None) -> Dict:
        """Di chuyển chéo trên-trái (WA)"""
        return self.move('upleft', num_steps)
    
    def move_upright(self, num_steps: int = None) -> Dict:
        """Di chuyển chéo trên-phải (WD)"""
        return self.move('upright', num_steps)
    
    def move_downleft(self, num_steps: int = None) -> Dict:
        """Di chuyển chéo dưới-trái (SA)"""
        return self.move('downleft', num_steps)
    
    def move_downright(self, num_steps: int = None) -> Dict:
        """Di chuyển chéo dưới-phải (SD)"""
        return self.move('downright', num_steps)
    
    # ==================== State Binding ====================
    
    def bind_all(self) -> None:
        """Bind tất cả state sang AngularJS"""
        state = self.get_state()
        
        # Bind từng phần
        self._bind('state', state)
        self._bind('ball', state.get('ball'))
        self._bind('board', state.get('board'))
        self._bind('entities', state.get('entities'))
        self._bind('gameStatus', self._get_game_status())
        self._bind('metrics', self._get_metrics())
    
    def bind_state(self) -> None:
        """Bind chỉ state (lightweight)"""
        self._bind('state', self.get_state())
    
    def bind_metrics(self) -> None:
        """Bind training metrics"""
        self._bind('metrics', self._get_metrics())
    
    def _bind(self, name: str, value: Any) -> None:
        """Internal bind helper"""
        if self.z is not None:
            # Convert numpy arrays to lists for JSON compatibility
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, dict):
                value = json.loads(json.dumps(value, cls=NumpyEncoder))
            
            self.z.angularBind(name, value)
    
    def get_state(self) -> Dict:
        """Get current state dict"""
        return self.env.get_state()
    
    def _get_game_status(self) -> Dict:
        """Get game status for UI"""
        return {
            'state': self.env.state_machine.current_state.name,
            'level': self.env.current_level.name if self.env.current_level else 'None',
            'level_id': self.env.current_level.id if self.env.current_level else None,
            'step': self.env._step_count,
            'episode': self._episode_count,
            'episode_reward': self.env._episode_reward,
            'done': self.env._done
        }
    
    def _get_metrics(self) -> Dict:
        """Get training metrics"""
        return {
            'episode_count': self._episode_count,
            'total_steps': self._total_steps,
            'history': self._metrics_history[-100:]  # Last 100 episodes
        }
    
    # ==================== Event Handlers ====================
    
    def _on_state_updated(self, event_data) -> None:
        """Handler cho STATE_UPDATED event"""
        pass  # Auto-bind được gọi trong step()
    
    def _on_episode_ended(self, event_data) -> None:
        """Handler cho EPISODE_ENDED event"""
        self._metrics_history.append({
            'episode': self._episode_count,
            'reward': event_data.data.get('reward', 0),
            'steps': event_data.data.get('steps', 0),
            'success': event_data.data.get('success', False)
        })
        
        # Bind updated metrics
        self.bind_metrics()
        self._bind('episodeEnded', {
            'success': event_data.data.get('success', False),
            'reward': event_data.data.get('reward', 0)
        })
    
    def _on_coin_collected(self, event_data) -> None:
        """Handler cho COIN_COLLECTED"""
        self._bind('lastEvent', {
            'type': 'coin_collected',
            'value': event_data.data.get('value', 0)
        })
    
    def _on_key_collected(self, event_data) -> None:
        """Handler cho KEY_COLLECTED"""
        self._bind('lastEvent', {
            'type': 'key_collected',
            'key_id': event_data.data.get('key_id')
        })
    
    def _on_goal_reached(self, event_data) -> None:
        """Handler cho GOAL_REACHED"""
        self._bind('lastEvent', {'type': 'goal_reached'})
    
    def _on_ball_fell(self, event_data) -> None:
        """Handler cho BALL_FELL"""
        self._bind('lastEvent', {
            'type': 'ball_fell',
            'hole_id': event_data.data.get('hole_id')
        })
    
    # ==================== Level Management ====================
    
    def get_available_levels(self) -> list:
        """Get danh sách level IDs"""
        return self.env.get_available_levels()
    
    def set_level(self, level_id: str) -> bool:
        """Set level"""
        return self.env.set_level(level_id)
    
    def next_level(self) -> bool:
        """Advance to next level"""
        return self.env.next_level()
    
    # ==================== Training Support ====================
    
    def get_observation_dim(self) -> int:
        """Get observation dimension"""
        return self.env.observation_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension"""
        return self.env.action_dim
    
    def sample_action(self) -> list:
        """Sample random action"""
        return [
            float(np.random.uniform(-1, 1)),
            float(np.random.uniform(-1, 1))
        ]
    
    def run_episode(
        self, 
        policy: Callable[[np.ndarray], np.ndarray] = None,
        render: bool = True,
        max_steps: int = None
    ) -> Dict:
        """
        Run 1 episode với policy.
        
        Args:
            policy: Function(obs) -> action. None = random policy.
            render: Whether to bind state mỗi step (cho visualization)
            max_steps: Override max steps
        
        Returns:
            Episode statistics
        """
        obs, info = self.env.reset()
        
        if render:
            self.bind_all()
        
        total_reward = 0.0
        steps = 0
        done = False
        
        max_steps = max_steps or self.env.config.max_steps
        
        while not done and steps < max_steps:
            # Get action
            if policy is not None:
                action = policy(obs)
            else:
                action = np.random.uniform(-1, 1, size=2)
            
            # Step
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                self.bind_all()
        
        return {
            'reward': total_reward,
            'steps': steps,
            'success': info.get('success', False),
            'final_distance': info.get('final_distance', 0)
        }


# ==================== Utility Functions ====================

def create_bridge(z, level_id: str = None) -> ZeppelinBridge:
    """
    Quick helper để tạo bridge trong Zeppelin.
    
    Usage:
        bridge = create_bridge(z, 'level_01_tutorial')
    """
    bridge = ZeppelinBridge(z)
    if level_id:
        bridge.init_game(level_id)
    return bridge
