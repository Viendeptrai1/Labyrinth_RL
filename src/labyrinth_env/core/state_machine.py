"""
State Machine pattern
Quản lý game flow và entity states
"""
from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Callable, Optional, Any
from dataclasses import dataclass, field


class GameState(Enum):
    """Các trạng thái của game"""
    LOADING = auto()      # Đang load level
    READY = auto()        # Sẵn sàng chơi
    PLAYING = auto()      # Đang chơi
    PAUSED = auto()       # Tạm dừng
    SUCCESS = auto()      # Đạt goal
    FAILED = auto()       # Rơi hố / hết thời gian
    RESETTING = auto()    # Đang reset


class EntityState(Enum):
    """Trạng thái của entity (ví dụ: Lock, Teleport)"""
    IDLE = auto()
    ACTIVE = auto()
    COOLDOWN = auto()
    LOCKED = auto()
    UNLOCKED = auto()
    COLLECTED = auto()
    DESTROYED = auto()


@dataclass
class StateTransition:
    """Định nghĩa transition giữa 2 states"""
    from_state: Enum
    to_state: Enum
    condition: Optional[Callable[[], bool]] = None
    on_transition: Optional[Callable[[], None]] = None


class StateMachine:
    """
    Generic state machine cho game flow hoặc entity states
    """
    def __init__(self, initial_state: Enum):
        self._current_state: Enum = initial_state
        self._previous_state: Optional[Enum] = None
        self._transitions: Dict[Enum, Dict[Enum, StateTransition]] = {}
        self._state_enter_callbacks: Dict[Enum, Callable] = {}
        self._state_exit_callbacks: Dict[Enum, Callable] = {}
        self._state_update_callbacks: Dict[Enum, Callable] = {}
        self._context: Dict[str, Any] = {}
    
    @property
    def current_state(self) -> Enum:
        return self._current_state
    
    @property
    def previous_state(self) -> Optional[Enum]:
        return self._previous_state
    
    def add_transition(
        self, 
        from_state: Enum, 
        to_state: Enum,
        condition: Callable[[], bool] = None,
        on_transition: Callable[[], None] = None
    ) -> 'StateMachine':
        """Thêm transition (fluent interface)"""
        if from_state not in self._transitions:
            self._transitions[from_state] = {}
        
        self._transitions[from_state][to_state] = StateTransition(
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            on_transition=on_transition
        )
        return self
    
    def on_enter(self, state: Enum, callback: Callable) -> 'StateMachine':
        """Callback khi enter state"""
        self._state_enter_callbacks[state] = callback
        return self
    
    def on_exit(self, state: Enum, callback: Callable) -> 'StateMachine':
        """Callback khi exit state"""
        self._state_exit_callbacks[state] = callback
        return self
    
    def on_update(self, state: Enum, callback: Callable) -> 'StateMachine':
        """Callback khi update trong state"""
        self._state_update_callbacks[state] = callback
        return self
    
    def can_transition_to(self, target_state: Enum) -> bool:
        """Kiểm tra có thể transition không"""
        if self._current_state not in self._transitions:
            return False
        if target_state not in self._transitions[self._current_state]:
            return False
        
        transition = self._transitions[self._current_state][target_state]
        if transition.condition is not None:
            return transition.condition()
        return True
    
    def transition_to(self, target_state: Enum, force: bool = False) -> bool:
        """
        Thực hiện transition sang state mới.
        Returns True nếu thành công.
        """
        if not force and not self.can_transition_to(target_state):
            return False
        
        # Exit callback
        if self._current_state in self._state_exit_callbacks:
            self._state_exit_callbacks[self._current_state]()
        
        # Transition callback
        if (self._current_state in self._transitions and 
            target_state in self._transitions[self._current_state]):
            transition = self._transitions[self._current_state][target_state]
            if transition.on_transition:
                transition.on_transition()
        
        # Update state
        self._previous_state = self._current_state
        self._current_state = target_state
        
        # Enter callback
        if target_state in self._state_enter_callbacks:
            self._state_enter_callbacks[target_state]()
        
        return True
    
    def update(self, dt: float = 0.0) -> None:
        """Update current state"""
        if self._current_state in self._state_update_callbacks:
            self._state_update_callbacks[self._current_state](dt)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context data (chia sẻ giữa states)"""
        self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data"""
        return self._context.get(key, default)
    
    def reset(self, initial_state: Enum) -> None:
        """Reset về state ban đầu"""
        self._current_state = initial_state
        self._previous_state = None
        self._context.clear()


def create_game_state_machine() -> StateMachine:
    """
    Factory function tạo state machine cho game flow.
    Định nghĩa sẵn các transitions hợp lệ.
    """
    sm = StateMachine(GameState.LOADING)
    
    # Định nghĩa transitions
    sm.add_transition(GameState.LOADING, GameState.READY)
    sm.add_transition(GameState.READY, GameState.PLAYING)
    sm.add_transition(GameState.PLAYING, GameState.PAUSED)
    sm.add_transition(GameState.PLAYING, GameState.SUCCESS)
    sm.add_transition(GameState.PLAYING, GameState.FAILED)
    sm.add_transition(GameState.PAUSED, GameState.PLAYING)
    sm.add_transition(GameState.PAUSED, GameState.RESETTING)
    sm.add_transition(GameState.SUCCESS, GameState.RESETTING)
    sm.add_transition(GameState.FAILED, GameState.RESETTING)
    sm.add_transition(GameState.RESETTING, GameState.LOADING)
    
    return sm
