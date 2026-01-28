"""
EventBus - Observer/Publisher-Subscriber pattern
Dùng để giao tiếp lỏng giữa các module (env, UI, metrics)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
from enum import Enum, auto
import time


class GameEvent(Enum):
    """Các loại event trong game"""
    # State events
    STATE_UPDATED = auto()
    EPISODE_STARTED = auto()
    EPISODE_ENDED = auto()
    LEVEL_LOADED = auto()
    
    # Game events
    BALL_MOVED = auto()
    COIN_COLLECTED = auto()
    KEY_COLLECTED = auto()
    LOCK_OPENED = auto()
    TELEPORT_ACTIVATED = auto()
    BALL_FELL = auto()  # Rơi hố
    GOAL_REACHED = auto()
    
    # Input events
    TILT_CHANGED = auto()
    RESET_REQUESTED = auto()
    
    # Training events
    STEP_COMPLETED = auto()
    METRICS_UPDATED = auto()
    CHECKPOINT_SAVED = auto()


@dataclass
class EventData:
    """Container cho event data"""
    event_type: GameEvent
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Central event bus - singleton pattern.
    Các module subscribe events và publish khi cần.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers: Dict[GameEvent, List[Callable]] = {}
            cls._instance._event_history: List[EventData] = []
            cls._instance._history_limit = 1000
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Lấy singleton instance"""
        return cls()
    
    def subscribe(self, event_type: GameEvent, callback: Callable[[EventData], None]) -> None:
        """Đăng ký lắng nghe event"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: GameEvent, callback: Callable) -> None:
        """Hủy đăng ký"""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def publish(self, event_type: GameEvent, **data) -> None:
        """Phát event tới tất cả subscribers"""
        event_data = EventData(event_type=event_type, data=data)
        
        # Lưu history
        self._event_history.append(event_data)
        if len(self._event_history) > self._history_limit:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    print(f"Error in event callback: {e}")
    
    def clear_subscribers(self, event_type: GameEvent = None) -> None:
        """Xóa subscribers (all hoặc theo type)"""
        if event_type is None:
            self._subscribers.clear()
        elif event_type in self._subscribers:
            self._subscribers[event_type].clear()
    
    def get_history(self, event_type: GameEvent = None, limit: int = 100) -> List[EventData]:
        """Lấy history events"""
        if event_type is None:
            return self._event_history[-limit:]
        return [e for e in self._event_history if e.event_type == event_type][-limit:]
    
    def clear_history(self) -> None:
        """Xóa history"""
        self._event_history.clear()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (dùng cho testing)"""
        cls._instance = None
