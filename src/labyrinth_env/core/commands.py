"""
Command pattern
Xử lý input và cho phép record/replay
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from collections import deque
import time
import numpy as np


@dataclass
class Command(ABC):
    """
    Base Command class.
    Mỗi command encapsulate 1 action có thể execute và undo.
    """
    timestamp: float = field(default_factory=time.time)
    executed: bool = False
    
    @abstractmethod
    def execute(self, context: Any) -> bool:
        """Thực thi command. Returns True nếu thành công."""
        pass
    
    def undo(self, context: Any) -> bool:
        """Undo command (optional). Returns True nếu thành công."""
        return False
    
    def to_dict(self) -> dict:
        """Serialize cho replay/logging"""
        return {
            'type': self.__class__.__name__,
            'timestamp': self.timestamp,
            'executed': self.executed
        }


@dataclass
class SetTiltCommand(Command):
    """
    Command để set góc nghiêng bàn.
    Action chính trong game Labyrinth.
    """
    target_pitch: float = 0.0  # Radians
    target_roll: float = 0.0   # Radians
    
    def execute(self, context: Any) -> bool:
        """Set tilt cho board entity trong context (World)"""
        if hasattr(context, 'set_board_tilt'):
            context.set_board_tilt(self.target_pitch, self.target_roll)
            self.executed = True
            return True
        return False
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'target_pitch': self.target_pitch,
            'target_roll': self.target_roll
        })
        return d


@dataclass
class ResetCommand(Command):
    """Command reset game/episode"""
    level_id: Optional[str] = None
    seed: Optional[int] = None
    
    def execute(self, context: Any) -> bool:
        if hasattr(context, 'reset'):
            context.reset(level_id=self.level_id, seed=self.seed)
            self.executed = True
            return True
        return False
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'level_id': self.level_id,
            'seed': self.seed
        })
        return d


@dataclass
class PauseCommand(Command):
    """Command pause/resume game"""
    pause: bool = True
    
    def execute(self, context: Any) -> bool:
        if hasattr(context, 'set_paused'):
            context.set_paused(self.pause)
            self.executed = True
            return True
        return False


class CommandQueue:
    """
    Queue để buffer và xử lý commands.
    Hỗ trợ record/replay.
    """
    def __init__(self, max_history: int = 10000):
        self._pending: deque = deque()
        self._history: List[Command] = []
        self._max_history = max_history
        self._recording = False
    
    def push(self, command: Command) -> None:
        """Thêm command vào queue"""
        self._pending.append(command)
    
    def process(self, context: Any) -> List[Command]:
        """
        Xử lý tất cả pending commands.
        Returns list of executed commands.
        """
        executed = []
        while self._pending:
            cmd = self._pending.popleft()
            if cmd.execute(context):
                executed.append(cmd)
                if self._recording:
                    self._history.append(cmd)
                    if len(self._history) > self._max_history:
                        self._history.pop(0)
        return executed
    
    def process_one(self, context: Any) -> Optional[Command]:
        """Xử lý 1 command"""
        if self._pending:
            cmd = self._pending.popleft()
            if cmd.execute(context):
                if self._recording:
                    self._history.append(cmd)
                return cmd
        return None
    
    def clear(self) -> None:
        """Xóa pending commands"""
        self._pending.clear()
    
    def start_recording(self) -> None:
        """Bắt đầu ghi lại commands"""
        self._recording = True
    
    def stop_recording(self) -> None:
        """Dừng ghi"""
        self._recording = False
    
    def get_recording(self) -> List[Command]:
        """Lấy history đã record"""
        return self._history.copy()
    
    def clear_recording(self) -> None:
        """Xóa history"""
        self._history.clear()
    
    def replay(self, commands: List[Command], context: Any) -> None:
        """Replay list of commands"""
        for cmd in commands:
            cmd.executed = False  # Reset
            self._pending.append(cmd)
        self.process(context)
    
    @property
    def pending_count(self) -> int:
        return len(self._pending)
    
    @property
    def history_count(self) -> int:
        return len(self._history)


# Utility: Convert keyboard/mouse input to commands
class InputMapper:
    """
    Map raw input (key presses, mouse) to Commands.
    Có thể customize mapping.
    """
    def __init__(self, max_tilt: float = 0.15):  # ~8.6 degrees
        self.max_tilt = max_tilt
        self._current_pitch = 0.0
        self._current_roll = 0.0
        self._tilt_speed = 0.5  # radians/second
    
    def key_to_tilt_delta(self, key: str, dt: float) -> Optional[SetTiltCommand]:
        """
        Map WASD/Arrow keys to tilt changes.
        Returns SetTiltCommand or None.
        """
        delta = self._tilt_speed * dt
        
        pitch_delta = 0.0
        roll_delta = 0.0
        
        key_lower = key.lower()
        if key_lower in ('w', 'arrowup'):
            pitch_delta = -delta  # Nghiêng về phía trước
        elif key_lower in ('s', 'arrowdown'):
            pitch_delta = delta   # Nghiêng về phía sau
        elif key_lower in ('a', 'arrowleft'):
            roll_delta = -delta   # Nghiêng trái
        elif key_lower in ('d', 'arrowright'):
            roll_delta = delta    # Nghiêng phải
        else:
            return None
        
        # Cập nhật và clamp
        self._current_pitch = np.clip(
            self._current_pitch + pitch_delta, 
            -self.max_tilt, 
            self.max_tilt
        )
        self._current_roll = np.clip(
            self._current_roll + roll_delta,
            -self.max_tilt,
            self.max_tilt
        )
        
        return SetTiltCommand(
            target_pitch=self._current_pitch,
            target_roll=self._current_roll
        )
    
    def mouse_to_tilt(self, dx: float, dy: float, sensitivity: float = 0.001) -> SetTiltCommand:
        """
        Map mouse drag to tilt.
        dx, dy là pixel movement.
        """
        self._current_pitch = np.clip(
            self._current_pitch + dy * sensitivity,
            -self.max_tilt,
            self.max_tilt
        )
        self._current_roll = np.clip(
            self._current_roll + dx * sensitivity,
            -self.max_tilt,
            self.max_tilt
        )
        
        return SetTiltCommand(
            target_pitch=self._current_pitch,
            target_roll=self._current_roll
        )
    
    def reset_tilt(self) -> SetTiltCommand:
        """Reset tilt về 0"""
        self._current_pitch = 0.0
        self._current_roll = 0.0
        return SetTiltCommand(target_pitch=0.0, target_roll=0.0)
    
    def set_tilt_direct(self, pitch: float, roll: float) -> SetTiltCommand:
        """Set tilt trực tiếp (cho RL agent)"""
        self._current_pitch = np.clip(pitch, -self.max_tilt, self.max_tilt)
        self._current_roll = np.clip(roll, -self.max_tilt, self.max_tilt)
        return SetTiltCommand(
            target_pitch=self._current_pitch,
            target_roll=self._current_roll
        )
