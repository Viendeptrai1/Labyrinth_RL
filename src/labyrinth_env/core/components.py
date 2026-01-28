"""
Component definitions - Data containers cho Entity
Mỗi Component chứa data + logic nhỏ liên quan
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto
import numpy as np

if TYPE_CHECKING:
    from .entity import Entity


@dataclass
class Component:
    """Base component class"""
    entity: Optional['Entity'] = field(default=None, repr=False)
    
    def to_dict(self) -> dict:
        """Serialize cho frontend"""
        raise NotImplementedError


@dataclass
class TransformComponent(Component):
    """
    Vị trí và hướng trong không gian 3D
    Position: (x, y, z) - tâm của object
    Rotation: (pitch, roll, yaw) - góc nghiêng (radians)
    Scale: (sx, sy, sz) - kích thước
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]
    
    @property
    def z(self) -> float:
        return self.position[2]
    
    @property
    def pitch(self) -> float:
        """Nghiêng trước-sau (quanh trục X)"""
        return self.rotation[0]
    
    @property
    def roll(self) -> float:
        """Nghiêng trái-phải (quanh trục Z)"""
        return self.rotation[2]
    
    def to_dict(self) -> dict:
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'scale': self.scale.tolist()
        }


@dataclass 
class PhysicsComponent(Component):
    """
    Physics properties cho dynamic objects (ball)
    """
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    radius: float = 0.5  # Cho sphere collision
    friction: float = 0.3
    restitution: float = 0.2  # Độ nảy
    is_static: bool = False
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)
    
    def apply_force(self, force: np.ndarray) -> None:
        """F = ma => a = F/m"""
        self.acceleration += force / self.mass
    
    def to_dict(self) -> dict:
        return {
            'velocity': self.velocity.tolist(),
            'mass': self.mass,
            'radius': self.radius,
            'speed': self.speed
        }


class CollectibleType(Enum):
    """Loại vật phẩm thu thập được"""
    COIN = auto()
    KEY = auto()


@dataclass
class CollectibleComponent(Component):
    """
    Component cho items có thể thu thập: Coin, Key
    """
    collectible_type: CollectibleType = CollectibleType.COIN
    value: int = 100  # Điểm/giá trị
    key_id: Optional[str] = None  # ID của key (để match với lock)
    collected: bool = False
    
    def collect(self) -> int:
        """Thu thập item, trả về value"""
        if not self.collected:
            self.collected = True
            if self.entity:
                self.entity.active = False
            return self.value
        return 0
    
    def to_dict(self) -> dict:
        return {
            'type': self.collectible_type.name,
            'value': self.value,
            'key_id': self.key_id,
            'collected': self.collected
        }


class TriggerType(Enum):
    """Loại trigger"""
    HOLE = auto()      # Rơi xuống -> game over
    GOAL = auto()      # Đích -> win
    TELEPORT = auto()  # Dịch chuyển
    LOCK = auto()      # Cần key để mở


@dataclass
class TriggerComponent(Component):
    """
    Component cho các vùng trigger: Hole, Goal, Teleport, Lock
    """
    trigger_type: TriggerType = TriggerType.HOLE
    radius: float = 0.5  # Trigger radius
    
    # Cho Teleport
    target_position: Optional[np.ndarray] = None
    teleport_pair_id: Optional[str] = None
    cooldown: float = 1.0  # Thời gian chờ giữa 2 lần teleport
    last_triggered: float = -999.0
    
    # Cho Lock
    required_key_id: Optional[str] = None
    is_unlocked: bool = False
    
    # Reward/Penalty
    reward: float = 0.0
    
    def can_trigger(self, current_time: float) -> bool:
        """Kiểm tra có thể trigger không (cooldown)"""
        if self.trigger_type == TriggerType.LOCK and not self.is_unlocked:
            return False
        return current_time - self.last_triggered >= self.cooldown
    
    def trigger(self, current_time: float) -> Tuple[bool, float]:
        """
        Trigger và trả về (success, reward)
        """
        if not self.can_trigger(current_time):
            return False, 0.0
        self.last_triggered = current_time
        return True, self.reward
    
    def unlock(self) -> bool:
        """Mở khóa (cho Lock type)"""
        if self.trigger_type == TriggerType.LOCK:
            self.is_unlocked = True
            if self.entity:
                self.entity.active = False  # Lock biến mất sau khi mở
            return True
        return False
    
    def to_dict(self) -> dict:
        return {
            'type': self.trigger_type.name,
            'radius': self.radius,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'teleport_pair_id': self.teleport_pair_id,
            'required_key_id': self.required_key_id,
            'is_unlocked': self.is_unlocked,
            'reward': self.reward
        }


@dataclass
class InventoryComponent(Component):
    """
    Inventory của player (ball) - giữ keys đã nhặt
    """
    keys: list = field(default_factory=list)
    coins_collected: int = 0
    total_score: int = 0
    
    def add_key(self, key_id: str) -> None:
        """Thêm key vào inventory"""
        if key_id not in self.keys:
            self.keys.append(key_id)
    
    def has_key(self, key_id: str) -> bool:
        """Kiểm tra có key không"""
        return key_id in self.keys
    
    def use_key(self, key_id: str) -> bool:
        """Sử dụng key (xóa khỏi inventory)"""
        if key_id in self.keys:
            self.keys.remove(key_id)
            return True
        return False
    
    def add_score(self, points: int) -> None:
        """Cộng điểm"""
        self.total_score += points
    
    def add_coin(self, value: int) -> None:
        """Thu thập coin"""
        self.coins_collected += 1
        self.add_score(value)
    
    def to_dict(self) -> dict:
        return {
            'keys': self.keys.copy(),
            'coins_collected': self.coins_collected,
            'total_score': self.total_score
        }


@dataclass
class CollisionComponent(Component):
    """
    Collision shape cho static objects (walls)
    """
    shape: str = 'box'  # 'box', 'sphere', 'cylinder'
    size: np.ndarray = field(default_factory=lambda: np.ones(3))  # half-extents cho box
    is_solid: bool = True  # Có chặn ball không
    
    def to_dict(self) -> dict:
        return {
            'shape': self.shape,
            'size': self.size.tolist(),
            'is_solid': self.is_solid
        }
