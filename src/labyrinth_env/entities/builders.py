"""
Entity Builders - Factory functions để tạo các entity cụ thể (2D)
Đăng ký với EntityFactory để có thể tạo từ level config
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List

from ..core.entity import Entity, EntityType, EntityFactory
from ..core.components import (
    TransformComponent, PhysicsComponent, CollisionComponent,
    TriggerComponent, CollectibleComponent, InventoryComponent,
    TriggerType, CollectibleType
)


@EntityFactory.register('ball')
def create_ball(
    position: Tuple[float, float] = (0, 0),
    radius: float = 0.3,
    mass: float = 1.0,
    friction: float = 0.3,
    **kwargs
) -> Entity:
    """
    Tạo ball entity (player controlled) - 2D.
    Position: (x, y) trên mặt phẳng bàn.
    """
    ball = Entity(entity_type=EntityType.BALL)
    
    # Transform - position on board (2D)
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, radius * 2])
    )
    
    # Physics
    physics = PhysicsComponent(
        mass=mass,
        radius=radius,
        friction=friction,
        restitution=0.2,
        is_static=False
    )
    
    # Inventory để giữ keys và score
    inventory = InventoryComponent()
    
    ball.add_component(transform)
    ball.add_component(physics)
    ball.add_component(inventory)
    
    return ball


@EntityFactory.register('wall')
def create_wall(
    position: Tuple[float, float],
    size: Tuple[float, float] = (0.5, 0.5),
    **kwargs
) -> Entity:
    """
    Tạo wall entity (static obstacle) - 2D.
    Size là half-extents (box từ -size đến +size).
    """
    wall = Entity(entity_type=EntityType.WALL)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array(size, dtype=float) * 2  # Full size for rendering
    )
    
    collision = CollisionComponent(
        shape='box',
        size=np.array(size, dtype=float),  # Half-extents for collision
        is_solid=True
    )
    
    wall.add_component(transform)
    wall.add_component(collision)
    
    return wall


@EntityFactory.register('hole')
def create_hole(
    position: Tuple[float, float],
    radius: float = 0.4,
    penalty: float = -100.0,
    **kwargs
) -> Entity:
    """
    Tạo hole entity (trap - ball falls in = game over) - 2D.
    """
    hole = Entity(entity_type=EntityType.HOLE)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, radius * 2])
    )
    
    trigger = TriggerComponent(
        trigger_type=TriggerType.HOLE,
        radius=radius,
        reward=penalty,
        cooldown=0.0  # Instant trigger
    )
    
    hole.add_component(transform)
    hole.add_component(trigger)
    
    return hole


@EntityFactory.register('coin')
def create_coin(
    position: Tuple[float, float],
    value: int = 100,
    **kwargs
) -> Entity:
    """
    Tạo coin entity (collectible for score) - 2D.
    """
    coin = Entity(entity_type=EntityType.COIN)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([0.3, 0.3])
    )
    
    collectible = CollectibleComponent(
        collectible_type=CollectibleType.COIN,
        value=value
    )
    
    coin.add_component(transform)
    coin.add_component(collectible)
    
    return coin


@EntityFactory.register('key')
def create_key(
    position: Tuple[float, float],
    key_id: str,
    value: int = 50,
    **kwargs
) -> Entity:
    """
    Tạo key entity (collectible to unlock locks) - 2D.
    """
    key = Entity(entity_type=EntityType.KEY)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([0.25, 0.25])
    )
    
    collectible = CollectibleComponent(
        collectible_type=CollectibleType.KEY,
        key_id=key_id,
        value=value
    )
    
    key.add_component(transform)
    key.add_component(collectible)
    
    return key


@EntityFactory.register('lock')
def create_lock(
    position: Tuple[float, float],
    required_key_id: str,
    size: Tuple[float, float] = (0.5, 0.5),
    **kwargs
) -> Entity:
    """
    Tạo lock entity (barrier that requires key to open) - 2D.
    Acts like a wall until unlocked.
    """
    lock = Entity(entity_type=EntityType.LOCK)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array(size, dtype=float) * 2
    )
    
    # Collision như wall
    collision = CollisionComponent(
        shape='box',
        size=np.array(size, dtype=float),
        is_solid=True
    )
    
    # Trigger để detect key
    trigger = TriggerComponent(
        trigger_type=TriggerType.LOCK,
        radius=0.6,
        required_key_id=required_key_id,
        is_unlocked=False,
        reward=50.0  # Bonus khi mở khóa
    )
    
    lock.add_component(transform)
    lock.add_component(collision)
    lock.add_component(trigger)
    
    return lock


@EntityFactory.register('teleport')
def create_teleport(
    position: Tuple[float, float],
    target_position: Tuple[float, float],
    pair_id: str = None,
    radius: float = 0.4,
    cooldown: float = 1.5,
    **kwargs
) -> Entity:
    """
    Tạo teleport entity (transports ball to target location) - 2D.
    """
    teleport = Entity(entity_type=EntityType.TELEPORT)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, radius * 2])
    )
    
    trigger = TriggerComponent(
        trigger_type=TriggerType.TELEPORT,
        radius=radius,
        target_position=np.array(target_position, dtype=float),
        teleport_pair_id=pair_id,
        cooldown=cooldown,
        reward=0.0
    )
    
    teleport.add_component(transform)
    teleport.add_component(trigger)
    
    return teleport


@EntityFactory.register('goal')
def create_goal(
    position: Tuple[float, float],
    radius: float = 0.5,
    reward: float = 1000.0,
    **kwargs
) -> Entity:
    """
    Tạo goal entity (win condition - destination for ball) - 2D.
    """
    goal = Entity(entity_type=EntityType.GOAL)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, radius * 2])
    )
    
    trigger = TriggerComponent(
        trigger_type=TriggerType.GOAL,
        radius=radius,
        reward=reward,
        cooldown=0.0
    )
    
    goal.add_component(transform)
    goal.add_component(trigger)
    
    return goal


# ==================== Helper: Create wall segments (2D) ====================

def create_wall_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: float = 0.2
) -> Entity:
    """
    Tạo wall segment từ điểm start đến end (trên mặt phẳng XY).
    Tiện cho level design.
    """
    # Tính center và size
    cx = (start[0] + end[0]) / 2
    cy = (start[1] + end[1]) / 2
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    # Rotation angle
    angle = np.arctan2(dy, dx)
    
    wall = create_wall(
        position=(cx, cy),
        size=(length/2, thickness/2)
    )
    
    # Set rotation
    transform = wall.get_component(TransformComponent)
    if transform:
        transform.rotation[0] = angle  # Rotate in 2D plane
    
    return wall


def create_rectangular_border(
    width: float,
    height: float,
    wall_thickness: float = 0.2
) -> List[Entity]:
    """
    Tạo 4 walls bao quanh board (2D).
    """
    half_w = width / 2
    half_h = height / 2
    
    walls = [
        # Top wall
        create_wall(
            position=(0, half_h),
            size=(half_w, wall_thickness/2)
        ),
        # Bottom wall
        create_wall(
            position=(0, -half_h),
            size=(half_w, wall_thickness/2)
        ),
        # Left wall
        create_wall(
            position=(-half_w, 0),
            size=(wall_thickness/2, half_h)
        ),
        # Right wall
        create_wall(
            position=(half_w, 0),
            size=(wall_thickness/2, half_h)
        ),
    ]
    
    return walls
