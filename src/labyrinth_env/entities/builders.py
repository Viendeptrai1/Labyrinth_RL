"""
Entity Builders - Factory functions để tạo các entity cụ thể
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
    position: Tuple[float, float, float] = (0, 0.5, 0),
    radius: float = 0.3,
    mass: float = 1.0,
    friction: float = 0.3,
    **kwargs
) -> Entity:
    """
    Tạo ball entity (player controlled).
    Y position = radius để ball nằm trên mặt bàn.
    """
    ball = Entity(entity_type=EntityType.BALL)
    
    # Transform - position on board surface
    transform = TransformComponent(
        position=np.array([position[0], radius, position[2]], dtype=float),
        scale=np.array([radius * 2, radius * 2, radius * 2])
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
    position: Tuple[float, float, float],
    size: Tuple[float, float, float] = (0.5, 1.0, 0.5),
    **kwargs
) -> Entity:
    """
    Tạo wall entity (static obstacle).
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
    position: Tuple[float, float, float],
    radius: float = 0.4,
    penalty: float = -100.0,
    **kwargs
) -> Entity:
    """
    Tạo hole entity (trap - ball falls in = game over).
    """
    hole = Entity(entity_type=EntityType.HOLE)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, 0.1, radius * 2])
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
    position: Tuple[float, float, float],
    value: int = 100,
    **kwargs
) -> Entity:
    """
    Tạo coin entity (collectible for score).
    """
    coin = Entity(entity_type=EntityType.COIN)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([0.3, 0.3, 0.3])
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
    position: Tuple[float, float, float],
    key_id: str,
    value: int = 50,
    **kwargs
) -> Entity:
    """
    Tạo key entity (collectible to unlock locks).
    """
    key = Entity(entity_type=EntityType.KEY)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([0.25, 0.25, 0.25])
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
    position: Tuple[float, float, float],
    required_key_id: str,
    size: Tuple[float, float, float] = (0.5, 1.0, 0.5),
    **kwargs
) -> Entity:
    """
    Tạo lock entity (barrier that requires key to open).
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
    position: Tuple[float, float, float],
    target_position: Tuple[float, float, float],
    pair_id: str = None,
    radius: float = 0.4,
    cooldown: float = 1.5,
    **kwargs
) -> Entity:
    """
    Tạo teleport entity (transports ball to target location).
    """
    teleport = Entity(entity_type=EntityType.TELEPORT)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, 0.2, radius * 2])
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
    position: Tuple[float, float, float],
    radius: float = 0.5,
    reward: float = 1000.0,
    **kwargs
) -> Entity:
    """
    Tạo goal entity (win condition - destination for ball).
    """
    goal = Entity(entity_type=EntityType.GOAL)
    
    transform = TransformComponent(
        position=np.array(position, dtype=float),
        scale=np.array([radius * 2, 0.1, radius * 2])
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


# ==================== Helper: Create wall segments ====================

def create_wall_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: float = 0.2,
    height: float = 0.5
) -> Entity:
    """
    Tạo wall segment từ điểm start đến end (trên mặt phẳng XZ).
    Tiện cho level design.
    """
    # Tính center và size
    cx = (start[0] + end[0]) / 2
    cz = (start[1] + end[1]) / 2
    
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length = np.sqrt(dx*dx + dz*dz)
    
    # Rotation angle
    angle = np.arctan2(dz, dx)
    
    wall = create_wall(
        position=(cx, height/2, cz),
        size=(length/2, height/2, thickness/2)
    )
    
    # Set rotation
    transform = wall.get_component(TransformComponent)
    if transform:
        transform.rotation[1] = angle  # Rotate around Y axis
    
    return wall


def create_rectangular_border(
    width: float,
    height: float,
    wall_thickness: float = 0.2,
    wall_height: float = 0.5
) -> List[Entity]:
    """
    Tạo 4 walls bao quanh board.
    """
    half_w = width / 2
    half_h = height / 2
    
    walls = [
        # Top wall
        create_wall(
            position=(0, wall_height/2, -half_h),
            size=(half_w, wall_height/2, wall_thickness/2)
        ),
        # Bottom wall
        create_wall(
            position=(0, wall_height/2, half_h),
            size=(half_w, wall_height/2, wall_thickness/2)
        ),
        # Left wall
        create_wall(
            position=(-half_w, wall_height/2, 0),
            size=(wall_thickness/2, wall_height/2, half_h)
        ),
        # Right wall
        create_wall(
            position=(half_w, wall_height/2, 0),
            size=(wall_thickness/2, wall_height/2, half_h)
        ),
    ]
    
    return walls
