"""
World - Container quản lý tất cả entities và physics update (2D)
"""
from __future__ import annotations
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass, field
import numpy as np

from .entity import Entity, EntityType
from .components import (
    TransformComponent, PhysicsComponent, CollisionComponent,
    TriggerComponent, CollectibleComponent, InventoryComponent,
    TriggerType, CollectibleType
)
from .events import EventBus, GameEvent


@dataclass
class BoardConfig:
    """Cấu hình bàn chơi 2D"""
    width: float = 10.0      # Chiều rộng (X)
    height: float = 10.0     # Chiều cao (Y) - top-down view
    max_tilt: float = 0.15   # Góc nghiêng tối đa (radians, ~8.6 degrees)
    
    # Hybrid movement settings - STRONG defaults for human play
    force_magnitude: float = 80.0   # Lực đẩy ball khi input
    friction_coefficient: float = 0.90  # Hệ số ma sát
    max_velocity: float = 15.0      # Tốc độ tối đa


# Direction vectors cho 8 hướng di chuyển (2D: x, y)
# UI render: y tăng = ball đi lên màn hình
DIRECTIONS = {
    'up': np.array([0.0, 1.0]),        # W - đi lên (tăng Y)
    'down': np.array([0.0, -1.0]),     # S - đi xuống (giảm Y)
    'left': np.array([-1.0, 0.0]),     # A - đi trái (giảm X)
    'right': np.array([1.0, 0.0]),     # D - đi phải (tăng X)
    'upleft': np.array([-0.707, 0.707]),      # WA
    'upright': np.array([0.707, 0.707]),      # WD
    'downleft': np.array([-0.707, -0.707]),   # SA
    'downright': np.array([0.707, -0.707]),   # SD
    'none': np.array([0.0, 0.0]),      # Không di chuyển
}
    

class World:
    """
    World container - quản lý entities và physics simulation (2D).
    Đây là "source of truth" cho game state.
    """
    
    def __init__(self, board_config: BoardConfig = None):
        self.board_config = board_config or BoardConfig()
        self._entities: Dict[str, Entity] = {}
        self._entities_by_type: Dict[EntityType, List[Entity]] = {t: [] for t in EntityType}
        
        # Board tilt state
        self._board_pitch: float = 0.0
        self._board_roll: float = 0.0
        
        # Physics constants
        self.gravity = 9.81
        self.dt = 1.0 / 60.0  # Fixed timestep
        
        # Time tracking
        self._time: float = 0.0
        self._step_count: int = 0
        
        # Event bus reference
        self._event_bus = EventBus.get_instance()
        
        # Ball entity reference (shortcut)
        self._ball: Optional[Entity] = None
    
    # ==================== Entity Management ====================
    
    def add_entity(self, entity: Entity) -> Entity:
        """Thêm entity vào world"""
        self._entities[entity.id] = entity
        self._entities_by_type[entity.entity_type].append(entity)
        
        # Cache ball reference
        if entity.entity_type == EntityType.BALL:
            self._ball = entity
        
        return entity
    
    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        """Xóa entity khỏi world"""
        if entity_id not in self._entities:
            return None
        
        entity = self._entities.pop(entity_id)
        self._entities_by_type[entity.entity_type].remove(entity)
        
        if entity is self._ball:
            self._ball = None
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Lấy entity theo ID"""
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Lấy tất cả entities của 1 type"""
        return self._entities_by_type[entity_type].copy()
    
    def get_all_entities(self) -> List[Entity]:
        """Lấy tất cả entities"""
        return list(self._entities.values())
    
    def clear(self) -> None:
        """Xóa tất cả entities"""
        self._entities.clear()
        for t in EntityType:
            self._entities_by_type[t].clear()
        self._ball = None
    
    @property
    def ball(self) -> Optional[Entity]:
        """Quick access to ball entity"""
        return self._ball
    
    # ==================== Board Tilt Control ====================
    
    def set_board_tilt(self, pitch: float, roll: float) -> None:
        """Set góc nghiêng bàn (clamped)"""
        self._board_pitch = np.clip(pitch, -self.board_config.max_tilt, self.board_config.max_tilt)
        self._board_roll = np.clip(roll, -self.board_config.max_tilt, self.board_config.max_tilt)
        
        self._event_bus.publish(
            GameEvent.TILT_CHANGED,
            pitch=self._board_pitch,
            roll=self._board_roll
        )
    
    @property
    def board_tilt(self) -> Tuple[float, float]:
        """Get current tilt (pitch, roll)"""
        return (self._board_pitch, self._board_roll)
    
    # ==================== Hybrid Movement ====================
    
    def apply_ball_force(self, direction: str) -> None:
        """
        Apply force vào ball theo hướng (2D).
        Direction: 'up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'none'
        """
        if self._ball is None or not self._ball.active:
            return
        
        ball_physics = self._ball.get_component(PhysicsComponent)
        if ball_physics is None:
            return
        
        # Get direction vector (2D)
        dir_vector = DIRECTIONS.get(direction, DIRECTIONS['none'])
        
        # Apply impulse directly to velocity (stronger, more responsive)
        impulse = dir_vector * self.board_config.force_magnitude * 0.1
        ball_physics.velocity += impulse
        
        # Clamp velocity
        speed = np.linalg.norm(ball_physics.velocity)
        if speed > self.board_config.max_velocity:
            ball_physics.velocity = ball_physics.velocity / speed * self.board_config.max_velocity
    
    def step_with_friction(self, dt: float = None) -> Dict:
        """
        Physics step với friction-based movement (không dùng tilt).
        Ball có momentum và dần dừng lại do friction.
        """
        if dt is None:
            dt = self.dt
        
        self._time += dt
        self._step_count += 1
        
        info = {
            'collisions': [],
            'triggers': [],
            'collections': []
        }
        
        if self._ball is None or not self._ball.active:
            return info
        
        ball_transform = self._ball.get_component(TransformComponent)
        ball_physics = self._ball.get_component(PhysicsComponent)
        ball_inventory = self._ball.get_component(InventoryComponent)
        
        if ball_transform is None or ball_physics is None:
            return info
        
        # 1. Apply friction (giảm velocity mỗi frame)
        ball_physics.velocity *= self.board_config.friction_coefficient
        
        # 2. Stop if very slow
        if np.linalg.norm(ball_physics.velocity) < 0.01:
            ball_physics.velocity = np.zeros(2)
        
        # 3. Update position
        new_position = ball_transform.position + ball_physics.velocity * dt
        
        # 4. Collision với walls
        new_position, collided = self._handle_wall_collisions(
            new_position, ball_physics, ball_transform.position
        )
        if collided:
            info['collisions'].append('wall')
        
        # 5. Collision với board bounds
        new_position = self._clamp_to_board(new_position, ball_physics.radius)
        
        # 6. Update position
        ball_transform.position = new_position
        
        # 7. Check triggers (holes, goal, teleports)
        trigger_info = self._check_triggers(ball_transform, ball_inventory)
        info['triggers'] = trigger_info
        
        # 8. Check collectibles (coins, keys)
        collect_info = self._check_collectibles(ball_transform, ball_inventory)
        info['collections'] = collect_info
        
        # Publish state update
        self._event_bus.publish(GameEvent.STATE_UPDATED, time=self._time, step=self._step_count)
        
        return info
    
    # ==================== Physics Update ====================
    
    def step(self, dt: float = None) -> Dict:
        """
        Update physics cho 1 timestep (tilt-based).
        Returns dict với info về các events xảy ra.
        """
        if dt is None:
            dt = self.dt
        
        self._time += dt
        self._step_count += 1
        
        info = {
            'collisions': [],
            'triggers': [],
            'collections': []
        }
        
        if self._ball is None or not self._ball.active:
            return info
        
        ball_transform = self._ball.get_component(TransformComponent)
        ball_physics = self._ball.get_component(PhysicsComponent)
        ball_inventory = self._ball.get_component(InventoryComponent)
        
        if ball_transform is None or ball_physics is None:
            return info
        
        # 1. Tính gravity dựa trên tilt (2D)
        # Khi bàn nghiêng, thành phần gravity chiếu xuống mặt bàn
        ax = self.gravity * np.sin(self._board_roll)   # Nghiêng trái-phải -> X
        ay = self.gravity * np.sin(self._board_pitch)  # Nghiêng trước-sau -> Y
        
        # 2. Apply gravity và friction
        ball_physics.acceleration[0] = ax
        ball_physics.acceleration[1] = ay
        
        # Friction (đơn giản)
        friction_force = -ball_physics.friction * ball_physics.velocity
        ball_physics.acceleration += friction_force / ball_physics.mass
        
        # 3. Semi-implicit Euler integration
        ball_physics.velocity += ball_physics.acceleration * dt
        new_position = ball_transform.position + ball_physics.velocity * dt
        
        # 4. Collision với walls
        new_position, collided = self._handle_wall_collisions(
            new_position, ball_physics, ball_transform.position
        )
        if collided:
            info['collisions'].append('wall')
        
        # 5. Collision với board bounds
        new_position = self._clamp_to_board(new_position, ball_physics.radius)
        
        # 6. Update position
        ball_transform.position = new_position
        
        # 7. Check triggers (holes, goal, teleports)
        trigger_info = self._check_triggers(ball_transform, ball_inventory)
        info['triggers'] = trigger_info
        
        # 8. Check collectibles (coins, keys)
        collect_info = self._check_collectibles(ball_transform, ball_inventory)
        info['collections'] = collect_info
        
        # 9. Reset acceleration cho frame tiếp theo
        ball_physics.acceleration = np.zeros(2)
        
        # Publish state update
        self._event_bus.publish(GameEvent.STATE_UPDATED, time=self._time, step=self._step_count)
        
        return info
    
    def _handle_wall_collisions(
        self, 
        new_pos: np.ndarray, 
        physics: PhysicsComponent,
        old_pos: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Xử lý va chạm với walls (2D AABB).
        Returns (adjusted_position, collided)
        """
        collided = False
        
        for wall in self._entities_by_type[EntityType.WALL]:
            if not wall.active:
                continue
            
            wall_transform = wall.get_component(TransformComponent)
            wall_collision = wall.get_component(CollisionComponent)
            
            if wall_transform is None or wall_collision is None:
                continue
            
            if not wall_collision.is_solid:
                continue
            
            # Simple AABB collision cho box walls (2D)
            if wall_collision.shape == 'box':
                wall_pos = wall_transform.position
                half_size = wall_collision.size
                ball_radius = physics.radius
                
                # Check overlap (2D)
                min_bound = wall_pos - half_size - ball_radius
                max_bound = wall_pos + half_size + ball_radius
                
                if (min_bound[0] <= new_pos[0] <= max_bound[0] and
                    min_bound[1] <= new_pos[1] <= max_bound[1]):
                    
                    collided = True
                    
                    # Push ball out và reflect velocity
                    # Tìm hướng đẩy ra ngắn nhất
                    penetrations = [
                        new_pos[0] - min_bound[0],  # left
                        max_bound[0] - new_pos[0],  # right
                        new_pos[1] - min_bound[1],  # bottom
                        max_bound[1] - new_pos[1],  # top
                    ]
                    min_idx = np.argmin(penetrations)
                    
                    if min_idx == 0:  # Push left
                        new_pos[0] = min_bound[0]
                        physics.velocity[0] = -physics.velocity[0] * physics.restitution
                    elif min_idx == 1:  # Push right
                        new_pos[0] = max_bound[0]
                        physics.velocity[0] = -physics.velocity[0] * physics.restitution
                    elif min_idx == 2:  # Push bottom
                        new_pos[1] = min_bound[1]
                        physics.velocity[1] = -physics.velocity[1] * physics.restitution
                    else:  # Push top
                        new_pos[1] = max_bound[1]
                        physics.velocity[1] = -physics.velocity[1] * physics.restitution
        
        return new_pos, collided
    
    def _clamp_to_board(self, pos: np.ndarray, radius: float) -> np.ndarray:
        """Giữ ball trong bounds của board (2D)"""
        half_w = self.board_config.width / 2 - radius
        half_h = self.board_config.height / 2 - radius
        
        pos[0] = np.clip(pos[0], -half_w, half_w)
        pos[1] = np.clip(pos[1], -half_h, half_h)
        
        return pos
    
    def _check_triggers(
        self, 
        ball_transform: TransformComponent,
        ball_inventory: InventoryComponent
    ) -> List[Dict]:
        """Check và xử lý triggers (holes, goal, teleports, locks) - 2D"""
        triggered = []
        ball_pos = ball_transform.position  # Already 2D
        
        # Check holes
        for hole in self._entities_by_type[EntityType.HOLE]:
            if not hole.active:
                continue
            hole_transform = hole.get_component(TransformComponent)
            hole_trigger = hole.get_component(TriggerComponent)
            
            if hole_transform is None or hole_trigger is None:
                continue
            
            dist = np.linalg.norm(ball_pos - hole_transform.position)
            if dist < hole_trigger.radius:
                success, reward = hole_trigger.trigger(self._time)
                if success:
                    triggered.append({
                        'type': 'hole',
                        'entity_id': hole.id,
                        'reward': reward
                    })
                    self._event_bus.publish(GameEvent.BALL_FELL, hole_id=hole.id)
        
        # Check goal
        for goal in self._entities_by_type[EntityType.GOAL]:
            if not goal.active:
                continue
            goal_transform = goal.get_component(TransformComponent)
            goal_trigger = goal.get_component(TriggerComponent)
            
            if goal_transform is None or goal_trigger is None:
                continue
            
            dist = np.linalg.norm(ball_pos - goal_transform.position)
            if dist < goal_trigger.radius:
                success, reward = goal_trigger.trigger(self._time)
                if success:
                    triggered.append({
                        'type': 'goal',
                        'entity_id': goal.id,
                        'reward': reward
                    })
                    self._event_bus.publish(GameEvent.GOAL_REACHED)
        
        # Check teleports
        for teleport in self._entities_by_type[EntityType.TELEPORT]:
            if not teleport.active:
                continue
            tp_transform = teleport.get_component(TransformComponent)
            tp_trigger = teleport.get_component(TriggerComponent)
            
            if tp_transform is None or tp_trigger is None:
                continue
            
            dist = np.linalg.norm(ball_pos - tp_transform.position)
            if dist < tp_trigger.radius and tp_trigger.can_trigger(self._time):
                if tp_trigger.target_position is not None:
                    success, reward = tp_trigger.trigger(self._time)
                    if success:
                        # Teleport ball
                        ball_transform.position = tp_trigger.target_position.copy()
                        triggered.append({
                            'type': 'teleport',
                            'entity_id': teleport.id,
                            'target': tp_trigger.target_position.tolist()
                        })
                        self._event_bus.publish(
                            GameEvent.TELEPORT_ACTIVATED,
                            from_id=teleport.id
                        )
        
        # Check locks
        for lock in self._entities_by_type[EntityType.LOCK]:
            if not lock.active:
                continue
            lock_transform = lock.get_component(TransformComponent)
            lock_trigger = lock.get_component(TriggerComponent)
            
            if lock_transform is None or lock_trigger is None:
                continue
            
            if lock_trigger.is_unlocked:
                continue
            
            dist = np.linalg.norm(ball_pos - lock_transform.position)
            if dist < lock_trigger.radius:
                # Check if player has key
                if ball_inventory and lock_trigger.required_key_id:
                    if ball_inventory.has_key(lock_trigger.required_key_id):
                        ball_inventory.use_key(lock_trigger.required_key_id)
                        lock_trigger.unlock()
                        triggered.append({
                            'type': 'lock_opened',
                            'entity_id': lock.id,
                            'key_id': lock_trigger.required_key_id
                        })
                        self._event_bus.publish(
                            GameEvent.LOCK_OPENED,
                            lock_id=lock.id
                        )
        
        return triggered
    
    def _check_collectibles(
        self,
        ball_transform: TransformComponent,
        ball_inventory: InventoryComponent
    ) -> List[Dict]:
        """Check và thu thập coins/keys (2D)"""
        collected = []
        ball_pos = ball_transform.position  # Already 2D
        
        # Check coins
        for coin in self._entities_by_type[EntityType.COIN]:
            if not coin.active:
                continue
            coin_transform = coin.get_component(TransformComponent)
            coin_collect = coin.get_component(CollectibleComponent)
            
            if coin_transform is None or coin_collect is None:
                continue
            
            dist = np.linalg.norm(ball_pos - coin_transform.position)
            if dist < 0.5:  # Collection radius
                value = coin_collect.collect()
                if value > 0 and ball_inventory:
                    ball_inventory.add_coin(value)
                    collected.append({
                        'type': 'coin',
                        'entity_id': coin.id,
                        'value': value
                    })
                    self._event_bus.publish(
                        GameEvent.COIN_COLLECTED,
                        coin_id=coin.id,
                        value=value
                    )
        
        # Check keys
        for key in self._entities_by_type[EntityType.KEY]:
            if not key.active:
                continue
            key_transform = key.get_component(TransformComponent)
            key_collect = key.get_component(CollectibleComponent)
            
            if key_transform is None or key_collect is None:
                continue
            
            dist = np.linalg.norm(ball_pos - key_transform.position)
            if dist < 0.5:
                key_id = key_collect.key_id
                key_collect.collect()
                if ball_inventory and key_id:
                    ball_inventory.add_key(key_id)
                    collected.append({
                        'type': 'key',
                        'entity_id': key.id,
                        'key_id': key_id
                    })
                    self._event_bus.publish(
                        GameEvent.KEY_COLLECTED,
                        key_id=key_id
                    )
        
        return collected
    
    # ==================== State Serialization ====================
    
    def get_state_snapshot(self) -> Dict:
        """
        Serialize toàn bộ world state cho frontend / observation (2D).
        """
        ball_data = None
        if self._ball and self._ball.active:
            ball_transform = self._ball.get_component(TransformComponent)
            ball_physics = self._ball.get_component(PhysicsComponent)
            ball_inventory = self._ball.get_component(InventoryComponent)
            
            ball_data = {
                'position': ball_transform.position.tolist() if ball_transform else [0, 0],
                'velocity': ball_physics.velocity.tolist() if ball_physics else [0, 0],
                'radius': ball_physics.radius if ball_physics else 0.5,
                'inventory': ball_inventory.to_dict() if ball_inventory else {}
            }
        
        return {
            'time': self._time,
            'step': self._step_count,
            'board': {
                'pitch': self._board_pitch,
                'roll': self._board_roll,
                'width': self.board_config.width,
                'height': self.board_config.height
            },
            'ball': ball_data,
            'entities': [
                e.to_dict() for e in self._entities.values() 
                if e.active and e.entity_type != EntityType.BALL
            ]
        }
    
    @property
    def time(self) -> float:
        return self._time
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def reset_time(self) -> None:
        """Reset time và step count"""
        self._time = 0.0
        self._step_count = 0
