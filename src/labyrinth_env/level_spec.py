"""
LevelSpec - Level definition và loader
Load level từ JSON files, validate và spawn entities
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import numpy as np

from .core.entity import EntityFactory
from .core.world import World, BoardConfig


@dataclass
class LevelSpec:
    """
    Specification cho 1 level.
    Có thể serialize/deserialize từ JSON.
    """
    id: str
    name: str
    difficulty: int = 1  # 1-10
    
    # Board config
    board_width: float = 10.0
    board_height: float = 10.0
    max_tilt: float = 0.15
    
    # Ball start position
    ball_start: tuple = (0.0, 0.0)  # (x, z) on board
    ball_radius: float = 0.3
    
    # Goal position
    goal_position: tuple = (0.0, 0.0)
    goal_radius: float = 0.5
    
    # Entities lists
    walls: List[Dict] = field(default_factory=list)
    holes: List[Dict] = field(default_factory=list)
    coins: List[Dict] = field(default_factory=list)
    keys: List[Dict] = field(default_factory=list)
    locks: List[Dict] = field(default_factory=list)
    teleports: List[Dict] = field(default_factory=list)
    
    # Time limit (seconds), 0 = no limit
    time_limit: float = 0.0
    
    # Reward config
    goal_reward: float = 1000.0
    hole_penalty: float = -100.0
    coin_value: int = 100
    time_penalty_per_step: float = -0.1
    
    # Hints/description
    description: str = ""
    hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            'id': self.id,
            'name': self.name,
            'difficulty': self.difficulty,
            'board': {
                'width': self.board_width,
                'height': self.board_height,
                'max_tilt': self.max_tilt
            },
            'ball': {
                'start': list(self.ball_start),
                'radius': self.ball_radius
            },
            'goal': {
                'position': list(self.goal_position),
                'radius': self.goal_radius,
                'reward': self.goal_reward
            },
            'walls': self.walls,
            'holes': self.holes,
            'coins': self.coins,
            'keys': self.keys,
            'locks': self.locks,
            'teleports': self.teleports,
            'time_limit': self.time_limit,
            'rewards': {
                'goal': self.goal_reward,
                'hole_penalty': self.hole_penalty,
                'coin_value': self.coin_value,
                'time_penalty': self.time_penalty_per_step
            },
            'description': self.description,
            'hints': self.hints
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LevelSpec':
        """Deserialize from dict"""
        board = data.get('board', {})
        ball = data.get('ball', {})
        goal = data.get('goal', {})
        rewards = data.get('rewards', {})
        
        return cls(
            id=data.get('id', 'unknown'),
            name=data.get('name', 'Unknown Level'),
            difficulty=data.get('difficulty', 1),
            board_width=board.get('width', 10.0),
            board_height=board.get('height', 10.0),
            max_tilt=board.get('max_tilt', 0.15),
            ball_start=tuple(ball.get('start', [0, 0])),
            ball_radius=ball.get('radius', 0.3),
            goal_position=tuple(goal.get('position', [0, 0])),
            goal_radius=goal.get('radius', 0.5),
            walls=data.get('walls', []),
            holes=data.get('holes', []),
            coins=data.get('coins', []),
            keys=data.get('keys', []),
            locks=data.get('locks', []),
            teleports=data.get('teleports', []),
            time_limit=data.get('time_limit', 0.0),
            goal_reward=rewards.get('goal', goal.get('reward', 1000.0)),
            hole_penalty=rewards.get('hole_penalty', -100.0),
            coin_value=rewards.get('coin_value', 100),
            time_penalty_per_step=rewards.get('time_penalty', -0.1),
            description=data.get('description', ''),
            hints=data.get('hints', [])
        )
    
    def save(self, filepath: str) -> None:
        """Save to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'LevelSpec':
        """Load from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class LevelLoader:
    """
    Load levels và spawn entities vào World.
    Quản lý curriculum (danh sách levels theo thứ tự difficulty).
    """
    
    def __init__(self, levels_dir: str = None):
        self.levels_dir = Path(levels_dir) if levels_dir else None
        self._levels_cache: Dict[str, LevelSpec] = {}
        self._curriculum: List[str] = []  # Level IDs theo thứ tự
    
    def register_level(self, spec: LevelSpec) -> None:
        """Đăng ký level spec"""
        self._levels_cache[spec.id] = spec
    
    def load_level_file(self, filepath: str) -> LevelSpec:
        """Load level từ file và cache"""
        spec = LevelSpec.load(filepath)
        self._levels_cache[spec.id] = spec
        return spec
    
    def load_all_from_directory(self) -> List[LevelSpec]:
        """Load tất cả levels từ thư mục"""
        if self.levels_dir is None or not self.levels_dir.exists():
            return []
        
        specs = []
        for json_file in sorted(self.levels_dir.glob('*.json')):
            try:
                spec = self.load_level_file(str(json_file))
                specs.append(spec)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Sort by difficulty
        specs.sort(key=lambda s: (s.difficulty, s.id))
        self._curriculum = [s.id for s in specs]
        
        return specs
    
    def get_level(self, level_id: str) -> Optional[LevelSpec]:
        """Lấy level spec theo ID"""
        return self._levels_cache.get(level_id)
    
    def get_curriculum(self) -> List[str]:
        """Lấy danh sách level IDs theo thứ tự curriculum"""
        return self._curriculum.copy()
    
    def set_curriculum(self, level_ids: List[str]) -> None:
        """Set custom curriculum"""
        self._curriculum = level_ids.copy()
    
    def spawn_level(self, level_id: str, world: World) -> bool:
        """
        Spawn tất cả entities của level vào world.
        Returns True nếu thành công.
        """
        spec = self.get_level(level_id)
        if spec is None:
            return False
        
        return self.spawn_from_spec(spec, world)
    
    def spawn_from_spec(self, spec: LevelSpec, world: World) -> bool:
        """Spawn entities từ LevelSpec vào World"""
        # Clear world
        world.clear()
        world.reset_time()
        
        # Update board config
        world.board_config.width = spec.board_width
        world.board_config.height = spec.board_height
        world.board_config.max_tilt = spec.max_tilt
        
        # Spawn ball
        ball = EntityFactory.create(
            'ball',
            position=(spec.ball_start[0], spec.ball_radius, spec.ball_start[1]),
            radius=spec.ball_radius
        )
        world.add_entity(ball)
        
        # Spawn goal
        goal = EntityFactory.create(
            'goal',
            position=(spec.goal_position[0], 0.0, spec.goal_position[1]),
            radius=spec.goal_radius,
            reward=spec.goal_reward
        )
        world.add_entity(goal)
        
        # Spawn walls
        for wall_data in spec.walls:
            pos = wall_data.get('position', [0, 0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0.5, pos[1]]
            size = wall_data.get('size', [0.5, 0.5, 0.5])
            
            wall = EntityFactory.create('wall', position=tuple(pos), size=tuple(size))
            world.add_entity(wall)
        
        # Spawn holes
        for hole_data in spec.holes:
            pos = hole_data.get('position', [0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0, pos[1]]
            radius = hole_data.get('radius', 0.4)
            
            hole = EntityFactory.create(
                'hole',
                position=tuple(pos),
                radius=radius,
                penalty=spec.hole_penalty
            )
            world.add_entity(hole)
        
        # Spawn coins
        for coin_data in spec.coins:
            pos = coin_data.get('position', [0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0.2, pos[1]]
            value = coin_data.get('value', spec.coin_value)
            
            coin = EntityFactory.create('coin', position=tuple(pos), value=value)
            world.add_entity(coin)
        
        # Spawn keys
        for key_data in spec.keys:
            pos = key_data.get('position', [0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0.2, pos[1]]
            key_id = key_data.get('key_id', 'key_1')
            
            key = EntityFactory.create('key', position=tuple(pos), key_id=key_id)
            world.add_entity(key)
        
        # Spawn locks
        for lock_data in spec.locks:
            pos = lock_data.get('position', [0, 0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0.5, pos[1]]
            required_key = lock_data.get('required_key_id', 'key_1')
            size = lock_data.get('size', [0.5, 0.5, 0.5])
            
            lock = EntityFactory.create(
                'lock',
                position=tuple(pos),
                required_key_id=required_key,
                size=tuple(size)
            )
            world.add_entity(lock)
        
        # Spawn teleports (pairs)
        for tp_data in spec.teleports:
            pos = tp_data.get('position', [0, 0])
            if len(pos) == 2:
                pos = [pos[0], 0.0, pos[1]]
            target = tp_data.get('target', [0, 0])
            if len(target) == 2:
                target = [target[0], 0.3, target[1]]
            pair_id = tp_data.get('pair_id', None)
            radius = tp_data.get('radius', 0.4)
            cooldown = tp_data.get('cooldown', 1.5)
            
            teleport = EntityFactory.create(
                'teleport',
                position=tuple(pos),
                target_position=tuple(target),
                pair_id=pair_id,
                radius=radius,
                cooldown=cooldown
            )
            world.add_entity(teleport)
        
        return True


# ==================== Built-in Levels ====================

def create_tutorial_level() -> LevelSpec:
    """Level 1: Tutorial - straight path to goal"""
    return LevelSpec(
        id='level_01_tutorial',
        name='Tutorial',
        difficulty=1,
        board_width=8.0,
        board_height=8.0,
        ball_start=(-3.0, 0.0),
        goal_position=(3.0, 0.0),
        goal_radius=0.6,
        walls=[
            {'position': [0, -3], 'size': [4.0, 0.5, 0.2]},
            {'position': [0, 3], 'size': [4.0, 0.5, 0.2]},
        ],
        description='Đưa bi vào lỗ. Dùng WASD hoặc phím mũi tên để nghiêng bàn.',
        hints=['Nghiêng bàn sang phải để bi lăn về phía goal']
    )


def create_simple_maze_level() -> LevelSpec:
    """Level 2: Simple maze with one turn"""
    return LevelSpec(
        id='level_02_simple_maze',
        name='Simple Maze',
        difficulty=2,
        board_width=10.0,
        board_height=10.0,
        ball_start=(-4.0, -4.0),
        goal_position=(4.0, 4.0),
        walls=[
            # Horizontal walls
            {'position': [0, -2], 'size': [3.0, 0.5, 0.2]},
            {'position': [0, 2], 'size': [3.0, 0.5, 0.2]},
            # Vertical walls
            {'position': [-2, 0], 'size': [0.2, 0.5, 2.0]},
            {'position': [2, 0], 'size': [0.2, 0.5, 2.0]},
        ],
        coins=[
            {'position': [0, 0], 'value': 100},
        ],
        description='Đi qua maze đơn giản để đến goal.',
    )


def create_holes_level() -> LevelSpec:
    """Level 3: Avoid holes"""
    return LevelSpec(
        id='level_03_holes',
        name='Watch Your Step',
        difficulty=3,
        board_width=10.0,
        board_height=10.0,
        ball_start=(-4.0, 0.0),
        goal_position=(4.0, 0.0),
        holes=[
            {'position': [-2, 0], 'radius': 0.5},
            {'position': [0, 1], 'radius': 0.4},
            {'position': [0, -1], 'radius': 0.4},
            {'position': [2, 0], 'radius': 0.5},
        ],
        coins=[
            {'position': [-1, 0], 'value': 150},
            {'position': [1, 0], 'value': 150},
        ],
        description='Tránh các hố trên đường đi!',
        hints=['Đi chậm và cẩn thận qua các hố']
    )


def create_key_lock_level() -> LevelSpec:
    """Level 4: Key and Lock mechanics"""
    return LevelSpec(
        id='level_04_key_lock',
        name='Locked Door',
        difficulty=4,
        board_width=10.0,
        board_height=10.0,
        ball_start=(-4.0, -4.0),
        goal_position=(4.0, 4.0),
        walls=[
            {'position': [2, 0], 'size': [0.2, 0.5, 4.0]},
        ],
        keys=[
            {'position': [-2, 2], 'key_id': 'gold_key'},
        ],
        locks=[
            {'position': [2, 2], 'required_key_id': 'gold_key', 'size': [0.3, 0.5, 0.5]},
        ],
        coins=[
            {'position': [0, 0], 'value': 100},
            {'position': [-2, -2], 'value': 100},
        ],
        description='Tìm chìa khóa để mở cửa.',
    )


def create_teleport_level() -> LevelSpec:
    """Level 5: Teleport introduction"""
    return LevelSpec(
        id='level_05_teleport',
        name='Portal',
        difficulty=5,
        board_width=12.0,
        board_height=12.0,
        ball_start=(-5.0, -5.0),
        goal_position=(5.0, 5.0),
        walls=[
            # Barrier in the middle
            {'position': [0, 0], 'size': [5.0, 0.5, 0.3]},
        ],
        teleports=[
            {'position': [-3, -2], 'target': [3, 2], 'pair_id': 'tp1'},
            {'position': [3, 2], 'target': [-3, -2], 'pair_id': 'tp1'},
        ],
        coins=[
            {'position': [-3, 3], 'value': 200},
            {'position': [3, -3], 'value': 200},
        ],
        description='Sử dụng portal để vượt qua chướng ngại vật.',
    )


def get_builtin_levels() -> List[LevelSpec]:
    """Trả về danh sách built-in levels theo curriculum"""
    return [
        create_tutorial_level(),
        create_simple_maze_level(),
        create_holes_level(),
        create_key_lock_level(),
        create_teleport_level(),
    ]
