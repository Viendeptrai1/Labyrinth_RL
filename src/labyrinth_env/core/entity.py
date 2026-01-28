"""
Entity base class - ECS-lite pattern
Mỗi game object là 1 Entity chứa các Component
"""
from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Type, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import uuid

if TYPE_CHECKING:
    from .components import Component


class EntityType(Enum):
    """Các loại entity trong game"""
    BALL = auto()
    WALL = auto()
    HOLE = auto()
    COIN = auto()
    KEY = auto()
    LOCK = auto()
    TELEPORT = auto()
    GOAL = auto()
    BOARD = auto()  # Tấm gỗ chính


@dataclass
class Entity:
    """
    Base Entity class theo ECS-lite pattern.
    Entity chỉ là container cho các Component.
    """
    entity_type: EntityType
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    active: bool = True
    _components: Dict[Type['Component'], 'Component'] = field(default_factory=dict)
    
    def add_component(self, component: 'Component') -> 'Entity':
        """Thêm component vào entity (fluent interface)"""
        component.entity = self
        self._components[type(component)] = component
        return self
    
    def get_component(self, component_type: Type['Component']) -> Optional['Component']:
        """Lấy component theo type"""
        return self._components.get(component_type)
    
    def has_component(self, component_type: Type['Component']) -> bool:
        """Kiểm tra entity có component không"""
        return component_type in self._components
    
    def remove_component(self, component_type: Type['Component']) -> None:
        """Xóa component"""
        if component_type in self._components:
            del self._components[component_type]
    
    @property
    def components(self):
        """Iterator qua tất cả components"""
        return self._components.values()
    
    def to_dict(self) -> dict:
        """Serialize entity để gửi sang frontend"""
        return {
            'id': self.id,
            'type': self.entity_type.name,
            'active': self.active,
            'components': {
                comp_type.__name__: comp.to_dict() 
                for comp_type, comp in self._components.items()
            }
        }


class EntityFactory:
    """
    Factory + Registry pattern để tạo entities từ config.
    Đăng ký builder cho từng entity type.
    """
    _builders: Dict[str, callable] = {}
    
    @classmethod
    def register(cls, entity_type: str):
        """Decorator để đăng ký builder"""
        def decorator(builder_func):
            cls._builders[entity_type] = builder_func
            return builder_func
        return decorator
    
    @classmethod
    def create(cls, entity_type: str, **config) -> Entity:
        """Tạo entity từ type và config"""
        if entity_type not in cls._builders:
            raise ValueError(f"Unknown entity type: {entity_type}")
        return cls._builders[entity_type](**config)
    
    @classmethod
    def available_types(cls) -> list:
        """Danh sách entity types đã đăng ký"""
        return list(cls._builders.keys())
