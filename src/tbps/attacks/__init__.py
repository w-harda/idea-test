"""攻击插件：导入/发现仅描述，不加载旧模块、模型或执行攻击。"""
from .registry import AttackRegistry, default_registry
from .base import AttackBinding, ImageRequest, TextRequest, PluginInfo, AttackOutput
