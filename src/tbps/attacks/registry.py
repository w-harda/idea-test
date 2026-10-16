"""可扩展注册表；发现不实例化，工厂必须显式收到运行依赖。"""
from .base import PluginInfo, need

class AttackRegistry:
    def __init__(self): self._entries = {}
    def register(self, info, factory):
        need(type(info) is PluginInfo and callable(factory), "descriptor/factory required")
        need(info.spec.id not in self._entries, "duplicate attack plugin")
        self._entries[info.spec.id] = (info,factory)
    def discover(self): return tuple(self._entries[k][0] for k in sorted(self._entries))
    def describe(self, name):
        need(name in self._entries, "unimplemented Attack Plugin: "+str(name))
        return self._entries[name][0]
    def create(self, name, **dependencies):
        info = self.describe(name)
        result = self._entries[name][1](**dependencies)
        need(result.info == info, "factory capability mismatch")
        return result
    def require_preparation(self, name):
        need(self.describe(name).offline_preparation_executable,
             "offline preparation execution is absent; read-only provenance audit only")

def default_registry():
    from .gv2 import GV2Plugin, INFO as G
    from .hv1 import HV1Plugin, INFO as H
    from .joint import JointPlugin, INFO as J
    registry = AttackRegistry()
    for info,factory in ((G,GV2Plugin),(H,HV1Plugin),(J,JointPlugin)): registry.register(info,factory)
    return registry
