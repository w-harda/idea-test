"""Joint只有组件组合；Gallery/Query身份独立，不要求正确图文配对。"""
from dataclasses import dataclass
from ..config import ATTACKS
from ..contracts import AttackBudget
from .base import PluginInfo, need
from .gv2 import GV2Plugin
from .hv1 import HV1Plugin

INFO = PluginInfo(ATTACKS["joint"],frozenset({"train_pairs","train_pids","source_gradients","source_features","source_rewards"}),
                  frozenset({"public_images","captions","caption_attributes"}))

@dataclass(frozen=True, slots=True)
class JointOutput:
    mode: str
    image: object
    text: object

class JointPlugin:
    info=INFO
    def __init__(self, *, image_plugin, text_plugin):
        need(type(image_plugin) is GV2Plugin and type(text_plugin) is HV1Plugin,
             "Joint must compose existing G/H plugins")
        need(image_plugin.provenance.execution_kind==text_plugin.provenance.execution_kind,
             "cannot mix CPU fixtures and real components")
        need(image_plugin.provenance.surrogate_sha256==text_plugin.provenance.surrogate_sha256,
             "G/H training surrogate mismatch")
        need(image_plugin.provenance.training_dataset==text_plugin.provenance.training_dataset,
             "G/H training dataset mismatch")
        need(image_plugin.provenance.sources_sha256==text_plugin.provenance.sources_sha256,
             "G/H source protocols must share one verified source bundle")
        self.image_plugin,self.text_plugin=image_plugin,text_plugin
        self.budget=AttackBudget(image_plugin.budget.image,text_plugin.budget.text)
    def generate(self, *, mode, image=None, text=None):
        need(mode in ("image_only","text_only","joint"), "explicit supported mode required")
        need((image is not None)==(mode in ("image_only","joint")) and
             (text is not None)==(mode in ("text_only","joint")), "mode/input modalities mismatch")
        return JointOutput(mode,self.image_plugin.generate(image) if image is not None else None,
                           self.text_plugin.generate(text) if text is not None else None)
