"""合成CPU后端；复用旧纯推理/约束函数，不构造真实G/H网络或加载权重。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import torch
from torch import nn
from tbps.artifacts import sha_file
from tbps.contracts import FileRef, PublicImageInput, PublicTextInput
from tbps.attacks.base import AttackBinding, ImageRequest, TextRequest
from tbps.attacks.provenance import AttackProvenance
from tbps.attacks.legacy_runtime import legacy_imports

H="1"*64
ROOT=Path(__file__).resolve().parents[2]

def support():
    config=json.loads((ROOT/"configs/framework/stage3a_plugins.server.json").read_text())
    legacy=Path(config["legacy_root"])
    pins={str(ROOT/Path(p).relative_to(legacy)):sha_file(ROOT/Path(p).relative_to(legacy))
          for p in config["runtime_sources"] if Path(p).is_relative_to(legacy)}
    return SimpleNamespace(root=ROOT,config={"runtime_sources":pins})

def provenance(name):
    return AttackProvenance(name,FileRef("/fixture/no-real-weights.pt",H),H,H,H,
                            "cuhk_pedes:attack_train:2048","cpu_fixture",H if name=="hv1" else None)

def image_request(root, sample_id="image-1", value=b"synthetic image bytes"):
    path=Path(root)/"fixture-image.bin";path.write_bytes(value)
    ref=FileRef(str(path),sha_file(path))
    return ImageRequest(PublicImageInput(ref,"rgb_0_1"),AttackBinding(H,sample_id,"image",ref.sha256))

def text_request(caption="red shirt",sample_id="query-1"):
    return TextRequest(PublicTextInput(caption),AttackBinding(H,sample_id,"text",hashlib.sha256(caption.encode()).hexdigest()))

class MockG(nn.Module):
    def __init__(self, amplitude=.03):
        super().__init__();self.amplitude=nn.Parameter(torch.tensor(amplitude))
        self.register_buffer("marker",torch.ones(1));self.mutate=False
    def forward(self, pixels):
        if self.mutate:self.marker.add_(1)
        return torch.ones_like(pixels)*self.amplitude

class GBackend:
    execution_kind="cpu_fixture"
    preprocess_id="cpu-fixture: synthetic RGB pixels"
    def __init__(self, module, pixels=None):
        self.model=MockG().eval().requires_grad_(False)
        self.pixels=pixels if pixels is not None else torch.linspace(0,1,192).reshape(1,3,8,8)
        self.generate_once=module.generate_once
        self.reads=[]
    def load_pixels(self, path):
        self.reads.append(path);return self.pixels.clone()

class ByteTokenizer:
    encoder={"<|startoftext|>":49406,"<|endoftext|>":49407}
    def encode(self,text):return list(text.encode("utf-8"))

class Frontend:
    def __init__(self, contracts, no_attrs=False, spans=None):
        self.contracts,self.no_attrs,self.spans=contracts,no_attrs,spans
        self.seen=[]
    def convert(self,caption):
        assert type(caption) is str
        self.seen.append(caption)
        C=self.contracts
        spans=self.spans if self.spans is not None else (
            (("color","red","upper","red",0,3),("garment_type","shirt","upper","shirt",4,9))
            if caption.startswith("red shirt") and not self.no_attrs else ())
        example=C.TextExample(caption,tuple(C.AttributeSpan(*a) for a in spans))
        return example,{"fixture":True}
    def __call__(self,caption):return self.convert(caption)[0]

class MockH(nn.Module):
    def __init__(self, modules, probe, mode="edit"):
        super().__init__();self.anchor=nn.Parameter(torch.tensor(1.))
        self.modules_=modules;self.probe=probe;self.mode=mode;self.mutate=False
        self.register_buffer("marker",torch.ones(1))
    def forward(self,example):
        if self.mutate:self.marker.add_(1)
        return self.anchor*0
    def attack(self,example):
        self(example)  # 只运行Mock，不实例化旧CharacterPolicy。
        constraints,policy=self.modules_
        k=policy.natural_k([1.]*len(example.attributes))
        edits=[];selected=[];decisions=[]
        if self.mode=="edit":
            attr=example.attributes[0]
            pos=next(p for p in range(attr.char_start,attr.char_end)
                     if constraints.CONFUSABLES.get(example.caption[p]))
            edits=[constraints.make_edit(example,0,pos,constraints.CONFUSABLES[example.caption[pos]][0])]
            selected=[attr.attribute_id]
        elif self.mode=="stop":
            decisions=[{"head":"attribute","index":len(example.attributes),"legal_indices":list(range(len(example.attributes)+1))}]
        elif self.mode=="position_noop":
            selected=[example.attributes[0].attribute_id]
            decisions=[{"head":"position","index":len(example.attributes[0].raw_text),"legal_indices":[len(example.attributes[0].raw_text)]}]
        adv=constraints.replay(example,edits,k)
        record={"original":example.caption,"adversarial":adv,"budget_k":k,
                "predicted_importance":tuple([1.]*len(example.attributes)),
                "attributes":tuple(asdict(a) for a in example.attributes),
                "selected_attribute_ids":tuple(selected),"edits":[asdict(e) for e in edits],
                "decisions":tuple(decisions),"tokenizer_audit":self.probe.pair_report(example.caption,adv)}
        return SimpleNamespace(record=lambda:record)

class HBackend:
    execution_kind="cpu_fixture"
    def __init__(self, old, mode="edit", no_attrs=False, spans=None):
        self.module=old("phase5_joint.backend")
        self.contracts=old("contracts");self.constraints=old("constraints")
        self.policy_module=old("policy")
        self.probe=old("tokenizer_probe").TokenizerProbe(ByteTokenizer())
        self.frontend=Frontend(self.contracts,no_attrs,spans)
        self.model=MockH((self.constraints,self.policy_module),self.probe,mode).eval().requires_grad_(False)
        self.attacker=self.module.TextAttacker(self.model,self.frontend,self.probe)
        self.seen=[]
    def attack(self,caption):
        self.seen.append(caption);return self.attacker.attack(caption)
    def validate(self,caption,record):
        return self.module.validate_action(caption,record,self.probe,self.frontend)
