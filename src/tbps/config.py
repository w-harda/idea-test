"""显式配置/权限校验及dry-run；描述符不是可执行插件。"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from .artifacts import read_json, json_bytes
from .contracts import (AttackSpec, ModelSpec, TokenizerSpec, AttackBudget, ImageBudget,
                        TextBudget, FileRef, ACCESS, count, text, sha256)

class ConfigError(ValueError): pass
class UnsupportedComponent(ConfigError): pass

# 仅已审计的历史身份描述；没有factory、模型导入或GPU入口。
ATTACKS = {
    "gv2": AttackSpec("gv2","historical-v2",frozenset({"image"}),"generator","offline",False,
                     frozenset({"train_pairs","source_gradients"}),frozenset({"public_images"}),"epoch"),
    "hv1": AttackSpec("hv1","historical-v1",frozenset({"text"}),"policy","scst",False,
                     frozenset({"train_pairs","train_pids","source_rewards"}),
                     frozenset({"captions","caption_attributes"}),"periodic"),
    "joint": AttackSpec("joint","historical-v1",frozenset({"image","text"}),"composite","multi_stage",False,
                       frozenset({"train_pairs","train_pids","source_gradients","source_rewards"}),
                       frozenset({"public_images","captions","caption_attributes"}),"component"),
    "ap_attack": AttackSpec("ap_attack","historical-baseline",frozenset({"image"}),"generator","offline",False,
                           frozenset({"source_gradients"}),frozenset({"public_images"}),"epoch"),
}
JOINT_COMPONENTS=("gv2","hv1")

IRRA = ModelSpec("irra","historical-official",(384,128),
    TokenizerSpec("IRRA Byte-BPE","19b06ade7532bcff961c4b90590197fcac0d466f08d0b6b801d3d19a36568d67",
                  77,"BPE token","retain_and_report"),512,"cosine","l2",
    frozenset({"image_encoding","text_encoding","tokenize","score","differentiable_images"}))

def exact(value, fields, label):
    if type(value) is not dict or set(value)!=set(fields):
        raise ConfigError(label+" fields must be exactly "+str(sorted(fields)))

def validate_access(mode, attack, surrogate, victim, surrogate_sha, victim_sha,
                    train_dataset, test_dataset, train_permissions, attack_permissions):
    """纯契约函数允许未来Adapter提供不同ModelSpec；不会注册或执行它们。"""
    modes={"white_box":(False,False),"cross_model":(True,False),
           "cross_dataset":(False,True),"cross_model_cross_dataset":(True,True)}
    if mode not in modes: raise ConfigError("explicit supported attack_mode required")
    components=tuple(ATTACKS[n] for n in JOINT_COMPONENTS) if attack.id=="joint" else (attack,)
    if attack.id=="joint":
        train_required=frozenset().union(*(a.train_requirements for a in components))
        infer_required=frozenset().union(*(a.attack_requirements for a in components))
        if not train_required<=attack.train_requirements or not infer_required<=attack.attack_requirements or attack.modalities!=frozenset({"image","text"}) or attack.training_kind!="multi_stage":
            raise ConfigError("incomplete Joint component capability declaration")
    different_model=(surrogate.id,surrogate_sha)!=(victim.id,victim_sha)
    different_data=train_dataset!=test_dataset
    if (different_model,different_data)!=modes[mode]:
        raise ConfigError("attack_mode conflicts with model/checkpoint or attack dataset axes")
    for p in (train_permissions,attack_permissions):
        if type(p) is not frozenset or not p<=ACCESS:raise ConfigError("unknown/mutable permission")
        if "test_labels" in p:raise ConfigError("Test labels are never strategy inputs")
    if not attack.train_requirements<=train_permissions or not attack.attack_requirements<=attack_permissions:
        raise ConfigError("required attack access not granted")
    if train_permissions & {"victim_gradients","victim_scores"}:
        raise ConfigError("training feedback must explicitly use surrogate")
    if different_model and attack_permissions & {"victim_gradients","victim_scores"}:
        raise ConfigError("transfer attack cannot access victim feedback")
    if attack.id in ("gv2","hv1","joint","ap_attack"):
        public=frozenset({"public_images","captions","caption_attributes"})
        if not attack_permissions<=public:raise ConfigError("historical generator/policy inference is public-input only")
    for phase_name,phase_permissions in (("train_requirements",train_permissions),("attack_requirements",attack_permissions)):
        for prefix,model in (("source",surrogate),("victim",victim)):
            if prefix+"_gradients" in phase_permissions:
                gradient_modalities=frozenset().union(*(a.modalities for a in components if prefix+"_gradients" in getattr(a,phase_name))) or attack.modalities
                required={"differentiable_"+("images" if modality=="image" else "text") for modality in gradient_modalities}
                if not required<=model.capabilities:raise ConfigError(prefix+" lacks declared modality gradients")
            if phase_permissions & {prefix+"_scores",prefix+"_rewards"} and "score" not in model.capabilities:
                raise ConfigError(prefix+" lacks declared scoring")
        if "source_features" in phase_permissions:
            required={modality+"_encoding" for modality in attack.modalities}
            if not required<=surrogate.capabilities:raise ConfigError("surrogate lacks modality feature encoding")

@dataclass(frozen=True, slots=True)
class ValidatedConfig:
    canonical: bytes
    attack: AttackSpec
    surrogate: ModelSpec
    victim: ModelSpec
    budget: AttackBudget
    def as_dict(self):
        return json.loads(self.canonical)
    def require_executable(self):
        if self.attack.id not in ("gv2","hv1","joint"):
            raise UnsupportedComponent("Attack Plugin is not implemented: "+self.attack.id)
        raise UnsupportedComponent("No Evaluation Engine; frozen G/H plugins use separate explicit-load APIs")
    def dry_run(self):
        data=self.as_dict()
        refs=[data[k] for k in ("attack_checkpoint","surrogate_checkpoint","victim_checkpoint")]
        refs += [data[k]["manifest"] for k in ("attack_train_dataset","attack_validation_dataset","attack_test_dataset")]
        missing=[r["path"] for r in refs if not Path(r["path"]).is_file()]
        return {"schema":"tbps-config-dry-run-1","configuration_valid":True,"executable":False,
                "asset_path_status":"NOT_READY" if missing else "PRESENT_NOT_SHA_VERIFIED",
                "missing":missing,"required_adapters":(["Attack"] if self.attack.id not in ("gv2","hv1","joint") else [])+["Evaluation Engine"],
                "available_adapters":{"dataset":["cuhk_pedes"],"model":["irra"],"attack":["gv2","hv1","joint"],"model_load":"explicit authorization only"},
                "attack":self.attack.id,"mode":data["attack_mode"],
                "note":"Only parsing/permissions; no model load, training, encoding or evaluation."}

def parse_config(data):
    required={"schema","attack_method","surrogate_model","victim_model","attack_train_dataset",
              "attack_validation_dataset","attack_test_dataset","attack_checkpoint","surrogate_checkpoint",
              "victim_checkpoint","attack_mode","attack_budget","seed","permissions"}
    exact(data,required,"configuration")
    if data["schema"]!="tbps-experiment-1":raise ConfigError("unsupported configuration schema")
    count(data["seed"],"seed",True)
    for k in ("attack_method","surrogate_model","victim_model"):text(data[k],k)
    if data["attack_method"] not in ATTACKS:raise UnsupportedComponent("unimplemented attack descriptor: "+data["attack_method"])
    if data["surrogate_model"]!="irra" or data["victim_model"]!="irra":
        raise UnsupportedComponent("Only IRRA is implemented; RaSa/APTM adapters are absent")
    attack=ATTACKS[data["attack_method"]]
    for key,role in (("attack_train_dataset","attack_train"),("attack_validation_dataset","attack_validation"),("attack_test_dataset","test")):
        d=data[key];exact(d,{"id","split","role","manifest"} ,key)
        if d["id"]!="cuhk_pedes":raise UnsupportedComponent("unimplemented dataset descriptor: "+str(d["id"]))
        if d["role"]!=role or d["split"]!=("test" if role=="test" else "train"):
            raise ConfigError("official split and internal role conflict")
        exact(d["manifest"],{"path","sha256"},"dataset manifest");FileRef(**d["manifest"])
    for key,model in (("surrogate_checkpoint",data["surrogate_model"]),("victim_checkpoint",data["victim_model"])):
        c=data[key];exact(c,{"path","sha256","model_id","model_training_dataset"},key)
        FileRef(c["path"],c["sha256"]);text(c["model_training_dataset"],"model weight training source")
        if c["model_id"]!=model:raise ConfigError("checkpoint/model mismatch")
    c=data["attack_checkpoint"]
    exact(c,{"path","sha256","attack_training_dataset","surrogate_model","surrogate_checkpoint_sha256",
             "selection_dataset","uses_test_selection"},"attack checkpoint")
    FileRef(c["path"],c["sha256"]);sha256(c["surrogate_checkpoint_sha256"])
    if c["uses_test_selection"] is not False:raise ConfigError("Test cannot select attack checkpoint")
    if c["attack_training_dataset"]!=data["attack_train_dataset"]["id"] or c["selection_dataset"]!=data["attack_validation_dataset"]["id"]:
        raise ConfigError("attack checkpoint dataset provenance mismatch")
    if c["surrogate_model"]!=data["surrogate_model"] or c["surrogate_checkpoint_sha256"]!=data["surrogate_checkpoint"]["sha256"]:
        raise ConfigError("attack/surrogate checkpoint provenance mismatch")
    b=data["attack_budget"];exact(b,{"image","text"},"attack_budget")
    image=None;text_budget=None
    if b["image"] is not None:
        exact(b["image"],{"norm","value","pixel_space"},"image budget");image=ImageBudget(**b["image"])
    if b["text"] is not None:
        exact(b["text"],{"unit","max_edits","selection_rule","constraints_id","per_attribute","per_shared_word"},"text budget")
        text_budget=TextBudget(**b["text"])
    budget=AttackBudget(image,text_budget)
    if (image is not None)!=("image" in attack.modalities) or (text_budget is not None)!=("text" in attack.modalities):
        raise ConfigError("budget modalities do not match attack")
    if image and (image.norm!="linf" or image.pixel_space!="rgb_0_1" or image.value>8/255):
        raise ConfigError("historical image attacks require rgb_0_1 linf <= 8/255")
    if text_budget and (text_budget.unit!="unicode_character" or text_budget.selection_rule!="H-natural-k-gamma3" or
                        text_budget.per_attribute!=1 or text_budget.per_shared_word!=1 or
                        text_budget.constraints_id!="hv1-visual-unicode-tokenizer-mask-v1"):
        raise ConfigError("historical H budget/constraint mismatch")
    permissions=data["permissions"];exact(permissions,{"train","attack_time"},"permissions")
    for values in permissions.values():
        if type(values) is not list or any(type(v) is not str for v in values) or len(set(values))!=len(values):
            raise ConfigError("permissions must be unique string arrays")
    validate_access(data["attack_mode"],attack,IRRA,IRRA,data["surrogate_checkpoint"]["sha256"],
                    data["victim_checkpoint"]["sha256"],data["attack_train_dataset"]["id"],
                    data["attack_test_dataset"]["id"],frozenset(permissions["train"]),frozenset(permissions["attack_time"]))
    return ValidatedConfig(json_bytes(data),attack,IRRA,IRRA,budget)

def main(argv=None):
    parser=argparse.ArgumentParser(description="CPU-only configuration dry-run; no execution option")
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    try:
        result=parse_config(read_json(args.config)).dry_run()
    except (ValueError,TypeError,OSError) as exc:
        parser.exit(2,"CONFIG_REJECTED "+str(exc)+"\n")
    print("CONFIG_VALID_DRY_RUN "+json.dumps(result,ensure_ascii=False,sort_keys=True))

if __name__=="__main__":main()
