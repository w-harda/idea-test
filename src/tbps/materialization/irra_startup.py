"""IRRA启动时序：真实CPU读取上下文退出并核验后，才允许CUDA与模型加载。"""
from dataclasses import dataclass
import sys
from .contracts import need

def isolation_snapshot():
    import torch
    import importlib.machinery
    return ((torch.load,torch.nn.Module.__init__,torch.nn.Module._call_impl,
             torch.cuda.init,torch.cuda._lazy_init,torch.autograd.grad,torch.autograd.backward,
             importlib.machinery.SourceFileLoader.get_code),
            tuple(sys.path),tuple(sys.meta_path),sys.dont_write_bytecode)

@dataclass(frozen=True)
class IRRAInputs:
    historical: dict
    feature_refs: tuple
    images: list
    texts: list
    source: object
    store: object

class CPUStartupGate:
    def __init__(self,emit=None):
        import torch
        from ..attacks.legacy_runtime import _LOCK
        need(not torch.cuda.is_initialized() and not _LOCK.locked(),"isolated CPU startup required")
        self.before=isolation_snapshot();self.completed=False;self.exited=False
        self.emit=emit or (lambda event:None)
    def checked(self):
        import torch
        need(not torch.cuda.is_initialized(),"CUDA initialized during CPU source validation")
        self.completed=True;self.emit("cpu_sources_verified")
    def exit_checked(self):
        import torch
        from ..attacks.legacy_runtime import _LOCK
        need(self.completed and not torch.cuda.is_initialized(),"CPU source phase incomplete or CUDA initialized early")
        need(isolation_snapshot()==self.before and not _LOCK.locked(),"CPU isolation contexts have not exited/restored")
        self.exited=True;self.emit("cpu_isolation_exited")
    def initialize_cuda(self):
        import torch
        from ..attacks.legacy_runtime import _LOCK
        need(self.completed and self.exited and isolation_snapshot()==self.before and not _LOCK.locked(),
             "CPU contexts must exit before CUDA initialization")
        need(not torch.cuda.is_initialized(),"CUDA initialized before startup gate")
        self.emit("cuda_initialization_call")
        torch.cuda.reset_peak_memory_stats()

def feature_paths(c,store):
    """白名单只允许本两样本物化RGB及对应历史六个紧凑批次。"""
    from .smoke_support import GALLERY,QUERY
    from ..engines.historical_sources import JointFullSources
    sources=JointFullSources(c["replay_config"]["path"])
    paths={store.reference(store.plan.binding(key,"image")).data.path for key in GALLERY}
    qidx=[next(i for i,q in enumerate(sources.manifest.queries) if q.query_id==key) for key in QUERY]
    for family,indices,size in (("clean_gallery",(0,15),64),("image_gallery",(0,15),64),
                               ("clean_text",qidx,sources.protocol["text_batch"]),
                               ("adv_text",qidx,sources.protocol["text_batch"])):
        paths.update(str(sources.root/f"features/{family}/{i//size:06d}/features.pt") for i in indices)
    return paths

def prepare_inputs(c,output,*,store_factory,feature_reader,emit=None):
    import torch
    from ..engines.historical_loader import feature_only_cpu,native_action_validator
    from ..models.irra_source import IRRASource
    from ..artifacts import read_json
    from .smoke_support import GALLERY,QUERY
    gate=CPUStartupGate(emit);gate.emit("cpu_read_started")
    store,legacy=store_factory(c,output)
    allowed=feature_paths(c,store)
    # 整个真实读取流程受限；内层原CPU/动作/特征门禁保留，不接受模型文件。
    with feature_only_cpu(allowed):
        historical,feature_refs=feature_reader(c)
        images=[store.load(store.reference(store.plan.binding(key,"image")))[0] for key in GALLERY]
        with native_action_validator(legacy.config_ref) as validate:
            texts=[store.load(store.reference(store.plan.binding(key,"text")),validate=validate)[1] for key in QUERY]
        source=IRRASource.from_dict(read_json(c["dataset_sources"]["path"])["irra"])
        source.verify();store.verify_plan(force=True)
        # Loader负责批次/来源；这里检查进入真实编码的选中行，不能信任Mock接口的形状。
        need(set(historical)=={"clean_gallery","image_gallery","clean_text","adv_text"},"four historical families required")
        for x in historical.values():
            need(x.shape==(2,512) and x.dtype==torch.float32 and x.device.type=="cpu"
                 and not x.requires_grad and bool(torch.isfinite(x).all()),"selected historical feature contract")
        gate.checked()
    # 显式核验包括内层cpu_only/feature_only_cpu/native_action_validator的恢复状态。
    gate.exit_checked()
    return IRRAInputs(historical,feature_refs,images,texts,source,store),gate
