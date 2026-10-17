"""全量G执行的资源与神经调用监督；不实现攻击算法。"""
from pathlib import Path
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import os
import signal
import subprocess
import time
from tbps.artifacts import ArtifactError
from tbps.engines.replay_control import ResourceGuard,ResourceLimit,rss_bytes
from tbps.materialization.full_prepare import fs_resources
from tbps.materialization.irra_only import ram
from tbps.materialization.contracts import need

class FullGuard(ResourceGuard):
    """实际可用系统RAM/磁盘/时间监测，无进程RSS硬上限。"""
    def __init__(self,c,root,*,memory=ram,sample=rss_bytes,clock=time.monotonic):
        self.c,self.root,self.memory=c,root,memory
        self.minimum_available=None;self.last_disk=0;self.disk={}
        super().__init__(0,c["max_seconds"],sample=sample,clock=clock)
    def check(self):
        self.peak=max(self.peak,self.sample())
        available=self.memory()["MemAvailable"]
        self.minimum_available=available if self.minimum_available is None else min(available,self.minimum_available)
        if available<self.c["system_ram_reserve_bytes"]:self.reason="system RAM safety reserve"
        if self.clock()-self.started>=self.seconds:self.reason="7200-second execution budget"
        if self.clock()-self.last_disk>=1 or not self.disk:
            self.disk=fs_resources(self.root);self.last_disk=self.clock()
        if self.disk["available_bytes"]<self.c["runtime_disk_reserve_bytes"]:self.reason="disk safety reserve"
        if self.disk["available_inodes"]<1000:self.reason="inode safety reserve"
        if self.reason:raise ResourceLimit(self.reason)
    def snapshot(self):
        return {"peak_rss_bytes":self.peak,"elapsed_seconds":self.clock()-self.started,
            "rss_hard_limit_bytes":None,"minimum_system_available_bytes":self.minimum_available,
            "max_seconds":self.seconds,"stop_reason":self.reason,"disk":dict(self.disk)}

def own_gpu_only():
    """运行中只允许本GPU子进程；不抢占、不终止其他任务。"""
    rows=subprocess.run(["nvidia-smi","-i","0","--query-compute-apps=pid",
        "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.splitlines()
    pids=[int(r.strip()) for r in rows if r.strip()]
    need(all(p==os.getpid() for p in pids),"another GPU compute process appeared; stop")
    return pids

@contextmanager
def no_optimization():
    import torch
    attempts={}
    def stop(name):
        def fail(*a,**kw):
            attempts[name]=attempts.get(name,0)+1
            raise ArtifactError("full generation forbids "+name)
        return fail
    with ExitStack() as stack:
        for owner,name in ((torch.Tensor,"backward"),(torch.autograd,"backward"),
            (torch.autograd,"grad"),(torch.autograd.functional,"jvp"),
            (torch.func,"jvp"),(torch.optim.Optimizer,"__init__")):
            stack.enter_context(patch.object(owner,name,new=stop(name)))
        yield attempts
        need(not attempts,"optimization attempted")

@contextmanager
def g_load_only(checkpoint,artifact_root,counts):
    """真实权重仅G10一次；像素读回与权重加载分开计数。"""
    import torch
    original=torch.load;root=Path(artifact_root).resolve()
    def load(path,*args,**kwargs):
        p=Path(path).resolve()
        if p==Path(checkpoint).resolve():
            need(counts["G_load_attempts"]==0,"one G10 weight deserialization only")
            counts["G_load_attempts"]+=1
        else:
            need(p.is_relative_to(root) and p.name=="pixels.pt" and not args
                 and kwargs=={"map_location":"cpu","weights_only":True},"other model/feature load forbidden")
            counts["pixel_tensor_loads"]+=1
        return original(path,*args,**kwargs)
    with patch.object(torch,"load",new=load):yield

class GObservation:
    def __init__(self,expected,old_pixels=None,on_mismatch=None,*,image_sources=None,preprocess_sources=None):
        self.expected=tuple(expected);self.started=0;self.completed=0;self.active=None
        self.old_pixels=old_pixels or {};self.comparisons={};self.records=[]
        self.current_started=None;self.on_mismatch=on_mismatch
        self.image_sources=image_sources;self.preprocess_sources=preprocess_sources
        self.preprocess_calls=0
    def before(self,module,args):
        import torch
        need(self.active is not None and len(args)==1 and len(args[0])==1,"one bound Gallery image")
        need(torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),"G inference-only")
        need(self.started<len(self.expected) and self.active==self.expected[self.started],"actual G order/limit")
        self.started+=1
    def after(self,module,args,result):
        self.completed+=1
    @contextmanager
    def attach(self,plugin):
        import torch
        from tbps.artifacts import checked,FileRef
        from tbps.attacks.legacy_runtime import _LegacyG
        import inspect
        allowed=set(plugin.backend.model.modules());original_call=torch.nn.Module._call_impl
        original=plugin.generate;original_reader=plugin.backend.load_pixels
        reading=[None];resize_type=None;resize_forward=None
        if self.preprocess_sources is not None:
            from torchvision import transforms
            sources=self.preprocess_sources
            need(self.image_sources is not None and set(sources)=={"backend","pixel_reader","torchvision"},
                 "explicit verified preprocessing sources/inputs required")
            for value in sources.values():checked(FileRef(**value))
            need(type(plugin.backend) is _LegacyG and original_reader.__func__ is _LegacyG.load_pixels
                 and inspect.getsourcefile(_LegacyG)==sources["backend"]["path"],
                 "only original verified image reader backend")
            need(plugin.backend.pixel_reader.__name__=="pixels_from_pil"
                 and inspect.getsourcefile(plugin.backend.pixel_reader)==sources["pixel_reader"]["path"],
                 "only pinned original pixels_from_pil")
            resize_type=transforms.Resize;resize_forward=resize_type.forward
            need(inspect.getsourcefile(resize_type)==sources["torchvision"]["path"]
                 and inspect.getsourcefile(resize_forward)==sources["torchvision"]["path"]
                 and resize_forward.__qualname__=="Resize.forward",
                 "pinned torchvision Resize source required")
        def read(path):
            if resize_type is None:return original_reader(path)
            need(reading[0] is None,"nested image preprocessing forbidden")
            p=Path(path).resolve()
            matched=[(k,v) for k,v in self.image_sources.items() if Path(v.path).resolve()==p]
            need(len(matched)==1,"preprocessing image must be a bound Gallery member")
            key,source=matched[0]
            need(self.started<len(self.expected) and key==self.expected[self.started]
                 and self.active in (None,key),"preprocessing input/order mismatch")
            before=checked(source);reading[0]=key
            try:
                value=original_reader(path)
                need(value.shape==(1,3,384,128) and value.dtype==torch.float32 and
                     bool(torch.isfinite(value).all()) and bool(((value>=0)&(value<=1)).all()),
                     "preprocessing must return original FP32 RGB")
                need(checked(source)==before,"image changed during preprocessing")
                return value
            finally:reading[0]=None
        def module_call(module,*a,**kw):
            if module in allowed:
                need(reading[0] is None,"G forward forbidden inside pixel reader")
            else:
                from PIL import Image
                from torchvision.transforms import InterpolationMode
                need(reading[0] is not None and type(module) is resize_type and not kw and len(a)==1
                     and isinstance(a[0],Image.Image) and a[0].mode=="RGB"
                     and type(module.size) in (tuple,list) and tuple(module.size)==(384,128) and module.interpolation is InterpolationMode.BILINEAR
                     and module.max_size is None and module.antialias is True
                     and "forward" not in vars(module) and type(module).forward is resize_forward
                     and not module._forward_hooks and not module._forward_pre_hooks
                     and not module._backward_hooks and not list(module.parameters()) and not list(module.buffers()),
                     "non-G neural forward forbidden")
                self.preprocess_calls+=1
            return original_call(module,*a,**kw)
        def generate(request):
            need(self.active is None,"nested G generation")
            self.active=request.binding.sample_id;self.current_started=time.perf_counter()
            try:
                value=original(request)
                trace=value.trace()
                if self.active in self.old_pixels:
                    actual={"original_pixels_sha256":trace["input_pixels_sha256"],
                        "adversarial_pixels_sha256":trace["adversarial_pixels_sha256"],
                        "effective_delta_sha256":trace["effective_delta_sha256"]}
                    wanted={k:self.old_pixels[self.active][k] for k in actual}
                    self.comparisons[self.active]={"actual":actual,"historical":wanted,"exact":actual==wanted}
                    if actual!=wanted and self.on_mismatch:self.on_mismatch(self.active,value,self.comparisons[self.active])
                    need(actual==wanted,"historical two-image pixel SHA mismatch: "+self.active)
                self.records.append({"image_id":self.active,"G_forward_samples":1,
                    "generation_seconds":time.perf_counter()-self.current_started,
                    "original_pixels_sha256":trace["input_pixels_sha256"],
                    "adversarial_pixels_sha256":trace["adversarial_pixels_sha256"],
                    "effective_delta_sha256":trace["effective_delta_sha256"],"linf":trace["linf"]})
                return value
            finally:self.active=None
        a=plugin.backend.model.register_forward_pre_hook(self.before)
        b=plugin.backend.model.register_forward_hook(self.after)
        try:
            with patch.object(torch.nn.Module,"_call_impl",new=module_call),patch.object(plugin,"generate",new=generate),patch.object(plugin.backend,"load_pixels",new=read):
                yield
        finally:a.remove();b.remove()
    def report(self):
        return {"started":self.started,"completed":self.completed,"records":self.records,
                "verified_preprocess_Resize_calls":self.preprocess_calls,
                "historical_two_image_comparisons":self.comparisons}
