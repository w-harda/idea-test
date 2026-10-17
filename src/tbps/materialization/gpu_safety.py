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
    def __init__(self,expected,old_pixels=None,on_mismatch=None):
        self.expected=tuple(expected);self.started=0;self.completed=0;self.active=None
        self.old_pixels=old_pixels or {};self.comparisons={};self.records=[]
        self.current_started=None;self.on_mismatch=on_mismatch
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
        allowed=set(plugin.backend.model.modules());original_call=torch.nn.Module._call_impl
        original=plugin.generate
        def module_call(module,*a,**kw):
            need(module in allowed,"non-G neural forward forbidden")
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
            with patch.object(torch.nn.Module,"_call_impl",new=module_call),patch.object(plugin,"generate",new=generate):
                yield
        finally:a.remove();b.remove()
    def report(self):
        return {"started":self.started,"completed":self.completed,"records":self.records,
                "historical_two_image_comparisons":self.comparisons}
