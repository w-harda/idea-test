"""CPU重放资源监视与只读来源保护；不加载模型或执行评分。"""
from pathlib import Path
import os
import signal
import threading
import time
from ..artifacts import ArtifactError,checked,FileRef,sha_file,read_json

class ResourceLimit(ArtifactError):pass

def rss_bytes():
    fields={}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:","VmHWM:")):
            k,v,*_=line.split();fields[k[:-1]]=int(v)*1024
    return max(fields.values())

class ResourceGuard:
    """后台仅采样本进程标量；信号使主线程安全退出，finally停止采样线程。"""
    def __init__(self,max_rss_bytes,max_seconds,*,sample=rss_bytes,clock=time.monotonic):
        self.limit=max_rss_bytes;self.seconds=max_seconds;self.sample=sample;self.clock=clock
        self.started=clock();self.peak=0;self.reason=None;self.event=threading.Event()
    def check(self):
        self.peak=max(self.peak,self.sample())
        if self.peak>=self.limit:self.reason="RSS budget reached"
        if self.clock()-self.started>=self.seconds:self.reason="time budget reached"
        if self.reason:raise ResourceLimit(self.reason)
    def snapshot(self):
        return {"peak_rss_bytes":self.peak,"elapsed_seconds":self.clock()-self.started,
                "max_rss_bytes":self.limit,"max_seconds":self.seconds,"stop_reason":self.reason}
    def __enter__(self):
        self.check();self.previous={}
        def interrupt(signum,frame):
            if signum==signal.SIGUSR1:raise ResourceLimit(self.reason or "resource watchdog")
            self.reason="interrupted by signal "+str(signum);raise InterruptedError(self.reason)
        for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM):
            self.previous[sig]=signal.signal(sig,interrupt)
        def watch():
            while not self.event.wait(.05):
                try:self.check()
                except ResourceLimit:
                    os.kill(os.getpid(),signal.SIGUSR1);return
        self.thread=threading.Thread(target=watch,name="cpu-replay-budget",daemon=True)
        self.thread.start();return self
    def __exit__(self,*exc):
        self.event.set();self.thread.join()
        for sig,fn in self.previous.items():signal.signal(sig,fn)

def historical_snapshot(path):
    """旧保护清单仅SHA/mtime，不修改旧文件。"""
    before=read_json(path)
    for name,v in before.items():
        state=checked(FileRef(name,v["sha256"]))
        if state["mtime_ns"]!=v["mtime_ns"]:raise ArtifactError("historical mtime changed: "+name)
    return before

def unchanged(snapshot):
    for name,v in snapshot.items():
        state=checked(FileRef(name,v["sha256"]))
        if state["mtime_ns"]!=v["mtime_ns"]:raise ArtifactError("historical source changed: "+name)
    return len(snapshot)

def compact_heap():
    """只释放本进程已不再引用的CPU临时块，不改变Tensor数值。"""
    import gc,ctypes
    gc.collect()
    trim=getattr(ctypes.CDLL(None),"malloc_trim",None)
    if trim is not None:trim(0)
