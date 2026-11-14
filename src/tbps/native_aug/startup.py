"""受控冷导入与原CPUStartupGate；原隔离断言保持。"""
from contextlib import ExitStack
from unittest.mock import patch
import sys
from .common import need

def bootstrap():
    import torch,importlib
    need(not torch.cuda.is_initialized(),'bootstrap must precede CUDA')
    from tbps.materialization.irra_startup import isolation_snapshot
    import tempfile
    from pathlib import Path
    before=isolation_snapshot()
    paths=list(sys.path);cold='torchvision' not in sys.modules
    attempts=[]
    def deny(label):
        def stop(*a,**kw):attempts.append(label);raise RuntimeError('bootstrap forbids '+label)
        return stop
    with ExitStack() as stack:
        for owner,name in [(torch,'load'),(torch.jit,'load'),(torch.nn.Module,'_call_impl'),(torch.cuda,'init'),(torch.cuda,'_lazy_init')]:
            stack.enter_context(patch.object(owner,name,deny(name)))
        importlib.import_module('torchvision')
    need(not attempts and not torch.cuda.is_initialized(),'bootstrap loaded weights/NN/CUDA')
    after=isolation_snapshot();added=[str(v) for v in sys.path if v not in paths]
    vendor=Path(torch.__file__).resolve().parents[1]/'setuptools/_vendor';temporary=Path(tempfile.gettempdir()).resolve()
    need(after[0]==before[0] and after[2:]==before[2:],'bootstrap changed callable/meta/bytecode state')
    need([v for v in sys.path if v in paths]==paths and len(added)<=2,'bootstrap changed original search path order')
    need(all(Path(v).resolve()==vendor or (Path(v).resolve().is_relative_to(temporary) and Path(v).resolve()!=temporary and Path(v).is_dir()) for v in added),'unexpected bootstrap path addition')
    return {'cold_torchvision_import':cold,'sys_path_added':[str(v) for v in sys.path if v not in paths],
            'forbidden_attempts':attempts,'CUDA':False,'weights_loaded':0,'neural_forwards':0}
