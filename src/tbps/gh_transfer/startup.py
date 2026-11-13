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

def prepare(path,method,model,scope):
    import torch
    boot=bootstrap()
    # 两个入口共用同一冷启动闭包；这些公共模块只定义契约，不加载模型。
    from tbps.materialization.irra_startup import CPUStartupGate
    from tbps.engines.native_cache import NativeFeatureCache
    from tbps.engines.native_score_cache import ChunkCachedScorer
    from tbps.engines.aptm_score_cache import APTMSetScorer
    from tbps.engines.evaluation import EvaluationEngine,publish_result
    from tbps.models.irra import fp32_runtime
    from tbps.alternating.parents import no_optimizer_steps
    from tbps.smoke_irra import model_snapshot
    from . import evaluation
    torch.set_num_threads(2);torch.manual_seed(42)
    gate=CPUStartupGate()
    c,mod,source=evaluation.preflight(path,model)
    coll,store=evaluation.collection(c,method);m=evaluation.manifest(c,scope)
    gate.checked();gate.exit_checked()
    report={'status':'GH_TRANSFER_COLD_CPU_STARTUP_PASS','method':method,'model':model,'scope':scope,
            'bootstrap':boot,'cpu_sources_verified':gate.completed,'cpu_isolation_exited':gate.exited,
            'CUDA':torch.cuda.is_initialized(),'real_model_loads':0,'gallery':len(m.gallery),'query':len(m.queries)}
    print('TRANSFER_STARTUP_GATE_PASS',method,model,scope,flush=True)
    return c,mod,source,coll,store,m,gate,report
