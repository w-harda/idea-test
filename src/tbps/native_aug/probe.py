"""真实官方权重的CPU接线探针：输入梯度、原生文本前缀/JVP；不是正式攻击结果。"""
from pathlib import Path
import torch,time
from .common import *
def run(c,name):
    from importlib import import_module
    from .training import native_bundle
    from .native import Source,jvp
    from .compat import source_session
    from tbps.models.irra import fp32_runtime
    torch.set_num_threads(2);need(not torch.cuda.is_initialized(),'CPU probe')
    n,bundle=native_bundle(c,name)
    mod=import_module('tbps.models.'+name);source=(mod.RasaSource if name=='rasa' else mod.APTMSource)(c['targets'][name]['path'])
    loader=mod.load_rasa if name=='rasa' else mod.load_aptm
    start=time.monotonic()
    with fp32_runtime(),loader(source,device='cpu') as adapter,source_session(n,bundle,c,name):
        wrapper=Source(adapter,name).eval().requires_grad_(False)
        caption='A man wearing a blue shirt and black pants.'
        H,mask=wrapper.prefix([caption]);need(H.shape==(1,wrapper.max_tokens,768),'native prefix768')
        B=torch.randn_like(H)*mask[:,:,None];B=B/B.double().norm().float()
        vector,check=jvp(wrapper,caption,{'B':B})
        # CPU原生输入梯度，不训练G/H或Victim。
        pixels=torch.rand(1,3,384,128,requires_grad=True)
        feature=wrapper(pixels);need(feature.shape==(1,256),'native projected dimension256')
        gradient,=torch.autograd.grad(feature[0,0],pixels)
        need(torch.isfinite(gradient).all() and gradient.norm()>0,'native image input gradient')
        counts=dict(wrapper.checkpoint_counts)
        need(counts['recompute_block_calls']>0,'real activation checkpoint backward path')
        wrapper.close()
        baseline_pixels=pixels.detach().clone().requires_grad_(True)
        baseline_feature=wrapper(baseline_pixels)
        baseline_gradient,=torch.autograd.grad(baseline_feature[0,0],baseline_pixels)
        need(torch.equal(feature.detach(),baseline_feature.detach()),'checkpoint forward changed native features')
        need(torch.allclose(gradient,baseline_gradient,atol=1e-7,rtol=1e-5),'checkpoint changed input gradient')
        gradient_error=float((gradient-baseline_gradient).abs().max())
        wrapper.check_frozen(final=True)
        report={'status':'NATIVE_AUG_REAL_CPU_BACKEND_PROBE_PASS','source':name,'feature_dim':256,'hidden_dim':768,
            'prefix_layer':wrapper.prefix_index+1,'JVP':check,'image_input_gradient_norm':float(gradient.norm()),
            'checkpoint_features_exact':True,'checkpoint_gradient_max_abs_error':gradient_error,'checkpoint_counts':counts,'memory_placement':wrapper.memory_placement,
            'tested_sources':[asdict(ref(Path(__file__).with_name(f))) for f in ('native.py','compat.py','guard.py','startup.py')],'source_configuration':c['targets'][name],
            'victim_frozen':True,'G_H_updates':0,'CUDA':False,'elapsed_seconds':time.monotonic()-start}
        d=CONTROL/'cpu_probes'/name;d.mkdir(parents=True,exist_ok=True);(d/'report.json').write_bytes(json_bytes(report))
        print(report,flush=True);return report
