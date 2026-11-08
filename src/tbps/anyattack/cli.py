"""所有GPU动作必须显式--allow-inference；审计永不进入神经网络。"""
import argparse,sys,importlib
import torch
from .core import *

def preflight(ctx):
    from tbps.models.irra_source import IRRASource
    from tbps.models.rasa import RasaSource,native_modules as rasa_modules,official_tokenizer as rasa_tokenizer
    from tbps.models.aptm import APTMSource,native_modules as aptm_modules,official_tokenizer as aptm_tokenizer
    def forbidden(*a,**k):raise ArtifactError('CPU import preflight forbids model/weight loading')
    # Transformers在导入时创建无权重激活常量；允许构造，仍拒绝所有前向/权重/CUDA。
    with no_training(cpu=True),patch.object(torch,'load',forbidden):
        need(not any(n=='models' or n.startswith('models.') for n in sys.modules),'isolated source namespace')
        sys.path.insert(0,ctx['c']['runtime_root']);module=importlib.import_module('models.model')
        need(module.Decoder is not None and module.CLIPEncoder is not None,'official import closure')
        for n in list(sys.modules):
            if n=='models' or n.startswith('models.'):del sys.modules[n]
        sys.path.remove(ctx['c']['runtime_root'])
        IRRASource.from_dict(ctx['d']['irra']).verify()
        for cls,modules,tokenizer,k in [(RasaSource,rasa_modules,rasa_tokenizer,'rasa_source'),
                                      (APTMSource,aptm_modules,aptm_tokenizer,'aptm_source')]:
            source=cls(ctx['c'][k]['path'])
            with modules(source) as (_,tok,utils):
                value=tokenizer(tok,source)(['a person in blue.'],padding='max_length',truncation=True,max_length=50,return_tensors='pt')
                need(value.input_ids.shape==(1,50),'native offline tokenizer')
    print('ANYATTACK_CPU_PREFLIGHT_PASS',ctx['scope'],len(ctx['manifest'].gallery),len(ctx['manifest'].queries),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--scope',choices=['smoke','full'],default='smoke')
    a.add_argument('--action',choices=['preflight','generate','audit-generation','evaluate','audit-evaluation'],default='preflight')
    a.add_argument('--model',choices=['irra','rasa','aptm']);a.add_argument('--allow-inference',action='store_true');args=a.parse_args()
    torch.set_num_threads(2)
    ctx=context(args.config,args.scope)
    if args.action=='preflight':preflight(ctx)
    elif args.action=='generate':
        need(args.allow_inference,'explicit inference authorization');from .generate import run;run(ctx)
    elif args.action=='evaluate':
        need(args.allow_inference and args.model,'explicit victim/inference authorization');from .evaluate import run;run(ctx,args.model)
    else:
        with no_training(cpu=True):
            if args.action=='audit-generation':audit_generation(ctx)
            else:
                need(args.model,'explicit victim');from .evaluate import audit;audit(ctx,args.model)
if __name__=='__main__':main()
