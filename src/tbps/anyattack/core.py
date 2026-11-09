"""官方AnyAttack免训练迁移：协议、图像桥接及只读CPU审计。"""
from pathlib import Path
from dataclasses import asdict, replace
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import hashlib, io, random, time, resource
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tbps.artifacts import ArtifactStore, ArtifactError, FileRef, checked, read_json, json_bytes, digest
from tbps.engines.materialized_cache import ref
from tbps.engines.provenance import payload_digest
from tbps.datasets.cuhk_pedes import CuhkPedes

EPS=8/255
def need(ok,message):
    if not ok:raise ArtifactError(message)

def tensor_bytes(value):
    b=io.BytesIO();torch.save(value,b);return b.getvalue()

def pixels(path,hw):
    # 与官方demo的PIL Resize((224,224))+ToTensor一致；carrier仅改变目标尺寸。
    with Image.open(path) as im:
        x=im.convert('RGB').resize((hw[1],hw[0]),Image.Resampling.BILINEAR)
        return pil_to_tensor(x).float().div(255).unsqueeze(0)

def mapping(gallery,seed=42):
    groups={}
    for i,g in enumerate(gallery):groups.setdefault(g.pid,[]).append(i)
    keys=list(groups);random.Random(seed).shuffle(keys)
    order=[i for key in keys for i in groups[key]];shift=max(map(len,groups.values()))
    need(shift*2<=len(order),'wrong-PID bijection requires largest group <= half Gallery')
    result=[None]*len(order)
    for i,carrier in enumerate(order):result[carrier]=order[(i+shift)%len(order)]
    need(sorted(result)==list(range(len(gallery))) and
         all(gallery[i].pid!=gallery[j].pid for i,j in enumerate(result)),'target-reference PID separation/bijection')
    return result

def strict_project(original,adversarial):
    """将FP32候选投到实数8/255以内的可表示端点；审计不加容差。"""
    lower64=(original.double()-EPS).clamp(0,1);upper64=(original.double()+EPS).clamp(0,1)
    lower=lower64.float();upper=upper64.float()
    lower=torch.where(lower.double()<lower64,torch.nextafter(lower,torch.full_like(lower,float('inf'))),lower)
    upper=torch.where(upper.double()>upper64,torch.nextafter(upper,torch.full_like(upper,float('-inf'))),upper)
    result=torch.maximum(lower,torch.minimum(adversarial,upper))
    need(float((result.double()-original.double()).abs().max())<=EPS,'strict actual RGB Linf <=8/255')
    return result

def bridge(original,raw):
    need(original.shape==(1,3,384,128) and raw.shape==(1,3,224,224),'native carrier/Decoder shape')
    for v in (original,raw):
        need(v.device.type=='cpu' and v.dtype==torch.float32 and torch.isfinite(v).all() and not v.requires_grad,'finite CPU FP32 bridge')
    need(original.min()>=0 and original.max()<=1,'original RGB range')
    clamped=raw.clamp(-EPS,EPS)
    resized=torch.nn.functional.interpolate(clamped,size=(384,128),mode='bilinear',align_corners=False).clamp(-EPS,EPS)
    adv=strict_project(original,(original+resized).clamp(0,1))
    return {'original':original.clone(),'raw_noise':raw.clone(),'clamped_noise':clamped,
            'resized_noise':resized,'adversarial':adv,'effective_delta':adv-original}

@contextmanager
def no_training(cpu=False):
    """审计允许weights_only CPU Tensor读取；拒绝神经调用、CUDA和所有导数。"""
    attempts=[]
    def deny(*a,**kw):attempts.append(True);raise ArtifactError('training/forbidden CPU neural operation')
    with ExitStack() as s:
        for obj,name in [(torch.Tensor,'backward'),(torch.autograd,'backward'),(torch.autograd,'grad'),
                         (torch.optim.Optimizer,'__init__')]:s.enter_context(patch.object(obj,name,deny))
        if cpu:
            need(not torch.cuda.is_initialized(),'fresh CPU process required')
            for obj,name in [(torch.nn.Module,'_call_impl'),(torch.cuda,'init'),(torch.cuda,'_lazy_init')]:
                s.enter_context(patch.object(obj,name,deny))
            original_load=torch.load
            def load(*a,**kw):
                need(kw.get('map_location')=='cpu' and kw.get('weights_only') is True,'CPU Tensor-only load')
                return original_load(*a,**kw)
            s.enter_context(patch.object(torch,'load',load))
        with torch.inference_mode():yield
        need(not attempts,'forbidden operation attempted')
        if cpu:need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')

def context(path,scope):
    c=read_json(path,'tbps-anyattack-pretrained-1')
    need(c['decoder']['sha256']=='bed1f25d46b4eaef0c3d4eb39c3853b2241991cc048b3b2fff21b7b8c38e0070'
         and c['clip']['sha256']=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
         and c['eps']==EPS and c['seed']==42 and c['training'] is False and c['fine_tuning'] is False and c['generation_mode']=='reuse_sealed_raw_noise'
         and c['max_text_replacements_per_query']==1 and c['actual_text_replacements_per_query']==0,
         'fixed official Decoder/CLIP/zero-training protocol')
    for k in ('decoder','clip','dataset_sources','rasa_source','aptm_source','bpe','raw_parent_config'):checked(FileRef(**c[k]))
    for p,h in c['producer_sources'].items():checked(FileRef(p,h))
    for p,h in c['official_files'].items():checked(FileRef(p,h))
    need(scope in ('smoke','full'),'explicit scope')
    d=read_json(c['dataset_sources']['path']);ds=CuhkPedes(FileRef(**d['annotation']),d['image_root'])
    full=ds.manifest('test',expected_counts=(3074,6156));targets=mapping(full.gallery,c['seed'])
    m=full if scope=='full' else replace(full,gallery=full.gallery[:10],queries=full.queries[:10])
    need(all(q.paired_image_id in {g.image_id for g in m.gallery} for q in m.queries),'Smoke positives covered')
    return dict(c=c,path=path,d=d,ds=ds,full=full,manifest=m,targets=targets,scope=scope)

def store_at(ctx,suffix):
    base=ctx['c']['output_base'];return ArtifactStore(Path(base)/ctx['scope']/suffix,new_base=base)

def source_protocol(ctx):
    return {'schema':'tbps-anyattack-generation-eps8-1','config':asdict(ref(ctx['path'])),
        'scope':ctx['scope'],'manifest':asdict(ctx['manifest']),
        'full_gallery_sha256':digest([asdict(g) for g in ctx['full'].gallery]),
        'target_indices':ctx['targets'][:len(ctx['manifest'].gallery)],
        'target_rule':'seed42 PID-group shuffle; rotate largest-group length; bijective wrong-PID references',
        'test_PID_access':'only target-reference mapping; no PID/caption/victim feedback to neural source',
        'decoder':'official coco_cos.pt; raw unnormalized CLIP ViT-B/32 image embedding',
        'geometry':'PIL BILINEAR RGB carrier384x128/reference224x224; clamp native224 noise; CPU bilinear align_corners=False to384x128; clamp; RGB add and [0,1] clamp; strict inward endpoint projection at8/255',
        'epsilon_nominal':EPS,'floating_point':'FP32 inward endpoint projection; double-measured actual Linf <=8/255; zero audit tolerance',
        'raw_parent_config':ctx['c']['raw_parent_config'],'generation_mode':'reuse_sealed_raw_noise',
        'max_text_replacements_per_query':1,'text_budget_unit':'word','actual_text_replacements_per_query':0,
        'text':'all original captions unchanged','training':False,'fine_tuning':False,'victim_feedback':False}

def key(i):return 'image-'+str(i).zfill(6)

def load_image(ctx,i):
    s=store_at(ctx,'generation');s.verify(key(i))
    x=torch.load(s.root/key(i)/'image.pt',map_location='cpu',weights_only=True)
    md=read_json(s.root/key(i)/'details.json')
    need({k:payload_digest(v) for k,v in x.items()}==md['payload_sha256'],'image Tensor content identity')
    return x,md

def historical(ctx):
    """原16/255产物只读：完整验证来源，提取预算裁剪前的Decoder输出。"""
    c=ctx['c'];parent=read_json(c['raw_parent_config']['path'],'tbps-anyattack-pretrained-1')
    need(parent['eps']==16/255 and parent['decoder']==c['decoder'] and parent['clip']==c['clip'] and parent['seed']==c['seed'],
         'exact historical official weights/seed; parent budget16/255')
    for path,h in parent['producer_sources'].items():checked(FileRef(path,h))
    s=ArtifactStore(Path(parent['output_base'])/ctx['scope']/'generation',new_base=parent['output_base'])
    s.verify('protocol');s.verify('completed');need(not s.pending(),'historical generation pending')
    p=read_json(s.root/'protocol/protocol.json');n=len(ctx['manifest'].gallery)
    need(p['manifest']==read_json_bytes(asdict(ctx['manifest'])) and p['target_indices']==ctx['targets'][:n],
         'historical manifest/reference mapping exact')
    r=read_json(s.root/'completed/report.json')
    need(r['status']=='PASS' and r['encoder_forwards']==r['decoder_forwards']==n and r['model_before']==r['model_after']
         and r['frozen'] and not r['training'] and not r['fine_tuning'],'historical frozen official generation')
    audit=ArtifactStore(Path(parent['output_base'])/ctx['scope']/'generation_cpu_audit',new_base=parent['output_base'])
    audit.verify('result');need(read_json(audit.root/'result/report.json')['status']=='PASS','historical independent source audit gate')
    return s,r,ref(audit.root/'result/manifest.json')

def historical_image(store,i):
    store.verify(key(i));md=read_json(store.root/key(i)/'details.json')
    x=torch.load(store.root/key(i)/'image.pt',map_location='cpu',weights_only=True)
    need({k:payload_digest(v) for k,v in x.items()}==md['payload_sha256'],'historical raw noise content SHA')
    return x,md,ref(store.root/key(i)/'manifest.json')

def audit_generation(ctx):
    t=time.monotonic();s=store_at(ctx,'generation');s.verify('protocol');s.verify('completed')
    need(read_json(s.root/'protocol/protocol.json')==read_json_bytes(source_protocol(ctx)),'source protocol exact')
    report=read_json(s.root/'completed/report.json');n=len(ctx['manifest'].gallery)
    parent,pr,pa=historical(ctx)
    need(report['status']=='PASS' and report['encoder_forwards']==report['decoder_forwards']==report['model_loads']==0 and
         report['model_before']==report['model_after']==pr['model_before'] and report['frozen'] is True,
         'no new neural inference/training; original source identity exact')
    need(report['parent_generation']==asdict(ref(parent.root/'completed/manifest.json')),'historical generation identity')
    hashes=[];maximum=0
    for i,g in enumerate(ctx['manifest'].gallery):
        x,md=load_image(ctx,i);old,omd,oref=historical_image(parent,i);j=ctx['targets'][i];target=ctx['full'].gallery[j]
        need(md['carrier']==asdict(g) and md['reference']==asdict(target) and g.pid!=target.pid and md['parent_manifest']==asdict(oref),
             'carrier/reference/parent identity')
        checked(FileRef(str(ctx['ds'].image_path(target.image_rel_path)),target.image_sha256))
        original=pixels(ctx['ds'].image_path(g.image_rel_path),(384,128))
        need(torch.equal(x['original'],original) and torch.equal(old['original'],original),'original pixels exact')
        need(torch.equal(x['raw_noise'],old['raw_noise']) and torch.equal(x['embedding'],old['embedding']),
             'raw Decoder output and CLIP embedding unchanged')
        replay=bridge(original,old['raw_noise'])
        need(set(x)==set(replay)|{'embedding'} and x['embedding'].shape==(1,512) and torch.isfinite(x['embedding']).all() and
             all(torch.equal(x[k],v) for k,v in replay.items()),'complete eps8 clamp/geometry/inward projection replay')
        actual=float((x['adversarial'].double()-original.double()).abs().max());maximum=max(maximum,actual)
        need(actual<=EPS and md['linf_double']==actual,'actual RGB budget8/255; no tolerance')
        need(x['adversarial'].min()>=0 and x['adversarial'].max()<=1,'RGB range')
        hashes.append(payload_digest(x['adversarial']))
        del x,old,replay,original
    need(hashes==report['adversarial_sha256'] and not s.pending(),'complete source index/no pending')
    result={'status':'PASS','scope':ctx['scope'],'gallery':n,'query':len(ctx['manifest'].queries),
            'max_linf_double':maximum,'epsilon_nominal':EPS,'budget_tolerance':0,
            'source_weight_state_unchanged':True,'all_original_captions':True,'max_text_replacements_per_query':1,
            'actual_text_replacements_per_query':0,'all_targets_different_PID':True,'raw_noise_unchanged':True,
            'parent_generation':asdict(ref(parent.root/'completed/manifest.json')),
            'elapsed_seconds':time.monotonic()-t,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'backward':0,'optimizer':0,'JVP':0,'model_loads':0,'CUDA_initialized':torch.cuda.is_initialized()}
    out=store_at(ctx,'generation_cpu_audit');out.publish('result',{'report.json':json_bytes(result)},
        {'kind':'independent_eps8_cpu_audit'},sources=(ref(s.root/'completed/manifest.json'),ref(ctx['path']),pa))
    print('GENERATION_CPU_AUDIT_PASS',result,flush=True);return result

def read_json_bytes(x):
    import json
    return json.loads(json_bytes(x))
