"""固定来源、配对Checkpoint及只读输入契约。"""
from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,checked,sha_file,read_json,digest,json_bytes

METRICS=('R@1','R@5','R@10','mAP')
METHODS=('cooperative','attraction','augmentation')
MODELS=('rasa','aptm')
def need(ok,message):
    if not ok:raise ArtifactError(message)
def ref(p):return FileRef(str(Path(p).resolve()),sha_file(p))
def utf8(t):return hashlib.sha256(t.encode('utf-8')).hexdigest()
def plain(x):return json.loads(json_bytes(x))
def config(path):
    c=read_json(path,'gh-cross-model-2')
    need(c['methods']==list(METHODS) and c['models']==list(MODELS) and c['round']==60,'fixed three methods/two targets/Round60')
    need(c['image_epsilon']==8/255 and c['text_budget']=='original Natural-K; one Unicode character per original native word','original budgets')
    need(c['modes']==['Clean','Joint'] and c['precision']=='fp32/math/no_tf32' and c['query_chunk']==128 and c['cache_chunk']==16,'native frozen evaluation')
    need(c['retry_real_failure'] is False and c['no_training'] is True,'no retry/training')
    root=Path(__file__).resolve().parents[3]
    files=list((root/'src/tbps').rglob('*.py'))+list((root/'operations').glob('gh_cross_model_v*.py'))
    need(c['producers']=={str(p):sha_file(p) for p in files},'complete versioned source pins')
    for v in c['targets'].values():checked(FileRef(**v))
    for v in c['legacy_attempt'].values():checked(FileRef(**v))
    checked(FileRef(**c['annotation']))
    return c

def manifest(c,scope='full'):
    from tbps.datasets.cuhk_pedes import CuhkPedes
    m=CuhkPedes(FileRef(**c['annotation']),c['image_root']).manifest('test',expected_counts=(3074,6156))
    need(scope in ('smoke','full'),'explicit scope')
    return replace(m,gallery=m.gallery[:10],queries=m.queries[:10]) if scope=='smoke' else m

def historical(c,method):
    need(method in METHODS,'known frozen method')
    h=c['historical'][method]
    for value in h['proofs'].values():checked(FileRef(**value))
    checked(FileRef(**h['configuration']))
    f=read_json(h['configuration']['path'])
    for name in ('producers','tools','runtime_pins'):
        for p,s in f[name].items():checked(FileRef(p,s))
    for r in f['fixed_resources'].values():checked(FileRef(**r))
    base=Path(h['output_base']);test=base/'test/round_60'
    for root,key in ((base/'train/checkpoints','round_60'),(base/'train_cpu_audit/round_60','completed'),
                     (test,'completed'),(test/'cpu_audit','completed'),(test/'attack','claim'),(test/'attack','completed')):
        ArtifactStore(root,new_base=base).verify(key)
    summary=read_json(test/'completed/summary.json');audit=read_json(test/'cpu_audit/completed/report.json')
    train=read_json(base/'train_cpu_audit/round_60/completed/report.json')
    prefix={'cooperative':'GH_COOP','attraction':'GH_ATTRACTION','augmentation':'GH_AUG'}[method]
    need(train['status']==prefix+'_TRAIN_CPU_AUDIT_PASS' and train['cursor']=={'round':60,'g_updates':484080,'h_updates':122220,'h_visits':122880},'historical train audit')
    need(train['configuration_sha256']==h['configuration']['sha256'],'historical train identity')
    need(audit['status']==prefix+'_JOINT_TEST_CPU_AUDIT_PASS' and audit['metrics']==summary['metrics']
         and audit['round']==summary['round']==60 and audit['scope']==summary['scope']=='official_test'
         and audit['gallery']==summary['gallery']==3074 and audit['query']==summary['query']==6156
         and audit['configuration_sha256']==summary['configuration_sha256']==h['configuration']['sha256'],'audited historical Full')
    cp=read_json(base/'train/checkpoints/round_60/manifest.json')
    for side in ('G','H'):
        r=FileRef(**h[side]);checked(r)
        need(summary[side]==asdict(r) and cp['files_sha256'][side+'.pt']==r.sha256,'paired Round60 identity')
    need(cp['provenance']['configuration']==h['configuration']['sha256'] and cp['provenance']['round']==60,'checkpoint manifest provenance')
    full=manifest(c)
    need(read_json(test/'attack/claim/dataset_manifest.json')==plain(asdict(full)),'original attack full manifest/order')
    done=read_json(test/'attack/completed/report.json')
    need(done=={'Adam_steps':0,'G_forwards':3074,'images':3074,'models_frozen':True,'paired_image_used_by_H':False,'texts':6156},'sealed pure inference source')
    session={k:f['native_template'][k] for k in ('managed_roots','import_paths')}
    session.update(runtime_pins=f['runtime_pins'],producers=f['producers'])
    return h,f,test,session

def collection(c,method):
    binding=c['input_collections'][method]
    need(Path(binding['root']).name==method,'method-specific input collection')
    if binding['completed'] is not None:checked(FileRef(**binding['completed']))
    s=ArtifactStore(binding['root'],new_base=binding['base']);s.verify('completed')
    report=read_json(s.root/'completed/report.json');v=read_json(s.root/'completed/collection.json')
    need(report['status']=='GH_TRANSFER_INPUT_CPU_AUDIT_PASS' and report['method']==method
         and report['round']==60 and (report['gallery'],report['query'])==(3074,6156),'complete original input audit')
    need(report['collection_sha256']==digest(v) and v['historical']==c['historical'][method],'sealed collection method binding')
    return v,s

def image_value(item):
    import torch
    checked(FileRef(**item['data']))
    raw=torch.load(item['data']['path'],map_location='cpu',weights_only=True)
    from tbps.alternating.native_backend import assert_pixels
    assert_pixels(raw['original'],raw['adversarial'],raw['effective_delta'])
    from tbps.engines.provenance import payload_digest
    need(payload_digest(raw['adversarial'])==item['adversarial_sha256'],'actual RGB input digest')
    return raw

def text_value(item):
    checked(FileRef(**item['data']))
    a=read_json(item['data']['path'])
    need(utf8(a['original'])==item['original_sha256'] and utf8(a['adversarial'])==item['adversarial_sha256'],'actual UTF8 input digest')
    return a
