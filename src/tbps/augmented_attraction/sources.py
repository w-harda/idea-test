"""独立来源与CPU预检；GPU执行须通过Full入口显式授权。"""
from tbps.alternating import fresh_sources as old
from .objectives import Settings,require
from dataclasses import asdict
from .views import ViewSettings
Grant=old.Grant

def augment(f,c):
    require(f['method']=='GH-PushPull-Augmentation-v1','explicit attraction method identity')
    settings=Settings(**f['attraction']);c['attraction']=asdict(settings)
    c['training_configuration']['attraction']=asdict(settings)
    upstreams=old.read_json(f['fixed_resources']['augmentation_upstreams']['path'])
    for upstream in upstreams['upstreams'].values():
        for item in upstream['files']:old.checked(old.FileRef(item['path'],item['sha256']))
    c['augmentation']=asdict(ViewSettings(**f['augmentation']));c['training_configuration']['augmentation']=c['augmentation']
    k=sum(e['K'] for e in old.read_json(c['prototype']['g_manifest'])['entries'] if e['role']=='train' and e['row_id'] in c['prototype']['samples'])
    n=len(c['prototype']['samples']);rounds=c['prototype']['rounds'];v=c['augmentation']['views']
    c['smoke']['max_G_forwards']=rounds*v*(k+n+2048)
    c['smoke']['max_IRRA_images']=rounds*v*(k+2*n+2048)
    return f,c

def load(path,scope):return augment(*old.load(path,scope))
def preflight(path,scope):
    f,c,b,g,report=old.preflight(path,scope);augment(f,c)
    report.update(status='GH_AUG_CPU_PREFLIGHT_PASS',method=f['method'],attraction=f['attraction'],
        phase_order='G1 -> current G1 three-view Bank -> H1 -> current H1 context -> G2',GPU_experiment_started=False,
        augmentation=f['augmentation'],original_Train_rows=len(c['prototype']['samples']),
        augmented_Train_views=3*len(c['prototype']['samples']),actual_neural_sample_limits=c['smoke'])
    return f,c,b,g,report
