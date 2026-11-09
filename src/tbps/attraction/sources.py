"""独立来源与CPU预检；GPU执行须通过Full入口显式授权。"""
from tbps.alternating import fresh_sources as old
from .objectives import Settings,require
from dataclasses import asdict
Grant=old.Grant

def augment(f,c):
    require(f['method']=='GH-Adversarial-Attraction-v1','explicit attraction method identity')
    settings=Settings(**f['attraction']);c['attraction']=asdict(settings)
    c['training_configuration']['attraction']=asdict(settings)
    return f,c

def load(path,scope):return augment(*old.load(path,scope))
def preflight(path,scope):
    f,c,b,g,report=old.preflight(path,scope);augment(f,c)
    report.update(status='GH_ATTRACTION_CPU_PREFLIGHT_PASS',method=f['method'],attraction=f['attraction'],
        phase_order='G1 -> current G1 Bank -> H1 -> current H1 context -> G2',GPU_experiment_started=False)
    return f,c,b,g,report
