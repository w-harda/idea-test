"""独立预检配置；本次入口只提供CPU预检，不启动新实验。"""
from tbps.alternating import fresh_sources as old
from .objectives import Settings,require
from dataclasses import asdict

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
