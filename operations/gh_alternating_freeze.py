"""显式冻结新配置；只可在新Attempt未消费时修订来源。"""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import sha_file,json_bytes,read_json
from tbps.alternating.configured_sources import derive

def freeze(rebind=False):
    oldpath=Path('/home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json')
    old=read_json(oldpath);out='/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v1-20261106'
    cfg=ROOT/'configs/framework/gh_alternating_v1.server.json'
    for scope in ('pilot_v1','train_v1'):
        if Path(out,scope).exists():raise ValueError('Attempt consumed; source rebinding forbidden')
    if cfg.exists():
        if not rebind:raise ValueError('config exists; explicit unconsumed rebind required')
        archive=Path('/home/lzf/ldx/tmp/gh_alternating_v1_cpu_20261106/config_revisions')
        dest=archive/('v'+str(len(list(archive.glob('v*')))+1));dest.mkdir(parents=True)
        import shutil
        for p in [cfg,*map(Path,read_json(cfg)['native_configs'].values())]:shutil.copy2(p,dest/p.name)
    f={'schema':'gh-alternating-v1','name':'GH-Alternating-v1','rounds':50,'pilot_rounds':2,
       'g_passes_per_round':1,'h_passes_per_round':1,'g_learning_rate':2e-4,'h_learning_rate':1e-4,
       'g_scheduler':{'kind':'constant','step_size':1,'gamma':1.0},'h_scheduler':{'kind':'constant','step_size':1,'gamma':1.0},
       'checkpoint_interval_rounds':1,'seed':42,'image_epsilon':8/255,'word_budget':1,
       'word_definition':'original whitespace-separated positions; preserve separators/punctuation/hyphens',
       'character_rule':'preserve original <=1 Unicode replacement per attribute/shared word',
       'long_training_authorized':False,'initialization':'independent G60/H10 complete Adam and stage RNG; not old failed Smoke continuation',
       'output_base':out,'outputs':{s:out+'/'+s+'_v1' for s in ('pilot','train')},
       'controls':{s:'/home/lzf/ldx/tmp/gh_alternating_v1_'+s+'_20261106' for s in ('pilot','train')},
       'audits':{s:out+'/'+s+'_cpu_audit_v1' for s in ('pilot','train')},'ready_receipt':out+'/readiness/ready.json',
       'resource_probe_directory':'/home/lzf/ldx',
       'resources':{'pilot':{'minimum_ram_bytes':8*1024**3,'minimum_disk_bytes':25*1024**3,'minimum_free_inodes':80000},
                    'train':{'minimum_ram_bytes':32*1024**3,'minimum_disk_bytes':250*1024**3,'minimum_free_inodes':2000000}},
       'GPU_policy':'shared_observe_only_no_VRAM_floor','native_configs':{s:str(ROOT/('configs/framework/gh_alternating_v1.'+s+'.native.server.json')) for s in ('pilot','train')},
       'references':{'formal_parent':{'path':str(oldpath),'sha256':sha_file(oldpath)}},
       'producers':{str(p):sha_file(p) for p in (ROOT/'src/tbps').rglob('*.py')},
       'tools':{str(p):sha_file(p) for p in (ROOT/'operations').glob('gh_alternating*.py')},
       'evaluation':{'authorized':False,'target_feedback':False,'official_Test_checkpoint_selection':False},
       'checkpoint_policy':'every completed round remains recoverable; interval publishes reference-only milestone without duplicate tensors'}
    cfg.write_bytes(json_bytes(f))
    for scope,p in f['native_configs'].items():Path(p).write_bytes(json_bytes(derive(f,scope,cfg)))
    print('CONFIGURATION',str(cfg),sha_file(cfg))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rebind-unconsumed',action='store_true');a=ap.parse_args();freeze(a.rebind_unconsumed)
