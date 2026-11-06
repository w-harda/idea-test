"""新版本入口：默认CPU，Pilot一次，长训练和显式恢复分开授权。"""
import argparse
import json
import os
import fcntl
from pathlib import Path
import resource
import signal
import subprocess
import shutil
import time
import traceback
from tbps.artifacts import ArtifactStore, FileRef, sha_file, read_json, json_bytes, digest
from .contracts import require
from .configured_sources import prepare, load, FormalGrant
from .formal_state import FormalJournal
from .native_sources import source_key


def resources(f,scope):
    """GPU只观察；RAM/磁盘/inode仍核验，没有最低空闲显存准入条件。"""
    gpu=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=name,memory.free,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=20).strip()
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=20).strip()
    mem={x.split(":")[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    base=Path(f["resource_probe_directory"]);disk=shutil.disk_usage(base);ino=os.statvfs(base)
    limits=f["resources"][scope]
    checks={"RAM":mem['MemAvailable']>=limits['minimum_ram_bytes'],"disk":disk.free>=limits['minimum_disk_bytes'],
            "inodes":ino.f_favail>=limits['minimum_free_inodes']}
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return {"policy":"shared_observe_only_no_VRAM_floor","GPU":gpu,"compute_processes":processes,
        "available_RAM_bytes":mem['MemAvailable'],"free_disk_bytes":disk.free,"free_inodes":ino.f_favail,
        "CPU_user_seconds":usage.ru_utime,"CPU_system_seconds":usage.ru_stime,"peak_RSS_bytes":usage.ru_maxrss*1024,
        "checks":checks,"resource_eligible":all(checks.values())}

def ready(f,path):
    p=Path(f["ready_receipt"])
    if not p.is_file():return False
    v=read_json(p)
    require(v["configuration_sha256"]==sha_file(path) and v["status"]=="READY_FOR_SEPARATE_TRAINING_AUTHORIZATION", "Pilot READY identity")
    from tbps.artifacts import checked
    checked(FileRef(**v["pilot_audit"]))
    a=read_json(v["pilot_audit"]["path"])
    require(a["status"]=="GH_ALTERNATING_V1_PILOT_CPU_AUDIT_PASS","Pilot audit required")
    return True

def run(path,scope,*,allow_pilot=False,allow_training=False,resume_round=None,stop_after=None):
    # 能力检查先于任何模型加载/新Attempt。
    f,c=load(path,scope)
    require(allow_pilot if scope=="pilot" else allow_training and ready(f,path),"explicit Pilot/Train authorization required")
    require(stop_after is None or scope=="pilot" and stop_after==1 and resume_round is None,"planned Pilot boundary only")
    control=Path(f['controls'][scope]);control.mkdir(parents=True,exist_ok=True)
    with (control/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        f,c,bundle,gate,pre=prepare(path,scope)
        snap=resources(f,scope);require(snap['resource_eligible'],'RAM/disk/inode insufficient; no GPU Attempt')
        root=Path(f['outputs'][scope]);journal=FormalJournal(root/'rounds',sha_file(path),f['output_base'])
        previous=None;parent=None
        if resume_round is None:
            require(not root.exists(),'Attempt consumed; no automatic retry')
        else:
            require(type(resume_round) is int and resume_round>=1,'explicit committed boundary required')
            require(not list(root.glob('FAIL*')),'real failure preserved; no automatic retry')
            parent,previous=journal.load_boundary(resume_round,bundle.state['parents'])
            require(all(int(p.name.split('_')[-1])<=resume_round for p in (root/'round_io').glob('round_*'))
                and all(int(p.name.split('_')[-1])<=resume_round for p in (root/'B').glob('bank_*')),'later IO blocks replay')
            from .configured_audit import verify_bank
            verify_bank(root,previous,c)
        store=ArtifactStore(root,new_base=f['output_base'])
        grant=FormalGrant(c,scope,allow_pilot=allow_pilot,allow_training=allow_training,ready=scope=='pilot' or ready(f,path))
        if resume_round is None:
            store.publish('claim',{'preflight.json':json_bytes(pre),'resources.json':json_bytes(snap)},
                {'configuration':sha_file(path),'native_source':source_key(c),'scope':scope})
        else:journal.record_resume(resume_round)
        identity={'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'tty':os.ttyname(0) if os.isatty(0) else None}
        suffix='initial' if resume_round is None else 'resume_'+str(resume_round)
        (control/('process_'+suffix+'.json')).write_bytes(json_bytes(identity))
        stop=[False]
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        start=time.monotonic()
        def progress(label,index,visited,total):
            import torch
            data={'phase':label,'round':index,'visited':visited,'total':total,'elapsed_seconds':time.monotonic()-start,
                'cursor':dict(backend.cursor),'counts':dict(backend.counts),'resources':resources(f,scope),
                'GPU_allocated_bytes':torch.cuda.memory_allocated(),'GPU_reserved_bytes':torch.cuda.memory_reserved(),**identity}
            (control/'status.json').write_bytes(json_bytes(data));print('GH_ALTERNATING_STAGE',label,index,visited,total,flush=True)
            with (control/'resource_stages.jsonl').open('a') as out:out.write(json.dumps(data,ensure_ascii=False)+'\n')
        try:
            import torch
            from tbps.models.irra import fp32_runtime
            from .configured_training import loaded,run_rounds
            torch.set_num_threads(2);gate.initialize_cuda()
            with fp32_runtime(),loaded(c,bundle,grant) as backend:
                backend.progress=progress;backend.resume_ledger=previous;backend.load_reward_inputs()
                if parent is None:
                    parent=backend.complete_state(0)
                    journal.initialize({'name':f['name'],'configuration':sha_file(path),'native_source':source_key(c)},parent)
                    store.publish('initial_state',{'state.pt':__import__('tbps.alternating.formal_state',fromlist=['tensor_bytes']).tensor_bytes(parent)},
                        {'state':__import__('tbps.alternating.state',fromlist=['tree_sha']).tree_sha(parent),'parents':parent['parents'],
                         'optimizer_mode':'restore moments/steps; explicit configured LR; new independent Scheduler'})
                state,complete=run_rounds(backend,journal,parent,c['prototype']['rounds'],start=resume_round or 0,
                    stop_after=stop_after,stop_requested=lambda:stop[0])
                report={'status':'GH_ALTERNATING_V1_GENERATED' if complete else 'STOPPED_AT_COMMITTED_BOUNDARY',
                    'scope':scope,'complete':complete,'cursor':state['cursor'],'counts':backend.counts,
                    'frozen_sources':backend.final_frozen(),'elapsed_seconds':time.monotonic()-start,
                    'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'resources':resources(f,scope),'configuration_sha256':sha_file(path),'planned_boundary_resume':resume_round is not None,
                    'not_retrieval_performance_evaluation':True}
            # 重新核验完整源码及配置；不创建新模型或重跑训练。
            load(path,scope)
            store.publish('generated' if complete else 'stopped_'+str(state['cursor']['round']),
                {'report.json':json_bytes(report)},{'configuration':sha_file(path),'scope':scope})
            print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
            return complete
        except BaseException:
            error=traceback.format_exc()
            with (root/('FAIL_'+suffix)).open('x') as out:out.write(error)
            (control/'status.json').write_bytes(json_bytes({'phase':'FAIL',**identity,'error':error}))
            raise

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True)
    a.add_argument('--action',choices=('preflight','pilot','train','audit'),default='preflight')
    a.add_argument('--scope',choices=('pilot','train'),default='pilot')
    a.add_argument('--allow-gpu-pilot',action='store_true');a.add_argument('--allow-training',action='store_true')
    a.add_argument('--resume-round',type=int);a.add_argument('--stop-after-round',type=int);v=a.parse_args()
    if v.action in ('pilot','train'):
        complete=run(v.config,v.action,allow_pilot=v.allow_gpu_pilot,allow_training=v.allow_training,
                     resume_round=v.resume_round,stop_after=v.stop_after_round)
        if not complete:raise SystemExit(3)
    elif v.action=='audit':
        from .configured_audit import audit
        print(json.dumps(audit(v.config,v.scope),ensure_ascii=False,indent=2))
    else:
        f,c,b,g,r=prepare(v.config,v.scope);r['resources']=resources(f,v.scope)
        print(json.dumps(r,ensure_ascii=False,indent=2));print(r['status'],flush=True)
