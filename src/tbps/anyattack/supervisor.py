"""免训练AnyAttack单次执行监督器；全部Smoke及审计通过后自动Full；真实失败不重试。"""
import argparse,subprocess,os,json,time,sys,traceback
from pathlib import Path

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--control',required=True);args=a.parse_args()
    c=json.loads(Path(args.config).read_text());control=Path(args.control);control.mkdir(parents=True,exist_ok=False)
    started={'kind':'AnyAttack official pretrained zero-shot inference; NOT training Attempt','pid':os.getpid(),
        'config':args.config,'started_unix':time.time(),'retry':False,'gpu_policy':'shared_observe_only'}
    with (control/'started.json').open('x') as f:json.dump(started,f,ensure_ascii=False,indent=2)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    stages=[]
    for scope in ('smoke','full'):
        stages += [(scope,'preflight',None),(scope,'generate',None),(scope,'audit-generation',None)]
        for model in ('irra','rasa','aptm'):stages += [(scope,'evaluate',model),(scope,'audit-evaluation',model)]
    state=dict(started,status='RUNNING',completed=[])
    try:
        for i,(scope,action,model) in enumerate(stages):
            name=f'{i:02d}_{scope}_{action}'+('_'+model if model else '')
            cmd=[sys.executable,'-u','-B','-m','tbps.anyattack.cli','--config',args.config,'--scope',scope,'--action',action]
            if model:cmd+=['--model',model]
            if action in ('generate','evaluate'):cmd+=['--allow-inference']
            state.update(stage=name,status='RUNNING');write(control/'status.json',state)
            start=time.time()
            with (control/(name+'.log')).open('xb') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(Path(__file__).resolve().parents[3]))
                state.update(child_pid=p.pid);write(control/'status.json',state);code=p.wait()
            receipt={'stage':name,'exit_code':code,'elapsed_seconds':time.time()-start};write(control/(name+'.json'),receipt)
            if code:raise RuntimeError(f'{name} exit {code}; FAIL_STOPPED_NO_RETRY')
            state['completed'].append(receipt)
        state.update(status='SUCCESS_ALL_THREE_FULL_CPU_AUDITED',finished_unix=time.time());write(control/'status.json',state)
    except BaseException as exc:
        state.update(status='FAIL_STOPPED_NO_RETRY',error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time())
        write(control/'status.json',state);raise
if __name__=='__main__':main()
