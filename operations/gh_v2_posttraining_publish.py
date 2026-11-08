"""六组审计通过后的独立仅报告发布；保留原final_report缺依赖FAIL。"""
from pathlib import Path
import sys,os,json,time,subprocess,argparse,traceback,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'operations')]
from gh_v2_recovery_v2 import CFG,BASE,REC,AUDIT,load,store,read_json,sha_file,require,unchanged,tools,verify_audit
from tbps.artifacts import FileRef,checked,json_bytes
from gh_v2_recovery_v2_supervisor import verify_version
PUBCTL=Path('/home/lzf/ldx/tmp/gh_v2_publication_20261108')
PLOTPY='/home/lzf/ldx/envs/deepmisranking/bin/python'
VERSION=ROOT/'configs/framework/gh_v2_publication_20261108.json'

def publication_version():
    v=read_json(VERSION,'gh-v2-publication-only-1')
    for p,h in v['tools'].items():checked(FileRef(p,h))
    for item in v['prior_control_files']:
        p=Path(item['path']);s=p.stat();require([s.st_ino,s.st_size,s.st_mtime_ns]==item['stat'],'prior recovery control stat changed');checked(FileRef(str(p),item['sha256']))
    verify_version();return {'path':str(VERSION),'sha256':sha_file(VERSION),'plot_environment':v['plot_environment']}

def final_report():
    import csv,io
    a=verify_audit();f,c=load(CFG,'test');rows=[];evidence=[]
    for n in f['test_rounds']:
        p=Path(f['test_output'])/f'round_{n:02d}'
        store(p).verify('completed');store(p/'cpu_audit').verify('completed')
        s=read_json(p/'completed/summary.json');v=read_json(p/'cpu_audit/completed/report.json')
        require(v['status']=='GH_V2_JOINT_TEST_CPU_AUDIT_PASS' and v['scope']==s['scope']=='official_test'
            and v['round']==s['round']==n and v['gallery']==s['gallery']==3074 and v['query']==s['query']==6156
            and v['configuration_sha256']==s['configuration_sha256']==sha_file(CFG) and v['metrics']==s['metrics'],'six complete independent official Joint results')
        cp=next(x for x in a['checkpoints'] if x['round']==n)
        require(s['G']=={k:cp['G'][k] for k in ('path','sha256')} and s['H']=={k:cp['H'][k] for k in ('path','sha256')},'same paired model identities')
        rows.append({'Round':n,**s['metrics']});evidence.append({'round':n,'summary':{'path':str(p/'completed/summary.json'),'sha256':sha_file(p/'completed/summary.json')},'audit':v})
    protection=unchanged();out=ROOT/'docs/framework/results/gh_alternating_v2_recovery';out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['Round','R@1','R@5','R@10','mAP'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'METRICS.csv').write_text(stream.getvalue())
    import subprocess,os
    env={**os.environ,'MPLCONFIGDIR':str(PUBCTL/'matplotlib'),'XDG_CACHE_HOME':str(PUBCTL/'cache'),'PYTHONDONTWRITEBYTECODE':'1'}
    plot=subprocess.check_output([PLOTPY,'-B',str(ROOT/'operations/gh_v2_joint_plot.py'),'--metrics',str(out/'METRICS.csv'),
        '--sha256',sha_file(out/'METRICS.csv'),'--output',str(out)],env=env,text=True)
    print(plot,flush=True)
    ev={'configuration_sha256':sha_file(CFG),'audit':{'path':str(AUDIT/'completed/report.json'),'sha256':sha_file(AUDIT/'completed/report.json')},
        'tools':tools(),'publication_tool':publication_version(),'audit_tool_revision':2,'first_recovery_failure_preserved':True,'historical_preservation':protection,'test_results':evidence,'training_attempts':1,'new_training_updates':0}
    (out/'SOURCE_EVIDENCE.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
    lines=['# GH-Alternating-v2：60轮训练后审计恢复与六组Joint正式Test','',
      '**原60轮CPU审计PASS；六组完整Joint Test及独立CPU审计全部PASS。没有重新训练。**','',
      '训练仅原Attempt：484080次G更新、122880次H访问、122220次H更新，训练79478.77秒。原Supervisor的64MiB FAIL、started.json和日志保持。恢复工具及执行身份独立；原158份生产源码、496份依赖和冻结配置保持SHA。','',
      '账本总3890547998字节；第2轮首次超64MiB，超限轮为2/3/4/5/7/9/10。独立读取器在原Manifest和SHA验证后逐项完整严格解析G/H/conditions，检查重复键、UTF-8、非有限值、尾随/截断、字段结构；单项4MiB、块64Ki字符，没有修改全局read_json。','',
      f'独立CPU训练审计覆盖60轮、全部2048×60数据：状态/完整父态声明链、每样本K+1、当前H条件、原S_raw/选择、原JVP回执、冻结隔离、当前RewardBank和六组权重身份。审计耗时{a["seconds"]:.2f}秒，进程峰值RSS {a["peak_RSS_bytes"]/1024**2:.2f}MiB；新模型前向/更新/JVP/CUDA均0。','',
      '证据边界：第1–59轮原始Bank、完整状态和rollout按原流水线滚动策略已退休，逐轮核对不可变账本、退休Manifest、连续状态摘要和训练预算记录；这些轮次不能宣称重新检查已不存在的原始Tensor。第60轮保留的完整Bank/条件/rollout逐项独立读取。全部历史H sampled/greedy输出重算原候选合法性与Natural-K，回放Unicode编辑；每合法原词位≤1字符，无WordBudget1。','',
      '恢复工具第一版新增逐步H检查混用了原生state_sha与轮边界tree_sha，13秒停止、未进行Test；其FAIL和日志完整保留。第二版独立控制身份修正摘要域，使用冻结原state_sha验证整条H更新链，并在初始/六Checkpoint/最新状态同时桥接两种摘要，没有跳过连续性检查。','',
      '六组各自严格加载原G10/H10至G60/H60；每组独立生成3074 Gallery/6156 Query。CUHK-PEDES × IRRA、Joint-only、原生完整Gallery全部同PID正例检索、FP32、L∞≤8/255；无Clean/单模态/额外模型数据集，也不将Test结果反馈训练或选点。','',
      '| Round | R@1 | R@5 | R@10 | mAP |','|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+str(row['Round'])+' | '+' | '.join(f'{row[k]:.2f}' for k in ('R@1','R@5','R@10','mAP'))+' |')
    lines+=['','单位%；越低攻击越有效。原精度见METRICS.csv；这是单条seed42轨迹，不据此推断统计显著或单调收敛。','','![R@1](R1_by_round.png)','','![mAP](mAP_by_round.png)','',
      '原后处理Supervisor已完成六Test/六审计，final_report因原tbps-image缺少matplotlib停止；该FAIL保留。独立仅发布工具改用既有Python3.9.25/Matplotlib3.7.5环境纯CSV绘图，不修改科学环境、不重新评测。','',
      '历史文件SHA/stat复核全部保持；详情见DIAGNOSIS.json、TRAIN_AUDIT.json及SOURCE_EVIDENCE.json。权重、账本、原始输出和大型资产不上传Git。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'TRAIN_AUDIT.json').write_bytes(json_bytes(a))
    d=read_json(REC/'diagnosis/report.json');d.pop('protected_files');(out/'DIAGNOSIS.json').write_bytes(json_bytes(d))
    import re
    state=ROOT/'docs/PROJECT_STATE.md';s=state.read_text();block='<!-- GH_V2_CURRENT_BEGIN -->\n## 当前阶段：GH-v2原60轮审计恢复及六组Joint正式Test完成\n\n原训练60轮未重跑，独立流式CPU审计和六组正式Test/独立审计PASS。原64MiB失败现场保持。[恢复报告](framework/results/gh_alternating_v2_recovery/REPORT.md)。\n<!-- GH_V2_CURRENT_END -->'
    state.write_text(re.sub(r'<!-- GH_V2_CURRENT_BEGIN -->.*?<!-- GH_V2_CURRENT_END -->',block,s,flags=re.S))
    store(REC).publish('completed',{'report.json':json_bytes({'status':'POSTTRAINING_RECOVERY_SIX_JOINT_PASS','rows':rows,'preservation':protection})},{'configuration':sha_file(CFG)})
    return {'rows':rows,'report':str(out/'REPORT.md')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-verified-publication',action='store_true');a=ap.parse_args()
    require(a.allow_verified_publication,'explicit verified report-only publication');version=publication_version()
    with (PUBCTL/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (PUBCTL/'started.json').open('x') as out:json.dump({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'time':time.time(),'version':version,'training':False,'Test_generation':False},out)
        def status(phase,**extra):
            p=PUBCTL/'status.pending';p.write_text(json.dumps({'phase':phase,'time':time.time(),**extra},ensure_ascii=False,indent=2));os.replace(p,PUBCTL/'status.json')
        try:
            status('VERIFIED_SIX_AUDITS_REPORT_ONLY');result=final_report();publication_version()
            files=['docs/PROJECT_STATE.md','docs/framework/results/gh_alternating_v2_recovery']
            subprocess.run(['git','add',*files],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
            subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs: 发布GH-v2六组正式Joint结果及曲线，保留后处理FAIL'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','codex/gh-alternating-v2'],cwd=ROOT,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/gh-alternating-v2'],cwd=ROOT,text=True).split()[0]
            require(head==remote,'remote SHA exact');status('SUCCESS_SIX_JOINT_RESULTS_PUBLISHED',commit=head)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        except BaseException as exc:
            status('FAIL_STOPPED_NO_RETRY',error=repr(exc))
            with (PUBCTL/'FAIL').open('x') as out:out.write(traceback.format_exc())
            raise
if __name__=='__main__':main()
