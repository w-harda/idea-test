"""仅完成SVG格式门禁和Git发布；不运行审计、模型、绘图或评测。"""
from pathlib import Path
import sys,json,time,os,subprocess,traceback,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'operations')]
from gh_v2_recovery_v2 import REC,BASE,CFG,AUDIT,store,verify_audit,read_json,sha_file,require,unchanged
from gh_v2_posttraining_publish import publication_version
from gh_v2_joint_plot import verified_rows
from tbps.artifacts import FileRef,checked,json_bytes,digest
CTL=Path('/home/lzf/ldx/tmp/gh_v2_publication_finalize_20261108')
VERSION=ROOT/'configs/framework/gh_v2_publication_finalize_20261108.json'

def xml_semantics(raw):
    def visit(x):
        ws=lambda s:' '.join((s or '').split())
        return [x.tag,sorted((k,ws(v)) for k,v in x.attrib.items()),ws(x.text),ws(x.tail),[visit(c) for c in x]]
    return digest(visit(ET.fromstring(raw)))

def main():
    require(sys.argv[1:]==['--allow-authorized-finalize'],'explicit report-only finalization')
    CTL.mkdir(exist_ok=True)
    with (CTL/'started.json').open('x') as out:json.dump({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'time':time.time(),'training':False,'Test_generation':False,'source_sha256':sha_file(__file__),'version_sha256':sha_file(VERSION)},out)
    def status(phase,**extra):
        p=CTL/'status.pending';p.write_text(json.dumps({'phase':phase,'time':time.time(),**extra},ensure_ascii=False,indent=2));os.replace(p,CTL/'status.json')
    try:
        status('VALIDATING_EXISTING_RESULTS_AND_SVG')
        v=read_json(VERSION,'gh-v2-git-publication-finalize-1');require(v['source_sha256']==sha_file(__file__),'finalizer source SHA')
        for item in v['preserved_publication_files']:
            p=Path(item['path']);s=p.stat();require([s.st_ino,s.st_size,s.st_mtime_ns]==item['stat'],'old publication stat');checked(FileRef(str(p),item['sha256']))
        publication_version();verify_audit();store(REC).verify('completed')
        out=ROOT/'docs/framework/results/gh_alternating_v2_recovery'
        rows=verified_rows(out/'METRICS.csv',v['metrics_sha256'])
        for row in rows:
            n=row['Round'];d=BASE/'test'/f'round_{n:02d}';store(d).verify('completed');store(d/'cpu_audit').verify('completed')
            a=read_json(d/'cpu_audit/completed/report.json');s=read_json(d/'completed/summary.json')
            require(a['status']=='GH_V2_JOINT_TEST_CPU_AUDIT_PASS' and a['scope']=='official_test' and a['round']==n
                and a['gallery']==3074 and a['query']==6156 and a['metrics']==s['metrics']=={k:row[k] for k in ('R@1','R@5','R@10','mAP')},'six exact official results')
        old={n:(out/n).read_bytes() for n in ('R1_by_round.svg','mAP_by_round.svg')}
        archive=store(REC/'svg_format_archive');archive.publish('originals',old,{'derived_SVG_only':True,'no_scientific_artifact_changes':True})
        proof=[]
        for name,raw in old.items():
            clean=('\n'.join(line.rstrip() for line in raw.decode('utf-8').splitlines())+'\n').encode()
            require(xml_semantics(raw)==xml_semantics(clean),'SVG XML semantic equality')
            before=sha_file(out/name);(out/name).write_bytes(clean)
            proof.append({'file':name,'before_sha256':before,'after_sha256':sha_file(out/name),'XML_semantics_equal':True})
        for n,h in v['png_sha256'].items():checked(FileRef(str(out/n),h))
        preservation=unchanged();publication_version()
        e=read_json(out/'SOURCE_EVIDENCE.json');e['final_git_publication']={'source_sha256':sha_file(__file__),'version_sha256':sha_file(VERSION),'SVG_normalization':proof,'prior_publication_FAIL_preserved':True,'control':str(CTL)}
        (out/'SOURCE_EVIDENCE.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n')
        p=out/'REPORT.md';s=p.read_text();s+='\n纯发布阶段的Matplotlib SVG行末空格触发Git whitespace门禁，旧FAIL保留。独立Git收尾仅清除派生SVG行末空白，XML语义摘要清理前后相同，PNG和指标SHA保持；未重绘、重新评测或训练。最终发布控制：`'+str(CTL)+'`，状态见status.json。\n';p.write_text(s)
        p=ROOT/'docs/framework/GH_V2_POSTTRAINING_RECOVERY.md';s=p.read_text();s+='\n仅发布器生成曲线/报告后，Git门禁发现Matplotlib SVG行末空格，原发布FAIL仍保留。最终入口operations/gh_v2_publication_finalize.py只核对现有六结果、归档原SVG、清除派生SVG行末空白并验证XML语义等价，PNG/指标SHA保持，不重绘或运行科学计算。控制/home/lzf/ldx/tmp/gh_v2_publication_finalize_20261108；最终成功状态SUCCESS_ALL_SIX_JOINT_PUBLISHED。\n';p.write_text(s)
        files=['operations/gh_v2_publication_finalize.py','configs/framework/gh_v2_publication_finalize_20261108.json','docs/PROJECT_STATE.md','docs/framework/GH_V2_POSTTRAINING_RECOVERY.md','docs/framework/results/gh_alternating_v2_recovery']
        subprocess.run(['git','add',*files],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs: 发布GH-v2六组Joint正式结果及经校验曲线'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/gh-alternating-v2'],cwd=ROOT,check=True)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/gh-alternating-v2'],cwd=ROOT,text=True).split()[0]
        require(head==remote,'remote SHA exact');status('SUCCESS_ALL_SIX_JOINT_PUBLISHED',commit=head,historical_preservation=preservation,SVG_normalization=proof)
    except BaseException as exc:
        status('FAIL_STOPPED_NO_RETRY',error=repr(exc))
        with (CTL/'FAIL').open('x') as out:out.write(traceback.format_exc())
        raise
if __name__=='__main__':main()
