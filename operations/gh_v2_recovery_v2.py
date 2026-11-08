"""GH-v2原60轮的独立只读后处理恢复；没有训练入口。"""
from pathlib import Path
import argparse,json,os,resource,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes,digest
from tbps.alternating.contracts import require,BankIdentity
from tbps.alternating.fresh_sources import load
from gh_v2_recovery_stream import scan
CFG=ROOT/'configs/framework/gh_alternating_v2.server.json'
BASE=Path('/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107')
REC=BASE/'posttraining_recovery_20261108'
CTL=Path('/home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108_v2')
OLDCTL=Path('/home/lzf/ldx/tmp/gh_alternating_v2_supervisor_20261107')
AUDIT=REC/'train_cpu_audit_v2'

def store(p):return ArtifactStore(p,new_base=BASE)
def ref(p):return FileRef(str(p),sha_file(p))
def usage():return {'seconds':time.monotonic(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def tools():return {str(p):sha_file(p) for p in sorted((ROOT/'operations').glob('*gh_v2_recovery*.py'))}

def unchanged():
    store(REC).verify('diagnosis');d=read_json(REC/'diagnosis/report.json')
    for v in d['protected_files']:
        p=Path(v['path']);s=p.stat()
        require([s.st_ino,s.st_size,s.st_mtime_ns]==v['stat'],'historical stat changed: '+str(p))
        checked(FileRef(v['path'],v['sha256']))
    return {'historical_files':len(d['protected_files']),'all_SHA256_and_stat_exact':True}

def diagnosis():
    begin=time.monotonic();f,c=load(CFG,'train');r=BASE/'train'
    require(read_json(OLDCTL/'status.json')['phase']=='FAIL_STOPPED_NO_RETRY','original stopped failure required')
    require(not list(r.glob('FAIL*')) and not store(r).pending(),'training failure/pending')
    for directory,identity in ((r,'generated'),(r,'initial_state'),(r/'boundaries','completed'),(r/'boundaries','protocol')):store(directory).verify(identity)
    report=read_json(r/'generated/report.json');completed=read_json(r/'boundaries/completed/report.json')
    expected={'round':60,'g_updates':484080,'h_visits':122880,'h_updates':122220}
    require(report['complete'] and report['cursor']==completed['cursor']==expected,'sixty complete computation receipts')
    require(sorted(p.name for p in (r/'ledgers').glob('round_*'))==[f'round_{i:04d}' for i in range(1,61)],'all sixty ledgers')
    ledgers=[]
    for i in range(1,61):
        d=r/'ledgers'/f'round_{i:04d}';m=store(r/'ledgers').verify(d.name)
        h,counts,stats=scan(d/'ledger.json')
        require(counts=={'G':2048,'H':2048,'conditions':2048} and h['round']==i,'complete ledger structure')
        ledgers.append({'round':i,'path':str(d/'ledger.json'),'size_bytes':(d/'ledger.json').stat().st_size,
          'sha256':m['files_sha256']['ledger.json'],**stats,'array_counts':counts})
        print('DIAGNOSIS_LEDGER',i,ledgers[-1]['size_bytes'],flush=True)
    cps=[]
    for i in f['test_rounds']:
        m=store(r/'checkpoints').verify(f'round_{i:02d}')
        cps.append({'round':i,'manifest':m,'files':[{'path':str(r/'checkpoints'/f'round_{i:02d}'/n),'sha256':h,
            'size_bytes':(r/'checkpoints'/f'round_{i:02d}'/n).stat().st_size} for n,h in m['files_sha256'].items()]})
    files=[]
    for root in (BASE,OLDCTL):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_relative_to(REC):continue
            require(not p.is_symlink(),'historical symlink')
            s=p.stat();files.append({'path':str(p),'stat':[s.st_ino,s.st_size,s.st_mtime_ns],'sha256':sha_file(p)})
    result={'schema':'gh-v2-posttraining-diagnosis-1','status':'READONLY_STRUCTURE_SHA_PASS_NOT_NUMERICAL_AUDIT',
        'configuration_sha256':sha_file(CFG),'production_producers':len(f['producers']),'runtime_pins':len(f['runtime_pins']),
        'training_receipt':report,'completed_receipt':completed,'ledgers':ledgers,'checkpoints':cps,'protected_files':files,
        'first_above_64MiB':next(x['round'] for x in ledgers if x['size_bytes']>64*1024**2),
        'over_limit_rounds':[x['round'] for x in ledgers if x['size_bytes']>64*1024**2],
        'total_ledger_bytes':sum(x['size_bytes'] for x in ledgers),'seconds':time.monotonic()-begin,
        'peak_RSS_bytes':usage()['peak_RSS_bytes'],'tools':tools(),
        'numerical_budget_audit':'PENDING','historical_tensor_retirement':'original rolling policy; rounds1..59 bank/state/rollout retired, sealed ledgers and retirement manifests retained'}
    store(REC).publish('diagnosis',{'report.json':json_bytes(result)},{'configuration':sha_file(CFG),'read_only':True})
    print(json.dumps({k:v for k,v in result.items() if k not in ('protected_files','ledgers','checkpoints')},ensure_ascii=False),flush=True)
    return result


def audit():
    begin=time.monotonic();f,c=load(CFG,'train');r=BASE/'train';diag=read_json(REC/'diagnosis/report.json')
    store(REC).verify('diagnosis');require(not AUDIT.exists(),'new audit identity consumed')
    before_protection=unchanged()
    import torch
    torch.set_num_threads(2)
    from tbps.alternating.parents import cpu_restore_only
    from tbps.alternating.native_sources import native_session
    from tbps.alternating.fresh_audit import validate_action
    from tbps.alternating.configured_audit import verify_bank
    from tbps.alternating.state import tree_sha
    from tbps.ablations.gv2_epoch_state import finite_state,adam_steps
    from tbps.alternating.native_backend import assert_pixels
    require(not torch.cuda.is_initialized(),'separate CPU process')
    allowed=[r/'initial_state/state.pt',r/'boundaries/round_0060/state.pt']
    allowed+=list((r/'checkpoints').glob('round_*/*.pt'))
    allowed+=list((r/'B/bank_60').glob('image_*/data.pt'))+[r/'B/bank_60/G_state/state.pt',r/'B/bank_60/completed/features.pt']
    allowed+=list((r/'round_io/round_0060').glob('condition_*/condition.pt'))
    rows=[];cp_results=[];max_linf=0.;max_edits=0;actions=0;max_buffer=0
    gm={x['row_id']:x for x in read_json(c['prototype']['g_manifest'])['entries'] if x['role']=='train'}
    allowed+=list(Path(x['tensor_path']) for x in gm.values())
    qualification=read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries']
    require([x['identity']['row_id'] for x in qualification]==c['prototype']['train_ids']==c['prototype']['samples'],'original full Train order')
    q={x['identity']['row_id']:x for x in qualification};ids=c['prototype']['train_ids']
    store(r).verify('generated');gpu=read_json(r/'generated/report.json')
    require(gpu['complete'] and gpu['frozen_sources'] and gpu['historical_G_H_loads']==0 and gpu['configuration_sha256']==sha_file(CFG),'original complete frozen fresh GPU receipt')
    store(r/'boundaries').verify('completed');end_receipt=read_json(r/'boundaries/completed/report.json')
    store(r).verify('initial_state')
    with cpu_restore_only(allowed) as guards,native_session(c) as session,torch.no_grad():
        contracts=session.import_module('contracts');constraints=session.import_module('constraints');policy=session.import_module('policy')
        tokenizer=session.import_module('utils.simple_tokenizer').SimpleTokenizer(bpe_path=c['sources']['bpe']['path'])
        probe=session.import_module('tokenizer_probe').TokenizerProbe(tokenizer)
        initial=torch.load(r/'initial_state/state.pt',map_location='cpu',weights_only=False)
        proof=read_json(r/'initial_state/proof.json')
        zero={'round':0,'g_updates':0,'h_visits':0,'h_updates':0}
        require(initial['cursor']==zero and not initial['G_Adam']['state'] and not initial['H_Adam']['state']
            and initial['schedulers']=={'G':None,'H':None} and proof['trained_checkpoint_loads']==0
            and proof['empty_G_Adam'] and proof['empty_H_Adam'] and proof['word_budget1_installed'] is False,'fresh original initial state')
        require(tree_sha(initial['G'])==proof['initial_G'] and tree_sha(initial['H'])==proof['initial_H'],'initial model identity')
        baseline={}
        for row,g in gm.items():
            checked(FileRef(g['metadata_path'],g['metadata_sha256']));checked(FileRef(g['tensor_path'],g['tensor_sha256']))
            value=torch.load(g['tensor_path'],map_location='cpu',weights_only=True)
            baseline[row]={'ids':value['selected_instance_ids'],'scores':value['S_raw'].tolist()}
        previous_g=proof['initial_G'];previous_h=proof['initial_H'];cursor=zero;state_sha=tree_sha(initial)
        from types import SimpleNamespace
        native_hash=session.import_module('feedback_backend').state_sha
        native_h=lambda value:native_hash(SimpleNamespace(state_dict=lambda:value))
        previous_native_h=native_h(initial['H'])
        initial_hash_bridge={'tree_sha':previous_h,'native_state_sha':previous_native_h}
        counts={k:0 for k in gpu['counts']};available_cache={};latest=None
        def check_action(a,row):
            nonlocal max_edits,actions
            require(a['original']==q[row]['caption'],'original training Caption identity')
            count=validate_action(a,contracts,constraints);actions+=1;max_edits=max(max_edits,count)
            ex=contracts.from_text_spans(a['original'],a['attributes'])
            key=digest({'caption':a['original'],'attrs':a['attributes']})
            if key not in available_cache:
                available_cache[key]=[i for i in range(len(ex.attributes)) if constraints.candidates(ex,ex.caption,i,set(),probe)]
            supports=available_cache[key]
            raw=a['predicted_importance'];require(len(raw)==len(ex.attributes),'importance/all Canonical mapping')
            require(a['budget_k']==policy.natural_k([raw[i] for i in supports]),'original gamma3/beta1 Natural-K')
            require(len(a['selected_attribute_ids'])==len(set(a['selected_attribute_ids']))<=a['budget_k'],'without replacement attribute budget')
            require(set(e['attribute_id'] for e in a['edits'])<=set(a['selected_attribute_ids']),'edits belong to selected attributes')
            for dec in a['decisions']:require(dec['index'] in dec['legal_indices'],'masked decision support')
            ta=a['tokenizer_audit'];require(not ta['new_truncation'],'no new truncation')
            return count
        for i in range(1,61):
            tick=time.monotonic();d=r/'ledgers'/f'round_{i:04d}';m=store(r/'ledgers').verify(d.name)
            require(m['provenance']['configuration']==sha_file(CFG),'ledger configuration source')
            claim=store(r/'boundaries').verify(f'claim_{i:04d}');cd=read_json(r/'boundaries'/f'claim_{i:04d}/claim.json')
            require(cd=={'round':i,'parent_state':state_sha},'complete committed parent-state claim chain')
            require(claim['provenance']['configuration']==sha_file(CFG),'claim configuration')
            acc={'g':0,'h':0,'jvp':0,'G_state':previous_g,'H_state':previous_native_h,'banks':set(),'G_rows':[], 'H_rows':[], 'condition_rows':[], 'condition_meta':[]}
            def item(key,j,x):
                nonlocal max_linf
                require(j<2048,'unexpected extra Train row')
                row=ids[j]
                if key=='G':
                    k=gm[row]['K'];require(x['row_id']==row and x['pass_index']==1 and x['K']==k
                        and x['optimizer_steps']==x['G_forwards']==x['encoder_forwards']==k+1,'per-row original G K+1')
                    require(x['G_before']==acc['G_state'],'continuous within-G updates');acc['G_state']=x['G_after']
                    require(x['weights']=={'lambda_global':1.,'lambda_local':1.,'lambda_retain':1.},'original G losses')
                    require(len(x['records'])==len(x['gradient_observations'])==k+1,'every G update recorded')
                    for t,(record,obs) in enumerate(zip(x['records'],x['gradient_observations'])):
                        require(record==obs['record'] and record['finite_gradients'] is True and record['update_index']==t+1,'finite recorded gradients')
                        require(0<=record['cumulative_linf']<=8/255,'recorded strict training image budget')
                        require(record['phase']==('joint' if t==k else 'attribute'),'serial and original-image joint ordering')
                        require(record['instance_id']==(None if t==k else baseline[row]['ids'][t]),'original selected attribute order')
                        max_linf=max(max_linf,record['cumulative_linf'])
                    acc['g']+=k+1;acc['G_rows'].append(row)
                elif key=='H':
                    require(x['row_id']==row and x['pass_index']==1,'all H visits original order')
                    eligible=q[row]['eligible'];require(x['optimizer_steps_this_step']==int(eligible),'eligible H update count')
                    if not eligible:require(x.get('fixed_NOOP') is True,'ineligible fixed NOOP')
                    else:
                        require(not x.get('fixed_NOOP') and x['H_state_sha256_before']==acc['H_state'],'continuous within-H updates')
                        acc['H_state']=x['H_state_sha256_after'];acc['banks'].add(x['bank_sha256'])
                        require(x['G_calls']==x['IRRA_image_calls']==0 and x['IRRA_text_samples']==3,'H reward no image regeneration')
                        for mode in ('sampled','greedy'):check_action(x[mode],row)
                    acc['h']+=x['optimizer_steps_this_step'];acc['H_rows'].append(row)
                else:
                    require(x['row']==row,'all conditions original order');ev=x['evidence'];ident=x['identity']
                    require(ident['original_caption_sha256']==__import__('hashlib').sha256(q[row]['caption'].encode()).hexdigest(),'original condition Caption identity')
                    require(ident['source_space_sha256']==c['condition_space_sha256'],'condition frozen source space')
                    require(ident['selected_g_ids']==baseline[row]['ids'] and ident['scores']==baseline[row]['scores'],'fixed original G selection/S_raw')
                    changed=ident['original_caption_sha256']!=ident['context_caption_sha256']
                    if changed:
                        require(ev['fresh_localization'] and ev['old_JVP_relabelled'] is False and ev['new_JVPs']==gm[row]['K'],'fresh H context JVP evidence')
                        require(len(ev['JVP_checks'])==ev['new_JVPs'] and all(t['zero_exact'] and 0<=t['central_fd_relative_error']<=.02 for t in ev['JVP_checks']),'all original JVP zero/FD checks')
                    else:require(ev['new_JVPs']==ev['new_text_encodings']==0,'unchanged Caption no new JVP')
                    acc['jvp']+=ev['new_JVPs'];acc['condition_rows'].append(row)
                    if i==60:
                        io=store(r/'round_io/round_0060');io.verify(f'condition_{j:04d}');io.verify(f'rollout_{j:04d}')
                        require(read_json(io.root/f'condition_{j:04d}/evidence.json')==x,'latest condition sealed original evidence')
                        value=torch.load(io.root/f'condition_{j:04d}/condition.pt',map_location='cpu',weights_only=True)
                        require(tree_sha(value)==x['payload_content_sha256'] and value['K']==gm[row]['K'],'latest condition original Tensor')
                        act=read_json(io.root/f'rollout_{j:04d}/action.json')
                        require(act['original']==q[row]['caption'] and __import__('hashlib').sha256(act['adversarial'].encode()).hexdigest()==ident['context_caption_sha256'],'latest G rollout/context binding')
                        if q[row]['eligible']:check_action(act['native_action'],row)
                        else:require(act['original']==act['adversarial'] and not act['edits'],'latest fixed NOOP')
            h,n,stats=scan(d/'ledger.json',item)
            require(set(h)=={'G_after','G_before','G_changed','H_after','H_before','H_changed','bank','condition_H_state','end_cursor','forward_counts_before','forward_counts_cumulative','g_orders','h_orders','isolation','isolation_fingerprints','learning_rates','round','seconds','start_cursor'},'complete ledger field structure')
            require(n=={'G':2048,'H':2048,'conditions':2048} and acc['G_rows']==acc['H_rows']==acc['condition_rows']==ids,'complete all2048 coverage')
            require(h['round']==i and h['G_before']==previous_g and h['H_before']==previous_h and h['start_cursor']==cursor,'continuous G/H round state')
            require(acc['G_state']==h['G_after'],'all G per-row states reach boundary')
            previous_native_h=acc['H_state']
            require(h['g_orders']==h['h_orders']==[ids] and acc['g']==8068 and acc['h']==2037,'full original one pass counts')
            require(h['end_cursor']=={'round':i,'g_updates':cursor['g_updates']+8068,'h_visits':cursor['h_visits']+2048,'h_updates':cursor['h_updates']+2037},'cumulative G/H counts')
            require(h['G_changed'] and h['H_changed'],'both models changed over each round')
            require(h['condition_H_state']==previous_h and all(v is True for v in h['isolation'].values()),'current H condition/frozen isolation')
            fp=h['isolation_fingerprints'];require(fp['inactive_H_before_G']==fp['inactive_H_after_G']
                and fp['inactive_G_before_H']==fp['inactive_G_after_H'],'inactive models and Optimizer fingerprints exact')
            require(fp['inactive_H_before_G'][0]==previous_h and fp['inactive_G_before_H'][0]==h['G_after'],'phase identity')
            bank=BankIdentity(**{**h['bank'],'ordered_train_ids':tuple(h['bank']['ordered_train_ids'])})
            bank.validate(ids,h['G_after'],i,'B',space=c['prototype']['source_space_sha256'],clean=c['prototype']['clean_gallery_sha256'])
            require(acc['banks']=={bank.adversarial_sha256},'all H updates current full RewardBank')
            require(h['forward_counts_before']==counts and h['forward_counts_cumulative']['JVP']==counts['JVP']+acc['jvp'],'forward/JVP continuous accounting')
            counts=h['forward_counts_cumulative'];require(h['learning_rates']=={'G':.0002,'H':.0001},'original fixed learning rates')
            if i<60:
                store(r/'boundaries').verify(f'retired_{i:04d}');ret=read_json(r/'boundaries'/f'retired_{i:04d}/retirement.json')
                bm=ret['boundary_manifest'];require(ret['round']==i and ret['replaced_by']==i+1
                    and bm['files_sha256']['ledger.json']==m['files_sha256']['ledger.json'] and bm['provenance']==m['provenance'],'sealed retired boundary ledger/state identity')
                require(bm['sources']==[{'path':str(r/'B'/f'bank_{i}/completed/manifest.json'),'sha256':digest(ret['bank_manifest'])}]
                    and ret['bank_manifest']['provenance']['bank']==h['bank'],'retired complete Bank manifest identity')
                require(not (r/'boundaries'/f'round_{i:04d}').exists() and not (r/'B'/f'bank_{i}').exists(),'original explicit rolling retirement')
            if i in f['test_rounds']:
                cm=store(r/'checkpoints').verify(f'round_{i:02d}');pair={'round':i,'ledger_sha256':m['files_sha256']['ledger.json']}
                for side in ('G','H'):
                    p=r/'checkpoints'/f'round_{i:02d}'/(side+'.pt');v=torch.load(p,map_location='cpu',weights_only=False)
                    finite_state(v['state_dict'])
                    require(v['schema']=='gh-v2-eval-weight-1' and v['round']==i and v['model']==side
                        and v['configuration_sha256']==sha_file(CFG) and v['training_ledger_sha256']==pair['ledger_sha256']
                        and tree_sha(v['state_dict'])==h[side+'_after']==cm['provenance'][side+'_state_sha'],'paired G/H endpoint exact identity')
                    if side=='H':require(native_h(v['state_dict'])==previous_native_h,'native H update chain reaches paired checkpoint')
                    pair[side]={'path':str(p),'sha256':cm['files_sha256'][side+'.pt'],'state_sha256':h[side+'_after']}
                cp_results.append(pair)
            state_sha=m['provenance']['state_sha'];previous_g=h['G_after'];previous_h=h['H_after'];cursor=h['end_cursor'];latest=h
            max_buffer=max(max_buffer,stats['max_buffer_characters'])
            rows.append({'round':i,'G_updates':acc['g'],'H_visits':2048,'H_updates':acc['h'],'new_JVPs':acc['jvp'],
                'G_after':previous_g,'H_after':previous_h,'H_native_after':previous_native_h,'ledger_sha256':m['files_sha256']['ledger.json'],
                'bank':h['bank']['adversarial_sha256'],'audit_seconds':time.monotonic()-tick})
            checked(FileRef(str(d/'ledger.json'),m['files_sha256']['ledger.json']))
            print('RECOVERY_TRAIN_AUDIT_ROUND_PASS',i,cursor,flush=True)
        store(r/'boundaries').verify('round_0060');last=torch.load(r/'boundaries/round_0060/state.pt',map_location='cpu',weights_only=False)
        for name in ('G','H','G_Adam','H_Adam'):finite_state(last[name])
        require(tree_sha(last)==state_sha==end_receipt['state_sha'] and last['cursor']==cursor==gpu['cursor'],'latest full state and completion exact')
        require(native_h(last['H'])==previous_native_h,'latest H native chain endpoint')
        require(tree_sha(last['G'])==previous_g and tree_sha(last['H'])==previous_h and last['parents']==initial['parents'],'latest G/H/parents identity')
        require(adam_steps(last['G_Adam'])==[484080] and max(adam_steps(last['H_Adam']))==122220,'latest dual Adam step accounting')
        require(last['schedulers']=={'G':None,'H':None} and last['training_configuration']==initial['training_configuration'],'original no Scheduler/fixed configuration')
        for side in ('G','H'):require(set(last[side+'_RNG'])=={'python','numpy','torch_cpu','cuda','sampler'} and len(last[side+'_RNG']['cuda'])==1,'complete stage RNG')
        bank_result=verify_bank(r,latest,c)
        require(counts==gpu['counts'] and counts['G']==counts['IRRA_image']==606960 and counts['H']==366660,'full actual counts and 2048banks')
    load(CFG,'train');preservation=unchanged();require(not torch.cuda.is_initialized(),'CPU audit did not initialize CUDA')
    result={'status':'GH_V2_TRAIN_CPU_AUDIT_PASS','schema':'gh-v2-posttraining-stream-audit-1','configuration_sha256':sha_file(CFG),
        'rounds':rows,'covered_rounds':60,'covered_Train_visits':122880,'state_continuity':True,'initial_H_hash_bridge':initial_hash_bridge,'H_hash_semantics':'native state_sha within/across step chain; tree_sha boundary chain; both independently recomputed at initial, six paired endpoints and latest state','G_updates':484080,'H_visits':122880,'H_updates':122220,
        'checkpoints':cp_results,'fresh_initialization':proof,'counts':counts,'latest_bank_audit':bank_result,
        'maximum_recorded_training_Linf':max_linf,'maximum_observed_H_edited_words':max_edits,'H_action_outputs_checked':actions,
        'text_budget':'original native Natural-K recomputed from original candidate availability; Unicode replay; <=1 char/native word',
        'retired_evidence_scope':'rounds1..59 raw bank/state/rollout tensors retired by original policy; sealed ledger/retirement manifests verified; all retained H sampled/greedy actions replayed; round60 full bank/conditions/rollouts read independently',
        'historical_preservation':preservation,'seconds':time.monotonic()-begin,'peak_RSS_bytes':usage()['peak_RSS_bytes'],
        'stream_max_buffer_characters':max_buffer,'tools':tools(),'model_forwards':0,'optimizer_updates':0,'new_JVPs':0,'CUDA':False,
        'source_compatibility':'original158 producers+496 pins+configuration unmodified; independent operations tool identity recorded; no SHA bypass'}
    store(AUDIT).publish('completed',{'report.json':json_bytes(result)}, {'configuration':sha_file(CFG),'posttraining_recovery':True},sources=(ref(REC/'diagnosis/manifest.json'),))
    return result


def verify_audit():
    store(AUDIT).verify('completed');a=read_json(AUDIT/'completed/report.json')
    require(a['status']=='GH_V2_TRAIN_CPU_AUDIT_PASS' and a['covered_rounds']==60
        and a['configuration_sha256']==sha_file(CFG) and a['G_updates']==484080
        and a['H_updates']==122220 and a['H_visits']==122880,'complete matching recovery CPU audit gate')
    return a

def evaluation(n):
    require(n in (10,20,30,40,50,60),'fixed paired endpoints only');a=verify_audit()
    f,c=load(CFG,'test');store(BASE/'train/checkpoints').verify(f'round_{n:02d}')
    for side in ('G','H'):
        v=next(v for v in a['checkpoints'] if v['round']==n)[side];checked(FileRef(v['path'],v['sha256']))
    import torch
    from unittest.mock import patch
    from tbps.alternating import fresh_evaluation as original
    from tbps.alternating.parents import no_optimizer_steps,cpu_restore_only,ParentBundle
    from tbps.alternating.native_sources import native_session,bindings,strict_import_preflight
    from tbps.alternating.state import tree_sha
    from tbps.materialization.irra_startup import CPUStartupGate
    def restored_preflight(path,scope):
        require(Path(path)==CFG and scope=='test','evaluation-only restored checkpoint preflight')
        f,c=load(path,scope);strict_import_preflight(c)
        cp=store(BASE/'train/checkpoints');cp.verify(f'round_{n:02d}')
        refs=[cp.root/f'round_{n:02d}'/(x+'.pt') for x in ('G','H')]
        with native_session(c) as session,cpu_restore_only(refs):
            b=bindings(session);gc=read_json(f['fixed_resources']['G_original_config']['path']);hc=read_json(f['fixed_resources']['H_original_config']['path'])
            tokenizer=b.get('utils.simple_tokenizer').SimpleTokenizer(bpe_path=c['sources']['bpe']['path'])
            probe=b.get('tokenizer_probe').TokenizerProbe(tokenizer)
            # 只构建架构后立即严格载入现有端点；不创建训练初始状态、Adam或阶段RNG。
            g=b.get('image_only_generator').ImageOnlyGenerator(**gc['architecture'])
            h=b.get('policy').CharacterPolicy(probe,**hc['architecture'])
            for side,module,p in (('G',g,refs[0]),('H',h,refs[1])):
                v=torch.load(p,map_location='cpu',weights_only=False)
                require(v['round']==n and v['model']==side and v['configuration_sha256']==sha_file(CFG),'restored paired checkpoint')
                module.load_state_dict(v['state_dict'],strict=True);require(tree_sha(module.state_dict())==tree_sha(v['state_dict']),'restored state exact')
                module.eval().requires_grad_(False)
            bundle=ParentBundle(g,h,None,None,None,{},b,{}, {'posttraining_checkpoint_restore':True},probe,b.get('text_frontend').CaptionFrontend())
        bundle.modules=session.modules
        f['audits']={**f['audits'],'train':str(AUDIT)}
        gate=CPUStartupGate();gate.checked();gate.exit_checked()
        return f,c,bundle,gate,{'restored_G_H_round':n,'new_training_state':False}
    with no_optimizer_steps(),patch.object(original,'preflight',restored_preflight):
        return original.evaluate(CFG,n,allow=True)

def test_audit(n):
    verify_audit()
    from tbps.alternating.fresh_audit import audit_test
    return audit_test(CFG,n)

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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,color in (('R@1','R1_by_round','#2662bb'),('mAP','mAP_by_round','#c54b50')):
        fig,ax=plt.subplots(figsize=(6.4,4));ax.plot([x['Round'] for x in rows],[x[metric] for x in rows],marker='o',color=color)
        ax.set(xlabel='Alternating training round',ylabel=metric+' (%)',xticks=f['test_rounds']);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    ev={'configuration_sha256':sha_file(CFG),'audit':{'path':str(AUDIT/'completed/report.json'),'sha256':sha_file(AUDIT/'completed/report.json')},
        'tools':tools(),'audit_tool_revision':2,'first_recovery_failure_preserved':True,'historical_preservation':protection,'test_results':evidence,'training_attempts':1,'new_training_updates':0}
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
    ap=argparse.ArgumentParser();ap.add_argument('--action',required=True,choices=('audit','test','test-audit','report','verify'));ap.add_argument('--round',type=int);a=ap.parse_args()
    begin=time.monotonic()
    from gh_v2_recovery_v2_supervisor import verify_version
    verify_version()
    try:
        result={'audit':audit,'test':lambda:evaluation(a.round),'test-audit':lambda:test_audit(a.round),'report':final_report,'verify':unchanged}[a.action]()
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:print('RECOVERY_ACTION_RESOURCES',a.action,{'seconds':time.monotonic()-begin,'peak_RSS_bytes':usage()['peak_RSS_bytes']},flush=True)
if __name__=='__main__':main()
