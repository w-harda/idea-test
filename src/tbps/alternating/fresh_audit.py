"""v2独立CPU审计；滚动退休的旧Tensor以已封存账本/退休证据核对。"""
from pathlib import Path
from tbps.artifacts import ArtifactStore,FileRef,sha_file,read_json,json_bytes,digest
from .contracts import require
from .fresh_sources import load
from .state import tree_sha
from .native_sources import native_session
from .parents import no_optimizer_steps
from .configured_audit import verify_bank

def validate_action(action,contracts,constraints):
    example=contracts.from_text_spans(action['original'],action['attributes'])
    records=tuple(constraints.EditRecord(**{**x,'source_span':tuple(x['source_span'])}) for x in action['edits'])
    require(constraints.replay(example,records,action['budget_k'])==action['adversarial'],'original Natural-K Unicode replay')
    words=[contracts.word_at(example.caption,x.position) for x in records]
    require(len(words)==len(set(words)) and len(words)<=action['budget_k'],'one character per selected/shared word')
    return len(words)

def audit(path,scope):
    f,c=load(path,scope)
    import torch
    torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'independent CPU process required')
    root=Path(f['outputs'][scope]);store=ArtifactStore(root,new_base=f['output_base'])
    require(not list(root.glob('FAIL*')) and not store.pending(),'FAIL/pending retained')
    store.verify('generated');report=read_json(root/'generated/report.json')
    require(report['complete'] and report['frozen_sources'] and report['historical_G_H_loads']==0
        and report['configuration_sha256']==sha_file(path),'complete fresh GPU receipt')
    store.verify('initial_state');initial=torch.load(root/'initial_state/state.pt',map_location='cpu',weights_only=False)
    proof=read_json(root/'initial_state/proof.json')
    require(initial['cursor']=={'round':0,'g_updates':0,'h_visits':0,'h_updates':0}
        and not initial['G_Adam']['state'] and not initial['H_Adam']['state'] and proof['trained_checkpoint_loads']==0
        and initial['schedulers']=={'G':None,'H':None},'fresh empty initial state')
    require(tree_sha(initial['G'])==proof['initial_G'] and tree_sha(initial['H'])==proof['initial_H'],'fresh initial model fingerprints')
    if scope=='smoke':
        store.verify('dynamic_budget_probe');multi=read_json(root/'dynamic_budget_probe/report.json')
        require(multi['status']=='ORIGINAL_DYNAMIC_K_MULTIWORD_PASS' and len(multi['changed_word_positions'])>=2,'multiword budget proof')
    n=c['prototype']['rounds'];expectedg=27 if scope=='smoke' else 8068;expectedh=7 if scope=='smoke' else 2037
    bs=ArtifactStore(root/'boundaries',new_base=f['output_base']);ls=ArtifactStore(root/'ledgers',new_base=f['output_base'])
    bs.verify('completed');previous_g=proof['initial_G'];previous_h=proof['initial_H'];prev=initial['cursor'];rows=[]
    max_edits=0;outputs=0
    from .parents import cpu_restore_only
    allowed=list((root/'checkpoints').glob('round_*/*.pt'))+list((root/'boundaries').glob('round_*/state.pt'))
    allowed+=list((root/'B').glob('bank_*/image_*/data.pt'))+list((root/'B').glob('bank_*/G_state/state.pt'))
    allowed+=list((root/'B').glob('bank_*/completed/features.pt'))
    with cpu_restore_only(allowed),native_session(c) as session,no_optimizer_steps(),torch.no_grad():
        contracts=session.import_module('contracts');constraints=session.import_module('constraints')
        for i in range(1,n+1):
            ls.verify(f'round_{i:04d}');log=read_json(root/'ledgers'/f'round_{i:04d}/ledger.json')
            require(log['G_before']==previous_g and log['H_before']==previous_h and log['start_cursor']==prev,'continuous updated-state chain')
            require(log['g_orders']==[c['prototype']['samples']] and log['h_orders']==[c['prototype']['samples']],'fixed full data order')
            gu=sum(x['optimizer_steps'] for x in log['G']);hu=sum(x['optimizer_steps_this_step'] for x in log['H'])
            require(gu==expectedg and hu==expectedh and len(log['H'])==len(c['prototype']['samples']),'original update accounting')
            require(log['end_cursor']=={'round':i,'g_updates':prev['g_updates']+gu,'h_visits':prev['h_visits']+len(log['H']),
                'h_updates':prev['h_updates']+hu},'zero-origin cumulative accounting')
            require(log['condition_H_state']==previous_h and all(log['isolation'].values()),'current H and freeze isolation')
            fp=log['isolation_fingerprints'];require(fp['inactive_H_before_G']==fp['inactive_H_after_G']
                and fp['inactive_G_before_H']==fp['inactive_G_after_H'],'inactive Optimizer/model exact')
            require(log['bank']['g_state_sha256']==log['G_after'] and log['bank']['ordered_train_ids']==c['prototype']['train_ids'],'current full G bank')
            for cond in log['conditions']:
                ev=cond['evidence'];ident=cond['identity']
                if ident['original_caption_sha256']!=ident['context_caption_sha256']:
                    require(ev['fresh_localization'] and ev['old_JVP_relabelled'] is False and ev['new_JVPs']==len(ident['selected_g_ids']), 'fresh context')
                    require(all(q['zero_exact'] and q['central_fd_relative_error']<=.02 for q in ev['JVP_checks']),'real JVP/FD')
            for h in log['H']:
                if h.get('fixed_NOOP'):continue
                require(h['bank_sha256']==log['bank']['adversarial_sha256'],'H reward latest bank')
                for mode in ('sampled','greedy'):
                    count=validate_action(h[mode],contracts,constraints);max_edits=max(max_edits,count);outputs+=1
            previous_g,previous_h,prev=log['G_after'],log['H_after'],log['end_cursor']
            rows.append({'round':i,'G_updates':gu,'H_updates':hu,'cursor':prev,'bank':log['bank'],
                'new_JVPs':sum(x['evidence']['new_JVPs'] for x in log['conditions']),'seconds':log['seconds']})
            if i<n:
                bs.verify(f'retired_{i:04d}');require(not (root/'boundaries'/f'round_{i:04d}').exists(),'old full state retention')
        bs.verify(f'round_{n:04d}');last=torch.load(root/'boundaries'/f'round_{n:04d}/state.pt',map_location='cpu',weights_only=False)
        require(last['cursor']==prev and tree_sha(last['G'])==previous_g and tree_sha(last['H'])==previous_h,'latest resume state')
        from tbps.ablations.gv2_epoch_state import adam_steps
        require(adam_steps(last['G_Adam'])==[prev['g_updates']] and max(adam_steps(last['H_Adam']))==prev['h_updates'],'latest dual Adam steps')
        require(last['schedulers']=={'G':None,'H':None},'original Scheduler settings')
        latest_log=read_json(root/'ledgers'/f'round_{n:04d}/ledger.json');bank=verify_bank(root,latest_log,c)
        for i in ([1] if scope=='smoke' else f['test_rounds']):
            cp=ArtifactStore(root/'checkpoints',new_base=f['output_base']);cp.verify(f'round_{i:02d}')
            log=read_json(root/'ledgers'/f'round_{i:04d}/ledger.json')
            for side in ('G','H'):
                value=torch.load(root/'checkpoints'/f'round_{i:02d}'/(side+'.pt'),map_location='cpu',weights_only=False)
                require(value['round']==i and value['model']==side and value['configuration_sha256']==sha_file(path)
                    and value['training_ledger_sha256']==sha_file(root/'ledgers'/f'round_{i:04d}/ledger.json')
                    and tree_sha(value['state_dict'])==log[side+'_after'],'paired evaluation Checkpoint identity')
    require(report['counts']['G']==report['counts']['IRRA_image']==n*(expectedg+2048),'actual G forwards')
    result={'status':'GH_V2_'+scope.upper()+'_CPU_AUDIT_PASS','configuration_sha256':sha_file(path),'rounds':rows,
        'fresh_initialization':proof,'word_budget':'original Dynamic-K; each native word <=1 character',
        'maximum_observed_H_edited_words':max_edits,'H_action_outputs_checked':outputs,'latest_bank_audit':bank,
        'rolling_old_tensor_audit':'sealed ledgers/retirement manifests; latest tensors independently checked',
        'not_Test_metrics':True}
    ArtifactStore(f['audits'][scope],new_base=f['output_base']).publish('completed',{'report.json':json_bytes(result)},
        {'configuration':sha_file(path),'scope':scope})
    return result


def audit_test(path,round_index,*,smoke=False):
    f,c=load(path,'test')
    import torch
    torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'separate CPU Test audit')
    from .fresh_evaluation import manifest_for
    from .native_backend import assert_pixels
    from tbps.engines.metrics import ranking
    from .parents import cpu_restore_only
    root=Path(f['test_output'])/('smoke' if smoke else f'round_{round_index:02d}')
    store=ArtifactStore(root,new_base=f['output_base']);require(not (root/'FAIL').exists(),'Test failure retained')
    store.verify('completed');summary=read_json(root/'completed/summary.json')
    require(summary['round']==round_index and summary['configuration_sha256']==sha_file(path) and summary['frozen_Target'],'Test paired endpoint/protocol')
    manifest=manifest_for(c,smoke);attack=ArtifactStore(root/'attack',new_base=f['output_base'])
    attack.verify('completed');engine=ArtifactStore(root/'results',new_base=f['output_base']);engine.verify('engine')
    protocol=read_json(root/'results/engine/protocol.json')
    require(protocol['modes']==['Joint'] and protocol['query_chunk']==128 and protocol['dataset_manifest']==__import__('json').loads(json_bytes(__import__('dataclasses').asdict(manifest))),'same native Joint protocol')
    allowed=list((root/'attack').glob('image_*/data.pt'))+[root/'results/engine/scores/Joint.pt']
    max_linf=0.;max_edits=0
    with cpu_restore_only(allowed),native_session(c) as session:
        contracts=session.import_module('contracts');constraints=session.import_module('constraints')
        for i,g in enumerate(manifest.gallery):
            attack.verify(f'image_{i:04d}');v=torch.load(root/'attack'/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)
            assert_pixels(v['original'],v['adversarial'],v['effective_delta']);max_linf=max(max_linf,float(v['effective_delta'].double().abs().max()))
        for i,q in enumerate(manifest.queries):
            attack.verify(f'text_{i:04d}');a=read_json(root/'attack'/f'text_{i:04d}/action.json')
            require(a['original']==q.caption and a['adversarial']==a['native_action']['adversarial'],'Test original Caption binding')
            max_edits=max(max_edits,validate_action(a['native_action'],contracts,constraints))
        scores=torch.load(root/'results/engine/scores/Joint.pt',map_location='cpu',weights_only=True)
        values,_=ranking(scores,manifest.evaluation_identity())
        require(all(values[k]==v for k,v in summary['metrics'].items()),'independent native ranking')
    result={'status':'GH_V2_JOINT_TEST_CPU_AUDIT_PASS','round':round_index,'scope':'smoke' if smoke else 'official_test',
        'configuration_sha256':sha_file(path),'gallery':len(manifest.gallery),'query':len(manifest.queries),
        'max_Linf':max_linf,'maximum_edited_words':max_edits,'metrics':summary['metrics'],'native_ranking_exact':True}
    ArtifactStore(root/'cpu_audit',new_base=f['output_base']).publish('completed',{'report.json':json_bytes(result)},
        {'configuration':sha_file(path),'round':round_index},sources=(FileRef(str(root/'completed/manifest.json'),sha_file(root/'completed/manifest.json')),))
    return result
