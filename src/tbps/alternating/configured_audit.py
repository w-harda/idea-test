"""新协议独立CPU审计；重用原Artifact与像素预算检查，不重跑模型/旧审计。"""
from pathlib import Path
from dataclasses import asdict
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,sha_file,digest
from .contracts import BankIdentity, require
from .state import tree_sha
from .word_budget import check_words
from .configured_sources import load
from .native_sources import source_key


def verify_bank(root,ledger,c):
    import torch
    from .native_backend import assert_pixels
    d=Path(root)/'B'/('bank_'+str(ledger['round']));s=ArtifactStore(d,new_base=c['output_base'])
    s.verify('G_state');s.verify('completed')
    bid=read_json(d/'completed/identity.json');require(bid==ledger['bank'],'bank ledger mismatch')
    bank=BankIdentity(**{**bid,'ordered_train_ids':tuple(bid['ordered_train_ids'])})
    bank.validate(c['prototype']['train_ids'],ledger['G_after'],ledger['round'],'B',
        space=c['prototype']['source_space_sha256'],clean=c['prototype']['clean_gallery_sha256'])
    g=torch.load(d/'G_state/state.pt',map_location='cpu',weights_only=True)
    require(tree_sha(g)==bank.g_state_sha256,'bank generating G Checkpoint identity')
    features=[];maximum=0.
    for i,row in enumerate(c['prototype']['train_ids']):
        s.verify(f'claim_{i:04d}');m=s.verify(f'image_{i:04d}')['provenance']
        v=torch.load(d/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)
        assert_pixels(v['original'],v['adversarial'],v['effective_delta'])
        require(m['row']==row and m['index']==i and m['G']==ledger['G_after'] and m['content_sha256']==tree_sha(v),'bank image content/order')
        x=v['feature'];require(x.shape==(1,512) and x.dtype==torch.float32 and torch.isfinite(x).all()
            and abs(float(x.norm())-1)<1e-5,'bank native feature')
        features.append(x);maximum=max(maximum,float(v['effective_delta'].double().abs().max()))
    saved=torch.load(d/'completed/features.pt',map_location='cpu',weights_only=True)
    require(tree_sha(torch.cat(features))==tree_sha(saved)==bank.adversarial_sha256 and not s.pending(),'full current bank')
    return {'images':len(features),'max_Linf':maximum,'G_checkpoint_bound':True,'ordered_members_bound':True}

def audit(path,scope):
    f,c=load(path,scope)
    import torch
    from .parents import cpu_restore_only
    root=Path(f['outputs'][scope]);store=ArtifactStore(root,new_base=f['output_base'])
    require(not list(root.glob('FAIL*')) and not store.pending(),'real failure/pending retained')
    store.verify('generated');generated=read_json(root/'generated/report.json')
    require(generated['complete'] and generated['frozen_sources'] and generated['configuration_sha256']==sha_file(path),'GPU completion identity')
    journal=ArtifactStore(root/'rounds',new_base=f['output_base']);journal.verify('protocol');journal.verify('completed')
    store.verify('initial_state');previous=torch.load(root/'initial_state/state.pt',map_location='cpu',weights_only=False)
    records=[];words=0;edited=0;all_gu=0;all_hu=0;previous_ledger=None
    # 不允许CPU审计执行任何神经前向、梯度、Optimizer或CUDA；仅载入允许状态文件。
    allowed=[root/'initial_state/state.pt']
    for i in range(1,c['prototype']['rounds']+1):
        allowed.append(root/'rounds'/f'round_{i:04d}/state.pt')
        allowed.extend((root/'B'/f'bank_{i}').glob('image_*/data.pt'))
        allowed.extend([root/'B'/f'bank_{i}/G_state/state.pt',root/'B'/f'bank_{i}/completed/features.pt'])
        allowed.extend((root/'round_io'/f'round_{i:04d}').glob('condition_*/condition.pt'))
    from .native_sources import native_session
    with cpu_restore_only(allowed), native_session(c) as legacy:
        contracts=legacy.import_module('contracts');constraints=legacy.import_module('constraints')
        def validate_action(action):
            ex=contracts.from_text_spans(action['original'],action['attributes'])
            records=tuple(constraints.EditRecord(**{**v,'source_span':tuple(v['source_span'])}) for v in action['edits'])
            require(constraints.replay(ex,records,action['budget_k'])==action['adversarial'],'native Unicode replay/metadata')
            check_words(action['original'],action['adversarial'])
        for i in range(1,c['prototype']['rounds']+1):
            d=root/'rounds'/f'round_{i:04d}';m=journal.verify(d.name)
            claim=journal.verify(d.name+'_claim');claim_data=read_json(root/'rounds'/(d.name+'_claim')/'claim.json')
            require(claim_data['parent_state']==tree_sha(previous),'committed full parent chain')
            state=torch.load(d/'state.pt',map_location='cpu',weights_only=False);log=read_json(d/'ledger.json')
            require(set(state)=={'G','H','G_Adam','H_Adam','G_RNG','H_RNG','cursor','parents','schedulers','training_configuration'},'complete state schema')
            require(tree_sha(state)==m['provenance']['state'] and state['cursor']==log['end_cursor'],'state content/cursor')
            require(state['training_configuration']==c['alternating_configuration'] and state['parents']==previous['parents'],'parent/variant configuration')
            require(tree_sha(previous['G'])==log['G_before'] and tree_sha(previous['H'])==log['H_before']
                and tree_sha(state['G'])==log['G_after'] and tree_sha(state['H'])==log['H_after'],'no initial-state reload; current H/G chain')
            require(log['condition_H_state']==log['H_before'] and all(log['isolation'].values()),'phase isolation/current H')
            fp=log['isolation_fingerprints']
            require(fp['inactive_H_before_G']==fp['inactive_H_after_G'] and fp['inactive_G_before_H']==fp['inactive_G_after_H'],'inactive model/Optimizer fingerprints')
            from tbps.ablations.gv2_epoch_state import adam_steps,finite_state
            for name in ('G','H','G_Adam','H_Adam'):finite_state(state[name])
            require(adam_steps(state['G_Adam'])==[state['cursor']['g_updates']]
                and max(adam_steps(state['H_Adam']))==state['cursor']['h_updates'],'absolute Adam steps')
            for side in ('G','H'):
                rng=state[side+'_RNG'];require(set(rng)=={'python','numpy','torch_cpu','cuda','sampler'}
                    and len(rng['cuda'])==1,'complete stage RNG')
                require(state['schedulers'][side]['last_epoch']==i,'Scheduler round accounting')
            for side,passes in (('g',f['g_passes_per_round']),('h',f['h_passes_per_round'])):
                require(len(log[side+'_orders'])==passes and all(len(o)==len(set(o)) and set(o)==set(c['prototype']['samples']) for o in log[side+'_orders']),'pass/member accounting')
            io=ArtifactStore(root/'round_io'/f'round_{i:04d}',new_base=f['output_base'])
            require(len(log['conditions'])==sum(map(len,log['g_orders'])),'G condition coverage')
            nonnoop=0
            for j,cond in enumerate(log['conditions']):
                io.verify(f'rollout_{j:04d}');io.verify(f'condition_{j:04d}')
                act=read_json(io.root/f'rollout_{j:04d}/action.json');a=check_words(act['original'],act['adversarial'])
                words+=1;edited+=a['changed_words'];nonnoop+=bool(a['changed_words'])
                if not act['native_action'].get('fixed_NOOP'):validate_action(act['native_action'])
                else:require(act['original']==act['adversarial'] and not act['edits'],'fixed NOOP altered')
                require(cond==read_json(io.root/f'condition_{j:04d}/evidence.json'),'condition evidence')
                v=torch.load(io.root/f'condition_{j:04d}/condition.pt',map_location='cpu',weights_only=True)
                require(tree_sha(v)==cond['payload_content_sha256'],'fresh condition payload')
                require(cond['identity']['context_caption_sha256']==__import__('hashlib').sha256(act['adversarial'].encode()).hexdigest(),'rollout/context identity')
                ev=cond['evidence'];changed=act['original']!=act['adversarial']
                if changed:
                    require(ev['fresh_localization'] and ev['old_JVP_relabelled'] is False and ev['new_text_encodings']>0
                        and ev['new_JVPs']==len(cond['identity']['selected_g_ids']),'no stale relabelled JVP')
                    require(all(x['zero_exact'] and x['central_fd_relative_error']<=.02 for x in ev['JVP_checks']),'real JVP zero/FD')
                else:require(ev['new_JVPs']==ev['new_text_encodings']==0,'explicit NOOP reuse')
                require(log['G'][j]['optimizer_steps']==v['K']+1,'original G K+1')
            bank=verify_bank(root,log,c)
            for h in log['H']:
                if h.get('fixed_NOOP'):continue
                require(h['bank_sha256']==log['bank']['adversarial_sha256'],'H latest G bank')
                for name in ('sampled','greedy'):
                    a=h[name];validate_action(a);require(len(a['edits'])<=1,'one Unicode character per word')
                    words+=1;edited+=int(bool(a['edits']))
                require(set(h['four_mode_paired_similarity'])=={'sampled','greedy'},'four similarity diagnostics')
            gu=sum(g['optimizer_steps'] for g in log['G']);hu=sum(h['optimizer_steps_this_step'] for h in log['H'])
            require(state['cursor']=={'round':i,'g_updates':previous['cursor']['g_updates']+gu,
                'h_visits':previous['cursor']['h_visits']+len(log['H']),'h_updates':previous['cursor']['h_updates']+hu},'visit/update ledger')
            all_gu+=gu;all_hu+=hu
            records.append({'round':i,'cursor':state['cursor'],'G_updates':gu,'H_updates':hu,'nonNOOP_contexts':nonnoop,
                'new_JVPs':sum(x['evidence']['new_JVPs'] for x in log['conditions']),'bank':bank,'seconds':log['seconds'],
                'G_changed':log['G_changed'],'H_changed':log['H_changed'],
                'same_text_reward_changes':sum(any(x!=0 for x in h.get('reward_delta_same_text',[])) for h in log['H']),
                'nonzero_H_gradients':sum(h.get('gradient_norm',0)>0 for h in log['H'])})
            previous=state;previous_ledger=log
    require(all(x['G_changed'] and x['H_changed'] for x in records),'parameter updates not demonstrated')
    require(generated['counts']['G']==generated['counts']['IRRA_image']==all_gu+len(c['prototype']['train_ids'])*len(records),'actual G/IRRA image counts')
    resumes=list((root/'rounds').glob('resume_*'))
    for d in resumes:journal.verify(d.name)
    result={'status':'GH_ALTERNATING_V1_'+scope.upper()+'_CPU_AUDIT_PASS','scope':scope,
        'configuration_sha256':sha_file(path),'rounds':records,'word_budget':{'checked_outputs':words,'edited_outputs':edited,'maximum_changed_words':1},
        'G_updates':all_gu,'H_updates':all_hu,'planned_fresh_process_boundary_resume':bool(resumes),
        'complete_dual_Adam_Scheduler_RNG_state':True,'GPU_report':generated,'retrieval_performance':'NOT_EVALUATED',
        'old_exact_Smoke_FAIL_unchanged':True,'formal_long_training':'NOT_RUN'}
    out=ArtifactStore(f['audits'][scope],new_base=f['output_base']);out.publish('completed',{'report.json':json_bytes(result)},
        {'configuration':sha_file(path),'scope':scope})
    if scope=='pilot':
        p=Path(f['audits'][scope])/'completed/report.json';proof=FileRef(str(p),sha_file(p))
        ArtifactStore(f['output_base'],new_base=f['output_base']).publish('readiness',{'ready.json':json_bytes({
            'status':'READY_FOR_SEPARATE_TRAINING_AUTHORIZATION','configuration_sha256':sha_file(path),
            'pilot_audit':asdict(proof),'automatically_start_training':False})},{'configuration':sha_file(path)},sources=(proof,))
    return result
