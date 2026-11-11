"""逐视图独立FP64重算；原始/局部/全局损失均覆盖，禁止模型重入。"""
import math
import torch
from torch.nn import functional as F
from tbps.artifacts import digest
from tbps.alternating.state import tree_sha
from .objectives import require
from .views import ViewSettings,recipe
ATOL=5e-5
RTOL=2e-4

def audit_attraction(log,cfg,contracts,constraints,anchors,bank_features,view_bank,augmentation):
    require(anchors.ndim==2 and bank_features.shape==anchors.shape and view_bank.shape==(len(anchors),3,anchors.shape[1]),'all original ordered candidates and three partner views')
    require(torch.equal(view_bank[:,0],bank_features) and torch.isfinite(view_bank).all(),'exact original view bank and finite three views')
    ids=log['bank']['ordered_train_ids'];require(len(ids)==len(anchors) and len(ids)==len(set(ids)),'unique complete Train candidates')
    positions={row:i for i,row in enumerate(ids)};anchor_sha=tree_sha(anchors);s=ViewSettings(**augmentation);view_sha=tree_sha(view_bank)
    require(len(log['bank_view_inputs'])==len(ids),'all bank view input identities')
    bindings={}
    for i,(row,item) in enumerate(zip(ids,log['bank_view_inputs'])):
        plan=recipe(s,row,log['round'],384,128)
        require(item['row']==row and item['recipe_sha256']==digest(plan) and item['feature_sha256']==tree_sha(view_bank[i]),'ordered bank views, recipe, feature SHA')
        require(len(item['view_linf'])==3 and all(math.isfinite(x) and 0<=x<=8/255 for x in item['view_linf']),'all Bank view strict budget records')
        bindings[row]=item
    c=F.normalize(anchors.double(),dim=1);max_error=0.;g_steps=0;h_steps=0
    def matrix(values):
        x=torch.tensor(values,dtype=torch.float64)
        require(x.ndim==2 and x.shape[1]==c.shape[1] and torch.isfinite(x).all() and (x.norm(dim=1)>0).all(),'complete finite feature vectors')
        return x
    def probs(x):return F.log_softmax(F.normalize(x,dim=1)@c.T/cfg['temperature'],dim=1)
    def evaluate(current,clean,partner):
        pa,p0,pt=probs(current),probs(clean),probs(partner)
        pull=(pt.exp()*(pt-pa)).sum(1);push=(p0.exp()*(p0-pa)).sum(1);baseline=(pt.exp()*(pt-p0)).sum(1)
        gain=baseline-pull;reward=cfg['pull_weight']*gain+cfg['push_weight']*push
        return {'pull_kl':pull,'push_kl':push,'baseline_pull_kl':baseline,'pull_gain':gain,'reward':reward,'loss':-reward,
            'partner_cosine':F.cosine_similarity(current,partner,dim=1),'clean_cosine':F.cosine_similarity(current,clean,dim=1)}
    def close(actual,expected,label):
        nonlocal max_error
        a,b=float(actual),float(expected);require(math.isfinite(a) and math.isfinite(b),'finite '+label)
        max_error=max(max_error,abs(a-b));require(math.isclose(a,b,rel_tol=RTOL,abs_tol=ATOL),label)
    def compare(logged,values,label):
        require(len(logged['per_view'])==3,'all per-view terms present')
        for k,v in values.items():
            close(logged[k],v.mean(),label+' mean '+k)
            for j in range(3):close(logged['per_view'][j][k],v[j],label+' individual '+k)
    batch=[]
    def flush():
        if not batch:return
        v=evaluate(matrix([z for row in batch for z in row[0]]),matrix([z for row in batch for z in row[1]]),matrix([row[2] for row in batch for _ in range(3)]))
        for i,row in enumerate(batch):compare(row[3],{k:x[i*3:(i+1)*3] for k,x in v.items()},'G')
        batch.clear()
    previous=log['G_before']
    for g,condition,row in zip(log['G'],log['conditions'],log['g_orders'][0]):
        require(g['row_id']==condition['row']==row and g['G_before']==previous,'G row/state continuity')
        require(g['K']==len(condition['identity']['selected_g_ids']) and len(g['records'])==g['K']+1==g['optimizer_steps'],'original K+1 updates')
        md=g['attraction'];require(md['settings']==cfg and md['teacher_H_state']==log['condition_H_state'] and md['anchor_sha256']==anchor_sha and md['candidate_count']==len(anchors),'current H/frozen anchors')
        require(md['condition_identity']==digest(condition['identity']) and md['view_recipe']==recipe(s,row,log['round'],384,128),'current H context and exact view recipe')
        require(md['clean_view_sha256']==bindings[row]['input_sha256'],'G and current Bank share exact augmented inputs')
        require(len(g['gradient_observations'])==len(g['records']),'actual gradient record coverage')
        clean=None
        for rec,obs in zip(g['records'],g['gradient_observations']):
            v=rec['attraction'];require(rec['finite_gradients'] and rec['G_gradient_l2']>=0 and rec['cumulative_linf']<=8/255,'finite G/budget')
            require(v['phase']==rec['phase'] and len(v['view_linf'])==3 and all(math.isfinite(x) and 0<=x<=8/255 for x in v['view_linf']),'all G views/phase/budget')
            require(all(obs['record'][k]==x for k,x in rec.items() if k!='attraction'),'actual optimized G step')
            require(len(v['view_features'])==len(v['clean_view_features'])==3 and v['image_feature']==v['view_features'][0],'three complete view features')
            require(torch.equal(torch.tensor(v['clean_view_features'][0]),anchors[positions[row]]),'original clean anchor identity')
            if clean is None:clean=v['clean_view_features']
            require(clean==v['clean_view_features'],'same clean view references across K+1')
            close(rec['global'],v['loss'],'view averaged Global')
            close(rec['total'],sum(g['weights']['lambda_'+k]*rec[k] for k in ('global','local','retain')),'weighted G total')
            batch.append((v['view_features'],v['clean_view_features'],md['partner_text_feature'],v));g_steps+=1
            if len(batch)>=32:flush()
        previous=g['G_after']
    flush();require(previous==log['G_after'] and len(log['G'])==len(log['conditions'])==len(log['g_orders'][0]),'all G coverage')
    previous_h=None
    for h,row in zip(log['H'],log['h_orders'][0]):
        require(h['row_id']==row,'H row identity')
        if h.get('fixed_NOOP'):
            require(h['optimizer_steps_this_step']==0 and h['IRRA_text_samples']==0,'original fixed NOOP');continue
        require(h['optimizer_steps_this_step']==1 and h['IRRA_text_samples']==3 and h['gradient_norm']>=0,'one SCST update/three texts')
        if previous_h is not None:require(h['H_state_sha256_before']==previous_h,'H policy continuity')
        previous_h=h['H_state_sha256_after'];md=h['attraction']
        require(md['settings']==cfg and md['teacher_G_state']==log['G_after'] and md['anchor_sha256']==anchor_sha and md['candidate_count']==len(anchors),'current G teacher and anchors')
        require(md['view_bank_sha256']==view_sha and md['R_sim_optimized'] is False and md['R_rank_optimized'] is False,'current three-view Bank and declared reward')
        partner=matrix(md['partner_view_features']);require(torch.equal(partner.float(),view_bank[positions[row]]) and md['partner_image_feature']==md['partner_view_features'][0],'all H partner views exact')
        text=matrix(md['text_features']);require(len(text)==3,'original/sample/greedy text features')
        rewards=[]
        for i,name in enumerate(('sampled','greedy'),1):
            values=evaluate(text[i:i+1].expand(3,-1),text[:1].expand(3,-1),partner);compare(md[name],values,'H '+name)
            reward=h[name+'_reward'];close(reward['total'],values['reward'].mean(),'actual H averaged reward')
            for k in ('pull_kl','push_kl','baseline_pull_kl','pull_gain'):close(reward[k],values[k].mean(),'actual H Reward '+k)
            rewards.append(values['reward'].mean())
        close(h['advantage'],rewards[0]-rewards[1],'SCST greedy baseline advantage')
        close(h['SCST_loss'],-float(h['advantage'])*h['sequence_log_prob'],'actual SCST objective');h_steps+=1
    require(len(log['H'])==len(log['h_orders'][0]),'all H coverage')
    return {'G_steps_independently_recomputed':g_steps,'G_view_terms_independently_recomputed':3*g_steps,
        'H_steps_independently_recomputed':h_steps,'H_view_terms_independently_recomputed':6*h_steps,
        'candidate_count':len(anchors),'views':3,'numeric_atol':ATOL,'numeric_rtol':RTOL,'max_absolute_scalar_difference':max_error,
        'image_budget_tolerance':0,'neural_forward_backward_optimizer_JVP':0}

def audit_view_bank(root,ledger,c):
    from pathlib import Path
    from tbps.artifacts import ArtifactStore,read_json
    from tbps.alternating.native_backend import assert_pixels
    from .views import apply_views
    d=Path(root)/'B'/f"bank_{ledger['round']}";store=ArtifactStore(d,new_base=c['output_base']);all_features=[]
    for i,(row,binding) in enumerate(zip(c['prototype']['train_ids'],ledger['bank_view_inputs'])):
        store.verify(f'image_{i:04d}');v=torch.load(d/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True);md=read_json(d/f'image_{i:04d}/metadata.json')
        clean=torch.cat([v['original'],v['augmented_original']]);adv=torch.cat([v['adversarial'],v['augmented_adversarial']]);delta=torch.cat([v['effective_delta'],v['augmented_effective_delta']])
        require(clean.shape==adv.shape==delta.shape==(3,3,384,128),'complete latest RGB view pixels')
        require(torch.equal(apply_views(v['original'],md['view_recipe']),clean),'exact CPU replay of augmentation')
        require(tree_sha(clean)==binding['input_sha256'] and digest(md['view_recipe'])==binding['recipe_sha256'],'latest view input identities')
        for j in range(3):assert_pixels(clean[j:j+1],adv[j:j+1],delta[j:j+1])
        require(tree_sha(v['view_features'])==binding['feature_sha256'] and torch.equal(v['view_features'][:1],v['feature']),'all latest view feature identities')
        all_features.append(v['view_features'])
    saved=torch.load(d/'completed/view_features.pt',map_location='cpu',weights_only=True)
    require(torch.equal(torch.stack(all_features),saved),'all latest view pixels/features ordered')
    return {'original_images':len(all_features),'total_view_images':3*len(all_features),'exact_recipe_pixel_replay':True,'all_views_Linf_max':max(max(x['view_linf']) for x in ledger['bank_view_inputs']),'image_budget_tolerance':0}
