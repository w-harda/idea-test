"""CPU独立FP64重算KL；只读已封存特征，禁止模型/梯度/JVP重入。"""
import math
import torch
from torch.nn import functional as F
from tbps.artifacts import digest
from tbps.alternating.state import tree_sha
from .objectives import require

ATOL=5e-5
RTOL=2e-4

def audit_attraction(log,cfg,contracts,constraints,anchors,bank_features):
    require(anchors.ndim==2 and bank_features.shape==anchors.shape and torch.isfinite(anchors).all() and torch.isfinite(bank_features).all(),'finite full ordered feature banks')
    ids=log['bank']['ordered_train_ids'];require(len(ids)==len(anchors) and len(ids)==len(set(ids)),'all Train candidate IDs')
    positions={row:i for i,row in enumerate(ids)};anchor_sha=tree_sha(anchors)
    # 与生产实现分离：FP64归一化、矩阵乘法和手写KL求和。
    c=F.normalize(anchors.double(),dim=1);max_error=0.;g_steps=0;h_steps=0
    def matrix(values):
        x=torch.tensor(values,dtype=torch.float64)
        require(x.ndim==2 and x.shape[1]==c.shape[1] and torch.isfinite(x).all() and (x.norm(dim=1)>0).all(),'complete finite feature vectors')
        return x
    def probabilities(values):return F.log_softmax(F.normalize(values,dim=1)@c.T/cfg['temperature'],dim=1)
    def evaluate(current,clean,partner):
        pa,p0,pt=probabilities(current),probabilities(clean),probabilities(partner)
        pull=(pt.exp()*(pt-pa)).sum(1);push=(p0.exp()*(p0-pa)).sum(1);baseline=(pt.exp()*(pt-p0)).sum(1)
        gain=baseline-pull;reward=cfg['pull_weight']*gain+cfg['push_weight']*push
        return {'pull_kl':pull,'push_kl':push,'baseline_pull_kl':baseline,'pull_gain':gain,'reward':reward,'loss':-reward,
            'partner_cosine':F.cosine_similarity(current,partner,dim=1),'clean_cosine':F.cosine_similarity(current,clean,dim=1)}
    def close(actual,expected,label):
        nonlocal max_error
        a,b=float(actual),float(expected);require(math.isfinite(a) and math.isfinite(b),'finite '+label)
        max_error=max(max_error,abs(a-b));require(math.isclose(a,b,rel_tol=RTOL,abs_tol=ATOL),label)
    batch=[]
    def flush():
        if not batch:return
        values=evaluate(matrix([v[0] for v in batch]),matrix([v[1] for v in batch]),matrix([v[2] for v in batch]))
        for i,(_,_,_,logged) in enumerate(batch):
            for key,value in values.items():close(logged[key],value[i], 'independent G '+key)
        batch.clear()
    previous=log['G_before']
    for g,condition,row in zip(log['G'],log['conditions'],log['g_orders'][0]):
        require(g['row_id']==condition['row']==row and g['G_before']==previous,'G identity/state chain')
        require(g['K']==len(condition['identity']['selected_g_ids']) and len(g['records'])==g['K']+1==g['optimizer_steps'],'original K+1 selected-attribute coverage')
        md=g['attraction'];require(md['settings']==cfg and md['teacher_H_state']==log['condition_H_state'] and md['anchor_sha256']==anchor_sha and md['candidate_count']==len(anchors),'current H and frozen anchors')
        require(md['condition_identity']==digest(condition['identity']),'G current H condition identity')
        clean=anchors[positions[row]].tolist();partner=md['partner_text_feature']
        require(len(g['gradient_observations'])==len(g['records']),'all original gradient observations')
        for rec,obs in zip(g['records'],g['gradient_observations']):
            values=rec['attraction'];require(rec['finite_gradients'] and rec['G_gradient_l2']>=0 and rec['cumulative_linf']<=8/255,'finite G and strict original pixel budget')
            require(values['phase']==rec['phase'],'attribute/joint phase identity')
            require(all(obs['record'][k]==v for k,v in rec.items() if k!='attraction'),'actual optimized G record')
            close(rec['global'],values['loss'],'new Global replaces original cosine')
            close(rec['total'],sum(g['weights']['lambda_'+k]*rec[k] for k in ('global','local','retain')),'actual G weighted total')
            batch.append((values['image_feature'],clean,partner,values));g_steps+=1
            if len(batch)>=64:flush()
        previous=g['G_after']
    flush();require(previous==log['G_after'] and len(log['G'])==len(log['conditions'])==len(log['g_orders'][0]),'all G rows/conditions')
    previous_h=None
    for h,row in zip(log['H'],log['h_orders'][0]):
        require(h['row_id']==row,'H row identity')
        if h.get('fixed_NOOP'):
            require(h['optimizer_steps_this_step']==0 and h['IRRA_text_samples']==0,'original fixed NOOP');continue
        require(h['optimizer_steps_this_step']==1 and h['IRRA_text_samples']==3 and h['gradient_norm']>=0,'one SCST step / original three text features')
        if previous_h is not None:require(h['H_state_sha256_before']==previous_h,'H per-row policy continuity')
        previous_h=h['H_state_sha256_after'];md=h['attraction']
        require(md['settings']==cfg and md['teacher_G_state']==log['G_after'] and md['anchor_sha256']==anchor_sha and md['candidate_count']==len(anchors),'H current G teacher and fixed anchors')
        require(md['R_sim_optimized'] is False and md['R_rank_optimized'] is False,'declared pull/push-only reward')
        features=matrix(md['text_features']);require(len(features)==3,'original/sample/greedy feature triplet')
        partner=matrix([md['partner_image_feature']]);expected=bank_features[positions[row]:positions[row]+1]
        require(torch.equal(partner.float(),expected.float()),'H paired current-G Bank feature exact')
        values=evaluate(features[1:],features[:1].expand(2,-1),partner.expand(2,-1))
        for i,name in enumerate(('sampled','greedy')):
            record=md[name]
            for key,v in values.items():close(record[key],v[i],'independent H '+name+' '+key)
            reward=h[name+'_reward'];close(reward['total'],values['reward'][i],'actual H '+name+' reward')
            for key in ('pull_kl','push_kl','baseline_pull_kl','pull_gain'):close(reward[key],values[key][i],'H Reward dataclass '+key)
        advantage=values['reward'][0]-values['reward'][1]
        close(h['advantage'],advantage,'SCST greedy baseline advantage')
        close(h['SCST_loss'],-float(h['advantage'])*h['sequence_log_prob'],'actual SCST objective')
        h_steps+=1
    require(len(log['H'])==len(log['h_orders'][0]),'all H rows')
    return {'G_steps_independently_recomputed':g_steps,'H_steps_independently_recomputed':h_steps,
        'candidate_count':len(anchors),'numeric_atol':ATOL,'numeric_rtol':RTOL,'max_absolute_scalar_difference':max_error,
        'image_budget_tolerance':0,'neural_forward_backward_optimizer_JVP':0}
