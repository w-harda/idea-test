"""不运行模型：逐记录核对协同损失、反事实编辑、预算和更新链。"""
import math
from .objectives import need

def close(a,b,label):need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),label)
def audit_cooperative(log,cfg,contracts,constraints):
 g_previous=log['G_before'];h_previous=None
 for g,condition,row in zip(log['G'],log['conditions'],log['g_orders'][0]):
  need(g['row_id']==condition['row']==row and g['G_before']==g_previous,'G row/state chain')
  need(g['optimizer_steps']==g['K']+1==len(g['records']),'original K+1 update membership')
  if 'identity' in condition:need(g['K']==len(condition['identity']['selected_g_ids']),'original selected K unchanged')
  metadata=g['cooperative'].get('metadata')
  if metadata is not None:
   need(metadata['row']==row and metadata['H_state']==log['condition_H_state'],'current H multi-caption context')
   if metadata['secondary'] is not None:
    from .audit import validate_action
    validate_action(metadata['secondary']['action'],contracts,constraints)
  weights=g['cooperative'].get('attribute_weights',[]);need(len(weights)==g['K'],'all original selected attributes weighted')
  if weights:close(sum(weights)/len(weights),1.,'mean-one residual weights')
  for j,record in enumerate(g['records']):
   v=record['cooperative'];need(record['finite_gradients'] and record['G_gradient_l2']>=0 and record['cumulative_linf']<=8/255,'G finite gradient/pixel budget')
   close(v['base_total'],record['global']+record['local']+record['retain'],'original loss terms')
   if j<g['K']:
    close(v['local_weight'],weights[j],'current original attribute weight')
    close(record['total'],record['global']+weights[j]*record['local']+record['retain'],'weighted attribute total')
   else:
    modes=v['four_modes'];close(v['synergy'],modes['s10']+modes['s01']-modes['s11']-modes['s00'],'four-mode interaction')
    scores=v['caption_scores'];need(len(scores)==v['caption_set_size'] and len(scores)>=2,'all multicaption targets')
    t=cfg['caption_temperature'];m=max(scores);smooth=m+t*(math.log(sum(math.exp((s-m)/t) for s in scores))-math.log(len(scores)))
    close(v['caption_set_loss'],smooth,'multicaption smooth maximum')
    close(v['extra'],-cfg['g_synergy_weight']*v['synergy']+cfg['caption_set_weight']*v['caption_set_loss'],'G cooperative objective')
    close(record['total'],v['base_total']+v['extra'],'G joint total')
   close(record['total'],v['total'],'actual optimized scalar')
  g_previous=g['G_after']
 need(g_previous==log['G_after'] and len(log['G'])==len(log['g_orders'][0])==len(log['conditions']),'complete G coverage')
 for h,row in zip(log['H'],log['h_orders'][0]):
  need(h['row_id']==row,'H row order')
  if h.get('fixed_NOOP'):need(h['optimizer_steps_this_step']==0,'original ineligible NOOP');continue
  need(h['optimizer_steps_this_step']==1 and h['gradient_norm']>=0,'single finite H optimizer step')
  if h_previous is not None:need(h['H_state_sha256_before']==h_previous,'H within-round continuous policy')
  h_previous=h['H_state_sha256_after'];v=h['cooperative'];raw=v['raw_rewards'];totals=v['augmented_totals']
  need(len(raw)==len(totals)==2+len(v['edit_credits']),'full counterfactual reward coverage')
  for reward,total in zip(raw,totals):
   close(reward['total'],reward['R_sim']+reward['R_rank'],'original H reward components')
   close(reward['synergy_increment'],reward['adv_sim_drop']-reward['clean_sim_drop'],'paired H interaction')
   close(total,reward['total']+cfg['h_synergy_weight']*reward['synergy_increment'],'augmented H reward')
  close(h['sampled_reward']['total'],totals[0],'sample reward');close(h['greedy_reward']['total'],totals[1],'greedy baseline')
  close(v['base_SCST_loss'],-(totals[0]-totals[1])*h['sequence_log_prob'],'SCST detached advantage')
  action=h['sampled'];edits=[constraints.EditRecord(**{**e,'source_span':tuple(e['source_span'])}) for e in action['edits']]
  ex=contracts.from_text_spans(action['original'],action['attributes'])
  need(len(edits)==len(v['counterfactual_texts'])==len(v['group_log_probs'])==len(v['edit_credits']),'one credit per actual edit')
  for i,(text,credit) in enumerate(zip(v['counterfactual_texts'],v['edit_credits'])):
   need(text==constraints.replay(ex,[e for j,e in enumerate(edits) if j!=i],action['budget_k']),'exact leave-one-edit-out Unicode replay')
   close(credit,totals[0]-totals[i+2],'counterfactual contribution')
  credit_loss=-cfg['edit_credit_weight']*sum(a*b for a,b in zip(v['edit_credits'],v['group_log_probs']))/max(1,len(edits))
  close(v['credit_loss'],credit_loss,'per-edit policy gradient scalar')
  close(h['SCST_loss'],v['base_SCST_loss']+credit_loss,'actual combined H objective')
  need(h['IRRA_text_samples']==3+len(edits),'bounded text feature batch')
 need(len(log['H'])==len(log['h_orders'][0]),'complete H coverage')
