"""只对已冻结映射/配对和现有结果做资格分层；不重配对/训练/IRRA forward。"""
import collections
import numpy as np
import factorized as f

def run():
 p=f.OUT;M=f.load(p/'mapped_labels.json');sp=f.load(f.V4/'outputs/split.json');pairs=f.load(p/'pairs.json');E=f.load(p/'energy.json');sub={}
 for r in ('upper','lower'):
  si=sp['slots'].index(r+'_clothing_type')
  for same in (True,False):
   pp=[x for x in pairs if x['target']==r+':clothing_type' and x['tier']=='strict' and (sp['records'][x['a']]['labels'][si]==sp['records'][x['b']]['labels'][si])==same];rows=[E[x['pair_id']] for x in pp]
   sub[r+'|old_type_same='+str(same)]={'pair_count':len(rows),'methods':{name:{'type':f.stats([x[name]['clothing_type'] for x in rows]),'color':f.stats([x[name]['color'] for x in rows]),'type_gt_color':sum(x[name]['clothing_type']>x[name]['color'] for x in rows)/len(rows) if rows else None} for name in ['semantic','flat','shared_basis_old_slot_z']}}
 f.save(p/'old_type_stratification.json',sub)
 groups=f.load(f.V52/'outputs/groups.json');oldenergy=f.load(f.V51/'outputs/energy.json');oldp=f.load(f.V51/'outputs/pairs.json');pg={x['pair_id']:x for g in oldp for x in g['pairs']}
 Q={s:np.array(c['geometry']['Q']) for s,c in f.load(p/'corpus.json').items()};R={s:np.array(c['geometry']['Q_random']) for s,c in f.load(p/'corpus.json').items()};Z=np.load(f.ROOT/'cache/region_z.fp32.npy',mmap_mode='r');rows=[]
 for g in groups:
  if g['target']!='lower_clothing_type' or g['has_full_body']:continue
  a,b=pg[g['pair_id']]['a'],pg[g['pair_id']]['b'];la,lb=M[a]['labels'],M[b]['labels']
  if la['lower:clothing_type'] is None or lb['lower:clothing_type'] is None or la['lower:color'] is None or la['lower:color']!=lb['lower:color'] or la['lower:clothing_type']==lb['lower:clothing_type']:continue
  d=np.array(Z[b,1],dtype=np.float64)-np.array(Z[a,1],dtype=np.float64);den=d@d
  if den<=0:continue
  v={s:float(np.sum((d@q)**2)/den) for s,q in Q.items()};rv={s:float(np.sum((d@q)**2)/den) for s,q in R.items()};oe=oldenergy[g['pair_id']]['semantic']
  rows.append({'pair_id':g['pair_id'],'new':v,'random':rv,'old':{'clothing_type':oe['lower_clothing_type'],'color':oe['lower_clothing_color']},'group':g['group']})
 ret={}
 for grp in ('all','A_clear_region','C_binding_risk'):
  xs=rows if grp=='all' else [x for x in rows if x['group']==grp];ret[grp]={'count':len(xs)}
  for method in ['new','random','old']:
   if not xs:continue
   t=np.array([x[method]['clothing_type'] for x in xs]);c=np.array([x[method]['color'] for x in xs]);ret[grp][method]={'type_median':float(np.median(t)),'color_median':float(np.median(c)),'margin_median':float(np.median(t-c)),'hit_type_gt_color':float((t>c).mean())}
 f.save(p/'old_lower_common_pairs.json',{'selection':'eligibility-only readout of fixed old lower strict non-full-body pairs; lower/lower different prototype type and known equal color. No new partners or energy filters. Added after primary result inspection; not claimed pre-registered.','metrics':ret,'rows':rows})
 # 对所有冻结主pair核查资格，验证没有按energy改标签/配对。
 for x in pairs:
  la,lb=M[x['a']]['labels'],M[x['b']]['labels'];assert la[x['target']] and lb[x['target']] and la[x['target']]!=lb[x['target']]
  if x['tier']=='strict':assert all(la[c] is not None and la[c]==lb[c] for c in x['controls'])
 assert len({x['pair_id'] for x in pairs})==len(pairs)==40198
 protocol=f.load(p/'protocol.json');frozen=f.load(p/'selection_frozen.json')
 assert all(f.digest(p/n)==h for n,h in frozen.items())
 assert all(f.digest(path)==h for path,h in protocol['old_sha256'].items())
 assert all(f.digest(f.ROOT/n)==h for n,h in protocol['source_sha256'].items())
 assert f.digest(f.ROOT/'cache/region_z.fp32.npy')==protocol['region_cache_sha256']
 import subprocess
 assert not subprocess.check_output(['git','diff','--stat'],cwd=f.REPO,text=True)
 assert not subprocess.check_output(['git','diff','--cached','--stat'],cwd=f.REPO,text=True)
 f.save(p/'final_audit.json',{'pair_count':len(pairs),'all_pair_constraints':True,'mapping_selection_unchanged':True,'old_inputs_unchanged':True,'source_unchanged':True,'training':False,'IRRA_forward_calls':0,'retrieval_calls':0,'official_tracked_staged_clean':True,'audit_results_source_sha256':f.digest(f.ROOT/'audit_results.py')})
if __name__=='__main__':run()
