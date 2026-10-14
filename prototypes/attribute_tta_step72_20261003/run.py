"""固定3句smoke；25句正式小诊断仅用户手动运行，无full模式。"""
import os,sys,json,copy,time,argparse,itertools,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
for key,sub in [('TORCH_HOME','torch'),('HF_HOME','huggingface'),('XDG_CACHE_HOME','xdg'),('CUDA_CACHE_PATH','cuda')]:os.environ.setdefault(key,'/home/lzf/ldx/cache/'+sub)
REPO=Path('/home/lzf/ldx/projects/idea-TBPS-test1');sys.path.insert(0,str(REPO/'src'))
from attribute_tta import *
from attributes.tta_adapter import _tokenize
from attributes.irra_surrogate import IRRASurrogate
from attributes.tta_irra import IRRATTAImageCallback
from attributes.attack_scheduler import SelectedAttribute as LegacyAttribute,AttackState,ImageAttackRequest
from PIL import Image
ROOT=Path(__file__).resolve().parent
SMOKE_IDS=('cuhk:34:0','cuhk:35:1','cuhk:0:1')
# 预先固定以覆盖自然k1/k2/k3、rank1 sleeve与upper/lower同axis，不按攻击结果选样。

def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def verify(inputs):
 for p,h in inputs.items():
  if digest(p)!=h:raise ValueError('frozen SHA mismatch: '+p)

def inputs_and_records():
 lock=load(P51/'provenance.lock.json')['frozen_sha256']
 # 不复用全量H/Z cache；消费的是既有25句保存的w/e与train-fitted basis。
 needed={p:h for p,h in lock.items() if Path(p).suffix=='.py' or p.endswith('semantic_bases.json') or p.endswith('reader_final.pt') or p.endswith('split.json') or p.endswith('best.pth') or p.endswith('irra_cuhk_official.yaml')}
 p61=load(P61/'protocol.lock.json')
 for name,h in p61['input_sha256'].items():needed[str(P51/'outputs/diagnostic'/name)]=h
 protocol=load(P61/'outputs/diagnostic/protocol.json')
 for name,h in protocol['new_source_sha256'].items():needed[str(P61/name)]=h
 needed[str(P61/'outputs/diagnostic/selection_payloads.json')]='053bb2b3b9e7c043b330293edbfbf66e9f18b3b36fcaaaa083e2c884117a04f2'
 for p in [REPO/'src/attributes/tta_irra.py',REPO/'src/attributes/tta_adapter.py',REPO/'src/attributes/generator_framework.py',REPO/'src/attributes/irra_surrogate.py',OFFICIAL/'attacker_TTA.py']:
  needed[str(p)]=digest(p)
 verify(needed)
 return needed,load(P51/'outputs/diagnostic/results.json'),[p for p in load(P61/'outputs/diagnostic/selection_payloads.json') if p['gamma']==3],load(BASIS)

def one_official(source,image,features,labels,*,steps,transforms,scales,seed):
 seed_all(seed);attack=load_official_attack(OFFICIAL)(None,_tokenize,imgs_eps=IMAGE_EPSILON,step_size=2/255);attack.N_trans=transforms
 bridge=FixedGuidanceBridge(source,features,labels)
 with torch.enable_grad():y,m=attack.img_attack(bridge,list(labels),image.detach(),image.detach(),[0]*len(labels),steps,torch.zeros_like(image),image.device,scales=scales)
 return project_image(image,y.detach()).detach(),m.detach()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('smoke','diagnostic'),default='smoke');p.add_argument('--allow-diagnostic',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
 if args.mode=='diagnostic' and not args.allow_diagnostic:raise SystemExit('diagnostic is user-run only; supply --allow-diagnostic')
 out=(args.output or ROOT/'outputs'/args.mode).resolve()
 if not out.is_relative_to(Path('/home/lzf/ldx')) or out.exists():raise ValueError('new output directory under /home/lzf/ldx required')
 hashes,rows,payloads,bases=inputs_and_records();look={r['row_id']:r for r in rows};plans={r['caption_id']:r for r in payloads};ids=SMOKE_IDS if args.mode=='smoke' else tuple(r['row_id'] for r in rows)
 assert len(ids)<=25
 steps=1 if args.mode=='smoke' else 10;transforms=1 if args.mode=='smoke' else 6;scales=None if args.mode=='smoke' else (.5,.75,1.25,1.5)
 out.mkdir(parents=True);t0=time.time()
 sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'));import learned_reader as v4
 v4.setup();torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim);versions={n:p._version for n,p in victim.named_parameters()}
 protocol={'mode':args.mode,'row_ids':list(ids),'steps':steps,'transforms_per_scale':transforms,'scales':scales,'image_epsilon':IMAGE_EPSILON,'step_size':2/255,'gamma':3,'beta':1,'k_max':4,'selection':'saved clean gamma3 plan, no score/TopK recomputation','guidance':'v_i=D_H normalized final text retrieval feature [w_i normalized(projected_context_i)^T]; loss=-image_feature dot v_i','guidance_disabled':'original single-caption TTA supervision, no attribute vectors','caption_role':'original forward linearization context only; no extra caption embedding added to attribute supervision','basis_rebuilt':False,'reader_retrained':False,'IRRA_precision':'fp32','torch_mha_fastpath':False,'sdp':'math for exact JVP/double backward; no weight changes','frozen_sha256':hashes,'source_sha256':{str(f):digest(f) for f in ROOT.glob('*.py')}};save(out/'protocol.json',protocol);results=[]
 for cid in ids:
  case_t0=time.time();record=look[cid];selected=prepared_plan(plans[cid],record,bases,'cuda');metadata_before=copy.deepcopy([a.metadata for a in selected]);qversions=[a.Q._version for a in selected]
  with Image.open(Path('/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs')/record['paired_image']) as im:image=victim.pixels(im).unsqueeze(0).cuda()
  transport=TextTransport(victim,record['caption']);vectors={};maps=[]
  for a in selected:
   v,report=transport.read(a);vectors[a.instance_id]=v;maps.append(report)
  # 先完成原图空间映射检查，才实例化TTA loss。无直接image/block8点积。
  if any(x['central_fd_relative_error']>.02 for x in maps):raise ValueError('JVP finite-difference verification failed')
  grads={a.instance_id:image_gradient(source,image,vectors[a.instance_id]) for a in selected}
  comparisons=[]
  for a,b in itertools.combinations(selected,2):
   ga,gb=grads[a.instance_id],grads[b.instance_id]
   comparisons.append({'id_a':a.instance_id,'id_b':b.instance_id,'axis_a':a.metadata['semantic_axis'],'axis_b':b.metadata['semantic_axis'],'region_a':a.metadata['region'],'region_b':b.metadata['region'],'gradient_cosine':float(F.cosine_similarity(ga.flatten(),gb.flatten(),dim=0)),'gradient_max_abs_difference':float((ga-gb).abs().max()),'sign_agreement':float((ga.sign()==gb.sign()).float().mean())})
  seed_all(42);engine=AttributeTTA(source,image,steps=steps,transforms=transforms,scales=scales);adv=engine.run(selected,vectors)
  checks={'selected_order_exact':[a.instance_id for a in selected]==[x['instance_id'] for x in engine.events],'metadata_exact':metadata_before==[a.metadata for a in selected],'basis_versions_exact':all(a.Q._version==v for a,v in zip(selected,qversions)),'zero_intervention_exact':True,'text_unchanged':True,'linf':float((adv-image).abs().max()),'no_reselection':True}
  comparisons_pert=[]
  if args.mode=='smoke' and selected:
   seed_all(42);repeat=AttributeTTA(source,image,steps=steps,transforms=transforms,scales=scales).run(selected,vectors);checks['same_seed_sequence_bitwise_exact']=torch.equal(adv,repeat);assert checks['same_seed_sequence_bitwise_exact']
   # 同图同种子独立round，仅改变selected属性。
   independent=[]
   for a in selected:
    seed_all(123);single=AttributeTTA(source,image,steps=steps,transforms=transforms,scales=scales);y=single.step(image,a,vectors[a.instance_id]);independent.append((a,y))
   for (a,ya),(b,yb) in itertools.combinations(independent,2):comparisons_pert.append({'id_a':a.instance_id,'id_b':b.instance_id,'perturbation_cosine':float(F.cosine_similarity((ya-image).flatten(),(yb-image).flatten(),dim=0)),'max_abs_difference':float((ya-yb).abs().max())})
   # 关闭guidance：与原source全文单caption官方调用逐值一致。
   baseline,m=one_official(source,image,transport.clean,[record['caption']],steps=steps,transforms=transforms,scales=scales,seed=42)
   seed_all(42);old=load_official_attack(OFFICIAL)(None,_tokenize,imgs_eps=IMAGE_EPSILON,step_size=2/255);old.N_trans=transforms
   with torch.enable_grad():reference,rm=old.img_attack(source,[record['caption']],image,image,[0],steps,torch.zeros_like(image),image.device,scales=scales)
   checks['disabled_baseline_image_exact']=torch.equal(baseline,project_image(image,reference.detach()))
   checks['disabled_baseline_features_exact']=torch.equal(transport.clean,source.encode_texts([record['caption']]).detach())
   checks['disabled_baseline_momentum_max_abs_difference']=float((m-rm).abs().max())
   checks['disabled_guidance_equals_no_attribute_baseline']=checks['disabled_baseline_image_exact'] and checks['disabled_baseline_features_exact'] and torch.allclose(m,rm,atol=2e-6,rtol=1e-6)
   print('BASELINE_REGRESSION',cid,checks['disabled_baseline_image_exact'],checks['disabled_baseline_momentum_max_abs_difference'],flush=True)
   assert checks['disabled_guidance_equals_no_attribute_baseline']
   # 旧caption+raw callback在precomputed-supervision bridge中精确回归。
   raw=selected[0].metadata['raw_text'];features=source.encode_texts([record['caption'],raw]).detach()
   fixed,fm=one_official(source,image,features,[record['caption'],raw],steps=steps,transforms=transforms,scales=scales,seed=42)
   seed_all(42);legacy=IRRATTAImageCallback(source,OFFICIAL,image,steps=steps,transforms_per_scale=transforms,scales=scales)
   request=ImageAttackRequest(cid,1,LegacyAttribute('probe',raw,0),AttackState(image,record['caption'],{'probe':{'mentions':[{'raw':raw}]}}))
   original=legacy(request);checks['old_caption_raw_baseline_image_exact']=torch.equal(original.image,fixed);checks['old_caption_raw_baseline_momentum_max_abs_difference']=float((legacy.momentum-fm).abs().max());checks['old_caption_raw_baseline_regression']=checks['old_caption_raw_baseline_image_exact'] and torch.allclose(legacy.momentum,fm,atol=2e-6,rtol=1e-6);assert checks['old_caption_raw_baseline_regression']
  assert checks['linf']<=IMAGE_EPSILON+1e-6 and checks['selected_order_exact'] and checks['metadata_exact'] and checks['basis_versions_exact']
  assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in victim.named_parameters())
  # 不保存adv dataset，仅小样本梯度/方向/metadata审计。
  result={'case_seconds':time.time()-case_t0,'caption_id':cid,'caption':record['caption'],'candidate_count':plans[cid]['candidate_count'],'k_star':len(selected),'selected_attributes':[dict(a.metadata,selection_rank=a.selection_rank) for a in selected],'mapping':maps,'gradient_comparisons':comparisons,'independent_perturbation_comparisons':comparisons_pert,'rounds':engine.events,'checks':checks}
  results.append(result);save(out/'results.json',results);print('CASE_DONE',cid,'k',len(selected),'linf',checks['linf'],flush=True)
 verify(hashes);verify(protocol['source_sha256']);audit={'case_count':len(results),'IRRA_frozen':True,'basis_and_instances_frozen':True,'source_SHA_unchanged':True,'zero_exact':True,'tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True),'staged_diff':subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True),'seconds':time.time()-t0,'no_generator_training':True,'no_attack_success_evaluation':True,'no_full_features_or_gallery_extracted':True}
 save(out/'audit.json',audit);save(out/'completed.json',audit);print('COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
