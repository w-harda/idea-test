"""原H7冻结推理；只读复用已审计G60图像/IRRA特征，新增Text-only/Joint原生评测。"""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import argparse,csv,io,json,time,traceback
import torch
from .gh_core import *
from tbps.baselines.cpgc_sep.victims import load_victim,encode_texts,frozen
from tbps.baselines.cpgc_sep.cache import NativeScorer,TBPSEvaluationEngine
from tbps.engines.evaluation import publish_result,load_result
from tbps.engines.metrics import ranking
from tbps.models.irra import fp32_runtime

def csv_data(ds,mode,groups,rounded=False):
 f=io.StringIO();w=csv.writer(f,lineterminator='\n');w.writerow(('Dataset','Victim','Mode',*KEYS))
 w.writerow((SPECS[ds][0],'IRRA',mode,*(format(groups[mode][k],'.2f') if rounded else groups[mode][k] for k in KEYS)))
 return f.getvalue().encode()

def preflight(c):
 require(not torch.cuda.is_initialized(),'CPU preflight only');base=Path(c['output_base']);s=ArtifactStore(base,new_base=base);require(not base.exists(),'fresh Attempt')
 with image.no_backward(cpu=True):
  ic=image_config(c);checked(FileRef(**c['H_checkpoint']));checked(FileRef(**c['G_checkpoint']))
  p,arch=H_payload(c)
  with frozen_H(c) as (h,_):
   require(sum(x.numel() for x in h.policy.parameters())==154594,'original H architecture')
   H_state=h.before
  targets={}
  for ds in DATASETS:
   src=VictimSource(ic,'irra',ds);src.verify();m=manifest(c,ds,'full');dep=dependency(c,ds,'smoke')
   targets[ds]={'Gallery':len(m.gallery),'Query':len(m.queries),'manifest_sha256':digest(asdict(m)),'IRRA':asdict(src.checkpoint),'smoke_image_audit':asdict(ref(dep[0].root/'cpu_audit/manifest.json'))}
  report={'status':'PASS','G_checkpoint':c['G_checkpoint'],'H_checkpoint':c['H_checkpoint'],'H_cursor':14336,'H_architecture':arch,'H_state_sha256':H_state,'H_parameters':154594,'policy_only_restored':True,'optimizer_restored':False,'CUDA':False,'targets':targets,'resources':resources(c)}
  s.publish('preflight',{'report.json':json_bytes(report),'protected.json':json_bytes(static_snapshot(c))},{'kind':'GH-transfer-CPU-preflight'},sources=(ref(c['path']),*tuple(FileRef(**v) for v in c['sources'].values())))
 require(not torch.cuda.is_initialized(),'no CPU CUDA initialization');print('GH_TRANSFER_CPU_PREFLIGHT_PASS',flush=True)

def gate_full(c):
 for ds in DATASETS:
  for mode in MODES:
   s=result_store(c,ds,'smoke',mode);s.verify('cpu_audit');a=read_json(s.root/'cpu_audit/report.json')
   require(a['status']=='PASS' and a['scope']=='smoke' and a['mode']==mode and a['config_sha256']==c['sha256'],'both real targets/two modes Smoke audits PASS required')

def generate(c,ds,scope):
 if scope=='full':gate_full(c)
 s=scope_store(c,ds,scope);require(not s.root.exists(),'no H Attempt retry');m=manifest(c,ds,scope)
 protocol={'schema':'H7-cross-dataset-text-1','H_checkpoint':c['H_checkpoint'],'G_checkpoint':c['G_checkpoint'],'manifest':asdict(m),'config_sha256':c['sha256'],'policy_input':'raw Caption string only','target_feedback':False,'test_label_policy_access':False}
 s.publish('protocol',{'protocol.json':json_bytes(protocol)},{'kind':'new-target-H7-text'},sources=(ref(c['path']),ref(Path(c['output_base'])/'preflight/manifest.json')))
 begin=time.monotonic();counts={'H_forwards':0,'G_forwards':0,'IRRA_forwards':0,'backward':0,'optimizer':0,'JVP':0};edited=noop=0
 with H_inference_guard(),fp32_runtime(),frozen_H(c) as (h,validate):
  before=h.before;current={'caption':None};native_calls=0
  def observe(module,args):
   nonlocal native_calls
   require(len(args)==1 and set(vars(args[0]))=={'caption','attributes'} and args[0].caption==current['caption'],'H input must be Caption-only')
   require(not torch.is_grad_enabled(),'H inference only');native_calls+=1
  hook=h.policy.register_forward_pre_hook(observe)
  with torch.inference_mode():
   for start in range(0,len(m.queries),32):
    rows=[];block=m.queries[start:start+32]
    s.publish(f'claim-{start:06d}',{'claim.json':json_bytes({'start':start,'count':len(block),'no_retry':True})},{'kind':'before-H-inference'},sources=(ref(s.root/'protocol/manifest.json'),))
    for q in block:
     current['caption']=q.caption;n=native_calls;action=h.attack(q.caption);validate(q.caption,action)
     require(native_calls-n==action['H_forward'] and action['H_forward'] in (0,1),'actual H forwards including legal NOOP')
     counts['H_forwards']+=action['H_forward'];edited+=int(action['adversarial']!=q.caption);noop+=int(action['adversarial']==q.caption)
     rows.append({'query_id':q.query_id,'action':action})
    s.publish(f'text-{start:06d}',{'rows.json':json_bytes(rows)},{'kind':'frozen-H-target-text','start':start,'count':len(block)},sources=(ref(s.root/'protocol/manifest.json'),ref(s.root/f'claim-{start:06d}/manifest.json')))
    progress(c,ds,scope,'H_TEXT_GENERATION',start+len(block),len(m.queries))
  hook.remove();done=h.finish();require(done['H_forward_samples']==native_calls==counts['H_forwards'],'real H count')
 report={'status':'PASS','scope':scope,'dataset':ds,'queries':len(m.queries),'counts':counts,'H_state_before':before,'H_state_after':done['H_state_sha256_after'],'H_checkpoint':c['H_checkpoint'],'edited':edited,'NOOP':noop,'target_model_loaded':False,'training':False,'seconds':time.monotonic()-begin}
 s.publish('text_completed',{'report.json':json_bytes(report)},{'kind':'all-new-target-text-generated'},sources=(ref(s.root/'protocol/manifest.json'),))
 print('GH_TRANSFER_NEW_TARGET_TEXT_PASS',ds,scope,flush=True)

class GHScorer(NativeScorer):
 def __init__(self,*args,c,ds,scope,mode,**kw):super().__init__(*args,**kw);self.c,self.ds,self.scope,self.mode=c,ds,scope,mode
 def score(self,q,g):
  v=super().score(q,g);qb,start,end=self.cursor;progress(self.c,self.ds,self.scope,'QUERY_SCORING',end,len(qb.ids),self.mode);return v

def evaluate(c,ds,scope,mode):
 if scope=='full':gate_full(c)
 old,m,a,native=dependency(c,ds,scope);textstore,_,rows=actions(c,ds,scope);s=result_store(c,ds,scope,mode);require(not s.root.exists(),'mode Attempt exists; no retry')
 s.publish('protocol',{'protocol.json':json_bytes({'dataset':ds,'scope':scope,'mode':mode,'image_audit':asdict(ref(old.root/'cpu_audit/manifest.json')),'H_text_completion':asdict(ref(textstore.root/'text_completed/manifest.json')),'config_sha256':c['sha256']})},{'kind':'GH-new-mode'},sources=(ref(c['path']),ref(old.root/'cpu_audit/manifest.json'),ref(textstore.root/'text_completed/manifest.json')))
 fs=ArtifactStore(textstore.root/'adv_features',new_base=c['output_base']);ic=image_config(c);counts={'new_image_samples':0,'new_text_samples':0,'G_forwards':0,'H_forwards':0,'backward':0,'optimizer':0,'JVP':0};resources(c)
 with image.no_backward(),fp32_runtime(),load_victim(ic,'irra',ds,'cuda:0',smoke=scope=='smoke') as (adapter,model,source):
  before=snapshot(model);actual={'image':0,'text':0}
  def forbid_image(*args):raise RuntimeError('IRRA image forward forbidden; reuse audited v2 image features')
  def observe_text(mod,args):actual['text']+=int(args[0].shape[1])
  ih=model.base_model.visual.register_forward_pre_hook(forbid_image);th=model.base_model.transformer.register_forward_pre_hook(observe_text)
  old,m,arrays,refs,adv_g_hash=original_features(c,ds,scope,adapter.space)
  text_hashes=tuple(hashlib.sha256(row['action']['adversarial'].encode()).hexdigest() for row in rows)
  if mode=='Text-only':
   require(not fs.root.exists(),'shared adversarial features only encoded once')
   fs.publish('protocol',{'protocol.json':json_bytes({'model_space':asdict(adapter.space),'H_checkpoint':c['H_checkpoint'],'text_hashes':text_hashes,'manifest_sha256':digest(asdict(m))})},{'kind':'shared-H-text-features'},sources=(ref(textstore.root/'text_completed/manifest.json'),source.checkpoint,ref(old.root/'cpu_audit/manifest.json')))
   features=[];feature_refs=[]
   for start in range(0,len(rows),8):
    block=rows[start:start+8];out=arrays['clean_text'][start:start+len(block)].detach().clone()
    changed=[j for j,row in enumerate(block) if row['action']['adversarial']!=m.queries[start+j].caption]
    if changed:
     out[changed]=encode_texts(adapter,[block[j]['action']['adversarial'] for j in changed],'irra');counts['new_text_samples']+=len(changed)
    key=f'text-{start:06d}';fs.publish(key,{'features.pt':tensor_bytes(out),'metadata.json':json_bytes({'start':start,'count':len(block),'new_text_samples':len(changed),'payload_sha256':payload_digest(out)})},{'kind':'H-native-text-feature-block'},sources=(ref(fs.root/'protocol/manifest.json'),))
    features.append(out);feature_refs.append(ref(fs.root/key/'manifest.json'));progress(c,ds,scope,'ENCODE_ADV_QUERY',start+len(block),len(rows),mode)
   adv_text=torch.cat(features);fs.publish('completed',{'report.json':json_bytes({'status':'PASS','queries':len(rows),'new_text_samples':counts['new_text_samples'],'payload_sha256':payload_digest(adv_text),'model_space':asdict(adapter.space)})},{'kind':'shared-adversarial-text-ready'},sources=(ref(fs.root/'protocol/manifest.json'),))
  else:
   fs.verify('completed');fp=read_json(fs.root/'protocol/protocol.json');require(fp['model_space']==asdict(adapter.space) and fp['text_hashes']==list(text_hashes),'Joint must reuse same target H texts/features')
   features=[];feature_refs=[]
   for p in sorted(fs.root.glob('text-*/features.pt')):
    fs.verify(p.parent.name);features.append(torch.load(p,map_location='cpu',weights_only=True).detach().clone());feature_refs.append(ref(p.parent/'manifest.json'))
   adv_text=torch.cat(features)
  require(actual['text']==counts['new_text_samples'],'actual text forward count')
  H_origin=AttackOrigin('hv1',FileRef(**c['H_checkpoint']),ref(textstore.root/'protocol/manifest.json'),H_BUDGET,digest(c['legacy_sources']))
  G_origin=AttackOrigin('gv2',FileRef(**c['G_checkpoint']),ref(old.root/'protocol/manifest.json'),G_BUDGET,digest(ic['legacy_sources']))
  original_t=tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in m.queries);original_g=tuple(g.image_sha256 for g in m.gallery)
  families=('adv_text','clean_gallery') if mode=='Text-only' else ('adv_text','adv_gallery');inputs={};banks={}
  for family in families:
   is_text=family=='adv_text';payload=adv_text if is_text else arrays[family];ids=tuple(q.query_id for q in m.queries) if is_text else tuple(g.image_id for g in m.gallery)
   hashes=text_hashes if is_text else (adv_g_hash if family=='adv_gallery' else original_g)
   binding=InputBinding(adapter.space.id,digest(asdict(m)),hashes,'text' if is_text else 'image',ids)
   origin=H_origin if is_text else (G_origin if family=='adv_gallery' else None)
   brefs=tuple(feature_refs)+(ref(fs.root/'completed/manifest.json'),) if is_text else tuple(refs[family])+(ref(old.root/'cpu_audit/manifest.json'),)
   inputs[family]=BoundRepresentation(RepresentationBatch(payload,adapter.space.id,binding.modality),binding,payload_digest(payload),brefs,origin,original_t if is_text else (original_g if origin else ()))
   banks[family]=SimpleNamespace(globals=payload,family=family,ids=ids,binding=binding)
  scorer=GHScorer(adapter,'irra',banks,store=s,config=c,c=c,ds=ds,scope=scope,mode=mode)
  protocol,result,scores=TBPSEvaluationEngine(m,scorer,source.checkpoint).run(inputs,modes=(mode,),query_chunk=128)
  publish_result(s,'result',protocol,result,scores,sources=(ref(s.root/'protocol/manifest.json'),ref(old.root/'cpu_audit/manifest.json')));del scores
  ih.remove();th.remove();frozen(model);require(snapshot(model)==before,'frozen target weights')
 report={'status':'PASS','dataset':ds,'scope':scope,'mode':mode,'Gallery':len(m.gallery),'Query':len(m.queries),'counts':counts,'actual_neural_samples':actual,'model_frozen':True,'model_before':digest(before),'model_after':digest(before),'model_space':asdict(adapter.space),'IRRA_checkpoint':asdict(source.checkpoint),'groups':result['groups'],'shared_text_protocol':asdict(ref(fs.root/'protocol/manifest.json')),'image_audit':asdict(ref(old.root/'cpu_audit/manifest.json'))}
 s.publish('completed',{'summary.json':json_bytes(report),'METRICS.csv':csv_data(ds,mode,result['groups']),'METRICS_2DP.csv':csv_data(ds,mode,result['groups'],True)},{'kind':'new-mode-scored-not-yet-audited'},sources=(ref(s.root/'result/manifest.json'),))
 print('GH_TRANSFER_SCORING_PASS',ds,scope,mode,flush=True)

def audit(c,ds,scope,mode):
 require(not torch.cuda.is_initialized(),'independent CPU audit only');s=result_store(c,ds,scope,mode);old,m,_,_=dependency(c,ds,scope)
 with image.no_backward(cpu=True):
  textstore,_,rows=actions(c,ds,scope,validate=True);gen=read_json(textstore.root/'text_completed/report.json')
  require(gen['status']=='PASS' and gen['queries']==len(m.queries) and gen['H_state_before']==gen['H_state_after'],'H source/freeze/full coverage')
  require(all(gen['counts'][k]==0 for k in ('G_forwards','IRRA_forwards','backward','optimizer','JVP')),'no generation feedback/training')
  for p in sorted(s.root.glob('*/manifest.json')):s.verify(p.parent.name)
  require(not s.pending() and not (s.root/'failure').exists(),'complete scoring mode')
  summary=read_json(s.root/'completed/summary.json');p=read_json(s.root/'result/protocol.json');_,report,scores=load_result(s,'result',expected_protocol_sha256=digest(p))
  require(p['modes']==[mode] and p['dataset_manifest']==json.loads(json_bytes(asdict(m))),'original manifest/mode')
  values,order=ranking(scores[mode],m.evaluation_identity());require(values==report['groups'][mode]==summary['groups'][mode],'full PID ranking CPU recomputation')
  require(summary['model_before']==summary['model_after'] and all(summary['counts'][k]==0 for k in ('G_forwards','H_forwards','new_image_samples','backward','optimizer','JVP')),'no forbidden model/optimizer/JVP activity')
  fs=ArtifactStore(textstore.root/'adv_features',new_base=c['output_base']);fs.verify('protocol');fs.verify('completed')
  blocks=[]
  for f in sorted(fs.root.glob('text-*/features.pt')):
   fs.verify(f.parent.name);blocks.append(torch.load(f,map_location='cpu',weights_only=True).detach().clone())
  adv_text=torch.cat(blocks);feature_report=read_json(fs.root/'completed/report.json');feature_protocol=read_json(fs.root/'protocol/protocol.json')
  require(adv_text.shape==(len(m.queries),512) and payload_digest(adv_text)==feature_report['payload_sha256']==p['inputs']['adv_text']['payload_sha256'],'complete shared adversarial text feature identity')
  require(feature_protocol['model_space']==summary['model_space'] and feature_protocol['H_checkpoint']==c['H_checkpoint'] and feature_protocol['manifest_sha256']==digest(asdict(m)),'native H feature provenance')
  require(summary['actual_neural_samples']['image']==0 and summary['actual_neural_samples']['text']==summary['counts']['new_text_samples'],'no re-encoded image samples')
  expected=tuple(hashlib.sha256(row['action']['adversarial'].encode()).hexdigest() for row in rows)
  require(feature_protocol['text_hashes']==list(expected),'shared feature source texts')
  require(p['inputs']['adv_text']['inputs']['ordered_sha256']==list(expected),'same target-generated H text identity')
  if mode=='Joint':
   other=result_store(c,ds,scope,'Text-only');other.verify('cpu_audit');other_summary=read_json(other.root/'completed/summary.json')
   require(other_summary['shared_text_protocol']==summary['shared_text_protocol'],'Text-only/Joint share exact H text/features')
  check_protection(c)
  s.publish('ranking',{'order.pt':tensor_bytes(order)},{'kind':'independent-full-PID-order'},sources=(ref(s.root/'result/manifest.json'),))
  a={'status':'PASS','scope':scope,'dataset':ds,'mode':mode,'Gallery':len(m.gallery),'Query':len(m.queries),'groups':{mode:values},'metrics_recomputed':True,'CUDA':False,'neural_forwards':0,'G_forwards':0,'H_forwards':0,'backward':0,'optimizer':0,'JVP':0,'config_sha256':c['sha256'],'shared_text_protocol':summary['shared_text_protocol'],'image_audit':summary['image_audit'],'H_checkpoint':c['H_checkpoint'],'G_checkpoint':c['G_checkpoint']}
  s.publish('cpu_audit',{'report.json':json_bytes(a)},{'kind':'GH-independent-CPU-audit'},sources=(ref(s.root/'completed/manifest.json'),ref(c['path'])))
 print('GH_TRANSFER_INDEPENDENT_CPU_AUDIT_PASS',ds,scope,mode,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--action',choices=('preflight','generate','evaluate','audit'));p.add_argument('--dataset',choices=DATASETS);p.add_argument('--scope',choices=('smoke','full'),default='smoke');p.add_argument('--mode',choices=MODES);p.add_argument('--allow-inference',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);c=config(a.config)
 if a.action in ('generate','evaluate') and not a.allow_inference:p.error('explicit frozen inference authorization required')
 if a.action=='preflight':preflight(c);return
 require(a.dataset in DATASETS,'target dataset required')
 try:
  if a.action=='generate':generate(c,a.dataset,a.scope)
  else:require(a.mode in MODES,'mode required');globals()[a.action](c,a.dataset,a.scope,a.mode)
 except BaseException as e:
  s=scope_store(c,a.dataset,a.scope) if a.action=='generate' else result_store(c,a.dataset,a.scope,a.mode)
  if not (s.root/'failure').exists():s.publish('failure',{'error.json':json_bytes({'error':repr(e),'traceback':traceback.format_exc()})},{'kind':'failure-preserve-no-retry'},sources=(ref(c['path']),))
  raise
if __name__=='__main__':main()
