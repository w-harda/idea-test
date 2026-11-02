"""冻结G在目标图像上重新前向，原生IRRA评分，分离CPU重新排名审计。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import ExitStack
from types import SimpleNamespace
import argparse,csv,io,json,time,traceback,resource
import torch
from .gv2_core import *
from tbps.baselines.cpgc_sep.victims import load_victim,encode_images,encode_texts,snapshot,frozen
from tbps.baselines.cpgc_sep.cache import NativeScorer,TBPSEvaluationEngine
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.models.base import RepresentationBatch,InputBinding
from tbps.engines.evaluation import publish_result,load_result
from tbps.engines.metrics import ranking
from tbps.models.irra import fp32_runtime
from tbps.attacks.gv2 import BUDGET

def gate_full(c):
 for ds in DATASETS:
  s=store_for(c,ds,'smoke');s.verify('cpu_audit')
  a=read_json(s.root/'cpu_audit/report.json')
  require(a['status']=='PASS' and a['dataset']==ds and a['scope']=='smoke' and a['config_sha256']==c['sha256'],'both real Smoke CPU audits required')

def csv_bytes(dataset,groups,rounded=False):
 b=io.StringIO();w=csv.writer(b,lineterminator='\n');w.writerow(('Dataset','Victim','Mode',*KEYS))
 for mode in ('Clean','Image-only'):
  label='Image-only (G-v2 E60)' if mode=='Image-only' else mode
  w.writerow((SPECS[dataset][0],'IRRA',label,*(format(groups[mode][k],'.2f') if rounded else groups[mode][k] for k in KEYS)))
 return b.getvalue().encode()

def generate(c,ds,scope):
 ensure_preflight(c)
 torch.manual_seed(c['seed'])
 if scope=='full':gate_full(c)
 manifest=get_manifest(c,ds,scope);store=store_for(c,ds,scope)
 require(not store.root.exists(),'Attempt exists; no retry/overwrite')
 report={'status':'RUNNING','dataset':ds,'scope':scope,'G_loads':0,'G_forwards':0,'backward':0,'optimizer':0,'JVP':0,'victim_forwards':0,'target_feedback':False,'resources':resources(c)}
 protocol={'schema':'gv2-cross-dataset-generation-1','manifest':asdict(manifest),'epoch':60,'checkpoint':c['checkpoint'],'config_sha256':c['sha256'],
  'native_pixels':'original PIL RGB -> frozen pixels_from_pil Resize384x128/ToTensor; FP32 RGB -> original generate_once',
  'G_input':'image only; no PID/Caption/features/target feedback','epsilon':8/255,'image_batch':1}
 store.publish('protocol',{'protocol.json':json_bytes(protocol)},{'kind':'target-Gallery-generation'},sources=(ref(c['path']),ref(Path(c['output_base'])/'preflight/manifest.json')))
 begin=time.monotonic()
 with no_backward(),fp32_runtime(),frozen_generator(c,'cuda:0') as (g,fn,pixel_reader):
  report['G_loads']=1;before=tree_hash(g.state_dict())
  def hook(module,args):
   require(len(args)==1 and tuple(args[0].shape)==(1,3,384,128) and args[0].dtype==torch.float32 and not torch.is_grad_enabled(),'original image only and no graph')
   report['G_forwards']+=1
  handle=g.register_forward_pre_hook(hook)
  from PIL import Image
  with torch.inference_mode():
   for i,row in enumerate(manifest.gallery):
    original_ref=FileRef(str(Path(c['datasets'][ds]['root'])/'imgs'/row.image_rel_path),row.image_sha256);checked(original_ref)
    # 不可变Claim在实际forward前提交，失败绝不自动重做。
    store.publish(f'claim-{i:06d}',{'claim.json':json_bytes({'index':i,'image_id':row.image_id,'one_forward':True})},{'kind':'one-G-forward-claim'},sources=(ref(store.root/'protocol/manifest.json'),))
    with Image.open(original_ref.path) as im:original=pixel_reader(im).unsqueeze(0)
    require(torch.equal(original,native_pixels(original_ref.path,'irra')),'target preprocess differs from frozen G preprocessing')
    count=report['G_forwards'];value=fn(g,original.to('cuda:0'));require(report['G_forwards']==count+1,'exactly one G forward')
    arrays={'original':original.detach().clone(),'adversarial':value['adversarial'].cpu().clone(),'effective_delta':value['effective_delta'].cpu().clone()}
    stats=image_checks(arrays['original'],arrays['adversarial'],arrays['effective_delta'],(384,128))
    md={'index':i,'image_id':row.image_id,'original':asdict(original_ref),'pixels':stats,'checkpoint':c['checkpoint'],'G_forwards':1,'backward':0,'optimizer':0,'JVP':0}
    store.publish(f'image-{i:06d}',{'pixels.pt':tensor_bytes(arrays),'metadata.json':json_bytes(md)},{'kind':'G-target-RGB'},sources=(ref(store.root/'protocol/manifest.json'),ref(store.root/f'claim-{i:06d}/manifest.json'),original_ref))
    del value,arrays,original
    progress(c,ds,scope,'GENERATE_GALLERY',i+1,len(manifest.gallery))
  handle.remove();after=tree_hash(g.state_dict());frozen(g)
  require(before==after and report['G_forwards']==len(manifest.gallery),'G frozen/count')
 report.update(status='PASS',G_state_before=before,G_state_after=after,seconds=time.monotonic()-begin,Gallery=len(manifest.gallery),Query=len(manifest.queries),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
 store.publish('generation_completed',{'report.json':json_bytes(report)},{'kind':'generated-all-target-images'},sources=(ref(store.root/'protocol/manifest.json'),))
 print('GV2_CROSS_GENERATION_PASS',ds,scope,json.dumps(report),flush=True)

class TransferScorer(NativeScorer):
 def __init__(self,*args,transfer_config,dataset,scope,**kw):
  super().__init__(*args,**kw);self.tc,self.ds,self.scope=transfer_config,dataset,scope
 def score(self,query,gallery):
  value=super().score(query,gallery)
  qb,start,end=self.cursor;gb=self.refs[id(gallery.payload)]
  progress(self.tc,self.ds,self.scope,'SCORE_QUERY_'+gb.family,end,len(qb.ids))
  return value

def encode_group(store,key,payload,details):
 store.publish(key,{'features.pt':tensor_bytes(payload),'details.json':json_bytes(details)},{'kind':'target-native-features'},sources=(ref(store.root/'protocol/manifest.json'),))

def evaluate(c,ds,scope):
 ensure_preflight(c)
 if scope=='full':gate_full(c)
 m=get_manifest(c,ds,scope);store=store_for(c,ds,scope);store.verify('generation_completed')
 require(not (store.root/'evaluation').exists(),'evaluation Attempt already claimed')
 store.publish('evaluation',{'claim.json':json_bytes({'dataset':ds,'scope':scope,'no_retry':True})},{'kind':'native-evaluation-claim'},sources=(ref(store.root/'generation_completed/manifest.json'),))
 begin=time.monotonic();counts={'image_samples':0,'text_samples':0,'backward':0,'optimizer':0,'JVP':0,'G_forwards':0};r={'resources':resources(c)}
 with no_backward(),fp32_runtime(),load_victim(c,'irra',ds,'cuda:0',smoke=scope=='smoke') as (adapter,model,source):
  before=snapshot(model);actual={'image':0,'text':0}
  # 观测官方视觉/文本Transformer的真实forward，不仅依赖外部计数器。
  def image_hook(mod,args):actual['image']+=len(args[0])
  def text_hook(mod,args):actual['text']+=int(args[0].shape[1])
  h1=model.base_model.visual.register_forward_pre_hook(image_hook)
  h2=model.base_model.transformer.register_forward_pre_hook(text_hook)
  p={'schema':'gv2-cross-dataset-native-evaluation-1','manifest_sha256':digest(asdict(m)),'model_space':asdict(adapter.space),'config_sha256':c['sha256'],'checkpoint':asdict(source.checkpoint),'native_precision':adapter.space.precision,'query_chunk':128,'image_batch':1,'text_batch':8,'Raw_Caption':True}
  store.publish('native_protocol',{'protocol.json':json_bytes(p)},{'kind':'native-IRRA-target-session'},sources=(ref(store.root/'protocol/manifest.json'),source.checkpoint))
  feature_parts={'clean_gallery':[],'adv_gallery':[],'clean_text':[]};refs={k:[] for k in feature_parts};advhash=[]
  pixelstore=VerifiedArtifactStore(store.root,new_base=c['output_base'])
  for start in range(0,len(m.gallery),32):
   clean=[];adv=[]
   for i in range(start,min(start+32,len(m.gallery))):
    arrays,md=load_pixels_unit(pixelstore,i)
    clean.append(encode_images(adapter,arrays['original'],'irra').detach().clone());adv.append(encode_images(adapter,arrays['adversarial'],'irra').detach().clone())
    advhash.append(md['pixels']['adversarial_pixels_sha256']);counts['image_samples']+=2;del arrays
    progress(c,ds,scope,'ENCODE_GALLERY',i+1,len(m.gallery))
   values={'clean_gallery':torch.cat(clean),'adv_gallery':torch.cat(adv)};key=f'gallery-feature-{start:06d}'
   encode_group(store,key,values,{'start':start,'end':start+len(clean),'payload_sha256':payload_digest(values)})
   for family in values:feature_parts[family].append(values[family]);refs[family].append(ref(store.root/key/'manifest.json'))
  for start in range(0,len(m.queries),8):
   rows=m.queries[start:start+8];x=encode_texts(adapter,[q.caption for q in rows],'irra').detach().clone()
   key=f'text-feature-{start:06d}';encode_group(store,key,x,{'start':start,'end':start+len(rows),'payload_sha256':payload_digest(x)})
   feature_parts['clean_text'].append(x);refs['clean_text'].append(ref(store.root/key/'manifest.json'));counts['text_samples']+=len(rows)
   progress(c,ds,scope,'ENCODE_QUERY',start+len(rows),len(m.queries))
  h1.remove();h2.remove()
  ids_g=tuple(g.image_id for g in m.gallery);ids_q=tuple(q.query_id for q in m.queries)
  original_g=tuple(g.image_sha256 for g in m.gallery);original_t=tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in m.queries)
  origin=AttackOrigin('gv2',FileRef(**c['checkpoint']),ref(store.root/'protocol/manifest.json'),BUDGET,digest(c['legacy_sources']))
  banks={};inputs={}
  for family,parts in feature_parts.items():
   x=torch.cat(parts);is_text=family=='clean_text';hashes=original_t if is_text else (tuple(advhash) if family=='adv_gallery' else original_g)
   binding=InputBinding(adapter.space.id,digest(asdict(m)),hashes,'text' if is_text else 'image',ids_q if is_text else ids_g)
   inputs[family]=BoundRepresentation(RepresentationBatch(x,adapter.space.id,binding.modality),binding,payload_digest(x),tuple(refs[family]),origin if family=='adv_gallery' else None,original_g if family=='adv_gallery' else ())
   banks[family]=SimpleNamespace(globals=x,family=family,ids=binding.ordered_ids,binding=binding)
  scorer=TransferScorer(adapter,'irra',banks,store=store,config=c,transfer_config=c,dataset=ds,scope=scope)
  engine=TBPSEvaluationEngine(m,scorer,source.checkpoint)
  protocol,result,scores=engine.run(inputs,modes=('Clean','Image-only'),query_chunk=128)
  publish_result(store,'result',protocol,result,scores,sources=(ref(store.root/'native_protocol/manifest.json'),));del scores
  frozen(model);require(snapshot(model)==before,'Victim parameters/buffers changed')
  require(counts['image_samples']==actual['image']==2*len(m.gallery) and counts['text_samples']==actual['text']==len(m.queries),'real native forward sample count')
 r.update(status='PASS',dataset=ds,scope=scope,counts=counts,actual_neural_samples=actual,model_frozen=True,model_state_before_sha256=digest(before),model_state_after_sha256=digest(before),Gallery=len(m.gallery),Query=len(m.queries),groups=result['groups'],checkpoint=asdict(source.checkpoint),model_space=asdict(adapter.space),seconds=time.monotonic()-begin,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 store.publish('completed',{'summary.json':json_bytes(r),'METRICS.csv':csv_bytes(ds,result['groups']),'METRICS_2DP.csv':csv_bytes(ds,result['groups'],True)},{'kind':'Full-scoring-completed-not-yet-audited'},sources=(ref(store.root/'result/manifest.json'),))
 print('GV2_CROSS_NATIVE_SCORING_PASS',ds,scope,json.dumps(r),flush=True)

def audit(c,ds,scope):
 require(not torch.cuda.is_initialized(),'independent CPU audit');torch.set_num_threads(2)
 ensure_preflight(c);m=get_manifest(c,ds,scope);s=store_for(c,ds,scope)
 with no_backward(cpu=True):
  verified=VerifiedArtifactStore(s.root,new_base=c['output_base']);units=0
  for p in sorted(s.root.glob('*/manifest.json')):
   verified.verify(p.parent.name);units+=1
  require(not s.pending() and not (s.root/'failure').exists(),'incomplete/failed Attempt')
  generation=read_json(s.root/'generation_completed/report.json');summary=read_json(s.root/'completed/summary.json')
  require(generation['status']==summary['status']=='PASS' and generation['G_forwards']==len(m.gallery) and generation['G_state_before']==generation['G_state_after'],'G forwards/freeze proof')
  require(generation['G_state_before']==ensure_preflight(c)['G_state_sha256'],'loaded trained E60 identity')
  require(all(generation[k]==summary['counts'][k]==0 for k in ('backward','optimizer','JVP')),'forbidden training/JVP')
  require(summary['counts']['G_forwards']==0 and summary['model_state_before_sha256']==summary['model_state_after_sha256'],'separate generation/evaluation freeze')
  maximum=0.0
  for i,g in enumerate(m.gallery):
   arrays,md=load_pixels_unit(verified,i);p=Path(c['datasets'][ds]['root'])/'imgs'/g.image_rel_path
   require(md['image_id']==g.image_id and md['original']==asdict(FileRef(str(p),g.image_sha256)) and md['checkpoint']==c['checkpoint'],'target image/G identity')
   checked(FileRef(str(p),g.image_sha256));fresh=native_pixels(p,'irra')
   require(torch.equal(fresh,arrays['original']),'original FP32 RGB mismatch')
   maximum=max(maximum,md['pixels']['linf']);del arrays,fresh
   progress(c,ds,scope,'CPU_PIXEL_AUDIT',i+1,len(m.gallery))
  protocol,report,scores=load_result(s,'result',expected_protocol_sha256=digest(read_json(s.root/'result/protocol.json')))
  require(protocol['dataset_manifest']==json.loads(json_bytes(asdict(m))),'original Caption/PID/ordering identity')
  require(protocol['model_space']==summary['model_space'] and protocol['modes']==['Clean','Image-only'],'target/scoring mode identity')
  recomputed={};orders={}
  for mode,matrix in scores.items():
   require(tuple(matrix.shape)==(len(m.queries),len(m.gallery)),'complete Query x Gallery matrix')
   metrics,order=ranking(matrix,m.evaluation_identity())
   require(metrics==report['groups'][mode]==summary['groups'][mode],'full PID rank/metric CPU recomputation mismatch')
   recomputed[mode]=metrics;orders[mode]=payload_digest(order)
   # 完整排名逐组保存，不只存first-hit；与评分矩阵绑定。
   s.publish('ranking-'+mode,{'order.pt':tensor_bytes(order)},{'kind':'CPU-full-query-order'},sources=(ref(s.root/'result/manifest.json'),));del order
  require((s.root/'completed/METRICS.csv').read_bytes()==csv_bytes(ds,recomputed) and (s.root/'completed/METRICS_2DP.csv').read_bytes()==csv_bytes(ds,recomputed,True),'CSV identity')
  tokens=token_hashes(VictimSource(c,'irra',ds),m)
  require(tokens['tokenizer_fingerprint']==ensure_preflight(c)['targets'][ds]['tokens']['tokenizer_fingerprint'],'Tokenizer identity')
  protected=verify_history(c)
  r={'status':'PASS','dataset':ds,'scope':scope,'Gallery':len(m.gallery),'Query':len(m.queries),'config_sha256':c['sha256'],'groups':recomputed,'ranking_payload_sha256':orders,'metrics_recomputed':True,'all_Gallery_G_forwards':len(m.gallery),'max_linf_FP32_real_difference':maximum,'CUDA':False,'neural_forwards':0,'training_updates':0,'backward':0,'optimizer':0,'JVP':0,'all_raw_Caption_preserved':True,'G_checkpoint':c['checkpoint'],'IRRA_checkpoint':summary['checkpoint'],'tokens':tokens,'immutable_units_verified':units,'protected_historical_files':protected}
  s.publish('cpu_audit',{'report.json':json_bytes(r)},{'kind':'independent-CPU-full-retrieval-audit'},sources=(ref(s.root/'completed/manifest.json'),ref(c['path'])))
 print('GV2_CROSS_INDEPENDENT_CPU_AUDIT_PASS',ds,scope,json.dumps(r),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--action',choices=('preflight','generate','evaluate','audit'),default='preflight');p.add_argument('--dataset',choices=DATASETS);p.add_argument('--scope',choices=('smoke','full'),default='smoke');p.add_argument('--allow-gpu',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);c=load_config(a.config)
 if a.action in ('generate','evaluate') and not a.allow_gpu:p.error('explicit GPU authorization required')
 if a.action=='preflight':preflight(c);return
 require(a.dataset in DATASETS,'dataset required')
 try:globals()[a.action](c,a.dataset,a.scope)
 except BaseException as e:
  s=store_for(c,a.dataset,a.scope)
  if not (s.root/'failure').exists():s.publish('failure',{'failure.json':json_bytes({'action':a.action,'error':repr(e),'traceback':traceback.format_exc()})},{'kind':'failed-Attempt-no-retry'},sources=(ref(c['path']),))
  raise
if __name__=='__main__':main()
