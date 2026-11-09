"""协同数学、原G/H实际CPU机制、分片完整性与单次更新回归。"""
import copy,hashlib,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,json_bytes,read_json
from tbps.cooperative.objectives import interaction,residual_weights,caption_set_loss,edit_credit_loss,group_edit_log_probs
from tbps.cooperative.g_step import cooperative_g
from tbps.cooperative.h_step import train_step
from tbps.cooperative.store import shard_ledger,read_ledger,SHARD_LIMIT
ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/framework/gh_cooperative_v1.server.json'
SETTINGS={'g_synergy_weight':.1,'h_synergy_weight':.1,'edit_credit_weight':.1,'caption_set_weight':.1,'residual_mix':.5,'attribute_temperature':1.,'caption_temperature':.1}

class MathTests(unittest.TestCase):
 def test_four_mode_derivative_has_both_online_terms(self):
  a=torch.tensor([[.3,.7]],requires_grad=True);clean=torch.tensor([[1.,0.]]);t=torch.tensor([[.8,.2]]);adv=torch.tensor([[.2,.8]])
  s,m=interaction(clean,a,t,adv);s.backward();self.assertTrue(torch.allclose(a.grad,t-adv))
  self.assertAlmostEqual(float(s),float(m['s10']+m['s01']-m['s11']-m['s00']))
 def test_detached_mean_one_and_K0_K1(self):
  for k in (0,1,3,7):
   x=torch.randn(1,5,requires_grad=True);j=torch.randn(k,5);p=torch.ones(k)/max(k,1)
   w,e=residual_weights(x,j,p);self.assertFalse(w.requires_grad);self.assertEqual(len(w),k)
   if k:self.assertAlmostEqual(float(w.mean()),1.,places=6)
 def test_zero_mix_original_local_weights(self):
  w,_=residual_weights(torch.randn(1,5),torch.randn(7,5),torch.ones(7)/7,mix=0);self.assertTrue(torch.equal(w,torch.ones(7)))
 def test_smooth_max_prioritizes_remaining_match(self):
  x=torch.tensor([[.8,.2]],requires_grad=True);caption_set_loss(x,torch.eye(2),.1).backward();self.assertGreater(float(x.grad[0,0]),float(x.grad[0,1]))
 def test_edit_credit_gradient_and_noop(self):
  a=torch.tensor(-2.,requires_grad=True);b=torch.tensor(-3.,requires_grad=True);loss=edit_credit_loss(a,[a,b],[1.,-2.],.1);loss.backward()
  self.assertAlmostEqual(float(a.grad),-.05);self.assertAlmostEqual(float(b.grad),.1)
  self.assertEqual(float(edit_credit_loss(a,[],[],.1)),0.)
 def test_stop_noop_grouping(self):
  ds=[{'head':'attribute'},{'head':'position'},{'head':'attribute'},{'head':'position'},{'head':'replacement'},{'head':'attribute'}]
  gs=group_edit_log_probs(ds,[torch.tensor(float(i)) for i in range(6)],[object()]);self.assertEqual(float(gs[0]),9.)
 def test_shards_cover_large_metadata_and_detect_mutation(self):
  with tempfile.TemporaryDirectory(dir='/home/lzf/ldx/tmp') as td:
   base=Path(td)/'tbps-framework-cooperative-test';root=base/'train/ledgers';ledger={'round':1,'G':[{'i':i,'payload':'x'*100000} for i in range(680)],'H':[],'conditions':[]}
   files=shard_ledger(ledger);self.assertGreater(sum(map(len,files.values())),64*1024**2);self.assertLess(max(map(len,files.values())),SHARD_LIMIT+1)
   store=ArtifactStore(root,new_base=base);store.publish('round_0001',files,{})
   got=read_ledger(root/'round_0001/ledger.json');self.assertEqual([x['i'] for x in got['G']],list(range(680)))
   chunk=next((root/'round_0001').glob('G_*.json'));chunk.write_bytes(chunk.read_bytes()+b' ')
   with self.assertRaises(ArtifactError):read_ledger(root/'round_0001/ledger.json')
 def test_rolling_journal_checkpoint_resume_and_failure_stop(self):
  from tbps.cooperative.store import FreshJournal
  from tbps.alternating.state import tree_sha
  with tempfile.TemporaryDirectory(dir='/home/lzf/ldx/tmp') as td:
   base=Path(td)/'tbps-framework-gh-cooperative-v1-20261109';run=base/'train';journal=FreshJournal(run,'f'*64,str(base))
   state={'G':{'v':torch.tensor([0.])},'H':{'v':torch.tensor([0.])},'parents':{'test':'identity'},'cursor':{'round':0}}
   journal.initialize({'test':True},state)
   for i in range(1,11):
    journal.begin(i,state);state=copy.deepcopy(state);state['cursor']['round']=i;state['G']['v'].add_(1)
    bank=ArtifactStore(run/'B'/f'bank_{i}',new_base=base);bank.publish('completed',{'id.json':json_bytes({'round':i})},{})
    io=run/'round_io'/f'round_{i:04d}';io.mkdir(parents=True)
    ledger={'round':i,'G_after':tree_sha(state['G']),'bank':{'g_state_sha256':tree_sha(state['G'])},'G':[],'H':[],'conditions':[]}
    ref=__import__('tbps.artifacts',fromlist=['FileRef']).FileRef(str(bank.root/'completed/manifest.json'),__import__('tbps.artifacts',fromlist=['sha_file']).sha_file(bank.root/'completed/manifest.json'))
    journal.commit(i,state,ledger,ref)
   journal.stop_boundary(10);restored,summary=journal.load_latest(state['parents']);self.assertEqual(tree_sha(restored),tree_sha(state))
   self.assertTrue((run/'checkpoints/round_10/G.pt').is_file());self.assertTrue((run/'ledgers/round_0001/ledger.json').is_file())
   self.assertFalse((run/'boundaries/round_0009').exists());self.assertFalse((run/'B/bank_9').exists())
   (run/'FAIL_injected').write_text('failure retained')
   with self.assertRaises(ValueError):journal.load_latest(state['parents'])
 def test_scheduler_ten_round_pause_and_exact_twenty_round_resume(self):
  import importlib.util
  from tbps.cooperative.store import FreshJournal
  from tbps.alternating.configured_training import run_rounds
  from tbps.alternating.state import tree_sha
  spec=importlib.util.spec_from_file_location('cooperative_fixture',Path(__file__).with_name('test_gh_alternating_v2.py'))
  fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);Backend=fixture.fake_backend()
  with tempfile.TemporaryDirectory(dir='/home/lzf/ldx/tmp') as td:
   def init(name):
    base=Path(td)/name/'tbps-framework-gh-cooperative-v1-20261109';b=Backend(base/'train');state=b.complete_state(0)
    j=FreshJournal(base/'train','f'*64,base);j.initialize({},state);return b,j,state
   b,j,state=init('split');state,done=run_rounds(b,j,state,20,stop_after=10);self.assertFalse(done)
   restored,ledger=j.load_latest(b.parents);next_backend=Backend(b.root);next_backend.counts.update(ledger['forward_counts_cumulative'])
   state,done=run_rounds(next_backend,j,restored,20,start=10);self.assertTrue(done)
   other,k,initial=init('continuous');expected,done=run_rounds(other,k,initial,20)
   self.assertEqual(tree_sha(state),tree_sha(expected));self.assertEqual(state['cursor']['round'],20)
 def test_previous_GH_v2_weights_rejected_by_new_fresh_guard(self):
  from tbps.cooperative.run import no_historical_weights
  with patch.object(torch,'load',return_value='read'):
   with no_historical_weights('/home/lzf/ldx/outputs/tbps-framework-gh-cooperative-v1-20261109'):
    with self.assertRaises(ValueError):torch.load('/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107/train/checkpoints/round_60/G.pt')
 def test_global_json_limit_remains(self):
  from tbps import artifacts
  with tempfile.TemporaryDirectory(dir='/home/lzf/ldx/tmp') as td:
   p=Path(td)/'large.json'
   with p.open('wb') as f:f.truncate(64*1024**2+1)
   with self.assertRaises(ArtifactError):read_json(p)

class NativeMechanismTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2)
  from tbps.cooperative.sources import load
  from tbps.alternating.fresh_sources import fresh_bundle
  from tbps.alternating.native_sources import native_session,strict_import_preflight
  cls.f,cls.c=load(CFG,'smoke');strict_import_preflight(cls.c);cls.bundle=fresh_bundle(cls.f,cls.c)
  cls.session_context=native_session(cls.c,reuse=cls.bundle.modules);cls.session=cls.session_context.__enter__();cls.b=cls.bundle.bindings
 @classmethod
 def tearDownClass(cls):cls.session_context.__exit__(None,None,None)
 def test_original_G_K_plus_one_and_function_restoration(self):
  gm=read_json(self.c['prototype']['g_manifest'])['entries'];serial=self.b.get('serial_training');old=(serial.attribute_losses,serial.joint_losses,serial._weighted)
  class G(torch.nn.Module):
   def __init__(self):super().__init__();self.v=torch.nn.Parameter(torch.ones(1,3,1,1)*.001)
   def forward(self,x):return self.v.tanh().expand_as(x)
  class E(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer('w',torch.randn(3,512))
   def forward(self,x):return torch.nn.functional.normalize(x.mean((2,3))@self.w,dim=1)
  for k in (0,1,3,7):
   entry=next(e for e in gm if e['role']=='train' and e['K']==k);v=torch.load(entry['tensor_path'],map_location='cpu',weights_only=True)
   g=G();e=E().eval();o=torch.optim.Adam(g.parameters(),lr=.0002);out={}
   extras={'clean_image':torch.nn.functional.normalize(torch.randn(1,512),dim=1),'clean_text':v['z_t'][None],
    'jvps':v['attribute_jvps'],'prior':v['score_weights'],'caption_set':v['z_t'][None]}
   with cooperative_g(serial,extras,SETTINGS,out) as details:result=serial.train_single_sample(g,e,o,torch.ones(1,3,8,8)*.5,v)

   for rec,vv in zip(result['records'],details):rec['cooperative']=vv
   result.update(cooperative=out,row_id='cpu-test',G_before='start',G_after='end')
   self.__class__.g_log=result
   self.assertEqual(result['optimizer_steps'],k+1);self.assertEqual(len(details),k+1);self.assertEqual(int(next(iter(o.state.values()))['step']),k+1)
   self.assertLessEqual(max(x['cumulative_linf'] for x in result['records']),8/255);self.assertEqual(old,(serial.attribute_losses,serial.joint_losses,serial._weighted))
 def test_official_annotation_array_and_secondary_caption_identity(self):
  from tbps.cooperative.backend import read_annotation
  annotation=read_annotation(self.c['prototype']['annotation'])
  entries=read_json(self.c['prototype']['g_manifest'])['entries'];mapping={e['row_id']:e for e in entries}
  for row in self.c['prototype']['samples']:
   raw=annotation[int(row.split(':')[1])];e=mapping[row]
   self.assertEqual(raw['split'],'train');self.assertEqual(raw['captions'][e['caption_index']],e['caption'])
 def test_backend_G_multicaption_wiring_on_CPU(self):
  from tbps.cooperative.backend import CooperativeBackend,read_annotation
  from tbps.alternating.conditions import LegacySteps
  from tbps.alternating.state import tree_sha
  gm=read_json(self.c['prototype']['g_manifest'])['entries'];entry=next(e for e in gm if e['row_id'] in self.c['prototype']['samples'] and e['K']>0)
  row=entry['row_id'];value=torch.load(entry['tensor_path'],map_location='cpu',weights_only=True)
  class G(torch.nn.Module):
   def __init__(self):super().__init__();self.v=torch.nn.Parameter(torch.ones(1,3,1,1)*.001)
   def forward(self,x):return self.v.tanh().expand_as(x)
  class E(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer('w',torch.randn(3,512))
   def forward(self,x):return torch.nn.functional.normalize(x.mean((2,3))@self.w,dim=1)
  class Victim:
   def encode_texts(self,texts):return torch.nn.functional.normalize(torch.randn(len(texts),512),dim=1)
  with tempfile.TemporaryDirectory(dir='/home/lzf/ldx/tmp') as td:
   base=Path(td)/'tbps-framework-cooperative-fixture';b=CooperativeBackend.__new__(CooperativeBackend)
   b.c=self.c;b.p=self.c['prototype'];b.bundle=self.bundle;b.native=self.b;b.g=G();b.ga=torch.optim.Adam(b.g.parameters(),lr=.0002)
   b.h=self.bundle.h.eval().requires_grad_(False);b.encoder=E().eval();b.victim=Victim();b.grant=SimpleNamespace(validate=lambda *a:None)
   b.cursor={'g_updates':0};b.clean=torch.nn.functional.normalize(torch.randn(2048,512),dim=1)
   b.annotation=read_annotation(b.p['annotation']);b.annotation_sha=hashlib.sha256(Path(b.p['annotation']).read_bytes()).hexdigest()
   b.entries={row:{'caption':entry['caption']}};b.original={row:SimpleNamespace(payload=value)};b.gentries={row:entry}
   b.store=ArtifactStore(base/'round',new_base=base);b.image=lambda unused:torch.ones(1,3,8,8)*.5
   b.steps=LegacySteps(self.b.get('serial_training').train_single_sample,None)
   condition=SimpleNamespace(payload=value,identity=SimpleNamespace(key='test-context'))
   log=b.g_update(row,condition);self.assertEqual(log['optimizer_steps'],value['K']+1)
   self.assertIsNotNone(log['cooperative']['metadata']['secondary']);self.assertEqual(log['records'][-1]['cooperative']['caption_set_size'],4)
   b.store.verify('cooperative_'+str(b.p['train_ids'].index(row)).zfill(4))
 def test_G_binding_restored_after_exception(self):
  s=self.b.get('serial_training');before=s._weighted
  with self.assertRaises(RuntimeError):
   with cooperative_g(s,{},SETTINGS,{}):raise RuntimeError('injected')
  self.assertIs(before,s._weighted)
 def test_original_H_single_step_multiedit_counterfactual(self):
  policy=self.bundle.h;policy.requires_grad_(True);phase=self.b.get('phase4_training');pm=self.b.get('policy');con=self.b.get('constraints')
  entries=read_json(Path(self.c['prototype']['history_root'])/'qualification/train.json')['entries']
  # 固定训练公开输入与独立随机种子；不以真实奖励或Test挑选样本。
  found=None
  for entry in entries[:64]:
   if not entry['eligible']:continue
   ex=self.b.get('formal_training.data').example_from(entry)
   for seed in range(1000,1064):
    with torch.no_grad():out=policy.rollout(ex,'sample',torch.Generator().manual_seed(seed))
    if len(out.edits)>=2:found=(ex,seed);break
   if found:break
  self.assertIsNotNone(found);ex,seed=found;runtime_mod=self.b.get('formal_training.runtime');probe=self.bundle.probe
  class Runtime:
   def __init__(self):self.probe=probe
   def assert_frozen(self):pass
   def encode_text(self,texts):
    rows=[]
    for text in texts:
     seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little');rows.append(torch.randn(512,generator=torch.Generator().manual_seed(seed)))
    return torch.nn.functional.normalize(torch.stack(rows),dim=1)
  rt=Runtime();rng=torch.Generator().manual_seed(47);clean=torch.nn.functional.normalize(torch.randn(4,512,generator=rng),dim=1)
  adv=torch.nn.functional.normalize(clean+.02*torch.randn(4,512,generator=rng),dim=1)
  ctx=runtime_mod.TrainGallery(clean,adv,torch.tensor([True,False,False,False]),0,'train',('train',)*4,'train',('train',)*4)
  def reward(k):return SimpleNamespace(runtime=rt,budget=k,evaluator=runtime_mod.RewardEvaluator(ctx,rt.encode_text,1.,1.))
  optimizer=torch.optim.Adam(policy.parameters(),lr=.0001);before=pm.choose;oldloss=phase.scst_loss
  log,features=train_step(phase,pm,con,policy,optimizer,ex,torch.Generator().manual_seed(seed),reward,1,'cpu-test',SETTINGS)
  self.assertGreaterEqual(len(log['sampled']['edits']),2);self.assertEqual(log['optimizer_steps_this_step'],1)
  self.assertEqual(log['IRRA_text_samples'],3+len(log['sampled']['edits']));self.assertIs(pm.choose,before);self.assertIs(phase.scst_loss,oldloss)
  for state in optimizer.state.values():self.assertEqual(int(state['step']),1)
  for i,text in enumerate(log['cooperative']['counterfactual_texts']):
   edits=[con.EditRecord(**{**e,'source_span':tuple(e['source_span'])}) for j,e in enumerate(log['sampled']['edits']) if j!=i]
   self.assertEqual(con.replay(ex,edits,log['K']),text)
  self.assertAlmostEqual(log['SCST_loss'],log['cooperative']['base_SCST_loss']+log['cooperative']['credit_loss'],places=5)
  self.assertFalse(features.requires_grad)
  from tbps.cooperative.audit_math import audit_cooperative
  ledger={'G':[self.g_log],'H':[log],'conditions':[{'row':'cpu-test'}],'g_orders':[['cpu-test']],'h_orders':[['cpu-test']], 'G_before':'start','G_after':'end'}
  audit_cooperative(ledger,SETTINGS,self.b.get('contracts'),con)
  bad=copy.deepcopy(ledger);bad['H'][0]['cooperative']['edit_credits'][0]+=.01
  with self.assertRaises(ArtifactError):audit_cooperative(bad,SETTINGS,self.b.get('contracts'),con)

if __name__=='__main__':unittest.main()
