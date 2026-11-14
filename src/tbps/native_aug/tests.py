"""只用合成CPU张量测试数学、预算及调度；不伪装真实攻击效果。"""
import sys,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .common import ROOT,config
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=config();sys.path[:0]=c['native_template']['import_paths']
    def test_profile_256_partner_gradients(self):
        from tbps.attraction.objectives import ProfileKL,Settings
        torch.manual_seed(4);anchors=torch.randn(12,256);profile=ProfileKL(anchors,Settings())
        x=torch.randn(3,256,requires_grad=True);clean=torch.randn(3,256,requires_grad=True);partner=torch.randn(3,256,requires_grad=True)
        v=profile.terms(x,clean,partner);v['loss'].mean().backward()
        self.assertGreater(float(x.grad.norm()),0);self.assertIsNone(clean.grad);self.assertIsNone(partner.grad)
    def test_native_dimension_heterogeneous(self):
        from .losses import attribute_losses
        for d in [7,256,512]:
            f=torch.randn(1,d,requires_grad=True);z=torch.randn(1,d);v=torch.randn(1,d)
            out=attribute_losses(f,z,v);out['total'].backward();self.assertIsNotNone(f.grad)
    def test_shape_mismatch_rejected(self):
        from .losses import attribute_losses
        with self.assertRaises(ValueError):attribute_losses(torch.randn(1,256),torch.randn(1,512),torch.randn(1,256))
    def test_schema_256_and_k0(self):
        from .condition_schema import build,validate
        ident=dict(row_id='cuhk:0:0',image_rel_path='train/a.jpg',caption_index=0,split='train',role='train',pid=1,caption='blue shirt')
        z=torch.nn.functional.normalize(torch.randn(256),dim=0)
        for k in [0,3]:
            s=[3.,2.,1.][:k];w=[v/sum(s) for v in s]
            x=build(ident,[str(i) for i in range(k)],s,w,torch.randn(k,256),z);validate(x)
            self.assertEqual(x['v_local'].shape,(256,))
    def test_native_uncapped_topk(self):
        from .native import select
        rows=[{'S_raw':1.,'instance_id':str(i)} for i in range(7)]
        self.assertEqual(len(select(rows)),7)
        self.assertEqual(select([]),[])
        self.assertEqual(len(select([{'S_raw':1000.},{'S_raw':1.}])),2)
    def test_original_pixel_projection_multiview(self):
        from budget_projection import add_projected_delta
        original=torch.rand(3,3,32,16);current=original
        for _ in range(5):
            current=add_projected_delta(original,current,(8/255)*torch.tanh(torch.randn_like(current))).detach()
            self.assertLessEqual(float((current.double()-original.double()).abs().max()),8/255)
    def test_generic_multiview_optimizer(self):
        from .g_step import train_single_sample
        from .condition_schema import build
        from tbps.attraction.objectives import ProfileKL,Settings
        class G(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.001))
            def forward(self,x):return self.w*torch.ones_like(x)
        class E(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.randn(3,256))
            def forward(self,x):return torch.nn.functional.normalize(x.mean((2,3))@self.w,dim=1)
        generator=G();encoder=E().eval().requires_grad_(False);im=torch.rand(3,3,16,16)
        identity=dict(row_id='cuhk:0:0',image_rel_path='train/a.jpg',caption_index=0,split='train',role='train',pid=1,caption='blue shirt')
        z=torch.nn.functional.normalize(torch.randn(256),dim=0);condition=build(identity,['a'],[1.],[1.],torch.randn(1,256),z)
        opt=torch.optim.Adam(generator.parameters(),lr=.0002)
        with torch.no_grad():clean=encoder(im)
        profile=ProfileKL(torch.randn(16,256),Settings());before=generator.w.detach().clone()
        result=train_single_sample(generator,encoder,opt,im,condition,profile=profile,clean_features=clean)
        self.assertEqual(result['optimizer_steps'],2);self.assertFalse(torch.equal(before,generator.w.detach()))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
    def test_smoke_recipe_uses_valid_positive_view_round(self):
        from tbps.augmented_attraction.views import recipe,ViewSettings,apply_views
        view=recipe(ViewSettings(),'train-smoke',1,16,16)
        image=torch.rand(1,3,16,16)
        actual=apply_views(image,view)
        self.assertEqual(actual.shape,(3,3,16,16));self.assertTrue(torch.equal(actual[:1],image))
    def test_strict_json_duplicate_nonfinite_trailing(self):
        from .audit import strict
        for line in ['{"x":1,"x":2}','{"x":NaN}','{"x":1} trailing']:
            with self.assertRaises(Exception):strict(line)
    def test_guard_no_new_truncation(self):
        from .guard import NativeGuard
        base=SimpleNamespace(legal_change=lambda a,b:True)
        tokenizer=lambda s,**kw:{'input_ids':list(range(len(s.split())))}
        guard=NativeGuard(base,tokenizer,lambda s:s,3)
        self.assertTrue(guard.legal_change('a b','a b c'))
        self.assertFalse(guard.legal_change('a b c','a b c d'))
        self.assertTrue(guard.legal_change('a b c d','a b c d'))
    def test_separate_evaluation_workers(self):
        from . import common
        with tempfile.TemporaryDirectory(prefix='tbps-framework-native-test-',dir='/home/lzf/ldx/tmp') as td:
            count=[]
            class P:
                def __init__(self,*args,**kw):count.append((args,kw));self.pid=100+len(count)
            with patch.object(common,'CONTROL',Path(td)),patch.object(common.subprocess,'Popen',P):
                d=Path(td)/'train_rasa';d.mkdir()
                jobs=common.spawn_eval('rasa')
                self.assertEqual(set(jobs),{'irra','rasa','aptm'});self.assertEqual(len(count),3)
                self.assertTrue(all(kw['start_new_session'] for _,kw in count))
                with self.assertRaises(Exception):common.spawn_eval('rasa')
    def test_zero_free_memory_does_not_gate_gpu_context(self):
        from . import common
        with tempfile.TemporaryDirectory(prefix='tbps-framework-native-test-',dir='/home/lzf/ldx/tmp') as td,patch.object(common,'CONTROL',Path(td)),patch.object(common.subprocess,'check_output',return_value='0, 24564, 24564\n'),patch.object(common.time,'sleep') as sleep,patch.object(torch.cuda,'is_initialized',return_value=False):
            entered=False
            with common.gpu('fixture'):entered=True
            self.assertTrue(entered);sleep.assert_not_called()
            rows=(Path(td)/'fixture/gpu_observations.jsonl').read_text().splitlines()
            self.assertEqual(len(rows),2)
            self.assertTrue(all(__import__('json').loads(x)['fixed_VRAM_floor'] is None for x in rows))
    def test_all_publish_calls_have_required_provenance(self):
        import ast
        for path in Path(__file__).parent.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='publish':
                    self.assertTrue(len(node.args)>=3 or any(x.arg=='provenance' for x in node.keywords),(path.name,node.lineno))
    def test_three_repaired_receipts_real_publish(self):
        from tbps.artifacts import ArtifactStore,json_bytes
        with tempfile.TemporaryDirectory(prefix='tbps-framework-native-test-',dir='/home/lzf/ldx/tmp') as td:
            for name in ('smoke_completed','bank_completed','evaluation_claim'):
                store=ArtifactStore(Path(td)/name,new_base=td)
                store.publish('receipt',{'log.json':json_bytes({'fixture':True})},{'scope':name})
                self.assertEqual(store.verify('receipt')['provenance'],{'scope':name})
    def test_reader_cpu_training_keeps_features_frozen(self):
        from learned_reader import SlotReader,ce_sum
        model=SlotReader([2,3],dim=8)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        hidden=torch.randn(4,5,8);mask=torch.ones(4,5,dtype=torch.bool)
        labels=torch.tensor([[0,1],[1,2],[0,0],[1,1]])
        before=model.queries.detach().clone()
        loss=ce_sum(model(hidden,mask)[2],labels);loss.backward();optimizer.step()
        self.assertIsNone(hidden.grad)
        self.assertFalse(torch.equal(before,model.queries.detach()))
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))
    def test_real_failure_never_retried(self):
        from . import common
        with tempfile.TemporaryDirectory(prefix='tbps-framework-native-test-',dir='/home/lzf/ldx/tmp') as td,patch.object(common,'CONTROL',Path(td)):
            d=Path(td)/'train_rasa';d.mkdir();(d/'FAIL').write_text('real failed')
            with self.assertRaises(Exception):common.wait_file(Path(td)/'missing','eval_rasa_to_irra',source='rasa')
if __name__=='__main__':unittest.main()
