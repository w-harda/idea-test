"""GH-v2恢复读取器的CPU语义与恶意输入回归。"""
import json,tempfile,unittest
from pathlib import Path
from gh_v2_recovery_stream import scan
from tbps.artifacts import ArtifactError,read_json

class StreamingTests(unittest.TestCase):
    def parse(self,raw,chunk=7):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json';p.write_bytes(raw if isinstance(raw,bytes) else raw.encode())
            got=[];h,c,s=scan(p,lambda k,i,v:got.append((k,i,v)),chunk=chunk)
            return h,c,s,got
    def test_chunk_boundaries(self):
        v={'G':[{'x':'汉字\"\\\n','number':1234.56e-10}]*21,'H':[],'conditions':[{'nested':{'a':[1,2,None,True]}}], 'scalar':-123.4e23}
        raw=json.dumps(v,ensure_ascii=False)
        for n in (1,2,3,7,31,65536):
            h,c,s,got=self.parse(raw,n)
            self.assertEqual(h,{'scalar':v['scalar']});self.assertEqual(c,{'G':21,'H':0,'conditions':1})
            self.assertEqual([x[2] for x in got if x[0]=='G'],v['G'])
            self.assertEqual(s['field_bytes']['G'],len(json.dumps(v['G'],ensure_ascii=False).encode()))
    def test_bad_inputs(self):
        cases=['[]','{"G":{},"H":[]}','{"G":[1]}','{"G":[],"G":[]}',
          '{"G":[{"a":1,"a":2}]}','{"G":[{"a":NaN}]}','{"G":[{"a":Infinity}]}',
          '{"G":[{"a":-Infinity}]}','{"G":[{"a":1e999}]}','{"G":[{"a":1}]} garbage',
          '{"G":[{"a":1},]}','{"G":[],}','{"G":[{"a":1}]','{"G":[{"a":1e}]}',
          b'{"G":[{"a":"\xff"}]}','{"x":truee}', '{"x":01}']
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises((ArtifactError,ValueError)):self.parse(raw)
    def test_large_total_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json'
            with p.open('w') as f:
                f.write('{"G":[')
                item=json.dumps({'nested':['x'*1024]*16})
                for i in range(4200):
                    if i:f.write(',')
                    f.write(item)
                f.write('],"H":[],"conditions":[]}')
            self.assertGreater(p.stat().st_size,64*1024*1024)
            with self.assertRaisesRegex(ArtifactError,'64MiB'):read_json(p)
            h,c,s=scan(p);self.assertEqual(c['G'],4200)
            self.assertLess(s['max_buffer_characters'],150000)
    def test_value_cap(self):
        with self.assertRaisesRegex(ArtifactError,'bounded reader limit'):
            self.parse('{"x":"'+'a'*(4*1024*1024+65537)+'"}',65536)


class RecoveryGateTests(unittest.TestCase):
    def test_no_training_cli(self):
        import ast
        tree=ast.parse(Path(__file__).with_name('gh_v2_recovery.py').read_text())
        constants=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('smoke',constants);self.assertNotIn('run_rounds',constants)
        self.assertNotIn('fresh_bundle',constants)
    def test_audit_gate_requires_exact_complete_totals(self):
        from unittest.mock import patch,MagicMock
        import gh_v2_recovery_v2 as r
        good={'status':'GH_V2_TRAIN_CPU_AUDIT_PASS','covered_rounds':60,'configuration_sha256':'a'*64,'G_updates':484080,'H_updates':122220,'H_visits':122880}
        for key,value in [('status','FAIL'),('covered_rounds',59),('configuration_sha256','b'*64),('G_updates',484079),('H_updates',122219),('H_visits',122879)]:
            v={**good,key:value}
            with patch.object(r,'store',return_value=MagicMock()),patch.object(r,'read_json',return_value=v),patch.object(r,'sha_file',return_value='a'*64):
                with self.assertRaises(ValueError):r.verify_audit()
        with patch.object(r,'store',return_value=MagicMock()),patch.object(r,'read_json',return_value=good),patch.object(r,'sha_file',return_value='a'*64):
            self.assertEqual(r.verify_audit(),good)
    def test_reject_unregistered_endpoints(self):
        import gh_v2_recovery_v2 as r
        for n in (None,1,7,11,59,61):
            with self.assertRaises(ValueError):r.evaluation(n)
    def test_recovery_source_pin_rejects_changes(self):
        from unittest.mock import patch
        import gh_v2_recovery_v2_supervisor as s
        with patch.object(s,'read_json',return_value={'configuration_sha256':'a'*64,'tools':{'x':'b'*64}}),patch.object(s,'sha_file',return_value='a'*64),patch.object(s,'tools',return_value={'x':'c'*64}):
            with self.assertRaisesRegex(ValueError,'source identity'):s.verify_version()
    def test_global_reader_and_producers_remain_frozen(self):
        import gh_v2_recovery_v2 as r
        f,c=r.load(r.CFG,'train')
        self.assertEqual(len(f['producers']),158);self.assertEqual(len(f['runtime_pins']),496)
        self.assertEqual(r.sha_file(r.CFG),'3ac664d8ac563f5ab71936c0863d1537906a4035a268a1b60974b48c361417ce')


class HashDomainTests(unittest.TestCase):
    def test_original_native_and_boundary_domains_bridge(self):
        import types,torch
        import gh_v2_recovery_v2 as r
        from tbps.alternating.native_sources import native_session
        from tbps.alternating.parents import cpu_restore_only
        from tbps.alternating.state import tree_sha
        f,c=r.load(r.CFG,'train');p=r.BASE/'train/initial_state/state.pt'
        with cpu_restore_only([p]),native_session(c) as session:
            h=torch.load(p,map_location='cpu',weights_only=False)['H']
            native=session.import_module('feedback_backend').state_sha
            self.assertEqual(tree_sha(h),'02dfb3a9a45c6da4b7e739cc4f80bee31230926498938a15337448510ea47d6a')
            self.assertEqual(native(types.SimpleNamespace(state_dict=lambda:h)),'9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f')
            h={k:v.clone() for k,v in h.items()};h[next(iter(h))].flatten()[0]+=1
            self.assertNotEqual(native(types.SimpleNamespace(state_dict=lambda:h)),'9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f')

if __name__=='__main__':unittest.main()
