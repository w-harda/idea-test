"""恢复入口CPU测试：中断身份、证据保护、重复执行拒绝和剩余阶段门禁。"""
import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('gh_attribute_response_recovery.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.train = self.base / 'train'
        for p in ('boundaries', 'ledgers', 'round_io/round_0031', 'B/bank_31'):
            (self.train / p).mkdir(parents=True)
        self.latest = {'round':30, 'state_sha':'a'*64}
        self.write('boundaries/latest.json', self.latest)
        self.write('boundaries/claim_0031/claim.json', {'round':31,'parent_state':'a'*64})
        self.write('ledgers/.pending-test/intent.json', {
            'artifact_id':'round_0031','provenance':{'configuration':r.CONFIG_SHA}})
        (self.train / 'round_io/round_0031/data.bin').write_bytes(b'uncommitted bytes')
        (self.train / 'B/bank_31/data.bin').write_bytes(b'bank bytes')
        for name in ('boundaries/round_0030/state.pt','B/bank_30/features.pt'):
            p=self.train/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'committed bytes')
    def write(self, name, value):
        p = self.train / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    def test_exact_interrupted_plan(self):
        latest, paths = r.recovery_plan(self.train)
        self.assertEqual(latest, self.latest); self.assertEqual(len(paths),4)
    def test_no_replay_committed_round(self):
        (self.train/'ledgers/round_0031').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'already published'):r.recovery_plan(self.train)
    def test_wrong_boundary(self):
        self.write('boundaries/latest.json', {'round':29})
        with self.assertRaisesRegex(RuntimeError,'Round30'):r.recovery_plan(self.train)
    def test_failure_not_bypassed(self):
        (self.train/'FAIL_resume').write_text('failure')
        with self.assertRaisesRegex(RuntimeError,'separate diagnosis'):r.recovery_plan(self.train)
    def test_extra_claim_not_removed(self):
        (self.train/'boundaries/claim_0032').mkdir()
        with self.assertRaisesRegex(RuntimeError,'unexpected'):r.recovery_plan(self.train)
    def test_wrong_pending_identity(self):
        self.write('ledgers/.pending-test/intent.json',{'artifact_id':'round_0032','provenance':{'configuration':r.CONFIG_SHA}})
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):r.recovery_plan(self.train)
    def test_wrong_parent(self):
        self.write('boundaries/claim_0031/claim.json',{'round':31,'parent_state':'b'*64})
        with self.assertRaisesRegex(RuntimeError,'parent mismatch'):r.recovery_plan(self.train)
    def test_unknown_pending_boundary_rejected(self):
        (self.train/'boundaries/.pending-unknown').mkdir()
        with self.assertRaisesRegex(RuntimeError,'pending boundary'):r.recovery_plan(self.train)
    def test_archive_preserves_files_and_control(self):
        old = self.base/'old'; old.mkdir(); (old/'started.json').write_text('original identity')
        archive = self.base/'archive'
        latest, paths = r.recovery_plan(self.train)
        before = list(r.file_index(paths))
        with patch.multiple(r, TRAIN=self.train, ARCHIVE=archive, OLD=old):
            r.archive_interruption({'latest':latest,'archive_plan':[str(p) for p in paths]})
            with self.assertRaisesRegex(RuntimeError,'consumed'):r.archive_interruption({})
        self.assertEqual((old/'started.json').read_text(),'original identity')
        self.assertEqual((archive/'original_control/started.json').read_text(),'original identity')
        self.assertEqual(r.read_json(self.train/'boundaries/latest.json'),latest)
        for item in before:
            p = archive/'uncommitted_round31'/Path(item['path']).relative_to(self.train)
            self.assertEqual(r.sha_file(p),item['sha256'])
            self.assertEqual(p.stat().st_mtime_ns,item['mtime_ns'])
        self.assertTrue(all(not p.exists() for p in paths))
        self.assertEqual((archive/'committed_round30/boundaries/round_0030/state.pt').read_bytes(),b'committed bytes')
        self.assertEqual((self.train/'boundaries/round_0030/state.pt').read_bytes(),b'committed bytes')
    def test_changed_boundary_prevents_archive(self):
        latest,paths = r.recovery_plan(self.train)
        self.write('boundaries/latest.json',{'round':30,'state_sha':'b'*64})
        with patch.multiple(r, TRAIN=self.train, ARCHIVE=self.base/'archive'):
            with self.assertRaisesRegex(RuntimeError,'changed'):
                r.archive_interruption({'latest':latest,'archive_plan':[str(p) for p in paths]})
        self.assertFalse((self.base/'archive').exists())
    def test_symbolic_evidence_rejected(self):
        (self.train/'round_io/round_0031/link').symlink_to(self.train/'boundaries/latest.json')
        with self.assertRaisesRegex(RuntimeError,'symlink'):list(r.file_index([self.train/'round_io/round_0031']))
    def test_only_remaining_nodes_and_audit_gates(self):
        stages = list(r.stages())
        self.assertEqual(len(stages),13)
        self.assertEqual([n for n,_,_ in stages if n.startswith('train_to')],['train_to_40','train_to_50','train_to_60'])
        for i,n in enumerate((40,50,60)):
            chunk = stages[i*4:i*4+4]
            self.assertEqual([s[0] for s in chunk],[f'train_to_{n}',f'audit_train_{n}',f'test_{n}',f'audit_test_{n}'])
            self.assertIn('--resume',chunk[0][1])
            self.assertEqual(chunk[0][2],3 if n<60 else 0)
        self.assertNotIn('--stop-after-round',stages[8][1])
    def test_driver_failure_does_not_allocate_or_modify_training(self):
        bad = type('Result',(),{'returncode':9,'stdout':'','stderr':'driver unavailable'})()
        with patch.object(r.subprocess,'run',return_value=bad) as call:
            with self.assertRaisesRegex(RuntimeError,'NVIDIA_DRIVER_UNAVAILABLE'):r.gpu_ready()
        self.assertEqual(call.call_count,1)
        self.assertTrue((self.train/'boundaries/claim_0031').exists())
    def test_no_minimum_vram_gate(self):
        good = type('Result',(),{'returncode':0,'stdout':'RTX 4090, 24563, 1','stderr':''})()
        cuda = type('Result',(),{'returncode':0,'stdout':'allocated','stderr':''})()
        with patch.object(r.subprocess,'run',side_effect=[good,cuda]) as call:
            self.assertEqual(r.gpu_ready()['cuda_allocation'],'allocated')
        self.assertEqual(call.call_count,2)
    def test_duplicate_json_rejected(self):
        (self.train/'boundaries/latest.json').write_text('{"round":30,"round":31}')
        with self.assertRaises(ValueError):r.recovery_plan(self.train)
if __name__ == '__main__':unittest.main()
