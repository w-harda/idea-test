import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tbps.config import *
from tbps.contracts import *
from framework_fixtures import config, H, I

class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-config-");self.addCleanup(self.tmp.cleanup)
        self.data=config(self.tmp.name)
    def rejected(self,fn,exception=ValueError):
        fn(self.data)
        with self.assertRaises(exception):parse_config(self.data)
    def test_dry_run_never_executable(self):
        c=parse_config(self.data);self.assertFalse(c.dry_run()["executable"]);self.assertEqual(c.dry_run()["asset_path_status"],"NOT_READY")
        with self.assertRaises(UnsupportedComponent):c.require_executable()
    def test_config_nested_immutable_copy(self):
        c=parse_config(self.data);self.data["seed"]=1
        d=c.as_dict();d["seed"]=2;self.assertEqual(c.as_dict()["seed"],42)
    def test_required_explicit_fields(self):
        for key in tuple(self.data):
            with self.subTest(key=key):
                d=dict(self.data);d.pop(key)
                with self.assertRaises(ValueError):parse_config(d)
    def test_no_run_or_unknown_config_fields(self):
        self.rejected(lambda d:d.update(action="run"))
    def test_unknown_attack(self):
        self.rejected(lambda d:d.update(attack_method="future_baseline"),UnsupportedComponent)
    def test_unknown_models_not_executable(self):
        for name in ("rasa","aptm"):
            with self.subTest(name=name):
                d=copy.deepcopy(self.data);d["victim_model"]=name
                with self.assertRaises(UnsupportedComponent):parse_config(d)
    def test_unknown_dataset(self):
        for name in ("icfg","rstp"):
            with self.subTest(name=name):
                d=copy.deepcopy(self.data);d["attack_test_dataset"]["id"]=name
                with self.assertRaises(UnsupportedComponent):parse_config(d)
    def test_white_box_checkpoint_axis(self):
        self.rejected(lambda d:d["victim_checkpoint"].update(sha256=H))
    def test_checkpoint_source_mismatch(self):
        self.rejected(lambda d:d["attack_checkpoint"].update(surrogate_checkpoint_sha256=H))
    def test_attack_training_source_mismatch(self):
        self.rejected(lambda d:d["attack_checkpoint"].update(attack_training_dataset="other"))
    def test_model_training_domain_distinct_from_attack_domain(self):
        self.data["victim_checkpoint"]["model_training_dataset"]="upstream separately documented source"
        self.assertEqual(parse_config(self.data).as_dict()["attack_train_dataset"]["id"],"cuhk_pedes")
    def test_test_selection_rejected(self):
        self.rejected(lambda d:d["attack_checkpoint"].update(uses_test_selection=True))
    def test_validation_role_rejected(self):
        self.rejected(lambda d:d["attack_validation_dataset"].update(role="attack_train"))
    def test_test_feedback_rejected(self):
        self.rejected(lambda d:d["permissions"]["attack_time"].append("test_labels"))
    def test_paired_inputs_rejected_in_h(self):
        self.data.update(attack_method="hv1")
        self.data["permissions"]={"train":["train_pairs","train_pids","source_rewards"],"attack_time":["captions","caption_attributes","paired_inputs"]}
        self.data["attack_budget"]={"image":None,"text":{"unit":"unicode_character","max_edits":None,"selection_rule":"H-natural-k-gamma3","constraints_id":"hv1-visual-unicode-tokenizer-mask-v1","per_attribute":1,"per_shared_word":1}}
        with self.assertRaises(ConfigError):parse_config(self.data)
        self.data["permissions"]["attack_time"].remove("paired_inputs");self.assertEqual(parse_config(self.data).attack.id,"hv1")
    def test_h_budget_not_silently_relaxed(self):
        self.test_paired_inputs_rejected_in_h()
        self.rejected(lambda d:d["attack_budget"]["text"].update(per_attribute=2))
    def test_missing_required_permission(self):
        self.rejected(lambda d:d["permissions"]["train"].remove("source_gradients"))
    def test_no_victim_training_feedback(self):
        self.rejected(lambda d:d["permissions"]["train"].append("victim_scores"))
    def test_unknown_permission(self):
        self.rejected(lambda d:d["permissions"]["attack_time"].append("secret_labels"))
    def test_explicit_pixel_space(self):
        self.rejected(lambda d:d["attack_budget"]["image"].update(pixel_space="normalized_clip"))
    def test_modality_budget_mismatch(self):
        self.rejected(lambda d:d["attack_budget"].update(image=None))
    def test_boolean_seed_rejected(self):
        self.rejected(lambda d:d.update(seed=True))
    def test_future_cross_model_contract_permissions(self):
        attack=AttackSpec("fixture","v1",frozenset({"image"}),"iterative",None,True,frozenset(),frozenset({"public_images"}),None)
        model=ModelSpec("another","v1",(200,100),None,1024,"native","native",frozenset({"score"}))
        validate_access("cross_model",attack,IRRA,model,I,H,"trainset","trainset",frozenset(),frozenset({"public_images"}))
        with self.assertRaises(ConfigError):
            validate_access("cross_model",attack,IRRA,model,I,H,"trainset","trainset",frozenset(),frozenset({"public_images","victim_scores"}))
    def test_cross_dataset_axis(self):
        validate_access("cross_dataset",ATTACKS["gv2"],IRRA,IRRA,I,I,"a","b",frozenset({"train_pairs","source_gradients"}),frozenset({"public_images"}))
        with self.assertRaises(ConfigError):
            validate_access("white_box",ATTACKS["gv2"],IRRA,IRRA,I,I,"a","b",frozenset({"train_pairs","source_gradients"}),frozenset({"public_images"}))
    def test_capability_missing_rejected(self):
        m=ModelSpec("mock","v1",None,None,None,"native","none",frozenset({"score"}))
        with self.assertRaises(ConfigError):
            validate_access("white_box",ATTACKS["gv2"],m,m,I,I,"a","a",frozenset({"train_pairs","source_gradients"}),frozenset({"public_images"}))
    def test_import_and_dry_run_no_model_or_filesystem_side_effects(self):
        script="""import builtins, pathlib, sys
original=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in {'torch','prototypes','transformers'}:raise AssertionError('model import '+name)
 return original(name,*a,**k)
builtins.__import__=guard
import tbps.contracts, tbps.config, tbps.artifacts, tbps.compat.legacy_assets
assert 'torch' not in sys.modules
print('SIDE_EFFECT_FREE_IMPORT')
"""
        p=subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True,check=True)
        self.assertIn("SIDE_EFFECT_FREE_IMPORT",p.stdout)
    def test_cli_only_dry_run(self):
        p=Path(self.tmp.name)/"config.json";p.write_text(json.dumps(self.data))
        r=subprocess.run([sys.executable,"-B","-m","tbps.config","--config",str(p)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn("CONFIG_VALID_DRY_RUN",r.stdout)
        r=subprocess.run([sys.executable,"-B","-m","tbps.config","--config",str(p),"--action","run"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)

    def test_attack_time_gradient_capability_required(self):
        a=AttackSpec("fixture","v1",frozenset({"text"}),"iterative",None,True,frozenset(),frozenset({"captions","source_gradients"}),None)
        m=ModelSpec("fixture","v1",None,None,None,"native","none",frozenset({"score"}))
        with self.assertRaises(ConfigError):
            validate_access("white_box",a,m,m,I,I,"a","a",frozenset(),frozenset({"captions","source_gradients"}))
    def test_attack_time_score_capability_required(self):
        a=AttackSpec("fixture","v1",frozenset({"image"}),"iterative",None,True,frozenset(),frozenset({"public_images","victim_scores"}),None)
        m=ModelSpec("fixture","v1",None,None,None,"native","none",frozenset())
        with self.assertRaises(ConfigError):
            validate_access("white_box",a,m,m,I,I,"a","a",frozenset(),frozenset({"public_images","victim_scores"}))

if __name__=="__main__":unittest.main()
