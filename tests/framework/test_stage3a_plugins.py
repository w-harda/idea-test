from dataclasses import replace, asdict
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

from tbps.artifacts import ArtifactError, ArtifactStore, sha_file, read_json, digest, json_bytes
from tbps.contracts import (AttackSpec, AttackBudget, ImageBudget, TextBudget, ModelSpec,
                            PublicTextInput, PublicImageInput, FileRef, AttributeSpan)
from tbps.config import IRRA
from tbps.attacks import default_registry, AttackRegistry
from tbps.attacks.base import PluginInfo, AttackBinding, ImageRequest, TextRequest, AttackPlugin
from tbps.attacks.gv2 import GV2Plugin, BUDGET as G_BUDGET, INFO as G_INFO
from tbps.attacks.hv1 import HV1Plugin, BUDGET as H_BUDGET, INFO as H_INFO
from tbps.attacks.joint import JointPlugin
from tbps.attacks.provenance import AttackProvenance, LegacyAttackSources, IRRA_SHA, G_SHA, H_SHA
from tbps.attacks.configuration import validate_generation
from tbps.attacks.legacy_runtime import legacy_imports, load_legacy_plugins
from tbps.attacks.storage import publish_output
from stage3a_fixtures import (ROOT,H,support,provenance,image_request,text_request,
                              GBackend,HBackend)


class PluginContractTests(unittest.TestCase):
    def test_discovery_has_no_instances(self):
        registry=default_registry()
        self.assertEqual([x.spec.id for x in registry.discover()],["gv2","hv1","joint"])
        self.assertFalse(hasattr(registry,"optimizer"))
    def test_register_new_plugin_without_editing_G_H(self):
        before=(sha_file(ROOT/"src/tbps/attacks/gv2.py"),sha_file(ROOT/"src/tbps/attacks/hv1.py"))
        spec=AttackSpec("new_fixture","v1",frozenset({"text"}),"policy",None,False,
                        frozenset(),frozenset({"captions"}),None)
        info=PluginInfo(spec,frozenset(),frozenset({"captions"}))
        class Fixture:
            def __init__(self):self.info=info;self.budget=H_BUDGET
            def generate(self,request):return request.public.caption
        registry=default_registry();registry.register(info,Fixture)
        self.assertEqual(registry.create("new_fixture").generate(text_request()),"red shirt")
        self.assertIsInstance(registry.create("new_fixture"),AttackPlugin)
        self.assertFalse(hasattr(registry.create("new_fixture"),"train"))
        self.assertEqual(before,(sha_file(ROOT/"src/tbps/attacks/gv2.py"),sha_file(ROOT/"src/tbps/attacks/hv1.py")))
    def test_duplicate_and_unknown_plugin_refused(self):
        registry=default_registry()
        with self.assertRaises(ArtifactError):registry.register(G_INFO,GV2Plugin)
        for name in ("sga","mtga","ap_attack","unknown"):
            with self.subTest(name=name),self.assertRaises(ArtifactError):registry.describe(name)
    def test_factory_capability_lie_rejected(self):
        registry=AttackRegistry()
        registry.register(G_INFO,lambda:type("Bad",(),{"info":H_INFO})())
        with self.assertRaises(ArtifactError):registry.create("gv2")
    def test_no_default_real_backend_or_training(self):
        with self.assertRaises(TypeError):default_registry().create("gv2")
        for info in default_registry().discover():
            self.assertFalse(info.spec.attack_time_optimization)
            self.assertFalse(info.surrogate_for_generation)
            self.assertFalse(info.gradients_for_generation)
            self.assertFalse(info.rewards_for_generation)
            self.assertFalse(info.offline_preparation_executable)
            self.assertFalse(info.training_executable)
            with self.assertRaises(ArtifactError):default_registry().require_preparation(info.spec.id)
    def test_offline_online_permissions_differ(self):
        self.assertIn("source_gradients",G_INFO.offline_preparation_access)
        self.assertEqual(G_INFO.online_access,frozenset({"public_images"}))
        self.assertIn("source_rewards",H_INFO.offline_preparation_access)
        self.assertNotIn("train_pids",H_INFO.online_access)
    def test_bad_descriptor_access_mismatch(self):
        with self.assertRaises(ArtifactError):PluginInfo(G_INFO.spec,frozenset(),frozenset({"captions"}))
        with self.assertRaises(ArtifactError):PluginInfo(G_INFO.spec,frozenset({"unknown"}),G_INFO.online_access)
        with self.assertRaises(ArtifactError):replace(G_INFO,training_executable="true")
    def test_input_binding_namespaced_no_labels(self):
        fields=set(AttackBinding.__dataclass_fields__)
        self.assertFalse(fields & {"pid","paired_index","gallery","reward","S_raw"})
        self.assertNotEqual(text_request(sample_id="q1").binding.id,text_request(sample_id="q2").binding.id)
        with self.assertRaises(TypeError):AttackBinding(H,"q","text",H,pid=1)
    def test_public_caption_rejects_identity_records(self):
        with self.assertRaises(ArtifactError):TextRequest({"caption":"red shirt","pid":1},text_request().binding)
        with self.assertRaises(ArtifactError):replace(text_request(),binding=AttackBinding(H,"q","text","2"*64))
    def test_external_attribute_preselection_refused(self):
        public=PublicTextInput("red shirt",(AttributeSpan("color","red","red",0,3),))
        with self.assertRaises(ArtifactError):TextRequest(public,text_request().binding)
    def test_image_content_hash_and_pixel_space(self):
        with tempfile.TemporaryDirectory() as root:
            q=image_request(root)
            with self.assertRaises(ArtifactError):replace(q,binding=replace(q.binding,input_sha256="2"*64))
            with self.assertRaises(ArtifactError):replace(q,public=PublicImageInput(q.public.content,"normalized"))
    def test_checkpoint_provenance_fixed_constants(self):
        for name,sha in (("gv2",G_SHA),("hv1",H_SHA)):
            p=AttackProvenance(name,FileRef("/readonly/weights",sha),H,H,IRRA_SHA,
                              "cuhk_pedes:attack_train:2048","legacy_unverified",H if name=="hv1" else None)
            self.assertEqual(p.checkpoint.sha256,sha)
            with self.assertRaises(ArtifactError):replace(p,checkpoint=FileRef("/readonly/wrong",H))
            with self.assertRaises(ArtifactError):replace(p,surrogate_sha256=H)
        with self.assertRaises(ArtifactError):replace(provenance("hv1"),execution_kind="legacy_unverified",checkpoint=FileRef("h",H_SHA),surrogate_sha256=IRRA_SHA,selection_sha256=None)
    def test_default_load_gate_before_import_and_source_access(self):
        with patch("tbps.attacks.legacy_runtime.legacy_imports",side_effect=AssertionError("forbidden")):
            with self.assertRaises(ArtifactError):
                with load_legacy_plugins(None):pass
        with self.assertRaises(ArtifactError):
            with legacy_imports(None):pass
    def test_unknown_loading_components_and_device_rejected(self):
        for components,device in ((("unknown",),"cpu"),(("gv2","gv2"),"cpu"),(("gv2",),"cuda")):
            with self.subTest(components=components),self.assertRaises(ArtifactError):
                with load_legacy_plugins(None,components=components,device=device,allow_model_load=True):pass
    def test_import_discovery_has_no_torch_legacy_or_gpu(self):
        code="""import builtins,sys
old=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {'torch','torchvision','prototypes','policy','inference_once','model'}:raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=guarded
from tbps.attacks import default_registry
import tbps.attacks.legacy_runtime, tbps.attacks.provenance, tbps.attacks.storage
assert len(default_registry().discover())==3
assert 'torch' not in sys.modules
print('ATTACK_DISCOVERY_NO_SIDE_EFFECTS')
"""
        p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,check=True)
        self.assertIn("ATTACK_DISCOVERY_NO_SIDE_EFFECTS",p.stdout)


    def test_binding_from_actual_manifest_preserves_caption_identity(self):
        from stage2_fixtures import dataset
        with tempfile.TemporaryDirectory() as root:
            ds,_=dataset(root);manifest=ds.manifest("train")
            q=manifest.queries[1]
            binding=AttackBinding.from_manifest(manifest,q.query_id,"text")
            request=TextRequest(ds.public_text(manifest,q.query_id),binding)
            self.assertEqual(request.public.caption,q.caption)
            with self.assertRaises(ArtifactError):
                TextRequest(PublicTextInput(manifest.queries[0].caption),binding)
            with self.assertRaises(ArtifactError):
                AttackBinding.from_manifest(manifest,"not-member","text")


    def test_old_AP_descriptor_still_not_an_executable_plugin(self):
        from tbps.config import ValidatedConfig, ATTACKS, UnsupportedComponent
        configuration=ValidatedConfig(b"{}",ATTACKS["ap_attack"],IRRA,IRRA,G_BUDGET)
        with self.assertRaisesRegex(UnsupportedComponent,"not implemented"):
            configuration.require_executable()

class GenerationConfigTests(unittest.TestCase):
    def validate(self,**changes):
        args=dict(budget=G_BUDGET,dataset_adapter_id="cuhk_pedes",
                  permissions=G_INFO.online_access,attack_mode="white_box",
                  surrogate=IRRA,surrogate_sha256=IRRA_SHA,victim=IRRA,
                  victim_sha256=IRRA_SHA,victim_adapter_ready=True)
        args.update(changes)
        return validate_generation(default_registry(),"gv2",**args)
    def test_irra_generation_configuration(self):
        report=self.validate()
        self.assertTrue(report["generation_api_available"])
        self.assertFalse(report["evaluation_engine_available"])
        self.assertFalse(report["victim_feedback_to_attack"])
    def test_fusion_target_and_arbitrary_dimension_declared_not_executable(self):
        fusion=ModelSpec("fusion_fixture","v1",(123,87),None,None,"fusion","native",frozenset({"fusion","score"}))
        d=self.validate(victim=fusion,victim_sha256=H,attack_mode="cross_model",victim_adapter_ready=False)
        self.assertEqual(d["victim_scoring"],"fusion");self.assertFalse(d["victim_adapter_ready"])
        dim37=ModelSpec("dim37","v1",None,None,37,"dot","none",frozenset({"score"}))
        self.assertEqual(self.validate(victim=dim37,victim_sha256=H,attack_mode="cross_model",victim_adapter_ready=False)["victim_model_id"],"dim37")
    def test_unimplemented_target_cannot_be_marked_ready(self):
        rasa=replace(IRRA,id="rasa")
        with self.assertRaises(ArtifactError):self.validate(victim=rasa,victim_sha256=H,attack_mode="cross_model")
    def test_missing_or_extra_online_permissions_refused(self):
        for permissions in (frozenset(),G_INFO.online_access|{"victim_scores"},G_INFO.online_access|{"train_pairs"},G_INFO.online_access|{"source_gradients"}):
            with self.subTest(permissions=permissions),self.assertRaises(ArtifactError):self.validate(permissions=permissions)
    def test_unsupported_dataset_or_cross_dataset_refused(self):
        for c in ({"dataset_adapter_id":"icfg"},{"attack_mode":"cross_dataset"},{"surrogate_sha256":H}):
            with self.subTest(change=c),self.assertRaises(ArtifactError):self.validate(**c)
    def test_model_axis_must_match_mode(self):
        with self.assertRaises(ValueError):self.validate(victim_sha256="not-a-sha")
        with self.assertRaises(ArtifactError):self.validate(attack_mode="cross_model")
        with self.assertRaises(ArtifactError):self.validate(victim_sha256=H)
    def test_frozen_budget_cannot_be_changed(self):
        with self.assertRaises(ArtifactError):self.validate(budget=AttackBudget(image=ImageBudget("linf",4/255,"rgb_0_1")))

class LegacyFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=support()
        cls.session=legacy_imports(cls.sources,allow_import=True)
        cls.old=staticmethod(cls.session.__enter__())
        cls.inference=cls.old("inference_once")
        cls.backend_module=cls.old("phase5_joint.backend")
        cls.state_before={str(p): (sha_file(p),p.stat().st_mtime_ns) for p in
                          (ROOT/"prototypes/attribute_image_generator_v2_20261014/inference_once.py",
                           ROOT/"prototypes/attribute_text_policy_h_v1_20261015/phase5_joint/backend.py")}
    @classmethod
    def tearDownClass(cls):
        cls.session.__exit__(None,None,None)
        for name,(sha,mtime) in cls.state_before.items():
            p=Path(name)
            assert sha_file(p)==sha and p.stat().st_mtime_ns==mtime
        assert not torch.cuda.is_initialized()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-plugin-")
        self.addCleanup(self.temp.cleanup)
        self.g_backend=GBackend(self.inference)
        self.g=GV2Plugin(backend=self.g_backend,provenance=provenance("gv2"))
    def h(self,**kwargs):
        b=HBackend(self.old,**kwargs)
        return HV1Plugin(backend=b,provenance=provenance("hv1"))
    def test_G_delegates_exact_original_function(self):
        self.assertIs(self.g_backend.generate_once,self.inference.generate_once)
        r=self.g.generate(image_request(self.temp.name))
        self.assertEqual(r.payload.shape,(1,3,8,8))
        self.assertEqual(r.trace()["G_forward_samples"],1)
        self.assertFalse(r.payload.requires_grad)
        self.assertEqual(r.provenance()["execution_kind"],"cpu_fixture")
    def test_G_pixels_budget_and_clipping_original_space(self):
        r=self.g.generate(image_request(self.temp.name))
        self.assertGreaterEqual(float(r.payload.min()),0)
        self.assertLessEqual(float(r.payload.max()),1)
        self.assertLessEqual(float((r.payload.double()-self.g_backend.pixels.double()).abs().max()),8/255)
        self.assertEqual(float(r.payload.flatten()[-1]),1)
        self.assertTrue(torch.equal(r.effective_delta,r.payload-self.g_backend.pixels))
    def test_G_source_file_tamper_refused(self):
        request=image_request(self.temp.name)
        Path(request.public.content.path).write_bytes(b"changed")
        with self.assertRaises(ArtifactError):self.g.generate(request)
    def test_G_invalid_input_pixels_refused(self):
        for pixels in (torch.ones(1,3,8,8)*2,torch.zeros(1,512),torch.zeros(1,3,8,8).double()):
            self.g_backend.pixels=pixels
            with self.subTest(shape=pixels.shape),self.assertRaises(ArtifactError):
                self.g.generate(image_request(self.temp.name))
    def test_G_feature_vector_is_not_pixel_output(self):
        self.g_backend.generate_once=lambda model,pixels: (model(pixels),{"adversarial":torch.ones(1,512),"effective_delta":torch.ones(1,512)})[1]
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
    def test_G_state_mutation_refused(self):
        self.g_backend.model.mutate=True
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
    def test_G_extra_forward_refused(self):
        def twice(model,pixels):
            model(pixels);return self.inference.generate_once(model,pixels)
        self.g_backend.generate_once=twice
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
    def test_G_no_forward_refused(self):
        self.g_backend.generate_once=lambda model,pixels:{"adversarial":pixels,"effective_delta":torch.zeros_like(pixels)}
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
    def test_G_training_mode_and_grad_parameters_refused(self):
        self.g_backend.model.train()
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
        self.g_backend.model.eval();self.g_backend.model.amplitude.requires_grad_(True)
        with self.assertRaises(ArtifactError):self.g.generate(image_request(self.temp.name))
    def test_H_original_attacker_and_validator_reused(self):
        h=self.h()
        self.assertIs(type(h.backend.attacker),self.backend_module.TextAttacker)
        r=h.generate(text_request())
        self.assertNotEqual(r.payload,"red shirt")
        self.assertEqual(h.backend.seen,["red shirt"])
        self.assertTrue(all(type(s) is str for s in h.backend.frontend.seen))
    def test_H_span_unicode_trace_and_dynamic_budget(self):
        r=self.h().generate(text_request())
        action=r.trace()["action"]
        self.assertEqual(action["budget_k"],2)
        self.assertEqual(len(action["edits"]),1)
        edit=action["edits"][0]
        self.assertEqual(edit["source_span"],[0,3])
        self.assertEqual(edit["original_codepoint"],"U+0065")
        self.assertEqual(edit["replacement_codepoint"],"U+0435")
        self.assertEqual(r.payload,"rеd shirt")
        self.assertTrue(action["tokenizer_audit"]["model_tokens_changed"])
    def test_H_STOP_retained_and_recorded(self):
        r=self.h(mode="stop").generate(text_request())
        action=r.trace()["action"]
        self.assertEqual(r.payload,"red shirt");self.assertEqual(action["edits"],[])
        self.assertEqual(action["budget_k"],2);self.assertEqual(action["H_forward"],1)
        self.assertEqual(action["decisions"][0]["index"],len(action["attributes"]))
    def test_H_position_NOOP_keeps_selected_attribute(self):
        action=self.h(mode="position_noop").generate(text_request()).trace()["action"]
        self.assertEqual(len(action["selected_attribute_ids"]),1)
        self.assertEqual(action["edits"],[])
    def test_H_K0_or_ineligible_does_not_forward(self):
        for caption,kwargs in (("plain",{}),("red shirt",{"no_attrs":True}),("red shirt "*100,{})):
            h=self.h(**kwargs)
            r=h.generate(text_request(caption))
            action=r.trace()["action"]
            self.assertEqual(r.payload,caption);self.assertFalse(action["eligible"])
            self.assertEqual(action["budget_k"],0);self.assertEqual(action["H_forward"],0)
    def test_H_invalid_unicode_replay_refused(self):
        h=self.h();r=h.generate(text_request());record=r.trace()["action"]
        record["edits"][0]["replacement_char"]="!"
        with self.assertRaises(ValueError):h.backend.validate("red shirt",record)
    def test_H_wrong_original_or_codepoint_refused(self):
        h=self.h();record=h.generate(text_request()).trace()["action"]
        for key,value in (("original","blue shirt"),):
            bad=dict(record);bad[key]=value
            with self.assertRaises(ValueError):h.backend.validate("red shirt",bad)
        record["edits"][0]["replacement_codepoint"]="U+0000"
        with self.assertRaises(ValueError):h.backend.validate("red shirt",record)
    def test_H_same_word_and_same_attribute_second_edit_refused(self):
        h=self.h();C=h.backend.contracts;constraints=h.backend.constraints
        attrs=(C.AttributeSpan("color","ace","upper","ace",0,3),
               C.AttributeSpan("garment_type","ace","upper","ace",0,3))
        ex=C.TextExample("ace shirt",attrs)
        e1=constraints.make_edit(ex,0,0,"а");e2=constraints.make_edit(ex,1,1,"с")
        with self.assertRaises(ValueError):constraints.replay(ex,[e1,e2],2)
        with self.assertRaises(ValueError):constraints.replay(ex,[e1,e1],2)
    def test_H_repeated_span_preserved_without_relocation(self):
        h=self.h(spans=(("color","red","upper","red",10,13),))
        caption="red shirt red shirt"
        action=h.generate(text_request(caption)).trace()["action"]
        self.assertEqual(action["edits"][0]["source_span"],[10,13])
        self.assertEqual(action["edits"][0]["position"],11)
    def test_H_state_mutation_and_wrong_request_refused(self):
        h=self.h();h.backend.model.mutate=True
        with self.assertRaises(ArtifactError):h.generate(text_request())
        with self.assertRaises(ArtifactError):self.h().generate({"caption":"red shirt","pid":1})
    def test_H_count_forgery_refused(self):
        h=self.h();original=h.backend.attack
        def wrong(caption):
            record=original(caption);record["H_forward"]=0;return record
        h.backend.attack=wrong
        with self.assertRaises(ArtifactError):h.generate(text_request())
    def test_H_and_G_budget_configuration_rejected(self):
        with self.assertRaises(ArtifactError):HV1Plugin(backend=self.h().backend,provenance=provenance("hv1"),budget=G_BUDGET)
        with self.assertRaises(ArtifactError):GV2Plugin(backend=self.g_backend,provenance=provenance("hv1"))
        with self.assertRaises(ArtifactError):GV2Plugin(backend=self.g_backend,provenance=replace(provenance("gv2"),execution_kind="legacy_unverified",checkpoint=FileRef("g",G_SHA),surrogate_sha256=IRRA_SHA))
    def test_Joint_composes_exact_instances_separate_identities(self):
        h=self.h();j=JointPlugin(image_plugin=self.g,text_plugin=h)
        self.assertIs(j.image_plugin,self.g);self.assertIs(j.text_plugin,h)
        result=j.generate(mode="joint",image=image_request(self.temp.name,"unpaired-gallery"),
                          text=text_request(sample_id="independent-query"))
        self.assertEqual(result.image.binding.sample_id,"unpaired-gallery")
        self.assertEqual(result.text.binding.sample_id,"independent-query")
        self.assertNotEqual(result.image.artifact_id,result.text.artifact_id)
    def test_Joint_single_modal_modes_only_call_needed_component(self):
        h=self.h();j=JointPlugin(image_plugin=self.g,text_plugin=h)
        image=j.generate(mode="image_only",image=image_request(self.temp.name))
        self.assertIsNone(image.text);self.assertEqual(h.backend.seen,[])
        reads=len(self.g_backend.reads)
        text=j.generate(mode="text_only",text=text_request())
        self.assertIsNone(text.image);self.assertEqual(len(self.g_backend.reads),reads)
    def test_Joint_bad_mode_inputs_or_component_sources_refused(self):
        h=self.h();j=JointPlugin(image_plugin=self.g,text_plugin=h)
        with self.assertRaises(ArtifactError):j.generate(mode="joint",text=text_request())
        with self.assertRaises(ArtifactError):j.generate(mode="targeted")
        h.provenance=replace(h.provenance,surrogate_sha256="2"*64)
        with self.assertRaises(ArtifactError):JointPlugin(image_plugin=self.g,text_plugin=h)
    def test_pixel_and_text_artifact_publication_separate(self):
        store=ArtifactStore(Path(self.temp.name)/"attacks",new_base=self.temp.name)
        image=self.g.generate(image_request(self.temp.name))
        text=self.h().generate(text_request())
        a=publish_output(store,image);b=publish_output(store,text)
        self.assertEqual(a.content.path[-9:],"pixels.pt");self.assertTrue(b.content.path.endswith("caption.json"))
        store.verify(a.artifact_id);store.verify(b.artifact_id)
        pixels=torch.load(a.content.path,map_location="cpu",weights_only=True)
        self.assertTrue(torch.equal(pixels["adversarial"],image.payload))
        self.assertEqual(pixels["adversarial"].untyped_storage().nbytes(),pixels["adversarial"].numel()*4)
        self.assertEqual(read_json(b.content.path)["adversarial"],text.payload)
        with self.assertRaises(ArtifactError):publish_output(store,image)
    def test_new_artifact_tamper_detected(self):
        store=ArtifactStore(Path(self.temp.name)/"attacks",new_base=self.temp.name)
        r=self.h().generate(text_request());artifact=publish_output(store,r)
        Path(artifact.content.path).write_text("{}")
        with self.assertRaises(ArtifactError):store.verify(artifact.artifact_id)


    def test_mutated_output_cannot_be_published(self):
        store=ArtifactStore(Path(self.temp.name)/"attacks",new_base=self.temp.name)
        result=self.g.generate(image_request(self.temp.name))
        result=replace(result,payload=result.payload.clone()+.001)
        with self.assertRaisesRegex(ArtifactError,"payload changed"):
            publish_output(store,result)
        text=self.h().generate(text_request())
        with self.assertRaisesRegex(ArtifactError,"payload/action"):
            publish_output(store,replace(text,payload="wrong text"))
    def test_offline_condition_reader_is_never_called_online(self):
        with patch.object(LegacyAttackSources,"condition_record",side_effect=AssertionError("offline access")):
            self.g.generate(image_request(self.temp.name))
            self.h().generate(text_request())


    def test_Joint_source_bundle_mismatch_rejected(self):
        h=self.h();h.provenance=replace(h.provenance,sources_sha256="3"*64)
        with self.assertRaises(ArtifactError):JointPlugin(image_plugin=self.g,text_plugin=h)

class LegacyImportIsolationTests(unittest.TestCase):
    def test_collision_and_nested_session_refused_and_restore(self):
        sources=support();before=list(sys.path)
        with patch.dict(sys.modules,{"contracts":object()}):
            with self.assertRaises(ArtifactError):
                with legacy_imports(sources,allow_import=True):pass
        self.assertEqual(sys.path,before)
        with legacy_imports(sources,allow_import=True):
            with self.assertRaises(ArtifactError):
                with legacy_imports(sources,allow_import=True):pass
        self.assertEqual(sys.path,before)
    def test_exception_restores_imports_and_no_pyc(self):
        sources=support();before=list(sys.path);bytecode=sys.dont_write_bytecode
        with self.assertRaises(RuntimeError):
            with legacy_imports(sources,allow_import=True) as old:
                old("inference_once");raise RuntimeError("fixture")
        self.assertNotIn("inference_once",sys.modules)
        self.assertEqual(sys.path,before);self.assertEqual(sys.dont_write_bytecode,bytecode)
        self.assertFalse(torch.cuda.is_initialized())
    def test_unpinned_import_refused(self):
        sources=support()
        sources.config["runtime_sources"]={k:v for k,v in sources.config["runtime_sources"].items()
                                            if not k.endswith("/constraints.py")}
        with legacy_imports(sources,allow_import=True) as old:
            with self.assertRaisesRegex(ArtifactError,"unpinned"):old("constraints")
    def test_model_and_dataset_import_forbidden(self):
        with legacy_imports(support(),allow_import=True) as old:
            for name in ("model","datasets"):
                with self.subTest(name=name),self.assertRaises(ArtifactError):old(name)
    def test_source_SHA_mismatch_before_import(self):
        sources=support()
        path=str(ROOT/"prototypes/attribute_image_generator_v2_20261014/inference_once.py")
        sources.config["runtime_sources"][path]=H
        with legacy_imports(sources,allow_import=True) as old:
            with self.assertRaisesRegex(ArtifactError,"SHA"):old("inference_once")

if __name__=="__main__":unittest.main()
