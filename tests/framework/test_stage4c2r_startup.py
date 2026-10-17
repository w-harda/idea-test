"""执行真实child→stage_irra→prepare_inputs流程；仅磁盘CPU Fixture和模型/设备Mock。"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import FileRef,sha_file,checked,json_bytes,ArtifactError
from tbps.runtime_closure import cpu_only
from tbps.engines.historical_loader import feature_only_cpu
from tbps.materialization import smoke
from tbps.materialization.irra_startup import CPUStartupGate,isolation_snapshot

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-startup-");self.root=Path(self.tmp.name)
        self.active=False;self.trace=[];self.loaded=0;self.encoded=0;self.failure=None
        self.baseline=isolation_snapshot();self.reports={}
        self.vec=torch.nn.functional.normalize(torch.arange(1,513,dtype=torch.float32)[None],dim=-1)
        self.features=self.root/"features.pt";torch.save(self.vec.repeat(2,1),self.features)
        self.feature_ref=FileRef(str(self.features),sha_file(self.features))
        self.image_files={}
        for key in smoke.GALLERY:
            p=self.root/(key.replace(":","-")+".pt")
            x=torch.full((1,3,384,128),.5)
            torch.save({"original":x,"adversarial":x+.01,"effective_delta":(x+.01)-x},p)
            self.image_files[key]=FileRef(str(p),sha_file(p))
        self.source_file=self.root/"source";self.source_file.write_bytes(b"fixture official source")
        self.source_ref=FileRef(str(self.source_file),sha_file(self.source_file))
        self.dataset=self.root/"dataset.json";self.dataset.write_bytes(json_bytes({"irra":{"fixture":True}}))
        self.config={"dataset_sources":{"path":str(self.dataset)}}
        test=self
        class Store:
            plan=SimpleNamespace(binding=lambda key,kind:(key,kind))
            def reference(self,binding):return SimpleNamespace(data=test.image_files[binding[0]]) if binding[1]=="image" else binding
            def load(self,ref,validate=None):
                test.assertFalse(test.active);test.trace.append("image_read" if hasattr(ref,"data") else "action_read")
                if hasattr(ref,"data"):
                    checked(ref.data)
                    value=torch.load(ref.data.path,map_location="cpu",weights_only=True)
                    return {k:v.detach().clone() for k,v in value.items()},{}
                if test.failure=="action":raise ArtifactError("fixture bad Unicode action")
                original="red shirt";action={"original":original,"adversarial":"rеd shirt"}
                validate(original,action)
                return action["adversarial"],{"original_caption":original,"adversarial_caption":action["adversarial"]}
            def verify_plan(self,force=False):
                if test.failure=="source":raise ArtifactError("fixture source SHA conflict")
                checked(test.source_ref)
        self.store=Store()
        class Source:
            def verify(self):
                test.assertFalse(test.active);checked(test.source_ref);test.trace.append("source_verified")
        self.source=Source()
        class Backend:
            def encode_image(self,x):
                test.assertTrue(test.active);test.encoded+=len(x);return test.vec.repeat(len(x),1)
            def encode_text(self,x):
                test.assertTrue(test.active);test.encoded+=len(x);return test.vec.repeat(len(x),1)
        class Adapter:
            def __init__(self):
                self._backend=Backend();self.space=SimpleNamespace()
            def inference(self):return self
            def encode_images(self,x):return SimpleNamespace(payload=self._backend.encode_image(x))
            def encode_texts(self,texts):
                return SimpleNamespace(payload=self._backend.encode_text(torch.zeros(len(texts),77,dtype=torch.long))),None
        self.adapter=Adapter()
    def tearDown(self):self.tmp.cleanup()
    @contextmanager
    def validator(self,ref):
        self.assertFalse(self.active)
        # 实际cpu_only在前端构造位置进入/退出；外层实际feature_only_cpu持续保护读取。
        with cpu_only():
            self.assertFalse(self.active)
        def validate(original,action):
            self.assertFalse(self.active)
            if action["original"]!=original:raise ArtifactError("original caption mismatch")
            return action["adversarial"]
        yield validate
    def features_read(self,c):
        self.assertFalse(self.active);checked(self.feature_ref)
        x=torch.load(self.feature_ref.path,map_location="cpu",weights_only=True).detach().clone()
        if self.failure=="feature":x[0,0]=float("nan")
        return {k:x.clone() for k in ("clean_gallery","image_gallery","clean_text","adv_text")},(self.feature_ref,)
    def reset(self):
        self.assertEqual(isolation_snapshot(),self.baseline)
        self.trace.append("cuda_reset");self.active=True
    def model_load(self,*a,**kw):
        self.assertTrue(self.active);self.assertEqual(isolation_snapshot(),self.baseline)
        self.trace.append("mock_model_load");self.loaded+=1;return self.adapter
    def execute(self,failure=None):
        self.failure=failure
        allowed={self.feature_ref.path,*(r.path for r in self.image_files.values())}
        token="fixture-token";config_file=self.root/"config.json";config_file.write_bytes(b"{}")
        control={"token_sha256":smoke.digest(token),"config":{"path":str(config_file),"sha256":sha_file(config_file)},
                 "started_unix":__import__("time").time()}
        gate={"status":"MATERIALIZED_SMOKE_CPU_PREFLIGHT_PASS","GH":{"status":"PASS","runtime_source_count":75},
              "IRRA":{"namespace_restored":True}}
        root=self.root/"attempt"
        real_json=smoke.read_json
        def read(path,*a,**kw):
            p=str(path)
            if p.endswith("/attempt/control.json"):return control
            if p.endswith("/attempt/preflight.json"):return gate
            if p.endswith("/stage-text/report.json"):return {"status":"PASS"}
            return real_json(path,*a,**kw)
        class Records:
            def verify(_,name):return {}
            def publish(_,name,files,*a,**kw):
                self.reports[name]={k:json.loads(v) for k,v in files.items() if k.endswith(".json")}
        records=Records()
        original_to=torch.Tensor.to
        def cpu_device(x,*a,**kw):
            if a and a[0]=="cuda:0":
                self.assertTrue(self.active);a=("cpu",*a[1:])
            return original_to(x,*a,**kw)
        with patch.object(smoke,"OUTPUT",str(root)),patch.object(smoke,"record_store",return_value=records), \
             patch.object(smoke,"read_json",side_effect=read), \
             patch.object(smoke,"sha_file",side_effect=lambda p:sha_file(p) if Path(p).exists() else "a"*64), \
             patch.object(smoke,"config",return_value=self.config), \
             patch.object(smoke,"materialized",return_value=(self.store,SimpleNamespace(config_ref=self.source_ref))), \
             patch.object(smoke,"selected_features",side_effect=self.features_read), \
             patch("tbps.materialization.irra_startup.feature_paths",return_value=allowed), \
             patch("tbps.engines.historical_loader.native_action_validator",side_effect=self.validator), \
             patch("tbps.models.irra_source.IRRASource.from_dict",return_value=self.source), \
             patch("tbps.models.irra.IRRAAdapter.from_source",side_effect=self.model_load), \
             patch("tbps.smoke_irra.model_snapshot",return_value={"frozen":True}), \
             patch("tbps.materialization.smoke.asdict",side_effect=lambda v:{} if v is self.adapter.space else __import__("dataclasses").asdict(v)), \
             patch("tbps.smoke_irra.gpu_available",return_value={"fixture":True}), \
             patch("torch.cuda.is_initialized",side_effect=lambda:self.active), \
             patch("torch.cuda.reset_peak_memory_stats",side_effect=self.reset), \
             patch("torch.cuda.max_memory_allocated",return_value=0),patch("torch.cuda.max_memory_reserved",return_value=0), \
             patch.object(torch.Tensor,"to",new=cpu_device), \
             patch.dict(__import__("os").environ,{"CUDA_VISIBLE_DEVICES":"0"}):
            # gate的baseline在实际prepare_inputs入口捕获，包含本CPU测试设备Mock。
            self.baseline=isolation_snapshot()
            code=smoke.child("irra",str(config_file),str(root),token)
        return code,self.reports["stage-irra"]["report.json"]
    def test_real_child_correct_order_reaches_mock_load_and_encodes(self):
        code,report=self.execute()
        self.assertEqual(code,0);self.assertEqual(self.loaded,1);self.assertEqual(self.encoded,8)
        self.assertEqual(report["startup_events"],["cpu_read_started","cpu_sources_verified","cpu_isolation_exited",
                                                  "cuda_initialization_call","irra_model_load_entry"])
        self.assertLess(self.trace.index("source_verified"),self.trace.index("cuda_reset"))
        self.assertLess(self.trace.index("cuda_reset"),self.trace.index("mock_model_load"))
        self.assertEqual(report["forwards"]["completed"],{"G":0,"H":0,"IRRA_image":4,"IRRA_text":4})
        self.assertEqual(set(report["scores"]),{"clean","image_only","text_only","joint"})
        print("CPU_STARTUP_TRACE "+json.dumps({"events":report["startup_events"],"mock_loads":self.loaded,
            "mock_encoded_samples":self.encoded,"real_CUDA":False},sort_keys=True))
    def test_source_failure_stops_before_cuda_and_load(self):
        code,report=self.execute("source");self.assertEqual(code,1);self.assertEqual(self.loaded,0)
        self.assertFalse(self.active);self.assertNotIn("cuda_reset",self.trace)
    def test_action_failure_stops_before_cuda_and_load(self):
        code,_=self.execute("action");self.assertEqual(code,1);self.assertEqual(self.loaded,0);self.assertFalse(self.active)
    def test_feature_failure_stops_before_cuda_and_load(self):
        code,_=self.execute("feature");self.assertEqual(code,1);self.assertEqual(self.loaded,0);self.assertFalse(self.active)
    def test_early_cuda_is_rejected_inside_actual_isolation(self):
        with patch("torch.cuda.reset_peak_memory_stats") as cuda:
            gate=CPUStartupGate()
            with feature_only_cpu(set()):
                with self.assertRaisesRegex(ArtifactError,"contexts must exit"):
                    gate.initialize_cuda()
            cuda.assert_not_called()
    def test_gate_requires_actual_function_restoration(self):
        gate=CPUStartupGate()
        with feature_only_cpu(set()):
            gate.checked()
            with self.assertRaisesRegex(ArtifactError,"not exited/restored"):gate.exit_checked()
        gate.exit_checked()
    def test_cuda_initialized_during_cpu_read_is_refused(self):
        gate=CPUStartupGate()
        with patch("torch.cuda.is_initialized",return_value=True):
            with self.assertRaises(ArtifactError):gate.checked()
    def test_actual_cpu_gate_still_forbids_model_load_and_neural_execution(self):
        with self.assertRaises(ArtifactError),cpu_only():
            torch.load("not-a-model")
        # cpu_only报告尝试后也拒绝整个上下文；上一行应由外层断言捕获。
