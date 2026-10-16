from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from tbps.artifacts import ArtifactError, sha_file, json_bytes
from tbps.contracts import FileRef
from tbps.models.irra_source import IRRASource, REQUIRED, OFFICIAL_IRRA_SHA, load_official_model
from stage2_fixtures import write

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage2-sources-");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        settings={"dataset_name":"CUHK-PEDES","pretrain_choice":"ViT-B/16","img_size":(384,128),"text_length":77,"stride_size":16,"cmt_depth":4,"vocab_size":49408,"loss_names":"sdm+mlm+id"}
        import yaml
        cfg=write(self.root,"config.yaml",yaml.dump(settings).encode(),True)
        code=tuple((n,write(self.root,n,b"# fixture; not official model implementation\n",True)) for n in REQUIRED)
        dummy=FileRef(str(self.root/"missing.pt"),"1"*64)
        self.source=IRRASource(str(self.root),replace(dummy,sha256=OFFICIAL_IRRA_SHA),cfg,dummy,dummy,code,"CUHK-PEDES official train")
    def test_settings_validate_without_model_import(self):
        self.assertEqual(self.source.settings()["img_size"],(384,128))
    def test_missing_asset_never_reports_verified(self):
        with self.assertRaises(FileNotFoundError):self.source.verify()
    def test_wrong_checkpoint_anchor_rejected(self):
        with self.assertRaises(ArtifactError):replace(self.source,checkpoint=replace(self.source.checkpoint,sha256="2"*64))
    def test_model_training_source_separate_and_required(self):
        with self.assertRaises(ArtifactError):replace(self.source,weight_training_dataset="2048 attack subset")
    def test_incomplete_source_closure_rejected(self):
        with self.assertRaises(ArtifactError):replace(self.source,code=self.source.code[1:])
    def test_source_path_collision_rejected(self):
        key,ref=self.source.code[0]
        with self.assertRaises(ArtifactError):replace(self.source,code=((key,replace(ref,path=str(self.root/"other.py"))),*self.source.code[1:]))
    def test_modified_config_sha_rejected(self):
        Path(self.source.config.path).write_text("text_length: 128\n")
        with self.assertRaises(ArtifactError):self.source.settings()
    def test_changed_architecture_even_repinned_rejected(self):
        import yaml
        d=self.source.settings();d["img_size"]=(224,224)
        p=Path(self.source.config.path);p.write_text(yaml.dump(d))
        s=replace(self.source,config=FileRef(str(p),sha_file(p)))
        with self.assertRaises(ArtifactError):s.settings()
    def test_device_not_implicitly_selected(self):
        with self.assertRaises(ArtifactError):load_official_model(self.source,device="cuda",allow_model_load=True)
    def test_mutable_code_mapping_rejected(self):
        with self.assertRaises(ArtifactError):replace(self.source,code=list(self.source.code))

if __name__=="__main__":unittest.main()
