"""仅小规模CPU合成缓存；不启动真实模型、Full/100或GPU。"""
import copy,gc,signal,weakref
from unittest.mock import patch
import torch
import eval_compat as compat
import eval_storage as storage
import evaluate as runner
import eval_metrics as metrics
from eval_common import io,HERE

from test_evaluation import Base,fake_protocol,reference,FakeBackend
import unittest

def stats(family,n):
    return {"text_samples":n if family=="text" else 0,"G_samples":0,
            "IRRA_image_samples":0 if family=="text" else n,
            **({} if family=="text" else {"linfs":[0.]*n})}

def legacy(out,family,index,p,rows,feature):
    target=storage.unit_path(out,family,index);target.mkdir(parents=True)
    torch.save({"features":feature,"stats":stats(family,len(rows))},target/"features.pt")
    meta={"family":family,"rows":rows,"protocol_sha256":io.digest(p),
          "sha256":io.sha(target/"features.pt"),"stats":stats(family,len(rows))}
    io.save_json(target/"metadata.json",meta)
    return target

class CompactStorageTests(Base):
    def test_historical_storage_reproduction_and_exact_compact_load(self):
        p=fake_protocol();base=reference()[0];view=base[2:4]
        target=legacy(self.out,"clean",0,p,p["gallery"][:2],view)
        old=torch.load(target/"features.pt",weights_only=True)
        self.assertEqual(old["features"].untyped_storage().nbytes(),base.numel()*4)
        before={str(x):io.sha(x) for x in target.iterdir()}
        value=storage.load_unit(self.out,"clean",0,p,p["gallery"][:2])
        self.assertTrue(torch.equal(value["features"],view))
        self.assertEqual(value["features"].untyped_storage().nbytes(),view.numel()*4)
        io.verify(before)

    def test_publish_view_has_independent_storage(self):
        p=fake_protocol();base=reference()[0];view=base[2:4]
        storage.publish(self.out,"clean",0,p,p["gallery"][:2],view,stats("clean",2))
        raw=torch.load(storage.unit_path(self.out,"clean",0)/"features.pt",weights_only=True)
        self.assertEqual(raw["features"].untyped_storage().nbytes(),view.numel()*4)
        self.assertTrue(torch.equal(raw["features"],view))
        base.zero_()
        self.assertFalse(torch.equal(raw["features"],view))

    def test_contiguous_alone_does_not_fix_cpu_row_view(self):
        view=reference()[0][2:4]
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.contiguous().untyped_storage().data_ptr(),view.untyped_storage().data_ptr())
        self.assertEqual(view.clone().untyped_storage().nbytes(),view.numel()*4)

    def test_original_loaded_tensor_released_before_return(self):
        p=fake_protocol();legacy(self.out,"clean",0,p,p["gallery"][:2],reference()[0][:2])
        original_load=storage.torch.load;refs=[]
        def load(*a,**kw):
            value=original_load(*a,**kw);refs.append(weakref.ref(value["features"]));return value
        with patch.object(storage.torch,"load",side_effect=load):
            result=storage.load_unit(self.out,"clean",0,p,p["gallery"][:2])
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertEqual(result["features"].untyped_storage().nbytes(),2*512*4)

    def test_mixed_formats_gather_exact_and_all_storages_compact(self):
        p=fake_protocol();base=reference()[0]
        for index,rows in storage.shards(p["gallery"],2):
            if index%2:storage.publish(self.out,"clean",index,p,rows,base[index*2:index*2+2],stats("clean",2))
            else:legacy(self.out,"clean",index,p,rows,base[index*2:index*2+2])
        result,_,_=storage.gather(self.out,"clean",p,p["gallery"],2)
        self.assertTrue(torch.equal(result,base))
        for index,rows in storage.shards(p["gallery"],2):
            v=storage.load_unit(self.out,"clean",index,p,rows)
            self.assertEqual(v["features"].untyped_storage().nbytes(),2*512*4)

    def test_mixed_cache_official_metrics_exact(self):
        p=fake_protocol();base=reference()[0];ids=[r["pid"] for r in p["gallery"]]
        for index,rows in storage.shards(p["gallery"],2):
            if index==0:legacy(self.out,"clean",index,p,rows,base[:2])
            else:storage.publish(self.out,"clean",index,p,rows,base[index*2:index*2+2],stats("clean",2))
        loaded,_,_=storage.gather(self.out,"clean",p,p["gallery"],2)
        self.assertEqual(metrics.metrics(base,base,ids,ids)[0],metrics.metrics(base,loaded,ids,ids)[0])

    def test_legacy_sha_mismatch_still_rejected(self):
        p=fake_protocol();t=legacy(self.out,"clean",0,p,p["gallery"][:2],reference()[0][:2])
        meta=io.load(t/"metadata.json");meta["sha256"]="0"*64;io.save_json(t/"metadata.json",meta)
        with self.assertRaises(ValueError):storage.load_unit(self.out,"clean",0,p,p["gallery"][:2])

    def test_read_stops_before_loading(self):
        def stop():raise runner.SafeStop("requested")
        with patch.object(storage.torch,"load",side_effect=AssertionError("must not load")):
            with self.assertRaises(runner.SafeStop):storage.load_unit(self.out,"text",0,{},[],check_stop=stop)

    def test_read_stops_after_one_small_load(self):
        p=fake_protocol();legacy(self.out,"clean",0,p,p["gallery"][:2],reference()[0][:2]);calls=[]
        def check():
            calls.append(1)
            if len(calls)==2:raise runner.SafeStop("requested")
        with self.assertRaises(runner.SafeStop):
            storage.load_unit(self.out,"clean",0,p,p["gallery"][:2],check_stop=check)
        self.assertEqual(len(calls),2)

    def test_gather_checks_stop_between_units(self):
        p=fake_protocol()
        for i,rows in storage.shards(p["gallery"],2):legacy(self.out,"clean",i,p,rows,reference()[0][i*2:i*2+2])
        loads=[];load=storage.load_unit
        def tracked(*a,**kw):loads.append(a[2]);return load(*a,**kw)
        def check():
            if len(loads)>=2:raise runner.SafeStop("requested")
        with patch.object(storage,"load_unit",side_effect=tracked):
            with self.assertRaises(runner.SafeStop):storage.gather(self.out,"clean",p,p["gallery"],2,check_stop=check)
        self.assertEqual(loads,[0,1])

class CancellationTests(Base):
    def test_real_SIGTERM_sets_flag_and_check_raises(self):
        with runner.SignalBoundary() as stop:
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(stop.requested)
            with self.assertRaises(runner.SafeStop):stop.check()

    def test_stop_in_clean_gather_then_resume_without_repeating_cache(self):
        p=fake_protocol();p["reuse_clean_reference"]=True;stop=runner.SignalBoundary()
        gather=runner.gather
        def stopped(*a,**kw):
            stop.request();return gather(*a,**kw)
        with patch.object(runner,"gather",side_effect=stopped):
            with self.assertRaises(runner.SafeStop):self.run_fake(p,stop=stop)
        before={str(x):io.sha(x) for x in (self.out/"features").rglob("*") if x.is_file()}
        self.assertEqual(FakeBackend.totals,[])
        self.run_fake(p);io.verify(before)
        self.assertEqual(len(FakeBackend.totals),30)

    def test_stop_after_clean_metrics_before_any_G(self):
        stop=runner.SignalBoundary();original=runner.metrics
        def interrupted(*a,**kw):
            value=original(*a,**kw);stop.request();return value
        with patch.object(runner,"metrics",side_effect=interrupted):
            with self.assertRaises(runner.SafeStop):self.run_fake(stop=stop)
        self.assertFalse(any(f.startswith("epoch") for f,r in FakeBackend.totals))

    def test_summary_gather_is_cancellable(self):
        p=fake_protocol();self.run_fake(p)
        hashes={str(x):io.sha(x) for x in self.out.rglob("*") if x.is_file()}
        def stop():raise runner.SafeStop("requested")
        with self.assertRaises(runner.SafeStop):
            runner.evaluate_cached(p,self.out,reference(),write=False,check_stop=stop)
        io.verify(hashes)

class CompatibilityTests(Base):
    def setUp(self):
        super().setUp()
        self.old={"mode":"full","reuse_clean_reference":True,"source_sha256":{"old":"a"},"checkpoint_sha256":{"1":"fixed"}}
        self.current=copy.deepcopy(self.old);self.current["source_sha256"]={"old":"b","added":"c"}
    def test_no_implicit_source_bypass(self):
        source=self.out/"unchanged_source";source.write_text("original")
        protocol={**self.old,"source_sha256":{str(source):"0"*64}}
        with self.assertRaises(ValueError):compat.verify_sources(protocol,self.out)
    def test_only_exact_non_source_contract_accepted(self):
        with patch.object(compat,"compatibility_sources",return_value=(self.current["source_sha256"],{})),patch.object(compat.io,"verify"):
            result=compat.accept_legacy_protocol(self.old,self.current,self.out)
        self.assertIs(result,self.old)
    def test_checkpoint_or_batch_change_rejected(self):
        for change in ({"checkpoint_sha256":{"1":"changed"}},{"image_batch":2},{"reuse_clean_reference":False}):
            current={**self.current,**change}
            with patch.object(compat,"compatibility_sources",return_value=(self.current["source_sha256"],{})):
                with self.assertRaises(ValueError):compat.accept_legacy_protocol(self.old,current,self.out)
    def test_unlisted_new_source_rejected(self):
        current=copy.deepcopy(self.current);current["source_sha256"]["rogue"]="d"
        with patch.object(compat,"compatibility_sources",return_value=(self.current["source_sha256"],{})):
            with self.assertRaises(ValueError):compat.accept_legacy_protocol(self.old,current,self.out)
    def test_all_other_sources_still_verified(self):
        with patch.object(compat,"compatibility_sources",return_value=({"immutable":"original"},{})),patch.object(compat.io,"verify",side_effect=ValueError("SHA")) as verify:
            with self.assertRaises(ValueError):compat.verify_sources(self.old,self.out,True)
        verify.assert_called_once_with({"immutable":"original"})
    def test_wrong_scope_rejected(self):
        with self.assertRaises(ValueError):compat.compatibility_sources(self.old,self.out)
    def test_original_protocol_change_rejected(self):
        with patch.object(compat,"EVAL_OUTPUTS",self.out.parent):
            # 正确目录名但协议摘要不在这次授权范围。
            with self.assertRaises(ValueError):compat.compatibility_sources(self.old,self.out.parent/"full")
    def test_cli_opt_in_default_false(self):
        self.assertFalse(runner.parser().parse_args([]).allow_storage_fix_compat)
        self.assertTrue(runner.parser().parse_args(["--allow-storage-fix-compat"]).allow_storage_fix_compat)
    def test_uncommitted_release_manifest_rejected(self):
        with patch.object(compat,"_git",return_value=b"uncommitted content"):
            with self.assertRaises(ValueError):compat.verify_release(io.load(compat.MANIFEST))

    def test_record_atomic_idempotent_and_old_evidence_unchanged(self):
        path=self.out/"protocol.json";io.save_json(path,self.old);before=io.sha(path)
        with patch.object(compat,"compatibility_sources",return_value=({},{"repair":"fixed"})),patch.object(compat,"_git",return_value=b"release\n"):
            a=compat.prepare_compatibility_record(self.old,self.out)
            b=compat.prepare_compatibility_record(self.old,self.out)
        self.assertEqual(a,b);self.assertEqual(io.sha(path),before)
        record=io.load(self.out/compat.RECORD)
        self.assertEqual(record["legacy_protocol_sha256"],compat.LEGACY_DIGEST)

    def test_record_cannot_be_overwritten(self):
        io.save_json(self.out/compat.RECORD,{"tampered":True})
        with patch.object(compat,"compatibility_sources",return_value=({},{})),patch.object(compat,"_git",return_value=b"release\n"):
            with self.assertRaises(ValueError):compat.prepare_compatibility_record(self.old,self.out)

    def test_new_producer_metadata_and_corrupt_record(self):
        p=fake_protocol();record={"manifest_sha256":"m","legacy_protocol_sha256":io.digest(p)}
        io.save_json(self.out/compat.RECORD,record)
        producer={"record_sha256":io.sha(self.out/compat.RECORD),"manifest_sha256":"m"}
        storage.publish(self.out,"clean",0,p,p["gallery"][:2],reference()[0][:2],stats("clean",2),producer=producer)
        self.assertEqual(io.load(storage.unit_path(self.out,"clean",0)/"metadata.json")["storage_fix_producer"],producer)
        io.save_json(self.out/compat.RECORD,{**record,"manifest_sha256":"changed"})
        with self.assertRaises(ValueError):storage.load_unit(self.out,"clean",0,p,p["gallery"][:2])

    def test_old_metadata_needs_no_new_sidecar(self):
        compat.verify_cache_producer(self.out,{"protocol_sha256":"old"})
    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
