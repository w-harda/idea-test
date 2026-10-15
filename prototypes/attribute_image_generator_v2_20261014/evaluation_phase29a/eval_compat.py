"""仅本次Storage修复的显式兼容；旧protocol/metadata/SHA一律不改。"""
import hashlib,json,subprocess
from pathlib import Path
from eval_common import HERE,ROOT,EVAL_OUTPUTS,io,safe

BASE_COMMIT="ab3de99a7209923cba23a5ed9f5fc017f70f3d2f"
LEGACY_DIGEST="1aceae9a0452faea888a2656d318d9084ae0048251fa1cdb084fb47cc8ae721d"
LEGACY_FILE_SHA="13f8287da3324c578198437a9e8c7d4501f6c2619b741db5f9636804c2aa55dc"
MANIFEST=HERE/"storage_fix_compatibility.json"
REPO=ROOT.parents[1]
RECORD="storage_fix_compatibility_record.json"
MODIFIED=("eval_storage.py","eval_protocol.py","evaluate.py")
ADDED=("eval_compat.py","test_storage_fix.py")

def _git(repo,*args):
    return subprocess.check_output(["git",*args],cwd=repo,stderr=subprocess.PIPE)

def verify_release(manifest,repo=REPO,manifest_path=MANIFEST,here=HERE):
    """清单和全部补丁字节必须已经发布在Git HEAD，拒绝任意未提交补丁。"""
    rel=str(manifest_path.relative_to(repo))
    if _git(repo,"show","HEAD:"+rel)!=manifest_path.read_bytes():
        raise ValueError("repair manifest is not the committed release")
    if manifest["baseline_commit"]!=BASE_COMMIT:
        raise ValueError("repair baseline changed")
    subprocess.run(["git","merge-base","--is-ancestor",BASE_COMMIT,"HEAD"],cwd=repo,
                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    prefix=here.relative_to(repo)
    expected_modified={str(prefix/n) for n in MODIFIED}
    expected_added={str(prefix/n) for n in ADDED}
    if set(manifest["modified_sources"])!=expected_modified or set(manifest["added_sources"])!=expected_added:
        raise ValueError("repair source allowlist changed")
    runtime={}
    for rel,entry in manifest["modified_sources"].items():
        original=_git(repo,"show",BASE_COMMIT+":"+rel)
        if hashlib.sha256(original).hexdigest()!=entry["old_sha256"]:
            raise ValueError("repair historical source SHA mismatch")
        runtime[str(repo/rel)]=entry["new_sha256"]
    for rel,sha in manifest["added_sources"].items():
        runtime[str(repo/rel)]=sha
    for path,sha in runtime.items():
        rel=str(Path(path).relative_to(repo))
        if hashlib.sha256(_git(repo,"show","HEAD:"+rel)).hexdigest()!=sha:
            raise ValueError("repair committed source SHA mismatch")
    io.verify(runtime)
    runtime[str(manifest_path)]=io.sha(manifest_path)
    return runtime

def compatibility_sources(protocol,out):
    out=safe(out)
    if out!=EVAL_OUTPUTS/"full" or protocol.get("mode")!="full" or not protocol.get("reuse_clean_reference"):
        raise ValueError("Storage compatibility only permits the original Full output")
    if io.digest(protocol)!=LEGACY_DIGEST or io.sha(out/"protocol.json")!=LEGACY_FILE_SHA:
        raise ValueError("original Full protocol evidence changed")
    manifest=io.load(MANIFEST)
    if (manifest["legacy_protocol_sha256"]!=LEGACY_DIGEST
            or manifest["legacy_protocol_file_sha256"]!=LEGACY_FILE_SHA):
        raise ValueError("repair protocol pin changed")
    delta=verify_release(manifest)
    locks=dict(protocol["source_sha256"])
    for rel,entry in manifest["modified_sources"].items():
        path=str(REPO/rel)
        if locks.get(path)!=entry["old_sha256"]:
            raise ValueError("repair original producer not locked")
    for rel in manifest["added_sources"]:
        if str(REPO/rel) in locks:
            raise ValueError("repair addition collides with original lock")
    locks.update(delta)
    return locks,delta

def verify_sources(protocol,out,allow_compat=False):
    locks=compatibility_sources(protocol,out)[0] if allow_compat else protocol["source_sha256"]
    io.verify(locks)

def accept_legacy_protocol(old,current,out):
    locks,_=compatibility_sources(old,out)
    if {k:v for k,v in old.items() if k!="source_sha256"}!={k:v for k,v in current.items() if k!="source_sha256"}:
        raise ValueError("Storage repair cannot change evaluation contract")
    if current["source_sha256"]!=locks:
        raise ValueError("changes outside the Storage repair allowlist")
    io.verify(locks)
    return old

def prepare_compatibility_record(protocol,out):
    locks,delta=compatibility_sources(protocol,out)
    payload={"schema":"gv2-storage-fix-compatibility-1","baseline_commit":BASE_COMMIT,
             "release_commit":_git(REPO,"rev-parse","HEAD").decode().strip(),
             "legacy_protocol_sha256":LEGACY_DIGEST,"legacy_protocol_file_sha256":LEGACY_FILE_SHA,
             "manifest_sha256":io.sha(MANIFEST),"repair_sources_sha256":delta,
             "unchanged_source_count":len(protocol["source_sha256"])-len(MODIFIED)}
    path=safe(Path(out)/RECORD)
    if path.exists():
        if io.load(path)!=payload:raise ValueError("immutable Storage compatibility record changed")
    else:
        io.save_json(path,payload)
    return {"record_sha256":io.sha(path),"manifest_sha256":payload["manifest_sha256"]}

def verify_cache_producer(out,metadata):
    producer=metadata.get("storage_fix_producer")
    if producer is None:return
    path=safe(Path(out)/RECORD)
    record=io.load(path)
    expected={"record_sha256":io.sha(path),"manifest_sha256":record["manifest_sha256"]}
    if producer!=expected or metadata["protocol_sha256"]!=record["legacy_protocol_sha256"]:
        raise ValueError("Storage repair cache producer evidence mismatch")
