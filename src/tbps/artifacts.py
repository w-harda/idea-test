"""仅新框架输出的原子不可变资产；旧资产只读；Tensor支持为显式惰性导入。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import os
import re
import uuid
from .contracts import FileRef, sha256

class ArtifactError(ValueError): pass
class PendingPublication(ArtifactError): pass

def json_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")

def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()

def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ArtifactError("duplicate JSON key: "+key)
        result[key]=value
    return result

def read_json(path, schema=None):
    path=Path(path)
    if path.stat().st_size>64*1024*1024:raise ArtifactError("metadata exceeds 64MiB")
    try:
        value=json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_object,
                         parse_constant=lambda s: (_ for _ in ()).throw(ArtifactError("nonfinite JSON: "+s)))
    except (UnicodeError,json.JSONDecodeError) as exc: raise ArtifactError("invalid JSON metadata") from exc
    if type(value) is not dict or (schema is not None and value.get("schema")!=schema):
        raise ArtifactError("metadata/schema mismatch")
    return value

def sha_file(path):
    """不反序列化Checkpoint/Tensor；只读流式SHA，并拒绝读取中改变。"""
    path=Path(path);before=path.stat()
    if not path.is_file():raise ArtifactError("regular file required")
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ArtifactError("source changed while reading")
    return h.hexdigest()

def checked(ref):
    if type(ref) is not FileRef:raise TypeError("FileRef required")
    if sha_file(ref.path)!=ref.sha256:raise ArtifactError("SHA mismatch: "+ref.path)
    stat=Path(ref.path).stat()
    return {**asdict(ref),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}

def _fsync(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _write(path,data):
    with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())

def _name(value):
    if type(value) is not str or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}",value) or value in (".",".."):
        raise ArtifactError("safe single-component artifact ID required")

def _filename(name):
    if type(name) is not str:raise ArtifactError("filename required")
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ("",".","..") for x in name.split("/")) or "\\" in name or ":" in name or not name:
        raise ArtifactError("unsafe relative filename")
    if name in ("manifest.json","completed.json","intent.json"):raise ArtifactError("reserved evidence filename")

class ArtifactStore:
    """new_base必须是新tbps-framework*命名空间；构造/verify不产生文件。"""
    def __init__(self,root,*,new_base,protected_roots=()):
        self.base=Path(new_base).resolve();self.root=Path(root).resolve()
        if not self.base.name.startswith("tbps-framework") or not self.root.is_relative_to(self.base):
            raise ArtifactError("dedicated new tbps-framework output root required")
        protected=(Path("/home/lzf/TBPS/Datasets"),Path("/home/lzf/Attack"),
                   Path("/home/lzf/ldx/projects/idea-TBPS-test1"),
                   Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014"),
                   *map(Path,protected_roots))
        if "prototypes" in self.root.parts or any(self.root.is_relative_to(p.resolve()) or self.base.is_relative_to(p.resolve()) for p in protected):
            raise ArtifactError("historical/source/data/model roots are read-only")
    def _path(self,*parts):
        p=self.root.joinpath(*parts)
        if not p.resolve().is_relative_to(self.root) or self.root.resolve()!=self.root:
            raise ArtifactError("output symlink escapes root")
        return p
    @contextmanager
    def _lock(self):
        # Linux实验端锁，惰性导入；不会因import模块而创建文件或CUDA状态。
        import fcntl
        self._path().mkdir(parents=True,exist_ok=True)
        with self._path(".writer.lock").open("a") as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ArtifactError("another writer owns output") from exc
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def pending(self):
        return tuple(sorted(p.name for p in self.root.glob(".pending-*"))) if self.root.exists() else ()
    def publish(self,artifact_id,files,provenance,*,sources=(),fault=None):
        _name(artifact_id)
        if type(files) is not dict or not files or any(type(v) is not bytes for v in files.values()):
            raise ArtifactError("nonempty bytes payload mapping required")
        for name in files:_filename(name)
        if type(sources) is not tuple or any(type(s) is not FileRef for s in sources):raise TypeError("immutable source refs required")
        if len({s.path for s in sources})!=len(sources):raise ArtifactError("duplicate source")
        # Validate JSON/provenance before acquiring lock or writing.
        if type(provenance) is not dict:raise ArtifactError("provenance object required")
        provenance=json.loads(json_bytes(provenance))
        for s in sources:checked(s)
        with self._lock():
            final=self._path(artifact_id)
            if final.exists():raise ArtifactError("immutable artifact already exists")
            for p in self.pending():
                intent=self._path(p,"intent.json")
                if not intent.exists() or read_json(intent).get("artifact_id")==artifact_id:
                    raise PendingPublication("uncommitted pending preserved; inspect before a new run")
            stage=self._path(".pending-"+uuid.uuid4().hex);stage.mkdir()
            _write(stage/"intent.json",json_bytes({"schema":"tbps-pending-1","artifact_id":artifact_id,"provenance":provenance}))
            locks={}
            for name,data in files.items():
                p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);_write(p,data)
                locks[name]=hashlib.sha256(data).hexdigest()
                if fault:fault("after_file")
            manifest={"schema":"tbps-artifact-1","artifact_id":artifact_id,"provenance":provenance,
                      "files_sha256":locks,"sources":[asdict(s) for s in sources]}
            _write(stage/"manifest.json",json_bytes(manifest))
            if fault:fault("before_receipt")
            _write(stage/"completed.json",json_bytes({"schema":"tbps-artifact-receipt-1",
                                                     "manifest_sha256":sha_file(stage/"manifest.json")}))
            for s in sources:checked(s)
            # intent仅留在未发布目录；完成后manifest/receipt承担相同绑定。
            (stage/"intent.json").unlink()
            for directory in sorted((p for p in stage.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):_fsync(directory)
            _fsync(stage)
            if fault:fault("before_publish")
            os.rename(stage,final);_fsync(self.root)
        return self.verify(artifact_id,expected_provenance=provenance)
    def verify(self,artifact_id,*,expected_provenance=None):
        _name(artifact_id);root=self._path(artifact_id)
        if not root.is_dir():raise ArtifactError("missing committed artifact")
        if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
            raise ArtifactError("symlinks are not immutable artifact payloads")
        done=read_json(root/"completed.json","tbps-artifact-receipt-1")
        sha256(done["manifest_sha256"])
        checked(FileRef(str(root/"manifest.json"),done["manifest_sha256"]))
        manifest=read_json(root/"manifest.json","tbps-artifact-1")
        if manifest.get("artifact_id")!=artifact_id:raise ArtifactError("artifact identity mismatch")
        if expected_provenance is not None and manifest["provenance"]!=json.loads(json_bytes(expected_provenance)):
            raise ArtifactError("provenance conflict")
        if type(manifest["files_sha256"]) is not dict or not manifest["files_sha256"]:raise ArtifactError("empty payload")
        expected=set(manifest["files_sha256"])|{"manifest.json","completed.json"}
        actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        if actual!=expected:raise ArtifactError("missing/unexpected payload file")
        for name,h in manifest["files_sha256"].items():
            _filename(name)
            p=root/name
            if not p.resolve().is_relative_to(root.resolve()):raise ArtifactError("payload escapes artifact")
            checked(FileRef(str(p),h))
        for s in manifest["sources"]:checked(FileRef(**s))
        return manifest
    def publish_tensor(self,artifact_id,tensor,provenance,*,sources=()):
        import torch
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!="cpu":
            raise ArtifactError("Stage 1 accepts CPU tensor only; no implicit GPU transfers")
        compact=tensor.detach().clone()
        stream=io.BytesIO();torch.save(compact,stream)
        return self.publish(artifact_id,{"tensor.pt":stream.getvalue()},provenance,sources=sources)
    def load_tensor(self,artifact_id,*,expected_provenance=None):
        import torch
        manifest=self.verify(artifact_id,expected_provenance=expected_provenance)
        if set(manifest["files_sha256"])!={"tensor.pt"}:raise ArtifactError("not a tensor artifact")
        value=torch.load(self._path(artifact_id,"tensor.pt"),map_location="cpu",weights_only=True)
        if not isinstance(value,torch.Tensor):raise ArtifactError("expected Tensor")
        # 兼容旧格式view：读取后立即clone，返回值不继续保留巨大的底层Storage。
        return value.detach().clone()
