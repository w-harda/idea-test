"""继承原不可变写入协议；只读SHA计算按inode/ctime/mtime/size失效，不删减校验。"""
from pathlib import Path
from dataclasses import asdict
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,read_json,json_bytes,sha_file,_name,_filename
from tbps.contracts import sha256
class DigestSession:
 def __init__(self):self.hashes={};self.hits=0;self.misses=0
 def checked(self,ref):
  if type(ref)is not FileRef:raise TypeError("FileRef required")
  p=Path(ref.path);s=p.stat()
  signature=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  key=(str(p.resolve()),signature)
  if key not in self.hashes:self.hashes[key]=sha_file(p);self.misses+=1
  else:self.hits+=1
  if self.hashes[key]!=ref.sha256:raise ArtifactError("SHA mismatch: "+ref.path)
  after=p.stat()
  if signature!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ArtifactError("source changed while verifying")
  return {**asdict(ref),"size":after.st_size,"mtime_ns":after.st_mtime_ns}
class VerifiedArtifactStore(ArtifactStore):
 def __init__(self,*a,session=None,**kw):
  super().__init__(*a,**kw);self.digest_session=session or DigestSession()
 def verify(self,artifact_id,*,expected_provenance=None):
  # Same checks/order as ArtifactStore.verify; only digest provider has stat-validated memoization.
  _name(artifact_id);root=self._path(artifact_id)
  if not root.is_dir():raise ArtifactError("missing committed artifact")
  if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):raise ArtifactError("symlinks are not immutable artifact payloads")
  done=read_json(root/"completed.json","tbps-artifact-receipt-1");sha256(done["manifest_sha256"])
  checked=self.digest_session.checked
  checked(FileRef(str(root/"manifest.json"),done["manifest_sha256"]))
  manifest=read_json(root/"manifest.json","tbps-artifact-1")
  if manifest.get("artifact_id")!=artifact_id:raise ArtifactError("artifact identity mismatch")
  if expected_provenance is not None and manifest["provenance"]!=__import__("json").loads(json_bytes(expected_provenance)):raise ArtifactError("provenance conflict")
  if type(manifest["files_sha256"])is not dict or not manifest["files_sha256"]:raise ArtifactError("empty payload")
  expected=set(manifest["files_sha256"])|{"manifest.json","completed.json"}
  actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
  if actual!=expected:raise ArtifactError("missing/unexpected payload file")
  for name,h in manifest["files_sha256"].items():
   _filename(name);p=root/name
   if not p.resolve().is_relative_to(root.resolve()):raise ArtifactError("payload escapes artifact")
   checked(FileRef(str(p),h))
  for source in manifest["sources"]:checked(FileRef(**source))
  return manifest
