"""仅新全量物化会话的来源验证缓存与成员索引；原Store/Schema不变。"""
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from unittest.mock import patch
import time
from ..artifacts import FileRef,checked,sha_file,ArtifactError,digest
from ..attacks.base import AttackBinding,ImageRequest,TextRequest
from ..contracts import PublicImageInput,PublicTextInput
from .store import MaterializedStore,utf8_sha
from .contracts import need

_LOCK=Lock()

def stamp(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

class VerifiedSources:
    """固定期望SHA；直接使用来源每次stat，完整来源stat≤1秒、首尾完整SHA。"""
    def __init__(self,refs,*,clock=time.monotonic,interval=1.0):
        need(0<interval<=1,"bounded full source stat interval")
        self.refs={r.path:r for r in refs}
        need(len(self.refs)==len(refs),"unique immutable session refs")
        self.clock,self.interval=clock,interval;self.last=0;self.states={};self.stamps={}
        self.stats={"source_SHA_initial":0,"source_SHA_final":0,"cached_checked":0,"full_stat_sweeps":0}
    def begin(self):
        for path,ref in self.refs.items():
            self.states[path]=checked(ref);self.stamps[path]=stamp(path);self.stats["source_SHA_initial"]+=1
        self.last=self.clock()
    def check(self,ref):
        if ref.path not in self.refs:return checked(ref)
        need(ref==self.refs[ref.path],"pinned source SHA conflict")
        need(stamp(ref.path)==self.stamps[ref.path],"source inode/size/mtime/ctime changed")
        self.stats["cached_checked"]+=1
        return dict(self.states[ref.path])
    def sweep(self,*,force=False):
        if force or self.clock()-self.last>=self.interval:
            for path in self.refs:need(stamp(path)==self.stamps[path],"session source changed")
            self.last=self.clock();self.stats["full_stat_sweeps"]+=1
    def finish(self):
        for path,ref in self.refs.items():
            need(checked(ref)==self.states[path] and stamp(path)==self.stamps[path],"source changed at final SHA boundary")
            self.stats["source_SHA_final"]+=1

class StreamingMaterializedStore(MaterializedStore):
    """相同Plan和单元格式；仅以原始Manifest构建索引，不改变顺序/成员。"""
    def __init__(self,store,plan):
        super().__init__(store,plan)
        self.gallery={g.image_id:g for g in plan.manifest.gallery}
        self.queries={q.query_id:q for q in plan.manifest.queries}
        self.bindings={}
        for kind,rows in (("image",self.gallery),("text",self.queries)):
            for key,row in rows.items():
                sha=row.image_sha256 if kind=="image" else utf8_sha(row.caption)
                self.bindings[kind,key]=AttackBinding(plan.manifest_sha256,key,kind,sha)
        self.session=None
    def unit_id(self,binding):
        need(binding==self.bindings.get((binding.modality,binding.sample_id)),"not an indexed materialization member")
        return digest({"plan":self.plan.id,"input":binding.id,"modality":binding.modality})
    def image_request(self,image_id):
        need(image_id in self.gallery,"unknown Gallery member")
        row=self.gallery[image_id]
        path=(Path(self.plan.image_root)/row.image_rel_path).resolve()
        need(path.is_relative_to(Path(self.plan.image_root).resolve()),"image escapes root")
        return ImageRequest(PublicImageInput(FileRef(str(path),row.image_sha256),"rgb_0_1"),self.bindings["image",image_id])
    def text_request(self,query_id):
        need(query_id in self.queries,"unknown Query member")
        return TextRequest(PublicTextInput(self.queries[query_id].caption),self.bindings["text",query_id])
    def verify_plan(self,*,force=False):
        if self.session is None:return super().verify_plan(force=force)
        if force:
            self.session.finish()
            return super().verify_plan(force=True)
        self.session.sweep()
        self.session.check(self._plan_ref)
        for path in self._protocol_files:self.session.check(self.session.refs[path])
        return self._plan_ref

class BulkSession:
    """作用域内只缓存明确来源FileRef；输出文件/未知FileRef仍完整SHA。异常也恢复模块。"""
    def __init__(self,store):
        need(type(store)is StreamingMaterializedStore,"explicit streaming Store required")
        self.store=store;self.cache=None;self.stack=None
    def __enter__(self):
        need(_LOCK.acquire(blocking=False),"bulk source session is not concurrent/reentrant")
        try:
            self.store.verify_plan(force=True)
            base=self.store.store.root/"protocol"
            manifest=self.store.store.verify("protocol")
            refs={r.path:r for r in self.store.plan.sources}
            refs.update({r["path"]:FileRef(**r) for r in manifest["sources"]})
            for p in base.iterdir():
                if p.is_file():refs[str(p)]=FileRef(str(p),sha_file(p))
            self.store._protocol_files=tuple(str(p) for p in base.iterdir() if p.is_file())
            self.cache=VerifiedSources(tuple(refs.values()));self.cache.begin()
            self.stack=ExitStack()
            import tbps.artifacts as a
            import tbps.materialization.store as s
            import tbps.materialization.legacy_text as t
            # 未修改全局实现；单进程受限会话结束后恢复，直接旧调用仍使用原策略。
            for module in (a,s,t):self.stack.enter_context(patch.object(module,"checked",new=self.cache.check))
            self.store.session=self.cache
            return self
        except BaseException:
            if self.stack:self.stack.close()
            _LOCK.release();raise
    def __exit__(self,*exc):
        try:
            self.cache.finish()
        finally:
            self.store.session=None
            self.stack.close();_LOCK.release()
