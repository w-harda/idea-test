"""官方代理utils父模块与冻结H的utils.simple_tokenizer子模块显式共存。"""
from contextlib import contextmanager
import sys
from .common import *
from tbps.alternating.native_sources import native_session
@contextmanager
def source_session(n,bundle,c,name):
    reuse=dict(bundle.modules)
    collisions=set(reuse)&set(sys.modules)
    if collisions:
        need(collisions=={'utils'},'unapproved legacy namespace collision: '+str(collisions))
        official=read_json(c['targets'][name]['path'])
        current=Path(sys.modules['utils'].__file__).resolve()
        need(current in {(Path(official['root'])/'utils.py').resolve(),(Path(official['root'])/'utils/__init__.py').resolve()} and str(current) in official['files'],'official utils identity: '+str(current))
        checked(FileRef(str(current),official['files'][str(current)]))
        # 冻结子模块和已绑定的类仍原样复用；保留官方父模块供其模型调用。
        reuse.pop('utils')
    with native_session(n,reuse=reuse) as session:yield session
