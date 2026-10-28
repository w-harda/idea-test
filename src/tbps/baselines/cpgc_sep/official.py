"""固定官方定义的无改写加载，隔离无关分类模型/训练入口的import副作用。"""
from pathlib import Path
import ast,types,hashlib,typing,math,copy,re,random,collections,pdb
from tbps.artifacts import ArtifactError,checked
from tbps.contracts import FileRef
def checked_tree(root,expected_commit,files):
 import subprocess
 if subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip()!=expected_commit:raise ArtifactError("official Commit mismatch")
 if subprocess.check_output(["git","-C",root,"status","--porcelain"],text=True).strip():raise ArtifactError("official checkout modified")
 for p,h in files.items():checked(FileRef(p,h))
def definitions(path,names,sha256,namespace):
 """编译原AST节点本身；不改函数体、不做Monkey Patch或Loss替换。"""
 path=Path(path)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:raise ArtifactError("official definition SHA")
 tree=ast.parse(path.read_text(),filename=str(path))
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ArtifactError("official definitions missing")
 module=ast.Module(body=nodes,type_ignores=[])
 exec(compile(module,str(path),"exec"),namespace)
 return namespace
def cpgc_text_class(c):
 import torch,numpy as np
 src=c["official"]["cpgc"];root=Path(src["root"])
 ns={"torch":torch,"np":np,"copy":copy,"re":re}
 definitions(root/"utils.py",["get_filter_words"],src["files"][str(root/"utils.py")],ns)
 ns["filter_words"]=ns["get_filter_words"]()
 definitions(root/"attacker.py",["TextAttacker"],src["files"][str(root/"attacker.py")],ns)
 return ns["TextAttacker"],ns["filter_words"]
def cpgc_pre_caption(c):
 src=c["official"]["cpgc"];p=Path(src["root"])/"dataset.py"
 return definitions(p,["pre_caption"],src["files"][str(p)],{"re":re})["pre_caption"]
def sep_ensemble_class(c):
 import torch,numpy as np
 src=c["official"]["sep"];p=Path(src["root"])/"SEP_utils.py"
 return definitions(p,["Ensemble"],src["files"][str(p)],{"torch":torch,"np":np,"math":math,"Any":typing.Any,"OrderedDict":collections.OrderedDict})["Ensemble"]
def sep_functions(c):
 import torch,numpy as np,nltk
 src=c["official"]["sep"];root=Path(src["root"])
 criteria=types.SimpleNamespace()
 p=root/"criteria.py";ns=definitions(p,["get_pos"],src["files"][str(p)],{"nltk":nltk})
 criteria.get_pos=ns["get_pos"]
 ns={"torch":torch,"np":np,"random":random,"math":math,"defaultdict":collections.defaultdict,"criteria":criteria,"VictimModel":object,"pdb":pdb}
 p=root/"SEP_Attack.py"
 return definitions(p,["get_change_ls","get_important_order","find_max_synonym","optimize","early_stop","attack"],src["files"][str(p)],ns)
