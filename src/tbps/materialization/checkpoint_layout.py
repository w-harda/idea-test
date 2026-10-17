"""仅ZIP目录与pickletools词法扫描，不执行pickle、不反序列化模型或读取Storage内容。"""
from pathlib import Path
import pickletools
import zipfile
from ..artifacts import ArtifactError

WIDTHS={"ByteStorage":1,"CharStorage":1,"BoolStorage":1,"ShortStorage":2,"HalfStorage":2,
        "BFloat16Storage":2,"IntStorage":4,"FloatStorage":4,"LongStorage":8,"DoubleStorage":8}

def checkpoint_layout(path):
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        data=[v for v in entries if v.filename.endswith("/data.pkl")]
        if len(data)!=1 or data[0].file_size>2*1024**2:raise ArtifactError("single bounded pickle metadata required")
        records={v.filename.split("/data/")[1]:v.file_size for v in entries if "/data/" in v.filename}
        # 仅记录原子值、memo引用和五字段storage描述；GLOBAL/REDUCE/BUILD绝不执行。
        memo={};candidate=[];last=None;section=None;model={};all_seen={};globals_seen=set()
        for op,arg,pos in pickletools.genops(archive.read(data[0])):
            name=op.name
            if name=="GLOBAL":
                last=("GLOBAL",arg);candidate.append(last);globals_seen.add(arg)
            elif name in ("BINUNICODE","SHORT_BINUNICODE","UNICODE","BININT","BININT1","BININT2","LONG1","LONG4"):
                last=arg;candidate.append(last)
                if arg=="model":section="model"
                elif arg in ("optimizer","scheduler","epoch"):section="other"
            elif name in ("BINPUT","LONG_BINPUT","PUT"):memo[int(arg)]=last
            elif name in ("BINGET","LONG_BINGET","GET"):
                last=memo.get(int(arg));candidate.append(last)
            elif name=="MARK":candidate=[]
            elif name=="TUPLE":last=tuple(candidate);candidate=[]
            elif name=="BINPERSID":
                if not (isinstance(last,tuple) and len(last)==5 and last[0]=="storage"):
                    raise ArtifactError("unrecognized storage metadata; refuse estimate")
                _,kind,key,device,count=last
                if not (isinstance(kind,tuple) and kind[0]=="GLOBAL" and kind[1].startswith("torch ")
                        and kind[1].split()[1] in WIDTHS and type(count)is int and count>=0 and type(key)is str):
                    raise ArtifactError("unknown storage description")
                nbytes=count*WIDTHS[kind[1].split()[1]]
                if records.get(key)!=nbytes:raise ArtifactError("ZIP storage bytes disagree with metadata")
                all_seen[key]=nbytes
                if section=="model":model[key]=nbytes
                last=None
            elif name not in ("PROTO",):last=None
        if not model or len(all_seen)!=len(records):raise ArtifactError("incomplete model/storage lexical inventory")
        return {"file_bytes":Path(path).stat().st_size,"storage_bytes":sum(records.values()),
            "model_storage_bytes":sum(model.values()),"other_storage_bytes":sum(records.values())-sum(model.values()),
            "storage_count":len(records),"model_storage_count":len(model),
            "pickle_bytes":data[0].file_size,"pickle_execution":False,"storage_contents_read":False,
            "globals_inspected_only":sorted(globals_seen)}
