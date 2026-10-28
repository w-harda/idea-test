"""公开资源流式中转：Windows不保存权重；服务器独立目录原子接收，不重试。"""
import argparse,hashlib,json,shlex,subprocess,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT="/home/lzf/ldx/checkpoints/cpgc_sep_official_20261028"
RESOURCES=[
 ("cpgc_uap_noise_39.pth","1cpV6JQ30y9Mb6CULDCrYzt75oXS6lmFR"),
 ("cpgc_uap_embedding_39.pth","1T81RN5ISz6rTKsXUY-W87EoAai-4E_qI"),
 ("cpgc_word_idx_ALBEF.npy","1QSpHcy6l_YtGLuXsgunZW7lIYX0auHra"),
 ("cpgc_word_embedding_ALBEF.pth","1J5YBp03kZ3cSfND4bdBxVnE8cR4qm6Eo"),
 ("rasa_icfg_checkpoint.pth","1lLB332ANq87v2jV2bLdsV7rQ7OBAnP9X"),
 ("rasa_rstp_checkpoint.pth","1e5KPmfoij22J2zZOZxhSodU4SNz76BjX"),
 ("sep_counter-fitted-vectors.txt","1bayGomljWb6HeYDMTDKXrh0HackKtSlx"),
 ("sep_mat.txt","1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx"),
 ("albef_flickr30k.pth","https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth")]
def receive_code(name,url,size):
 return '''
from pathlib import Path
import sys,hashlib,json,os,time
root=Path(%r);root.mkdir(exist_ok=True)
p=root/%r
assert not p.exists() and not Path(str(p)+".receipt.json").exists(),"existing resource refuses overwrite"
tmp=Path(str(p)+".download.pending")
h=hashlib.sha256();n=0
try:
 with tmp.open("xb") as f:
  while True:
   b=sys.stdin.buffer.read(1024*1024)
   if not b:break
   f.write(b);h.update(b);n+=len(b)
  f.flush();os.fsync(f.fileno())
 assert n==%r,("HTTP length differs",n,%r)
 os.link(tmp,p);tmp.unlink()
 receipt={"path":str(p),"url":%r,"bytes":n,"sha256":h.hexdigest(),"completed_unix":time.time()}
 with Path(str(p)+".receipt.json").open("x") as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps(receipt),flush=True)
except BaseException as e:
 fail=Path(str(p)+".download.FAIL.json")
 with fail.open("x") as f:json.dump({"bytes_received":n,"error":repr(e),"url":%r},f)
 raise
'''%(ROOT,name,size,size,url,url)
def transfer(item):
 name,ref=item
 url=ref if ref.startswith("https://") else "https://drive.usercontent.google.com/download?id="+ref+"&export=download&confirm=t"
 print("DOWNLOAD_BEGIN "+name,flush=True)
 req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
 with urllib.request.urlopen(req,timeout=60) as response:
  size=int(response.headers.get("Content-Length","0"))
  if size<=0 or "text/html" in response.headers.get("Content-Type",""):raise RuntimeError("not a binary resource / no exact length")
  command="python3 -B -c "+shlex.quote(receive_code(name,url,size))
  p=subprocess.Popen(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","-o","HostName=100.69.166.7","tbps-server",command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  h=hashlib.sha256();n=0;last=0;start=time.monotonic()
  try:
   while True:
    b=response.read(1024*1024)
    if not b:break
    p.stdin.write(b);h.update(b);n+=len(b)
    if n-last>=128*1024*1024:
     print("DOWNLOAD_PROGRESS "+json.dumps({"file":name,"bytes":n,"total":size,"seconds":time.monotonic()-start}),flush=True);last=n
   p.stdin.close()
   out=p.stdout.read().decode();err=p.stderr.read().decode();rc=p.wait()
   if rc:raise RuntimeError("SSH receiver exit "+str(rc)+" "+err[-2000:])
   d=json.loads(out);assert d["sha256"]==h.hexdigest() and d["bytes"]==n==size
   print("DOWNLOAD_COMPLETE "+json.dumps(d),flush=True);return d
  except BaseException:
   if p.stdin and not p.stdin.closed:p.stdin.close()
   p.wait(timeout=30)
   raise
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--allow-public-resource-download",action="store_true");a=parser.parse_args()
 if not a.allow_public_resource_download:parser.error("explicit download authorization flag required")
 failures=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  fs={pool.submit(transfer,item):item[0] for item in RESOURCES}
  for f in as_completed(fs):
   try:f.result()
   except BaseException as e:
    failures.append({"file":fs[f],"error":repr(e)});print("DOWNLOAD_FAIL "+json.dumps(failures[-1]),flush=True)
 print("RESOURCE_TRANSFER_FINISHED "+json.dumps({"failures":failures}),flush=True)
 return bool(failures)
if __name__=="__main__":sys.exit(main())
