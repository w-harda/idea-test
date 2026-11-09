"""用户按需执行一次：读取阶段、进程、当前日志与已审计节点，无后台轮询。"""
from pathlib import Path
import json,os
CTL=Path('/home/lzf/ldx/tmp/gh_cooperative_v1_20261109')
def show(path,label):
 print('\n'+label)
 if not path.exists():print('PENDING');return None
 value=json.loads(path.read_text());print(json.dumps(value,ensure_ascii=False,indent=2));return value
def main():
 status=show(CTL/'status.json','Supervisor');started=show(CTL/'started.json','启动身份')
 child=show(CTL/'child.json','当前子进程')
 if child:
  try:os.kill(child['pid'],0);print('child_alive=True')
  except ProcessLookupError:print('child_alive=False')
  label=child['label'];scope='train' if label.startswith('train_to_') else 'smoke' if label=='01_smoke' else None
  if scope:show(Path('/home/lzf/ldx/tmp/gh_cooperative_v1_'+scope+'_20261109/status.json'),'训练阶段/轮次/游标')
  log=CTL/(label+'.log')
  if log.exists():
   # 只读取末尾64KiB，避免载入整份长日志。
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode('utf-8',errors='replace')
   print('\n当前日志末尾：'+str(log));print('\n'.join(tail.splitlines()[-30:]))
 show(CTL/'results_so_far.json','已通过CPU审计的正式Test节点')
if __name__=='__main__':main()
