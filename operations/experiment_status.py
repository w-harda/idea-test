"""用户手动检查单次实验状态/当前日志；只读、无监控循环或模型导入。"""
from pathlib import Path
from collections import deque
import argparse,json

def main():
 a=argparse.ArgumentParser();a.add_argument('--control',required=True);a.add_argument('--log',action='store_true');a.add_argument('--publication',action='store_true');a.add_argument('--lines',type=int,default=60);args=a.parse_args()
 root=Path(args.control);path=root/('publication_status.json' if args.publication else 'status.json')
 if not path.exists():print('PENDING: '+str(path));return
 s=json.loads(path.read_text());print(json.dumps(s,ensure_ascii=False,indent=2))
 if args.log and not args.publication:
  stage=s.get('stage');log=root/(stage+'.log') if stage else None
  if log and log.exists():
   print('\nCURRENT_LOG '+str(log))
   with log.open(encoding='utf-8',errors='replace') as f:print(''.join(deque(f,maxlen=max(1,args.lines))),end='')
if __name__=='__main__':main()
