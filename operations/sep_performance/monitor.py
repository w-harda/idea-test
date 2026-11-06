"""只读Linux /proc与nvidia-smi采样；不attach、不发送信号。"""
import argparse,json,os,subprocess,time
from pathlib import Path

def proc(pid):
    p=Path('/proc')/str(pid)
    s=(p/'stat').read_text().rsplit(')',1)[1].split()
    io={}
    try:
        io={k:int(v) for k,v in (x.split(':') for x in (p/'io').read_text().splitlines())}
    except (FileNotFoundError,PermissionError):pass
    status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
    threads={}
    for task in (p/'task').iterdir():
        try:
            t=(task/'stat').read_text().rsplit(')',1)[1].split()
            threads[task.name]=int(t[11])+int(t[12])
        except FileNotFoundError:pass
    return dict(pid=pid,ticks=int(s[11])+int(s[12]),rss=int(s[21])*os.sysconf('SC_PAGE_SIZE'),
                swap=status.get('VmSwap','0 kB').strip(),threads=threads,io=io,
                voluntary=int(status.get('voluntary_ctxt_switches','0')),
                involuntary=int(status.get('nonvoluntary_ctxt_switches','0')),
                cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())

def host():
    text=Path('/proc/stat').read_text().splitlines()
    cores={x.split()[0]:list(map(int,x.split()[1:])) for x in text if x.startswith('cpu')}
    mem=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
    return {'cores':cores,'memory':{k:mem[k].strip() for k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')},
            'vmstat':{k:int(v) for k,v in (x.split() for x in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout','pgmajfault')}}
def gpu():
    def call(args):return subprocess.check_output(['nvidia-smi',*args],text=True).strip()
    return {'device':call(['--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader,nounits']),
            'processes':call(['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])}

def summarize(samples):
    cpu=[];utils=[];thread={};busy=[];gpu_mem=[];power=[];clocks=[]
    hz=os.sysconf('SC_CLK_TCK')
    for a,b in zip(samples,samples[1:]):
        dt=b['time']-a['time'];p,q=a['process'],b['process']
        cpu.append(100*(q['ticks']-p['ticks'])/hz/dt)
        for tid,ticks in q['threads'].items():
            if tid in p['threads']:thread.setdefault(tid,[]).append(100*(ticks-p['threads'][tid])/hz/dt)
        used=0.;idle=0
        for name,v in b['host']['cores'].items():
            old=a['host']['cores'][name]
            dif=[x-y for x,y in zip(v,old)];total=sum(dif[:8]);wait=dif[3]+dif[4]
            if name=='cpu':busy.append(100*(total-wait)/total if total else 0)
            else:
                if total and 100*(total-wait)/total<5:idle+=1
        b['idle_cores_below_5pct']=idle
    for s in samples:
        parts=s['gpu']['device'].split(', ')
        utils.append(float(parts[0]));gpu_mem.append(float(parts[1]));power.append(float(parts[2]));clocks.append(float(parts[3]))
    avg=lambda x:sum(x)/len(x) if x else 0
    return {'duration_seconds':samples[-1]['time']-samples[0]['time'],'samples':len(samples),'logical_cores':os.cpu_count(),
            'process_cpu_mean_pct_one_core_100':avg(cpu),'process_cpu_peak_pct':max(cpu,default=0),
            'process_mean_logical_cores':avg(cpu)/100,'host_cpu_busy_mean_pct':avg(busy),
            'idle_cores_below_5pct_mean':avg([s.get('idle_cores_below_5pct',0) for s in samples[1:]]),
            'thread_cpu_mean_pct':{k:avg(v) for k,v in sorted(thread.items(),key=lambda kv:-avg(kv[1]))[:12]},
            'gpu_util_mean_pct':avg(utils),'gpu_util_peak_pct':max(utils),'gpu_zero_util_fraction':sum(v==0 for v in utils)/len(utils),
            'gpu_memory_peak_MiB':max(gpu_mem),'gpu_memory_mean_MiB':avg(gpu_mem),'gpu_power_mean_W':avg(power),
            'gpu_clock_mean_MHz':avg(clocks),'rss_peak_bytes':max(s['process']['rss'] for s in samples),
            'io_delta':{k:samples[-1]['process']['io'].get(k,0)-v for k,v in samples[0]['process']['io'].items()},
            'context_switches':{k:samples[-1]['process'][k]-samples[0]['process'][k] for k in ('voluntary','involuntary')},
            'host_memory_last':samples[-1]['host']['memory'],'swap_pages_in_delta':samples[-1]['host']['vmstat']['pswpin']-samples[0]['host']['vmstat']['pswpin'],
            'swap_pages_out_delta':samples[-1]['host']['vmstat']['pswpout']-samples[0]['host']['vmstat']['pswpout']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=60)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    samples=[];start=time.monotonic()
    with (out/'samples.jsonl').open('x') as f:
        while True:
            s={'time':time.monotonic(),'wall_time':time.time(),'process':proc(a.pid),'host':host(),'gpu':gpu()}
            samples.append(s);f.write(json.dumps(s)+'\n');f.flush()
            if time.monotonic()-start>=a.seconds:break
            time.sleep(max(0,1-(time.monotonic()-s['time'])))
    result=summarize(samples);(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
