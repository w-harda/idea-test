"""从第30轮完整边界恢复原实验；独立控制身份，保留中断证据，真实错误不重试。"""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116')
PY = '/home/lzf/ldx/envs/tbps-image/bin/python'
CFG = SOURCE / 'configs/framework/gh_attribute_response.server.json'
BASE = Path('/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116')
TRAIN = BASE / 'train'
OLD = Path('/home/lzf/ldx/tmp/gh_attribute_response_20261116')
CTL = Path('/home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119')
ARCHIVE = BASE / 'recovery_evidence/20261119'
CONFIG_SHA = 'bd7af6e51f56514c79831d8a76cbdbca919c1adf52289e5a2d3d1aec80bfa975'
SOURCE_HEAD = '720c607e70075b9c2625cf1377f142cbe8792906'
sys.path.insert(0, str(SOURCE / 'src'))
from tbps.artifacts import ArtifactStore, read_json, sha_file, json_bytes

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def record(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('xb') as f:
            f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.pending')
        with tmp.open('wb') as f:
            f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

def status(phase, **extra):
    record(CTL / 'status.json', {'phase': phase, 'time': time.time(), **extra})

@contextmanager
def lock(path):
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def environment():
    return {**os.environ, 'PYTHONPATH': str(SOURCE / 'src'),
            'PYTHONDONTWRITEBYTECODE': '1',
            'MPLCONFIGDIR': str(CTL / 'matplotlib'),
            'XDG_CACHE_HOME': str(CTL / 'cache'), 'CUDA_CACHE_PATH': str(CTL / 'cuda_cache')}

def gpu_ready():
    p = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.free',
                        '--format=csv,noheader'], capture_output=True, text=True)
    require(p.returncode == 0, 'NVIDIA_DRIVER_UNAVAILABLE: ' + (p.stdout + p.stderr).strip())
    # 不设置空闲显存门槛；真正CUDA分配失败也直接停止。
    q = subprocess.run([PY, '-B', '-c',
        'import torch; x=torch.zeros(1,device="cuda:0"); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0))'],
        capture_output=True, text=True)
    require(q.returncode == 0, 'CUDA_ALLOCATION_FAILED: ' + (q.stdout + q.stderr)[-2000:])
    return {'nvidia_smi': p.stdout.strip(), 'cuda_allocation': q.stdout.strip()}

def recovery_plan(train):
    """只接受本次精确的未提交Round31，不接受任意失败或任意待发布目录。"""
    train = Path(train)
    latest = read_json(train / 'boundaries/latest.json')
    require(latest['round'] == 30, 'only the verified Round30 boundary is authorized')
    require(not list(train.glob('FAIL*')), 'real failure requires separate diagnosis')
    require(not (train / 'boundaries/completed').exists(), 'training already completed')
    require(not (train / 'ledgers/round_0031').exists()
            and not (train / 'boundaries/round_0031').exists(), 'Round31 already published; do not replay')
    later_claims = sorted(p.name for p in (train / 'boundaries').glob('claim_*')
                          if int(p.name.split('_')[-1]) > 30)
    later_io = sorted(p.name for p in (train / 'round_io').glob('round_*')
                      if int(p.name.split('_')[-1]) > 30)
    later_banks = sorted(p.name for p in (train / 'B').glob('bank_*')
                         if int(p.name.split('_')[-1]) > 30)
    require(later_claims == ['claim_0031'] and later_io == ['round_0031']
            and later_banks == ['bank_31'], 'unexpected interrupted-round paths')
    require(not list((train / 'boundaries').glob('.pending-*')), 'unexpected pending boundary')
    pending = sorted((train / 'ledgers').glob('.pending-*'))
    require(len(pending) == 1, 'exactly one interrupted Round31 ledger required')
    intent = read_json(pending[0] / 'intent.json')
    require(intent['artifact_id'] == 'round_0031'
            and intent['provenance']['configuration'] == CONFIG_SHA, 'pending ledger identity mismatch')
    claim = read_json(train / 'boundaries/claim_0031/claim.json')
    require(claim == {'round': 31, 'parent_state': latest['state_sha']}, 'interrupted parent mismatch')
    return latest, [train / 'boundaries/claim_0031', train / 'round_io/round_0031',
                    train / 'B/bank_31', pending[0]]

def file_index(roots):
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob('*'))
        for p in paths:
            require(not p.is_symlink(), 'symlink forbidden in recovery evidence')
            if not p.is_file():
                continue
            st = p.stat()
            yield {'path': str(p), 'sha256': sha_file(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}

def preflight():
    require(sha_file(CFG) == CONFIG_SHA, 'frozen configuration changed')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
            == SOURCE_HEAD, 'original source HEAD changed')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip(),
            'original worktree is not clean')
    latest, paths = recovery_plan(TRAIN)
    m = ArtifactStore(TRAIN / 'boundaries', new_base=BASE).verify('round_0030')
    require(m['provenance']['state_sha'] == latest['state_sha'], 'complete boundary state mismatch')
    require(sha_file(Path(latest['manifest']['path'])) == latest['manifest']['sha256'],
            'latest boundary manifest mismatch')
    ArtifactStore(TRAIN / 'boundaries', new_base=BASE).verify('claim_0031')
    auditroot = BASE / 'train_cpu_audit/round_30'
    ArtifactStore(auditroot, new_base=BASE).verify('completed')
    train = read_json(auditroot / 'completed/report.json')
    require(train['status'] == 'GH_AR_TRAIN_CPU_AUDIT_PASS'
            and len(train['rounds']) == 30 and train['cursor']['round'] == 30
            and train['configuration_sha256'] == CONFIG_SHA, '30-round CPU audit identity mismatch')
    results = []
    for n in (10, 20, 30):
        ArtifactStore(TRAIN / 'checkpoints', new_base=BASE).verify(f'round_{n:02d}')
        ArtifactStore(TRAIN / 'checkpoints', new_base=BASE).verify(f'teacher_{n:02d}')
        p = BASE / f'test/round_{n:02d}/cpu_audit'
        ArtifactStore(p, new_base=BASE).verify('completed')
        a = read_json(p / 'completed/report.json')
        require(a['status'] == 'GH_AR_JOINT_TEST_CPU_AUDIT_PASS'
                and a['round'] == n and a['configuration_sha256'] == CONFIG_SHA
                and a['scope'] == 'official_test' and a['gallery'] == 3074 and a['query'] == 6156,
                'existing Full audit identity mismatch')
        results.append({'round': n, 'metrics': a['metrics']})
    return {'status': 'RECOVERY_CPU_PREFLIGHT_PASS', 'configuration_sha256': CONFIG_SHA,
            'source_head': SOURCE_HEAD, 'latest': latest, 'archive_plan': [str(p) for p in paths],
            'existing_results': results, 'resume_from': 30, 'remaining_nodes': [40, 50, 60],
            'replay_uncommitted_round': 31, 'not_a_second_fresh_training_attempt': True}

def archive_interruption(proof):
    require(not ARCHIVE.exists(), 'recovery evidence consumed; no automatic retry')
    require(read_json(TRAIN / 'boundaries/latest.json') == proof['latest'], 'resume boundary changed')
    _, paths = recovery_plan(TRAIN)
    require([str(p) for p in paths] == proof['archive_plan'], 'archive plan changed')
    ARCHIVE.mkdir(parents=True)
    # 原Supervisor及训练控制文件全部复制，原started/失败/阶段日志仍原位保留。
    shutil.copytree(OLD, ARCHIVE / 'original_control',
                    ignore=shutil.ignore_patterns('cache', 'cuda_cache', 'matplotlib'))
    # 原滚动保留策略会在提交31轮后退休30轮边界，先另存完整30轮状态及Bank。
    committed_sources = [TRAIN / 'boundaries/round_0030', TRAIN / 'B/bank_30']
    for source in committed_sources:
        target = ARCHIVE / 'committed_round30' / source.relative_to(TRAIN)
        shutil.copytree(source, target)
    with (ARCHIVE / 'committed_round30_files.jsonl').open('x') as out:
        for item in file_index(committed_sources):
            target = ARCHIVE / 'committed_round30' / Path(item['path']).relative_to(TRAIN)
            require(sha_file(target) == item['sha256'], 'committed Round30 backup SHA mismatch')
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
        out.flush(); os.fsync(out.fileno())
    inventory = ARCHIVE / 'interrupted_files.jsonl'
    with inventory.open('x') as out:
        for item in file_index(paths):
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
        out.flush(); os.fsync(out.fileno())
    for p in paths:
        target = ARCHIVE / 'uncommitted_round31' / p.relative_to(TRAIN)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(p, target)
    count = 0
    with inventory.open() as f:
        for line in f:
            item = json.loads(line)
            p = ARCHIVE / 'uncommitted_round31' / Path(item['path']).relative_to(TRAIN)
            st = p.stat()
            require(sha_file(p) == item['sha256'] and st.st_size == item['size']
                    and st.st_mtime_ns == item['mtime_ns'], 'archived bytes/stat changed')
            count += 1
    record(ARCHIVE / 'completed.json', {'original_boundary': proof['latest'],
           'preserved_files': count, 'inventory_sha256': sha_file(inventory),
           'original_control_preserved': True, 'committed_round30_backup_verified': True,
           'committed_inventory_sha256': sha_file(ARCHIVE / 'committed_round30_files.jsonl')}, exclusive=True)

def stages():
    for n in (40, 50, 60):
        args = ['--action', 'train', '--allow-gpu', '--resume']
        if n < 60:
            args += ['--stop-after-round', str(n)]
        yield f'train_to_{n}', args, 3 if n < 60 else 0
        yield f'audit_train_{n}', ['--action', 'audit', '--scope', 'train', '--round', str(n)], 0
        yield f'test_{n}', ['--action', 'test', '--round', str(n), '--allow-gpu'], 0
        yield f'audit_test_{n}', ['--action', 'test-audit', '--round', str(n)], 0
    yield 'final_report', ['--action', 'report'], 0

def child(label, args, expected):
    status(label)
    with (CTL / (label + '.log')).open('x') as log:
        p = subprocess.Popen([PY, '-u', '-B', str(SOURCE / 'operations/gh_attribute_response.py'),
             '--config', str(CFG), *args], cwd=SOURCE, env=environment(), stdin=subprocess.DEVNULL,
             stdout=log, stderr=subprocess.STDOUT)
        record(CTL / 'child.json', {'pid': p.pid, 'label': label, 'time': time.time()})
        rc = p.wait()
    record(CTL / (label + '.exit.json'), {'returncode': rc, 'expected': expected}, exclusive=True)
    require(rc == expected, f'{label} exit {rc}; stopped without retry')

def publish():
    files = ['docs/PROJECT_STATE.md', 'docs/framework/results/gh_attribute_response']
    require(not subprocess.check_output(['git', 'diff', '--cached', '--name-only'],
                                       cwd=SOURCE, text=True).strip(), 'unexpected staged files')
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=SOURCE, check=True)
    subprocess.run(['git', '-c', 'user.name=w-harda', '-c', 'user.email=wharda@163.com',
        'commit', '-m', 'docs: 发布属性更新收益引导版60轮六节点Joint结果及恢复来源'], cwd=SOURCE, check=True)
    subprocess.run(['git', 'push', 'origin', 'codex/gh-attribute-response'], cwd=SOURCE, check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin',
        'refs/heads/codex/gh-attribute-response'], cwd=SOURCE, text=True).split()[0]
    require(head == remote, 'publication remote SHA mismatch')
    return head

def run():
    with lock(CTL / 'supervisor.lock'), lock(OLD / 'supervisor.lock'):
        require(not (CTL / 'started.json').exists(), 'recovery already consumed; do not retry')
        # 驱动不可用时，在移动原产物或消耗恢复身份之前拒绝。
        gpu = gpu_ready()
        child('cpu_source_preflight', ['--action', 'preflight', '--scope', 'train'], 0)
        with lock(OLD / 'train/execution.lock'):
            proof = preflight()
            record(CTL / 'started.json', {'pid': os.getpid(), 'pgid': os.getpgid(0),
                'sid': os.getsid(0), 'time': time.time(), 'source': str(SOURCE), 'source_head': SOURCE_HEAD,
                'recovery_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'tool_sha256': sha_file(Path(__file__)), 'configuration_sha256': CONFIG_SHA,
                'resume_from': 30, 'gpu': gpu, 'mode': 'original_training_continuation'}, exclusive=True)
            status('ARCHIVING_UNCOMMITTED_ROUND31')
            archive_interruption(proof)
        for label, args, expected in stages():
            child(label, args, expected)
        # 将恢复身份附在新报告，不改原训练配置或历史账本。
        evidence = SOURCE / 'docs/framework/results/gh_attribute_response/RECOVERY.json'
        record(evidence, {'started': read_json(CTL / 'started.json'),
                         'archive': read_json(ARCHIVE / 'completed.json'),
                         'old_supervisor_records_unchanged': True})
        head = publish()
        status('SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED', commit=head)

def show_status():
    for path, label in ((CTL / 'status.json', '恢复Supervisor'),
                        (CTL / 'started.json', '恢复身份'),
                        (CTL / 'child.json', '当前子进程')):
        print(label, json.dumps(read_json(path), ensure_ascii=False) if path.exists() else 'NOT_STARTED')
    childfile = CTL / 'child.json'
    if childfile.exists():
        p = read_json(childfile); proc = Path('/proc') / str(p['pid'])
        cmd = (proc / 'cmdline').read_bytes() if (proc / 'cmdline').exists() else b''
        print('child_alive', b'gh_attribute_response.py' in cmd)
        log = CTL / (p['label'] + '.log')
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 12000))
                print('\n'.join(f.read().decode(errors='replace').splitlines()[-6:]))
    path = OLD / 'train/status.json'
    if path.exists():
        p = read_json(path)
        print('训练阶段快照', {k: p.get(k) for k in ('phase', 'round', 'visited', 'total', 'cursor')})
    for n in (10, 20, 30, 40, 50, 60):
        p = BASE / f'test/round_{n:02d}/cpu_audit/completed/report.json'
        if p.exists():
            a = read_json(p); print(n, a.get('status'), a.get('metrics'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', choices=('preflight', 'launch', 'run', 'status'), default='status')
    ap.add_argument('--allow-recovery', action='store_true')
    a = ap.parse_args()
    if a.action == 'status':
        return show_status()
    CTL.mkdir(parents=True, exist_ok=True)
    if a.action == 'preflight':
        with lock(OLD / 'supervisor.lock'), lock(OLD / 'train/execution.lock'):
            value = preflight()
        record(CTL / 'preflight.json', value)
        print(json.dumps(value, ensure_ascii=False, indent=2)); return
    require(a.allow_recovery, 'explicit recovery authorization required')
    if a.action == 'launch':
        try:
            gpu_ready()
        except Exception as e:
            status('BLOCKED_NVIDIA_DRIVER_OR_CUDA', error=str(e))
            raise
        require(not (CTL / 'started.json').exists(), 'recovery already consumed')
        with (CTL / 'supervisor.log').open('xb') as log:
            p = subprocess.Popen([PY, '-u', '-B', str(Path(__file__).resolve()),
                '--action', 'run', '--allow-recovery'], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment())
        record(CTL / 'launch.json', {'pid': p.pid, 'time': time.time()}, exclusive=True)
        print('LAUNCHED', p.pid, 'status command:', PY, '-B', str(Path(__file__).resolve()), '--action status')
        return
    try:
        run()
    except BaseException as e:
        status('FAIL_STOPPED_NO_RETRY', error=repr(e)); raise

if __name__ == '__main__':
    main()
