"""Persistent, diagnostic-only ABC queue. Import/status are stdlib-only."""
import argparse
from contextlib import contextmanager
import csv
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
TASKS = ('D-A-7', 'D-B-7', 'D-C-7')
STOPPED = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def event(root, name, **detail):
    with (root/'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time_unix':time.time(), 'event':name, **detail}, ensure_ascii=False)+'\n')


def seal(root):
    root = Path(root)
    inventory = {p.relative_to(root).as_posix():{'sha256':sha(p), 'bytes':p.stat().st_size}
                 for p in sorted(root.rglob('*')) if p.is_file() and p != root/'artifacts.json'}
    write(root/'artifacts.json', inventory)
    return sha(root/'artifacts.json')


def verify_seal(root):
    root = Path(root)
    inventory = read(root/'artifacts.json')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'artifacts.json'}
    if actual != set(inventory):
        raise ValueError(f'File inventory mismatch: {root}')
    for name, metadata in inventory.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or (root/rel).is_symlink():
            raise ValueError('Unsafe artifact path')
        expected = metadata['sha256'] if isinstance(metadata,dict) else metadata
        if sha(root/rel) != expected or (isinstance(metadata,dict) and (root/rel).stat().st_size != metadata['bytes']):
            raise ValueError(f'Artifact changed: {root/rel}')


def stopped(root):
    if STOPPED or (root/'STOP').exists():
        raise InterruptedError('User stopped diagnostic queue; no further dispatch')


def process_identity(pid):
    """Linux PID/start tick pair protects against PID reuse."""
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        return {'pid':pid,'start_tick':text[text.rfind(')')+2:].split()[19]}
    except (OSError,IndexError):return None


def arm_parent_death(expected_parent):
    import ctypes
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'prctl(PR_SET_PDEATHSIG)')
    if os.getppid()!=expected_parent:
        raise InterruptedError('Controller exited before worker guard was armed')


def exit_state(state, code):
    if code and state.get('status') not in ('failed','interrupted','oom','memory_margin_exceeded'):
        return {**state,'status':'failed','error_type':'UnclassifiedWorkerExit','returncode':code,'retryable':False}
    return state


@contextmanager
def lock(root):
    with (root/'queue.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another controller owns this diagnostic queue') from exc
        yield


def source_identity(data_dir):
    source = Path(__file__).parent
    return {'sources':{p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*.py'))},
            'data':{name:sha(Path(data_dir)/name) for name in ('train2id.txt','valid2id.txt','entity2id.txt','relation2id.txt')}}


def validate_binding(root, catalog):
    if source_identity(catalog['data_dir']) != catalog['provenance']:
        raise ValueError('Source/data changed: create a new diagnostic version, do not mix results')
    if sha(root/'source/artifacts.json') != catalog['source_inventory_sha256']:
        raise ValueError('Source snapshot inventory changed')
    verify_seal(root/'source')


def ensure_run(root, data_dir):
    root, data_dir = Path(root), Path(data_dir).resolve()
    if root.exists():
        catalog = read(root/'catalog.json')
        if catalog['data_dir'] != str(data_dir):raise ValueError('Different data directory')
        validate_binding(root,catalog)
        return catalog
    root.mkdir(parents=True)
    source = Path(__file__).parent
    for path in sorted(source.rglob('*.py')):
        dest=root/'source'/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    digest=seal(root/'source')
    catalog={'protocol':'formal-abc-v2-diagnostic','data_dir':str(data_dir),'tasks':list(TASKS),
             'seed':7,'epochs':100,'validation_every':10,'threads':10,'candidate_formal_epochs':1000,
             'formal_execution_allowed':False,'max_attempts':2,'provenance':source_identity(data_dir),
             'source_inventory_sha256':digest,'created_unix':time.time()}
    write(root/'catalog.json',catalog)
    write(root/'catalog_seal.json',{'sha256':sha(root/'catalog.json')})
    write(root/'status.json',{'status':'created','stage':'preflight','pid':os.getpid()})
    event(root,'created',authorization='implementation/checks/three diagnostics only')
    return catalog


def gpu_processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def wait_gpu(root):
    announced=False
    while True:
        stopped(root)
        others=[pid for pid in gpu_processes() if pid != os.getpid()]
        if not others:return
        if not announced:
            event(root,'waiting_resource',compute_pids=others)
            write(root/'status.json',{'status':'waiting_resource','compute_pids':others,'pid':os.getpid()})
            announced=True
        time.sleep(10)


def launch(root, output, request, *, worker_command='_abc-worker', wait_for_gpu=True):
    stopped(root)
    if wait_for_gpu:wait_gpu(root)
    elif gpu_processes():raise RuntimeError('GPU already has a compute process')
    if 'expected_provenance' in request and source_identity(request['data_dir'])!=request['expected_provenance']:
        raise ValueError('Source/data changed while waiting for GPU')
    output.parent.mkdir(parents=True,exist_ok=True)
    req=output.with_name(output.name+'.request.json')
    if output.exists() or req.exists():raise FileExistsError(output)
    request={**request,'controller_pid':os.getpid()}
    write(req,request)
    write(root/'active.json',{'output':str(output),'request':str(req),'controller_pid':os.getpid()})
    cmd=[sys.executable,'-B','-m','src.py.experiments.formal_v1',worker_command,'--request',str(req.resolve())]
    event(root,'worker_start',output=str(output),kind=request['kind'],group=request['group'])
    with output.with_name(output.name+'.stdout.log').open('w') as out, output.with_name(output.name+'.stderr.log').open('w') as err:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,start_new_session=True,
            env={**os.environ,'PYTHONHASHSEED':str(request['seed']),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'10','MKL_NUM_THREADS':'10'})
        try:
            write(root/'active.json',{'output':str(output),'request':str(req),'controller_pid':os.getpid(),
                'controller_identity':process_identity(os.getpid()),'worker_pid':process.pid,'worker_identity':process_identity(process.pid)})
            while process.poll() is None:
                stopped(root)
                time.sleep(1)
        except BaseException:
            # Only this worker's private process group is terminated.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            if not (output/'status.json').exists():
                write(output/'status.json',{'status':'interrupted','error_type':'InterruptedError'})
            raise
    result=read(output/'status.json') if (output/'status.json').exists() else {'status':'running'}
    result=exit_state(result,process.returncode or (1 if result.get('status')=='running' else 0))
    if result.get('error_type')=='UnclassifiedWorkerExit':
        write(output/'status.json',result)
    event(root,'worker_end',output=str(output),returncode=process.returncode,status=result.get('status'),error_type=result.get('error_type'))
    (root/'active.json').unlink(missing_ok=True)
    return result


def dispatch(root, task, attempt, catalog):
    validate_binding(root,catalog)
    output=root/'jobs'/task/f'attempt-{attempt:02d}'
    request={'kind':'diagnostic','group':task.split('-')[1],'seed':7,'output':str(output.resolve()),
             'data_dir':catalog['data_dir'],'config':catalog['config'],'catalog_sha256':sha(root/'catalog.json'),
             'expected_environment':catalog['environment'],'expected_provenance':catalog['provenance']}
    return launch(root,output,request)


def retryable(state):
    return state.get('error_type')=='InfrastructureError' and state.get('status')!='interrupted'


def check_anchor(root,task,path):
    if not (root/'outcomes.json').exists():return
    previous=read(root/'outcomes.json').get(task,{})
    if previous.get('status')=='complete':
        if previous['path']!=path.relative_to(root).as_posix() or previous['artifacts_sha256']!=sha(path/'artifacts.json'):
            raise ValueError('Previously anchored job changed; refusing to replace digest')


def recover_running(root,path,state):
    if not (root/'active.json').exists():raise RuntimeError(f'Incomplete attempt has no lifecycle evidence: {path}')
    active=read(root/'active.json')
    if Path(active['output']).resolve()!=path.resolve():raise RuntimeError('Active record identifies another attempt')
    for key in ('worker_identity','controller_identity'):
        identity=active.get(key)
        if identity and process_identity(identity['pid'])==identity:raise RuntimeError('Prior controller/worker is still alive; refusing duplicate')
    state={**state,'status':'interrupted','error_type':'UnclassifiedControllerExit','error':'Controller/worker no longer alive; cause not classified, no automatic retry'}
    write(path/'status.json',state);event(root,'recovered_unclassified_exit',path=str(path))
    return state


def verify_job(root, task, path, catalog):
    verify_seal(path)
    state,manifest=read(path/'status.json'),read(path/'manifest.json')
    if state['status']!='complete' or state['epochs_completed']!=100 or state['validation_count']!=10:
        raise ValueError('Diagnostic did not complete 100 epochs / 10 validations')
    if manifest['group']!=task.split('-')[1] or manifest['seed']!=7 or manifest['kind']!='diagnostic':raise ValueError('Task identity mismatch')
    if manifest['config']!=catalog['config'] or manifest['provenance']!=catalog['provenance'] or manifest['environment']!=catalog['environment']:
        raise ValueError('Diagnostic source/config/environment mismatch')
    if manifest['freeze_sha256']!=sha(root/'catalog.json'):raise ValueError('Diagnostic catalog binding mismatch')
    if (path/'test_metrics.json').exists():raise ValueError('Diagnostic accessed test output')
    for epoch in range(10,101,10):
        metrics=read(path/'validation'/f'epoch-{epoch:04d}.json')
        if any(len(metrics['ranks'][side])!=17535 for side in ('head','tail')):raise ValueError('Incomplete validation ranks')
    if len(read(path/'batch_order.json'))!=100:raise ValueError('Incomplete positive-batch provenance')
    with (path/'epochs.csv').open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=100 or any(int(row['examples'])!=272115 for row in rows):raise ValueError('Incomplete training epoch coverage')
    return state


def run_one(root, task, catalog):
    root=Path(root)
    if task not in TASKS:raise ValueError('Only D-A-7, D-B-7, D-C-7 are authorized')
    stopped(root)
    last=None
    for attempt in (1,2):
        path=root/'jobs'/task/f'attempt-{attempt:02d}'
        if path.exists():
            last=read(path/'status.json')
            if last['status']=='complete':
                check_anchor(root,task,path)
                verify_job(root,task,path,catalog)
                return {**last,'path':path.relative_to(root).as_posix(),'artifacts_sha256':sha(path/'artifacts.json')}
            if last['status']=='running':last=recover_running(root,path,last)
            if not retryable(last):return {**last,'path':path.relative_to(root).as_posix()}
            continue
        if attempt==2 and not retryable(last or {}):break
        stopped(root)
        last=dispatch(root,task,attempt,catalog)
        if last['status']=='complete':
            verify_job(root,task,path,catalog)
            return {**last,'path':path.relative_to(root).as_posix(),'artifacts_sha256':sha(path/'artifacts.json')}
        if not retryable(last):break
    return {**(last or {'status':'failed'}),'path':path.relative_to(root).as_posix()}


def preflight(root,catalog):
    if (root/'preflight.json').exists():
        result=read(root/'preflight.json')
        if result['status']!='passed':raise ValueError('Previous preflight incomplete; preserve version and investigate')
        for row in result['attempts']:
            if sha(root/row['path']/'artifacts.json')!=row['artifacts_sha256']:raise ValueError('Preflight artifact anchor changed')
            verify_seal(root/row['path'])
        return result
    from .protocol import Config
    attempts=[]
    for size in (5000,2500,1000):
        passed=True
        for group in ('A','B','C'):
            cfg=replace(Config(),protocol='formal-abc-v2-diagnostic',batch_size=size,max_epochs=100)
            output=root/'preflight'/f'bs{size}-{group}'
            request={'kind':'probe','group':group,'seed':7,'output':str(output.resolve()),'data_dir':catalog['data_dir'],'config':cfg.to_dict(),'expected_provenance':catalog['provenance']}
            if output.exists():
                verify_seal(output);state=read(output/'status.json')
            else:state=launch(root,output,request)
            attempts.append({**state,'path':output.relative_to(root).as_posix(),'artifacts_sha256':sha(output/'artifacts.json') if (output/'artifacts.json').exists() else None})
            if state['status'] in ('oom','memory_margin_exceeded'):passed=False;break
            if state['status']!='passed':
                write(root/'preflight.json',{'status':'failed','attempts':attempts});raise RuntimeError(f'Preflight failed: {state}')
        if passed:
            result={'status':'passed','config':cfg.to_dict(),'environment':read(output/'environment.json'),'attempts':attempts}
            write(root/'preflight.json',result);return result
    write(root/'preflight.json',{'status':'failed','attempts':attempts})
    raise RuntimeError('No common batch size passed')


def run(root,data_dir,only=None):
    root=Path(root).resolve();catalog=ensure_run(root,data_dir)
    with lock(root):
        try:
            if sha(root/'catalog.json')!=read(root/'catalog_seal.json')['sha256']:raise ValueError('Catalog modified')
            if 'config' not in catalog:
                probe=preflight(root,catalog)
                catalog.update(config=probe['config'],environment=probe['environment'])
                write(root/'catalog.json',catalog);write(root/'catalog_seal.json',{'sha256':sha(root/'catalog.json')})
            outcomes=read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
            for task in ((only,) if only else TASKS):
                stopped(root)
                write(root/'status.json',{'status':'running','stage':'diagnostic','task':task,'pid':os.getpid()})
                outcomes[task]=run_one(root,task,catalog)
                write(root/'outcomes.json',outcomes)
            from .diagnostic_report import analyze
            summary=analyze(root)
            status='awaiting_review' if summary['complete'] else 'incomplete'
            write(root/'status.json',{'status':status,'stage':'diagnostic_report','formal_execution_allowed':False,'finished_unix':time.time()})
            event(root,'stopped_at_diagnostic_boundary',complete=summary['complete'])
            return summary
        except BaseException as exc:
            write(root/'status.json',{'status':'interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed','error_type':type(exc).__name__,'error':str(exc),'formal_execution_allowed':False})
            event(root,'controller_exit',error=repr(exc))
            raise


def validate_mapping(path):
    with Path(path).open() as stream:
        count=int(stream.readline());rows=[line.split() for line in stream if line.strip()]
    if len(rows)!=count or any(len(row)!=2 for row in rows):raise ValueError('Mapping length/columns mismatch')
    if len({row[0] for row in rows})!=count or {int(row[1]) for row in rows}!=set(range(count)):raise ValueError('Mapping must be bijective')


def worker(request):
    arm_parent_death(request['controller_pid'])
    import torch
    from . import data,runtime,protocol as p
    output=Path(request['output']);cfg=p.Config(**request['config'])
    torch.set_num_threads(10)
    if request['kind']=='diagnostic':
        if request['group'] not in ('A','B','C') or cfg.max_epochs!=100 or cfg.validation_every!=10 or request['seed']!=7:
            raise ValueError('Only frozen diagnostic tasks are accepted')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the actual worker')
        if source_identity(request['data_dir'])!=request['expected_provenance']:raise ValueError('Worker source/data mismatch')
        environment=p.environment()
        if request.get('expected_environment') and environment!=request['expected_environment']:raise ValueError('Worker environment changed')
        for name in ('entity2id.txt','relation2id.txt'):validate_mapping(Path(request['data_dir'])/name)
        if request['kind']=='diagnostic':
            return runtime.run_job(output,request['data_dir'],cfg,request['group'],7,'diagnostic','cuda',freeze_hash=request['catalog_sha256'])
        if request['kind']!='probe':raise ValueError('No formal passes allowed')
        output.mkdir(parents=True,exist_ok=False)
        write(output/'request.json',request);write(output/'environment.json',environment)
        ds=data.load(request['data_dir'])
        model,opt,sampler=runtime.build(ds,cfg,request['group'],7,'cuda')
        batch=next(data.batches(ds['train'],cfg.batch_size,7,0))
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):loss,_=runtime.step(model,opt,sampler,batch,cfg,'cuda')
        torch.cuda.synchronize()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite preflight loss')
        reserved=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
        state={'status':'passed' if reserved<.9*total else 'memory_margin_exceeded','group':request['group'],
               'batch_size':cfg.batch_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':reserved,'total_bytes':total}
        write(output/'status.json',state);seal(output);return state
    except BaseException as exc:
        output.mkdir(parents=True,exist_ok=True)
        if request['kind']=='probe':
            state={'status':'oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'failed','error_type':type(exc).__name__,'error':repr(exc)}
            write(output/'status.json',state);(output/'error.log').write_text(traceback.format_exc());seal(output)
            return state
        if not (output/'status.json').exists():
            write(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':repr(exc)})
        raise


def status(root):
    root=Path(root)
    result=read(root/'status.json')
    if (root/'active.json').exists():
        active=read(root/'active.json');path=Path(active['output'])/'status.json'
        result['active']={**active,'progress':read(path) if path.exists() else 'starting'}
    if (root/'outcomes.json').exists():result['outcomes']=read(root/'outcomes.json')
    return result


def main(argv):
    parser=argparse.ArgumentParser(description='ABC diagnostic-only queue')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run-diagnostics','run-task'):
        cmd=sub.add_parser(name);cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--data',type=Path,default=ROOT/'src/py/data/FB15K237')
        if name=='run-task':cmd.add_argument('--task',choices=TASKS,required=True)
    cmd=sub.add_parser('status');cmd.add_argument('--run',type=Path,required=True)
    cmd=sub.add_parser('_abc-worker');cmd.add_argument('--request',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='status':result=status(args.run)
    else:
        def interrupt(signum,frame):
            global STOPPED
            STOPPED=True
            raise InterruptedError(f'User signal {signum}')
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        if args.command=='_abc-worker':result=worker(read(args.request))
        else:result=run(args.output,args.data,getattr(args,'task',None))
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
