"""Persistent mechanism-only task controller. Status imports only the standard library."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from . import diagnostics as d

PROTOCOL='mechanism-v1'
DATA=d.ROOT/'src/py/data/FB15K237'
REFERENCE=d.ROOT/'output/results/formal_abc_v2/formal-20261106-01'
GROUPS=('M00','M10','M01','M11')
DOCUMENTS=('docs/superpowers/plans/2026-11-08-sampling-mechanisms-execution.md',
           'docs/superpowers/specs/2026-11-08-sampling-mechanisms-design.md',
           'docs/formal_v1/RESULTS_REFLECTION_2026-11-08.md',
           'docs/formal_v1/MECHANISM_EXECUTION_LEDGER.md',
           'docs/formal_v1/MECHANISM_USAGE.md',
           'docs/formal_v1/MECHANISM_CODE_REVIEW_2026-11-08.md')


def config():
    from .protocol import Config
    return Config(protocol=PROTOCOL,max_epochs=1000,batch_size=5000,validation_every=10)


def task_catalog():
    result=[{'id':'O-A','kind':'observe_a','group':'A','seed':601,'dependencies':[]},
            {'id':'O-TENSOR','kind':'observe','group':'tensor','seed':601,'dependencies':[]}]
    for seed,order in zip((401,402,403),(('M00','M10','M11','M01'),
                                       ('M10','M01','M00','M11'),('M01','M11','M10','M00'))):
        result.extend({'id':f'V-{group}-{seed}','kind':'validation','group':group,'seed':seed,
                       'dependencies':['O-A','O-TENSOR']} for group in order)
    return result


def environment():
    from .formal import environment as collect
    return collect()


def check(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source_before=d.source_identity(DATA)['sources']
    commands=[]
    for pattern in ('test_mechanism_*.py','test_formal_v1*.py'):
        command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern]
        name=pattern.replace('*','all')+'.log'
        with (output/name).open('w') as stream:
            result=subprocess.run(command,cwd=d.ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        commands.append({'command':command,'returncode':result.returncode,'log':name})
        if result.returncode:
            d.write(output/'verification.json',{'status':'failed','commands':commands})
            raise RuntimeError(f'CPU checks failed: {output/name}')
    source_after=d.source_identity(DATA)['sources']
    if source_before!=source_after:
        raise ValueError('Source changed during CPU checks')
    receipt={'status':'passed','sources':source_after,'commands':commands,
             'test_sources':{p.name:d.sha(p) for p in sorted((d.ROOT/'tests').glob('test_mechanism_*.py'))}}
    (output/'test_sources').mkdir()
    for path in sorted((d.ROOT/'tests').glob('test_*formal_v1*.py'))+sorted((d.ROOT/'tests').glob('test_mechanism_*.py')):
        shutil.copy2(path,output/'test_sources'/path.name)
    d.write(output/'verification.json',receipt)
    d.seal(output)
    return receipt


def _copy_source(output):
    source=Path(__file__).parent
    for path in sorted(source.rglob('*.py')):
        destination=output/path.relative_to(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    return d.seal(output)


def _copy_data(source, output):
    before=d.source_identity(source)['data']
    output.mkdir(parents=True)
    for name in before:
        shutil.copy2(Path(source)/name,output/name)
    if d.source_identity(output)['data']!=before or d.source_identity(source)['data']!=before:
        raise ValueError('Input changed during snapshot')
    return before


def resource_launch(root, output, request):
    from .formal import gpu_lock
    while True:
        d.wait_gpu(root)
        try:
            with gpu_lock():
                return d.launch(root,output,request,worker_command='_mechanism-worker',wait_for_gpu=False)
        except RuntimeError as exc:
            if str(exc) not in ('GPU already has a compute process','Another formal task or preflight holds this GPU'):
                raise
            d.event(root,'waiting_resource_race',reason=str(exc))
            time.sleep(10)


def prepare(checks, output, data_dir=DATA):
    checks,output=Path(checks).resolve(),Path(output).resolve()
    d.verify_seal(checks)
    acceptance=d.read(checks/'verification.json')
    if acceptance['status']!='passed' or acceptance['sources']!=d.source_identity(data_dir)['sources']:
        raise ValueError('CPU acceptance no longer matches source')
    if acceptance['test_sources']!={p.name:d.sha(p) for p in sorted((d.ROOT/'tests').glob('test_mechanism_*.py'))}:
        raise ValueError('Mechanism checks changed since CPU acceptance')
    if any(output.is_relative_to(path) for path in (checks,Path(data_dir).resolve(),Path(__file__).parent)):
        raise ValueError('Preparation output overlaps inputs')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(checks,output/'checks')
    _copy_data(data_dir,output/'inputs/FB15K237')
    data_path=(output/'inputs/FB15K237').resolve()
    for name in ('entity2id.txt','relation2id.txt'):
        d.validate_mapping(data_path/name)
    with d.lock(output):
        observed_environment=environment()
        # Execute real CUDA tests under the same cross-protocol GPU lock as workers.
        from .formal import gpu_lock
        d.wait_gpu(output)
        with gpu_lock(),(output/'cuda-checks.log').open('w') as log:
            command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_mechanism_cuda.py']
            result=subprocess.run(command,cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  env={**os.environ,'OMP_NUM_THREADS':'10','MKL_NUM_THREADS':'10','PYTHONDONTWRITEBYTECODE':'1'})
        cuda_log=(output/'cuda-checks.log').read_text()
        if result.returncode or 'skipped=' in cuda_log or 'Ran 0 tests' in cuda_log:
            raise RuntimeError('Actual CUDA checks failed or were skipped')
        provenance=d.source_identity(data_path)
        results=[]
        for group in GROUPS:
            task={'id':f'P-{group}-900','kind':'probe','group':group,'seed':900}
            target=output/'preflight'/task['id']/'attempt-01'
            request={'phase':'prepare','task':task,'kind':'probe','group':group,'seed':900,
                     'output':str(target),'data_dir':str(data_path),'config':config().to_dict(),
                     'expected_environment':observed_environment,'expected_provenance':provenance,
                     'binding':{'environment':observed_environment,'provenance':provenance}}
            result=resource_launch(output,target,request)
            if result['status']!='complete':
                d.write(output/'preparation.json',{'status':'failed','task':task,'result':result})
                raise RuntimeError(f'Preflight failed: {group}')
            d.verify_seal(target)
            results.append({'task':task,**result,'path':target.relative_to(output).as_posix(),
                            'artifacts_sha256':d.sha(target/'artifacts.json')})
        if d.source_identity(data_path)!=provenance or environment()!=observed_environment:
            raise ValueError('Preparation source/input/environment changed')
        budget={row['group']:{'measured_epoch_seconds':row['mean_epoch_seconds'],
                              'measured_validation_seconds':row['validation_seconds'],
                              'projected_validation_job_seconds':1000*row['mean_epoch_seconds']+100*row['validation_seconds'],
                              'overhead':'startup/checkpoint/logging to be refined from completed tasks'} for row in results}
        projected=3*sum(item['projected_validation_job_seconds'] for item in budget.values())
        receipt={'status':'passed','protocol':PROTOCOL,'config':config().to_dict(),
                 'environment':observed_environment,'provenance':provenance,'preflights':results,
                 'budget':{'per_group':budget,'projected_training_validation_hours':projected/3600,
                           'with_30_percent_reserve_hours':1.3*projected/3600,
                           'unknown':'O/O-A, additional saves, analysis and archive overhead'},
                 'cpu_checks_sha256':d.sha(output/'checks/artifacts.json'),
                 'cuda_checks_sha256':d.sha(output/'cuda-checks.log')}
        d.write(output/'preparation.json',receipt)
    d.seal(output)
    return receipt


def freeze(preparation, output, reference=REFERENCE):
    from .protocol import digest
    preparation,output,reference=(Path(p).resolve() for p in (preparation,output,reference))
    d.verify_seal(preparation)
    ready=d.read(preparation/'preparation.json')
    if ready['status']!='passed' or ready['config']!=config().to_dict():
        raise ValueError('Preparation is not a matching passed protocol')
    if d.source_identity(preparation/'inputs/FB15K237')!=ready['provenance'] or environment()!=ready['environment']:
        raise ValueError('Source/input/environment changed since preparation')
    if any(output.is_relative_to(p) for p in (preparation,reference,DATA,Path(__file__).parent)):
        raise ValueError('Frozen output overlaps an input')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(preparation,output/'preparation')
    shutil.copytree(preparation/'inputs',output/'inputs')
    anchors=[]
    for group,epoch in (('A',180),('C',1000)):
        source=reference/f'jobs/Q-{group}-101/attempt-01'
        inventory=d.read(source/'artifacts.json')
        expected=inventory['best.pt']
        expected=expected['sha256'] if isinstance(expected,dict) else expected
        outcomes=d.read(reference/'outcomes.json')
        if outcomes[f'Q-{group}-101']['artifacts_sha256']!=d.sha(source/'artifacts.json') or d.sha(source/'best.pt')!=expected:
            raise ValueError('Historical anchor differs from closed task identity')
        destination=output/'inputs/anchors'/f'{group}-101.pt'
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source/'best.pt',destination)
        import torch
        checkpoint=torch.load(destination,map_location='cpu',weights_only=True)
        if checkpoint['epoch']!=epoch:
            raise ValueError('Historical anchor epoch mismatch')
        anchors.append({'id':f'{group}-101','relative_path':destination.relative_to(output).as_posix(),
                        'sha256':expected,'epoch':epoch})
    source_inventory=_copy_source(output/'source')
    for name in DOCUMENTS:
        destination=output/'documents'/Path(name).name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(d.ROOT/name,destination)
    d.write(output/'task_catalog.json',task_catalog())
    write_cards(output,ready['budget'])
    body={'protocol':PROTOCOL,'config':ready['config'],'provenance':ready['provenance'],
          'environment':ready['environment'],'budget':ready['budget'],'anchors':anchors,
          'reference_freeze_sha256':d.read(reference/'freeze.json')['freeze_sha256'],
          'source_inventory_sha256':source_inventory,'input_inventory_sha256':d.seal(output/'inputs'),
          'preparation_inventory_sha256':d.sha(output/'preparation/artifacts.json'),
          'documents_inventory_sha256':d.seal(output/'documents'),
          'tasks_sha256':d.sha(output/'task_catalog.json'),'task_cards_sha256':d.sha(output/'TASKS.md'),
          'created_unix':time.time()}
    d.write(output/'freeze.json',{**body,'freeze_sha256':digest(body)})
    d.write(output/'status.json',{'status':'frozen_awaiting_start','completed':0,'tasks':14})
    return read_freeze(output)


def read_freeze(root):
    from .protocol import digest
    root=Path(root)
    record=d.read(root/'freeze.json')
    body={key:value for key,value in record.items() if key!='freeze_sha256'}
    if record['protocol']!=PROTOCOL or digest(body)!=record['freeze_sha256']:
        raise ValueError('Invalid mechanism freeze identity')
    if record['config']!=config().to_dict() or d.read(root/'task_catalog.json')!=task_catalog() or d.sha(root/'task_catalog.json')!=record['tasks_sha256']:
        raise ValueError('Mechanism task/configuration changed')
    if d.sha(root/'TASKS.md')!=record['task_cards_sha256']:
        raise ValueError('Mechanism task cards changed')
    for name,field in (('source','source_inventory_sha256'),('inputs','input_inventory_sha256'),
                       ('preparation','preparation_inventory_sha256'),('documents','documents_inventory_sha256')):
        if d.sha(root/name/'artifacts.json')!=record[field]:
            raise ValueError(f'Changed frozen {name} inventory')
        d.verify_seal(root/name)
    return record


def write_cards(root,budget):
    import shlex
    lines=['# 机制实验独立任务卡','','协议为 mechanism-v1，完成单任务即退出。','']
    for task in task_catalog():
        estimate='观测耗时尚无实测。'
        if task['kind']=='validation':
            seconds=budget['per_group'][task['group']]['projected_validation_job_seconds']
            estimate=f'训练和验证外推 {seconds/3600:.2f} 小时，含30%余量 {1.3*seconds/3600:.2f} 小时；保存等额外开销另计。'
        lines.extend([f'## {task["id"]}','','种类 '+task['kind']+'。'+estimate,
                      '依赖 '+(', '.join(task['dependencies']) or '已封存检查与输入')+'。',
                      f'材料位于 jobs/{task["id"]}/attempt-NN。','','```bash',
                      f'{shlex.quote(sys.executable)} -B -m src.py.experiments.formal_v1 mechanism run-task --run {shlex.quote(str(root))} --task {task["id"]}',
                      '```',''])
    (root/'TASKS.md').write_text('\n'.join(lines))


def verify_live(root,record):
    if d.source_identity(Path(root)/'inputs/FB15K237')!=record['provenance']:
        raise ValueError('Mechanism source or frozen data changed')
    if environment()!=record['environment']:
        raise ValueError('Mechanism environment changed')


def complete_outcome(root,path,state):
    return {**state,'path':path.relative_to(root).as_posix(),'artifacts_sha256':d.sha(path/'artifacts.json'),
            'sidecars':{name:d.sha(path.with_name(path.name+'.'+name)) for name in ('request.json','stdout.log','stderr.log')}}


def verify_result(root,task,outcome,record):
    from .mechanism_report import verify_job
    return verify_job(Path(root),task,outcome,record)


def dispatch(root,record,task,output):
    verify_live(root,record)
    anchors=[{**item,'path':str(root/item['relative_path'])} for item in record['anchors']]
    request={'phase':'frozen','run':str(root),'task':task,'kind':task['kind'],'group':task['group'],
             'seed':task['seed'],'output':str(output),'data_dir':str(root/'inputs/FB15K237'),
             'config':record['config'],'expected_environment':record['environment'],
             'expected_provenance':record['provenance'],'freeze_sha256':record['freeze_sha256'],
             'binding':{'environment':record['environment'],'provenance':record['provenance'],
                        'freeze_sha256':record['freeze_sha256'],'anchors':anchors}}
    return resource_launch(root,output,request)


def interruption(state):
    return state.get('status')=='interrupted' or state.get('error_type') in (
        'UnclassifiedWorkerExit','UnclassifiedControllerExit','UnsealedCompletion')


def run_one(root,record,task,resume=False):
    root=Path(root)
    d.stopped(root)
    directory=root/'jobs'/task['id']
    numbers=set()
    for path in directory.glob('attempt-*'):
        match=re.fullmatch(r'attempt-(\d+)(?:\.(?:request\.json|stdout\.log|stderr\.log))?',path.name)
        if match:
            numbers.add(int(match.group(1)))
    attempts=[directory/f'attempt-{number:02d}' for number in sorted(numbers)]
    infrastructure_failures=0
    last=None
    for path in attempts:
        if not (path/'status.json').exists():
            active=d.read(root/'active.json') if (root/'active.json').exists() else {}
            for field in ('worker_identity','controller_identity'):
                identity=active.get(field)
                if identity and d.process_identity(identity['pid'])==identity:
                    raise RuntimeError('Previous attempt process is still alive')
            path.mkdir(parents=True,exist_ok=True)
            d.write(path/'recovery.json',{'active_record':active,'recovered_unix':time.time()})
            d.write(path/'status.json',{'status':'interrupted','error_type':'UnclassifiedControllerExit',
                                      'error':'Reserved attempt interrupted before worker status was saved'})
            d.seal(path)
            d.event(root,'recovered_reserved_attempt',task=task['id'],path=str(path))
        state=d.read(path/'status.json')
        if state['status']=='complete' and not (path/'artifacts.json').exists():
            anchored=d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
            if anchored.get(task['id'],{}).get('status')=='complete':
                raise ValueError('Previously anchored complete inventory is missing')
            d.write(path/'recovery.json',{'prior_state':state,'recovered_unix':time.time()})
            state={**state,'status':'interrupted','error_type':'UnsealedCompletion'}
            d.write(path/'status.json',state)
            d.seal(path)
        if state['status']=='complete':
            outcome=complete_outcome(root,path,state)
            existing=d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
            if task['id'] in existing and existing[task['id']].get('status')=='complete' and existing[task['id']]!=outcome:
                raise ValueError('Previously anchored outcome changed')
            verify_result(root,task,outcome,record)
            return outcome
        if state['status']=='running':
            d.write(path/'recovery.json',{'prior_state':state,'active_record':d.read(root/'active.json') if (root/'active.json').exists() else {}})
            state=d.recover_running(root,path,state)
            d.seal(path)
        if state.get('error_type')=='InfrastructureError':
            infrastructure_failures+=1
        last={**state,'path':path.relative_to(root).as_posix()}
    if last and interruption(last) and not resume:
        return last
    if last and not interruption(last) and (last.get('error_type')!='InfrastructureError' or infrastructure_failures>=2):
        return last
    number=max([int(path.name.split('-')[1]) for path in attempts],default=0)+1
    while True:
        d.stopped(root)
        output=directory/f'attempt-{number:02d}'
        d.write(root/'status.json',{'status':'running','task':task['id'],'attempt':number,'pid':os.getpid()})
        result=dispatch(root,record,task,output)
        if result['status']=='complete':
            outcome=complete_outcome(root,output,result)
            verify_result(root,task,outcome,record)
            return outcome
        last={**result,'path':output.relative_to(root).as_posix()}
        if result.get('error_type')!='InfrastructureError':
            return last
        infrastructure_failures+=1
        if infrastructure_failures>=2:
            return last
        d.event(root,'retry_infrastructure',task=task['id'],previous_attempt=number)
        number+=1


def run(root,only=None,resume=False):
    root=Path(root).resolve()
    if (root/'artifacts.json').exists():
        raise FileExistsError('Closed mechanism run is immutable; use status or an external reanalysis directory')
    record=read_freeze(root)
    if only and only not in {task['id'] for task in task_catalog()}:
        raise ValueError('Task is outside the mechanism catalog')
    with d.lock(root):
        if (root/'STOP').exists():
            if not resume:
                raise InterruptedError('Explicit --resume required after user stop')
            (root/'STOP').unlink()
        if (root/'active.json').exists():
            active=d.read(root/'active.json')
            for field in ('worker_identity','controller_identity'):
                item=active.get(field)
                if item and d.process_identity(item['pid'])==item:
                    raise RuntimeError('Prior controller or worker is still alive')
        outcomes=d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
        tasks=[task for task in task_catalog() if not only or task['id']==only]
        try:
            for task in tasks:
                d.stopped(root)
                missing=[name for name in task['dependencies'] if outcomes.get(name,{}).get('status')!='complete']
                if missing:
                    outcomes[task['id']]={'status':'blocked','dependencies':missing}
                else:
                    for name in task['dependencies']:
                        dependency=next(item for item in task_catalog() if item['id']==name)
                        verify_result(root,dependency,outcomes[name],record)
                    outcomes[task['id']]=run_one(root,record,task,resume)
                d.write(root/'outcomes.json',outcomes)
                if interruption(outcomes[task['id']]):
                    raise InterruptedError(f'Explicit --resume required after interruption of {task["id"]}')
                if outcomes[task['id']].get('error_type') in ('OutOfMemoryError','MemoryError','FloatingPointError',
                                                             'ValueError','ArithmeticError','RuntimeError'):
                    raise RuntimeError('Scientific/preflight error requires investigation before further dispatch')
            complete=sum(item.get('status')=='complete' for item in outcomes.values())
            result={'status':'complete' if complete==14 else 'task_finished' if only else 'incomplete',
                    'completed':complete,'tasks':14,'finished_unix':time.time()}
            d.write(root/'status.json',result)
            return result
        except BaseException as exc:
            d.write(root/'status.json',{'status':'interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',
                                       'error_type':type(exc).__name__,'error':str(exc)})
            raise


def status(root):
    return d.status(root)


def worker(request):
    d.arm_parent_death(request['controller_pid'])
    if request['config']!=config().to_dict():
        raise ValueError('Worker configuration differs from mechanism protocol')
    if request['phase']=='frozen':
        root=Path(request['run']).resolve()
        record=read_freeze(root)
        expected=next((task for task in task_catalog() if task['id']==request['task']['id']),None)
        if request['task']!=expected or request['freeze_sha256']!=record['freeze_sha256']:
            raise ValueError('Worker task differs from frozen catalog')
        if Path(request['output']).resolve().parent!=root/'jobs'/expected['id']:
            raise ValueError('Worker output outside authorized task')
        anchors=[{**item,'path':str(root/item['relative_path'])} for item in record['anchors']]
        binding={'environment':record['environment'],'provenance':record['provenance'],
                 'freeze_sha256':record['freeze_sha256'],'anchors':anchors}
        if (Path(request['data_dir']).resolve()!=root/'inputs/FB15K237' or request['binding']!=binding or
                request['expected_provenance']!=record['provenance'] or request['expected_environment']!=record['environment']):
            raise ValueError('Worker input/environment/anchor binding differs from freeze')
        verify_live(root,record)
    elif request['phase']=='prepare':
        expected={'id':f'P-{request["group"]}-900','kind':'probe','group':request['group'],'seed':900}
        if request['group'] not in GROUPS or request['task']!=expected:
            raise ValueError('Unknown preparation task')
    else:
        raise ValueError('Unknown worker phase')
    if d.source_identity(request['data_dir'])!=request['expected_provenance'] or environment()!=request['expected_environment']:
        raise ValueError('Actual worker source/data/environment differs from request')
    from .mechanism_jobs import execute
    return execute(request['output'],request['data_dir'],config(),request['task'],request['binding'])


def main(argv):
    parser=argparse.ArgumentParser(description='Mechanism-only observations and validation matrix')
    commands=parser.add_subparsers(dest='command',required=True)
    cmd=commands.add_parser('check');cmd.add_argument('--output',type=Path,required=True)
    cmd=commands.add_parser('prepare');cmd.add_argument('--checks',type=Path,required=True);cmd.add_argument('--output',type=Path,required=True)
    cmd=commands.add_parser('freeze');cmd.add_argument('--preparation',type=Path,required=True);cmd.add_argument('--output',type=Path,required=True)
    for name in ('run-task','run-queue','status','stop','analyze'):
        cmd=commands.add_parser(name);cmd.add_argument('--run',type=Path,required=True)
        if name in ('run-task','run-queue'):cmd.add_argument('--resume',action='store_true')
        if name=='run-task':cmd.add_argument('--task',required=True)
        if name=='analyze':cmd.add_argument('--output',type=Path,required=True)
    cmd=commands.add_parser('_worker');cmd.add_argument('--request',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='status':
        result=status(args.run)
    elif args.command=='stop':
        if (args.run/'artifacts.json').exists():
            result={'status':'already_closed'}
        else:
            d.write(args.run/'STOP',{'requested_unix':time.time()})
            result={'status':'stop_requested'}
    else:
        def interrupt(signum,frame):
            d.STOPPED=True
            raise InterruptedError(f'User signal {signum}')
        signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
        if args.command=='check':result=check(args.output)
        elif args.command=='prepare':result=prepare(args.checks,args.output)
        elif args.command=='freeze':result=freeze(args.preparation,args.output)
        elif args.command in ('run-task','run-queue'):result=run(args.run,getattr(args,'task',None),args.resume)
        elif args.command=='_worker':result=worker(d.read(args.request))
        else:
            from .mechanism_report import analyze
            result=analyze(args.run,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(result.get('status') in ('failed','interrupted','incomplete'))
