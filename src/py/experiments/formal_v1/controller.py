"""Serial independent-process experiments and evidence-only analysis."""
from dataclasses import replace
from pathlib import Path
import csv,json,os,signal,statistics,subprocess,sys,time
import torch
from . import protocol as p,runtime,data
ROOT=Path(__file__).resolve().parents[4]

def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    others=[int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
    if others:raise RuntimeError(f'GPU already in use by processes {others}; no experiment dispatched')
    return result.stdout

def matrix(kind):
    if kind=='diagnostic':return [(g,7,'diagnostic') for g in ('cpu','balanced','tail')]
    if kind=='quality':seeds=[101,102,103];passes=['quality']
    elif kind=='timing':seeds=[201,202,203,204,205];passes=['throughput','trace']
    else:raise ValueError(kind)
    return [(g,s,mode) for mode in passes for i,s in enumerate(seeds) for g in (('cpu','balanced') if i%2==0 else ('balanced','cpu'))]

def read_preflight(path,prov):
    result=json.loads(Path(path).read_text())
    if result.get('status')!='passed' or result['provenance']!=prov or result.get('environment')!=p.environment():raise ValueError('Passing preflight for current source/data required')
    return p.Config(**result['config'])

def dispatch(job_root,data_dir,cfg,group,seed,kind,freeze_hash=None,timeout=86400):
    job_root=Path(job_root);request=job_root.with_suffix('.request.json');request.parent.mkdir(parents=True,exist_ok=True)
    if request.exists() or job_root.exists():raise FileExistsError(job_root)
    p.write_json(request,{'root':str(job_root.resolve()),'data':str(Path(data_dir).resolve()),'config':cfg.to_dict(),'group':group,'seed':seed,'kind':kind,'freeze_sha256':freeze_hash})
    cmd=[sys.executable,'-B','-m','src.py.experiments.formal_v1','_worker','--request',str(request.resolve())]
    with job_root.with_suffix('.stdout.log').open('w') as out,job_root.with_suffix('.stderr.log').open('w') as err:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,env={**os.environ,'PYTHONHASHSEED':str(seed),'PYTHONDONTWRITEBYTECODE':'1'})
        try:code=process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if not (job_root/'status.json').exists():raise RuntimeError(f'Worker exit {code} without status; see {job_root}.stderr.log')
    result=json.loads((job_root/'status.json').read_text())
    if code or result['status'] not in ('complete','passed','oom'):raise RuntimeError(f'Worker {job_root.name}: {result}')
    if result['status']=='complete':result['artifacts_sha256']=p.sha(job_root/'artifacts.json')
    return result

def probe(request):
    root=Path(request['root']);cfg=p.Config(**request['config']);group=request['group']
    p.new_run(root,request)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this execution environment')
        ds=data.load(request['data']);model,opt,sampler=runtime.build(ds,cfg,group,7,'cuda')
        batch=next(data.batches(ds['train'],cfg.batch_size,7,0));torch.cuda.reset_peak_memory_stats()
        for _ in range(2):loss,_=runtime.step(model,opt,sampler,batch,cfg,'cuda')
        torch.cuda.synchronize();reserved=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite preflight loss')
        result={'status':'passed' if reserved<.9*total else 'oom','group':group,'batch_size':cfg.batch_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':reserved,'total_bytes':total}
    except torch.cuda.OutOfMemoryError as exc:result={'status':'oom','group':group,'batch_size':cfg.batch_size,'error':str(exc)}
    except BaseException as exc:p.write_json(root/'status.json',{'status':'failed','error':repr(exc)});raise
    p.write_json(root/'status.json',result);return result

def preflight(root,data_dir):
    root=Path(root);prov=p.provenance(data_dir);p.new_run(root,{'provenance':prov,'environment':p.environment()})
    attempts=[]
    try:
        for size in (5000,2500,1000):
            cfg=replace(p.Config(),batch_size=size);passed=True
            for group in ('cpu','balanced','tail'):
                gpu_idle();result=dispatch(root/f'bs{size}_{group}',data_dir,cfg,group,7,'probe',timeout=600);attempts.append(result)
                if result['status']!='passed':passed=False;break
            if passed:
                value={'status':'passed','config':cfg.to_dict(),'provenance':prov,'attempts':attempts,'environment':p.environment()}
                p.write_json(root/'preflight.json',value);p.write_json(root/'status.json',{'status':'complete'});return value
        raise RuntimeError('No common batch size passed preflight')
    except BaseException as exc:p.write_json(root/'status.json',{'status':'failed','attempts':attempts,'error':repr(exc)});raise

def run_suite(root,data_dir,kind,preflight_path=None,frozen_path=None):
    root=Path(root);prov=p.provenance(data_dir)
    frozen=p.validate_frozen(frozen_path,prov) if kind!='diagnostic' else None
    cfg=p.Config(**frozen['config']) if frozen else read_preflight(preflight_path,prov)
    p.new_run(root,{'kind':kind,'config':cfg.to_dict(),'provenance':prov,'frozen':frozen})
    value={'status':'running','kind':kind,'config':cfg.to_dict(),'provenance':prov,'environment':p.environment(),'jobs':[],'frozen':frozen}
    p.write_json(root/'suite.json',value)
    try:
        for group,seed,mode in matrix(kind):
            if p.provenance(data_dir)!=prov:raise RuntimeError('Source or input changed during suite')
            gpu_idle();job=root/'jobs'/f'{mode}_{group}_{seed}'
            state=dispatch(job,data_dir,cfg,group,seed,mode,frozen['freeze_sha256'] if frozen else None,timeout=21600 if kind=='diagnostic' else 86400)
            state['path']=job.relative_to(root).as_posix();value['jobs'].append(state);p.write_json(root/'suite.json',value)
        value['status']='complete'
        if kind=='diagnostic':value['budget']=p.budget(value)
    except BaseException as exc:value.update(status='failed',error=repr(exc));raise
    finally:
        p.write_json(root/'suite.json',value);p.write_json(root/'status.json',{'status':value['status']})
    return value

def verify_job(root,expected_hash=None):
    root=Path(root);inventory=root/'artifacts.json'
    if expected_hash is not None and p.sha(inventory)!=expected_hash:raise ValueError('Artifact inventory changed')
    hashes=json.loads(inventory.read_text())
    required={'manifest.json','status.json','epochs.csv','initial_weights.json'}
    if not isinstance(hashes,dict) or not required<=set(hashes):raise ValueError('Required artifacts missing')
    actual={f.name for f in root.iterdir() if f.is_file() and f.name!='artifacts.json'}
    if actual!=set(hashes):raise ValueError('Artifact inventory does not match files')
    for name,digest in hashes.items():
        if Path(name).name!=name or (root/name).is_symlink():raise ValueError('Unsafe artifact path')
        if p.sha(root/name)!=digest:raise ValueError(f'Artifact changed: {root/name}')
    state=json.loads((root/'status.json').read_text());kind=state['kind']
    if kind in ('diagnostic','quality'):required|={'validation.csv','best.pt'}
    if kind=='quality':required.add('test_metrics.json')
    if kind=='trace':required.add('steps.csv')
    if not required<=set(hashes):raise ValueError('Pass-specific artifacts missing')
    return state

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def analyze(root):
    root=Path(root);suite=json.loads((root/'suite.json').read_text());kind=suite['kind']
    result={'kind':kind,'execution_status':suite['status'],'verdict':'INCOMPLETE','quality_target_supported':False,'quality_target_point_estimate':False,'jobs':len(suite['jobs'])}
    if suite['status']!='complete' or len(suite['jobs'])!=len(matrix(kind)):return result
    states={}
    for row in suite['jobs']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe job path')
        if not row.get('artifacts_sha256'):raise ValueError('Unanchored job artifact inventory')
        path=root/rel;s=verify_job(path,row['artifacts_sha256']);manifest=json.loads((path/'manifest.json').read_text())
        if {k:v for k,v in row.items() if k not in ('path','artifacts_sha256')}!=s:raise ValueError('Summary differs from sealed job status')
        expected_freeze=suite.get('frozen',{}).get('freeze_sha256') if suite.get('frozen') else None
        if (kind!='diagnostic' and not expected_freeze) or manifest.get('freeze_sha256')!=expected_freeze or s.get('freeze_sha256')!=expected_freeze:raise ValueError('Job frozen-protocol binding mismatch')
        if manifest.get('environment')!=suite.get('environment'):raise ValueError('Job environment differs from suite')
        epochs=read_csv(path/'epochs.csv')
        if len(epochs)!=s['epochs_completed'] or abs(sum(int(x['epoch_time_ns'])/1e9 for x in epochs)-s['training_seconds'])>1e-6:raise ValueError('Recorded training duration mismatch')
        if kind in ('diagnostic','quality'):
            validation=read_csv(path/'validation.csv')
            if len(validation)!=s['validation_count'] or abs(sum(float(x['evaluation_seconds']) for x in validation)-s['validation_seconds'])>1e-6:raise ValueError('Recorded validation duration mismatch')
        if s['status']!='complete' or manifest['provenance']!=suite['provenance'] or manifest['config']!=suite['config']:raise ValueError('Job provenance/config mismatch')
        key=s['group'],s['seed'],s['kind']
        if key in states:raise ValueError('Duplicate paired job')
        states[key]=(path,s)
    if set(states)!=set(matrix(kind)):raise ValueError('Expected matrix not complete')
    for seed in {key[1] for key in states}:
        initial=[json.loads((path/'initial_weights.json').read_text()) for key,(path,_) in states.items() if key[1]==seed]
        if any(x!=initial[0] for x in initial):raise ValueError('Paired initial weights differ')
    result['verdict']='DESCRIPTIVE'
    if kind=='quality':
        ratios=[];details=[];tests=set()
        for seed in (101,102,103):
            metrics={}
            for group in ('cpu','balanced'):
                path,state=states[group,seed,'quality'];metrics[group]=json.loads((path/'test_metrics.json').read_text());tests.add(state['test_sha256'])
            ratio=metrics['balanced']['overall']['mrr']/metrics['cpu']['overall']['mrr'];ratios.append(ratio);details.append({'seed':seed,'metrics':metrics,'ratio':ratio})
        if len(tests)!=1:raise ValueError('Test splits changed across jobs')
        primary=p.paired_summary(ratios);result.update(primary=primary,pairs=details,quality_target_point_estimate=primary['geometric_mean']>=.95,quality_target_supported=primary['ci95'][0]>=.95)
        result['verdict']='QUALITY_TARGET_SUPPORTED' if result['quality_target_supported'] else 'QUALITY_LOSS_SUPPORTED' if primary['ci95'][1]<.95 else 'INCONCLUSIVE'
    elif kind=='timing':
        ratios=[];dispersion=[];components=[]
        for seed in range(201,206):
            epochs={};sds={};means={}
            for group in ('cpu','balanced'):
                path,_=states[group,seed,'throughput'];epochs[group]=statistics.mean(int(x['epoch_time_ns']) for x in read_csv(path/'epochs.csv'))
                path,_=states[group,seed,'trace'];rows=[x for x in read_csv(path/'steps.csv') if x['is_partial']=='False' and int(x['batch_size'])==suite['config']['batch_size']]
                means[group]=statistics.mean(int(x['sampling_ns']) for x in rows)
                sds[group]=statistics.mean(statistics.pstdev(int(x['sampling_ns']) for x in rows if x['epoch']==e) for e in sorted({x['epoch'] for x in rows}))
            ratios.append(epochs['cpu']/epochs['balanced']);components.append(means['cpu']/means['balanced'])
            if sds['balanced']>0:dispersion.append(sds['cpu']/sds['balanced'])
        result.update(epoch_speedup=p.paired_summary(ratios),sampling_speedup=p.paired_summary(components),sd_compression=p.paired_summary(dispersion) if len(dispersion)==5 and min(dispersion)>0 else None)
    else:result['budget']=p.budget(suite)
    return result

def export_analysis(root,output):
    output=Path(output);value=analyze(root);output.mkdir(parents=True,exist_ok=False);p.write_json(output/'summary.json',value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suite=json.loads((Path(root)/'suite.json').read_text());fig,ax=plt.subplots()
    for row in suite['jobs']:
        file=Path(root)/row['path']/'validation.csv'
        if file.exists():
            rows=read_csv(file);ax.plot([float(x['training_seconds']) for x in rows],[float(x['mrr']) for x in rows],label=f"{row['group']} seed {row['seed']}")
    ax.set(xlabel='Training time (s), excluding evaluation',ylabel='Validation filtered MRR')
    if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(output/'validation_vs_training_time.png');plt.close(fig)
    (output/'report.md').write_text('# Formal v1 analysis\n\nExecution: '+value['execution_status']+'\n\nVerdict: '+value['verdict']+'\n\nSee summary.json for paired effects and uncertainty. Quality and speed are separate evidence objects; a quality result alone does not establish useful acceleration.\n')
    return value
