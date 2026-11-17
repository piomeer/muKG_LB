"""One job per process; shared training, explicit sampling and timing passes."""
import csv,json,math,random,time,traceback,subprocess,os
from pathlib import Path
import numpy as np
import torch
from . import core,data,protocol as p

class InfrastructureError(RuntimeError):
    """Recorded external GPU interference, eligible for one fresh attempt."""

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)
def gpu_snapshot(root,epoch,phase):
    """Boundary telemetry is outside epoch timing; transient interference can escape it."""
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    device=subprocess.run(['nvidia-smi','--query-gpu=timestamp,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    others=[int(x.strip()) for x in processes.splitlines() if x.strip().isdigit() and int(x.strip()) not in (os.getpid(),os.getppid())]
    with (root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'phase':phase,'device':device,'compute_pids':processes.strip(),'other_pids':others})+'\n')
    if others:raise InfrastructureError(f'GPU interference from processes {others}')

def build(dataset,cfg,group,seed,device):
    seed_all(seed);model=core.TransE(dataset['entities'],dataset['relations'],cfg.dim,cfg.margin).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    known=set(dataset['train'])
    if group in ('cpu','A'):
        sampler=core.CPUSampler(dataset['entities'],cfg.negatives,known)
    else:
        mode='balanced' if group in ('B','C') else group
        sampling_device='cpu' if group=='B' else device
        sampler=core.TensorSampler(dataset['entities'],dataset['relations'],cfg.negatives,known,mode=mode,device=sampling_device,max_rounds=cfg.max_rejection_rounds)
    return model,opt,sampler

def step(model,opt,sampler,batch,cfg,device,trace=False,*,split_sampling=False):
    components={}
    def clock():
        if trace:sync(device)
        return time.perf_counter_ns()
    begin=clock();start=begin;opt.zero_grad(set_to_none=True);now=clock();components['zero_grad_ns']=now-start
    start=now;negative=sampler.generate(batch)
    if split_sampling:
        now=clock();components['sampling_ns']=now-start;start=now
        negative=negative.to(device);now=clock();components['negative_transfer_ns']=now-start
    else:
        negative=negative.to(device);now=clock();components['sampling_ns']=now-start
    start=now;positive=batch.to(device);now=clock();components['positive_transfer_ns']=now-start
    start=now;ph,pr,pt=positive.unbind(-1);nh,nr,nt=negative.unbind(-1)
    loss=core.adversarial_loss(model(ph,pr,pt)[:,None],model(nh,nr,nt),cfg.adv)
    now=clock();components['forward_loss_ns']=now-start
    start=now;loss.backward();now=clock();components['backward_ns']=now-start
    start=now;opt.step();now=clock();components['optimizer_ns']=now-start
    components['total_step_ns']=now-begin
    return loss.detach(),components

def run_job(root,data_dir,cfg,group,seed,kind,device='cuda',freeze_hash=None):
    if group not in ('cpu','balanced','tail','A','B','C') or kind not in ('diagnostic','quality','throughput','trace'):raise ValueError('Unknown group/pass')
    if group in ('A','B','C') and kind!='diagnostic':raise ValueError('ABC authorization is diagnostic-only')
    if kind=='quality' and not freeze_hash:raise ValueError('Quality jobs require a frozen protocol')
    return _run_job(root,data_dir,cfg,group,seed,kind,device,freeze_hash)


def run_official_job(root,data_dir,cfg,group,seed,kind,freeze_hash,device='cuda',provenance=None):
    """Run one official task after the controller verifies its freeze binding."""
    if group not in ('A','B','C') or kind not in ('quality','throughput','trace') or not freeze_hash:
        raise ValueError('Official ABC task requires a validated freeze')
    return _run_job(root,data_dir,cfg,group,seed,kind,device,freeze_hash,official=True,provenance=provenance)


def _run_job(root,data_dir,cfg,group,seed,kind,device,freeze_hash,official=False,provenance=None):
    root=Path(root);started=time.perf_counter();prov=provenance if provenance is not None else p.provenance(data_dir)
    p.new_run(root,{'config':cfg.to_dict(),'group':group,'seed':seed,'kind':kind,'provenance':prov,'environment':p.environment(),'freeze_sha256':freeze_hash})
    state={'status':'running','group':group,'seed':seed,'kind':kind,'epochs_completed':0,'training_seconds':0.0,'validation_seconds':0.0,'test_seconds':0.0,'validation_count':0,'converged':False,'freeze_sha256':freeze_hash}
    try:
        ds=data.load(data_dir);timing=kind in ('throughput','trace')
        if timing and not official:
            wm,wo,ws=build(ds,cfg,group,seed,device);batch=next(data.batches(ds['train'],cfg.batch_size,seed,0))
            for _ in range(3):step(wm,wo,ws,batch,cfg,device)
            sync(device);del wm,wo,ws
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
        model,opt,sampler=build(ds,cfg,group,seed,device)
        initial={k:p.hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
        p.write_json(root/'initial_weights.json',initial)
        stop=None if official else p.EarlyStopping(cfg.min_epochs,cfg.patience,cfg.min_delta)
        epochs=(6 if official else 5) if timing else min(100,cfg.max_epochs) if kind=='diagnostic' else cfg.max_epochs
        if official and timing:state.update(warmup_seconds=0.0,measured_epochs=0)
        batch_orders={}
        for epoch in range(1,epochs+1):
            warmup=official and timing and epoch==1
            trace=kind=='trace' and not warmup
            trace_rows=[]
            if str(device).startswith('cuda'):gpu_snapshot(root,epoch,'before')
            model.train()
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            loss_sum=torch.zeros((),device=device);steps=examples=0
            sync(device);begin=time.perf_counter_ns()
            for index,batch in enumerate(data.batches(ds['train'],cfg.batch_size,seed,epoch)):
                loss,parts=step(model,opt,sampler,batch,cfg,device,trace=trace);loss_sum+=loss;steps+=1;examples+=len(batch)
                if trace:
                    row={'epoch':epoch,'step':index,'batch_size':len(batch),'is_partial':len(batch)!=cfg.batch_size,**parts}
                    if official:trace_rows.append(row)
                    else:append(root/'steps.csv',row)
            sync(device);elapsed=time.perf_counter_ns()-begin;loss_value=float(loss_sum.cpu())/steps
            if not math.isfinite(loss_value):raise FloatingPointError('Nonfinite training loss')
            for row in trace_rows:append(root/'steps.csv',row)
            state['epochs_completed']=epoch
            if warmup:state['warmup_seconds']=elapsed/1e9
            else:
                state['training_seconds']+=elapsed/1e9
                if official and timing:state['measured_epochs']+=1
            phase={'phase':'warmup' if warmup else 'measured'} if official and timing else {}
            append(root/'epochs.csv',{'epoch':epoch,'epoch_time_ns':elapsed,'mean_batch_loss':loss_value,'steps':steps,'examples':examples,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,**phase})
            if group in ('A','B','C'):
                order_hash=p.hashlib.sha256()
                for ordered in data.batches(ds['train'],cfg.batch_size,seed,epoch):
                    order_hash.update(ordered.numpy().tobytes())
                batch_orders[str(epoch)]=order_hash.hexdigest()
                p.write_json(root/'batch_order.json',batch_orders)
            should_stop=False
            if not timing and epoch%cfg.validation_every==0:
                sync(device);ev=time.perf_counter();metrics=core.evaluate(model,ds['valid'],set(ds['train'])|set(ds['valid']),ds['entities'],device,cfg.query_chunk,cfg.entity_chunk);sync(device)
                duration=time.perf_counter()-ev;state['validation_seconds']+=duration;state['validation_count']+=1
                value=metrics['overall']['mrr']
                if official:
                    if not math.isfinite(value):raise FloatingPointError('Nonfinite validation MRR')
                    improved=value>state.get('best_validation_mrr',float('-inf'))
                else:improved,should_stop=stop.observe(epoch,value)
                if group in ('A','B','C'):
                    (root/'validation').mkdir(exist_ok=True)
                    p.write_json(root/'validation'/f'epoch-{epoch:04d}.json',metrics)
                append(root/'validation.csv',{'epoch':epoch,'training_seconds':state['training_seconds'],'wall_seconds':time.perf_counter()-started,'evaluation_seconds':duration,'mrr':value,'head_mrr':metrics['head']['mrr'],'tail_mrr':metrics['tail']['mrr'],**{k:v for k,v in metrics['overall'].items() if k!='mrr'},**{f'{side}_{k}':v for side in ('head','tail') for k,v in metrics[side].items() if k!='mrr'}})
                if improved:torch.save({'model':model.state_dict(),'epoch':epoch,'validation_mrr':value},root/'best.pt')
                if improved:state.update(best_epoch=epoch,best_validation_mrr=value)
                print(json.dumps({'group':group,'seed':seed,'epoch':epoch,'validation_mrr':value,'training_seconds':state['training_seconds'],'evaluation_seconds':duration}),flush=True)
            if str(device).startswith('cuda'):gpu_snapshot(root,epoch,'after')
            state['wall_seconds']=time.perf_counter()-started;p.write_json(root/'status.json',state)
            if should_stop and kind!='diagnostic':state['converged']=True;break
        if kind=='quality':
            if not (root/'best.pt').exists():raise RuntimeError('No validation-selected checkpoint')
            checkpoint=torch.load(root/'best.pt',map_location=device,weights_only=True)
            if official and (checkpoint['epoch']!=state['best_epoch'] or checkpoint['validation_mrr']!=state['best_validation_mrr']):
                raise ValueError('Loaded checkpoint differs from earliest best validation')
            model.load_state_dict(checkpoint['model'])
            test_path=Path(data_dir)/'test2id.txt'
            expected_test=prov['data'].get('test2id.txt') if official else None
            if expected_test and p.sha(test_path)!=expected_test:raise ValueError('Test split differs from frozen identity')
            test=data.read_triples(test_path,ds['entities'],ds['relations'])
            if expected_test and p.sha(test_path)!=expected_test:raise ValueError('Test split changed during loading')
            if Path(data_dir).name.upper()=='FB15K237' and len(test)!=20466:raise ValueError('Official test size mismatch')
            if official and str(device).startswith('cuda'):gpu_snapshot(root,epochs,'before_test')
            sync(device);ev=time.perf_counter();result=core.evaluate(model,test,set(ds['train'])|set(ds['valid'])|set(test),ds['entities'],device,cfg.query_chunk,cfg.entity_chunk);sync(device)
            state['test_seconds']=time.perf_counter()-ev;state['test_sha256']=p.sha(test_path)
            if official and str(device).startswith('cuda'):gpu_snapshot(root,epochs,'after_test')
            if official:result['checkpoint_epoch']=state['best_epoch']
            p.write_json(root/'test_metrics.json',result)
        state.update(status='complete',wall_seconds=time.perf_counter()-started)
        p.write_json(root/'status.json',state)
        p.write_json(root/'artifacts.json',{f.relative_to(root).as_posix():p.sha(f) for f in root.rglob('*') if f.is_file() and f.name!='artifacts.json'})
        return {**state,'artifacts_sha256':p.sha(root/'artifacts.json')}
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',error=repr(exc),error_type=type(exc).__name__,wall_seconds=time.perf_counter()-started)
        p.write_json(root/'status.json',state);(root/'error.log').write_text(traceback.format_exc());raise
