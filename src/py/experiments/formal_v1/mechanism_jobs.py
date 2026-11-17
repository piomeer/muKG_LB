"""Independent mechanism workers with a validation-only training path."""
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from . import core, data, diagnostics as d, runtime
from .mechanism_sampling import PolicySampler


def weights_identity(model):
    return {name:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for name,value in model.state_dict().items()}


def build(dataset, cfg, group, seed, device):
    runtime.seed_all(seed)
    model = core.TransE(dataset['entities'],dataset['relations'],cfg.dim,cfg.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    sampler = PolicySampler(dataset['entities'],dataset['relations'],cfg.negatives,
                            set(dataset['train']),group,seed,device,cfg.max_rejection_rounds)
    return model,optimizer,sampler


def _training(root, dataset, cfg, task, state, device, probe=False):
    model,opt,sampler = build(dataset,cfg,task['group'],task['seed'],device)
    d.write(root/'initial_weights.json',weights_identity(model))
    order = {}
    epochs = 3 if probe else cfg.max_epochs
    for epoch in range(1,epochs+1):
        model.train()
        if str(device).startswith('cuda'):
            runtime.gpu_snapshot(root,epoch,'before')
            torch.cuda.reset_peak_memory_stats()
        runtime.sync(device)
        start = time.perf_counter_ns()
        loss_sum = torch.zeros((),device=device)
        examples=steps=short=0
        batch_hash=hashlib.sha256()
        for batch in data.batches(dataset['train'],cfg.batch_size,task['seed'],epoch):
            batch_hash.update(batch.numpy().tobytes())
            loss,_ = runtime.step(model,opt,sampler,batch,cfg,device)
            loss_sum += loss
            examples += len(batch)
            steps += 1
            short = len(batch) if len(batch)<cfg.batch_size else short
        runtime.sync(device)
        elapsed = time.perf_counter_ns()-start
        loss_value = float(loss_sum.cpu())/steps
        if not math.isfinite(loss_value):
            raise FloatingPointError('Nonfinite mechanism training loss')
        warmup = probe and epoch==1
        state['training_seconds'] += 0 if warmup else elapsed/1e9
        state['epochs_completed']=epoch
        if warmup:
            state['warmup_seconds'] = elapsed/1e9
        order[str(epoch)] = batch_hash.hexdigest()
        d.write(root/'batch_order.json',order)
        validate = epoch==3 if probe else epoch%cfg.validation_every==0
        if validate:
            runtime.sync(device)
            beginning=time.perf_counter()
            metrics=core.evaluate(model,dataset['valid'],set(dataset['train'])|set(dataset['valid']),
                                  dataset['entities'],device,cfg.query_chunk,cfg.entity_chunk)
            runtime.sync(device)
            duration=time.perf_counter()-beginning
            value=metrics['overall']['mrr']
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite validation MRR')
            state['validation_seconds'] += duration
            state['validation_count'] += 1
            (root/'validation').mkdir(exist_ok=True)
            d.write(root/'validation'/f'epoch-{epoch:04d}.json',metrics)
            runtime.append(root/'validation.csv',{'epoch':epoch,'training_seconds':state['training_seconds'],
                           'wall_seconds':time.perf_counter()-state['_started'], 'evaluation_seconds':duration,
                           **{f'{side}_{name}':value for side in ('head','tail','overall') for name,value in metrics[side].items()}})
            if value > state.get('best_validation_mrr',-math.inf):
                state.update(best_epoch=epoch,best_validation_mrr=value)
                torch.save({'model':model.state_dict(),'epoch':epoch,'validation_mrr':value},root/'best.pt')
            state['last_validation_mrr']=value
            print(json.dumps({'task':task['id'],'epoch':epoch,'validation_mrr':value,
                              'training_seconds':state['training_seconds']}),flush=True)
        allocated = torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0
        reserved = torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0
        state.update(peak_allocated_bytes=max(state.get('peak_allocated_bytes',0),allocated),
                     peak_reserved_bytes=max(state.get('peak_reserved_bytes',0),reserved))
        runtime.append(root/'epochs.csv',{'epoch':epoch,'phase':'warmup' if warmup else 'measured' if probe else 'training',
                       'epoch_time_ns':elapsed,'mean_batch_loss':loss_value,'examples':examples,'steps':steps,
                       'short_batch_size':short,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved})
        if str(device).startswith('cuda'):
            runtime.gpu_snapshot(root,epoch,'after')
        state['wall_seconds']=time.perf_counter()-state['_started']
        d.write(root/'status.json',{k:v for k,v in state.items() if not k.startswith('_')})
    torch.save({'model':model.state_dict(),'epoch':epochs,'validation_mrr':state.get('last_validation_mrr')},root/'last.pt')
    if probe:
        state.update(measured_epochs=2,mean_epoch_seconds=state['training_seconds']/2)


def _probe(root, dataset, cfg, task, state, device):
    model,opt,sampler=build(dataset,cfg,task['group'],task['seed'],device)
    batches=list(data.batches(dataset['train'],cfg.batch_size,900,0))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for batch in (batches[0],batches[-1]):
        loss,_=runtime.step(model,opt,sampler,batch,cfg,device)
        runtime.sync(device)
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite preflight update')
    reserved=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0
    allocated=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0
    del model,opt,sampler
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    _training(root,dataset,cfg,task,state,device,probe=True)
    if str(device).startswith('cuda'):
        state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],reserved)
        state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],allocated)
        state['total_bytes']=torch.cuda.get_device_properties(0).total_memory
        if state['peak_reserved_bytes']>=.9*state['total_bytes']:
            raise MemoryError('Preflight reserved memory exceeds 90% reserve threshold')


def execute(root, data_dir, cfg, task, binding, device='cuda'):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    state={'status':'running','task':task['id'],'kind':task['kind'],'group':task.get('group'),
           'seed':task.get('seed'),'epochs_completed':0,'validation_count':0,
           'training_seconds':0.0,'validation_seconds':0.0,'started_unix':time.time(),
           '_started':time.perf_counter()}
    d.write(root/'manifest.json',{'task':task,'config':cfg.to_dict(),**binding})
    d.write(root/'status.json',{k:v for k,v in state.items() if not k.startswith('_')})
    try:
        dataset=data.load(data_dir)
        if task['kind']=='validation':
            _training(root,dataset,cfg,task,state,device)
        elif task['kind']=='probe':
            _probe(root,dataset,cfg,task,state,device)
        else:
            from .mechanism_observe_jobs import observe_job
            state.update(observe_job(root,dataset,cfg,task,binding,device))
        state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',
                     error_type=type(exc).__name__,error=repr(exc))
        (root/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        state['wall_seconds']=time.perf_counter()-state.pop('_started')
        state['finished_unix']=time.time()
        d.write(root/'status.json',state)
        d.seal(root)
    return state
