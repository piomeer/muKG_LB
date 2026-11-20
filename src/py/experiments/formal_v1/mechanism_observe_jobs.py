"""Bounded observation jobs on frozen training inputs and parameter anchors."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from . import core, data, diagnostics as d, runtime
from .mechanism_jobs import weights_identity
from .mechanism_observe import gradient_observation, historical_observation, sampling_summary
from .mechanism_sampling import GROUPS, PolicySampler


def selected_batches(dataset, cfg):
    batches=list(data.batches(dataset['train'],cfg.batch_size,901,0))
    indices=list(dict.fromkeys([*range(min(4,len(batches))),len(batches)-1]))
    return [(index,batches[index]) for index in indices]


def append(root, row):
    with (root/'observations.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def observe_job(root, dataset, cfg, task, binding, device):
    batches=selected_batches(dataset,cfg)
    d.write(root/'batches.json',[{'index':i,'triples':batch.tolist()} for i,batch in batches])
    (root/'arrays').mkdir()
    known=set(dataset['train'])
    calls=0
    if task['kind']=='observe_a':
        for seed in (601,602,603):
            for index,batch in batches:
                value=seed*100003+index*10
                random.seed(value);np.random.seed(value)
                negative,summary=historical_observation(batch,dataset['entities'],cfg.negatives,known)
                name=f'A-{seed}-b{index:02d}.npz'
                np.savez_compressed(root/'arrays'/name,positive=batch.numpy().astype(np.int32),
                                    negative=negative.numpy().astype(np.int32))
                append(root,{'seed':seed,'batch_index':index,'arrays':f'arrays/{name}',**summary})
                calls+=1
        return {'observation_calls':calls,'positive_count':sum(len(b) for _,b in batches)*3}

    runtime.seed_all(401)
    model=core.TransE(dataset['entities'],dataset['relations'],cfg.dim,cfg.margin).to(device)
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    torch.save({'model':initial,'seed':401},root/'initial.pt')
    anchors=[{'id':'initial-401','path':str(root/'initial.pt')}]+binding.get('anchors',[])
    for anchor in anchors:
        checkpoint=torch.load(anchor['path'],map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model'])
        before=weights_identity(model)
        d.write(root/f'anchor-{anchor["id"]}.json',{'anchor':anchor,'weights':before})
        for group in GROUPS:
            for seed in (601,602,603):
                for index,batch in batches:
                    if str(device).startswith('cuda'):
                        runtime.gpu_snapshot(root,calls,'before_observation')
                    sampler=PolicySampler(dataset['entities'],dataset['relations'],cfg.negatives,known,
                                          group,seed,device,cfg.max_rejection_rounds,observe=True)
                    sampler.candidate_generator.manual_seed(seed*100003+index*10+1)
                    sampler.direction_generator.manual_seed(seed*100003+index*10+2)
                    negative=sampler.generate(batch)
                    direction=sampler.last_observation.pop('directions')
                    gradients,arrays=gradient_observation(model,batch.to(device),negative,direction,cfg.adv,
                                                          chunk=250,counterfactual=group in ('M00','M01'))
                    if gradients['gradient_reconstruction_relative_error']>1e-4:
                        raise ArithmeticError('Gradient decomposition does not reconstruct shared loss')
                    positive_cpu=batch.numpy()
                    negative_cpu=negative.cpu().numpy()
                    directions_cpu=direction.cpu().numpy()
                    summary=sampling_summary(positive_cpu,negative_cpu,directions_cpu,arrays['scores'],
                                             arrays['weights'],dataset['entities'],dataset['relations'])
                    name=f'{anchor["id"]}-{group}-{seed}-b{index:02d}.npz'
                    replacement=np.where(directions_cpu,negative_cpu[:,:,0],negative_cpu[:,:,2]).astype(np.int32)
                    np.savez_compressed(root/'arrays'/name,replacement=replacement,directions=directions_cpu,
                                        scores=arrays['scores'],weights=arrays['weights'])
                    append(root,{'anchor':anchor['id'],'group':group,'seed':seed,'batch_index':index,
                                 'arrays':f'arrays/{name}','sampling':sampler.last_observation,
                                 'summary':summary,'gradients':gradients})
                    calls+=1
                    d.write(root/'status.json',{'status':'running','task':task['id'],
                                               'observation_calls':calls,'anchor':anchor['id'],'group':group})
                    print(json.dumps({'task':task['id'],'observation':calls,'anchor':anchor['id'],
                                      'group':group,'seed':seed,'batch_index':index}),flush=True)
                    if str(device).startswith('cuda'):
                        runtime.gpu_snapshot(root,calls,'after_observation')
                    del sampler,negative,direction
        if weights_identity(model)!=before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Observation changed parameters or accumulated optimizer gradients')
    return {'observation_calls':calls,'anchor_count':len(anchors)}
