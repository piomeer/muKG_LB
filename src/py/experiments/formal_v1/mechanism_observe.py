"""Fixed-parameter observations. Loss weights retain the historical detach semantics."""
import math
import numpy as np
import torch
from torch.nn import functional as F

from . import core


def cosine(left, right):
    dot = sum(float((a*b).sum().item()) for a,b in zip(left,right))
    ll = sum(float(a.square().sum().item()) for a in left)
    rr = sum(float(b.square().sum().item()) for b in right)
    if ll == 0 or rr == 0:
        return None
    return dot / math.sqrt(ll*rr)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count':0, 'mean':None, 'min':None, 'max':None, 'p25':None, 'median':None, 'p75':None, 'p95':None}
    return {'count':int(values.size), 'mean':float(values.mean()),
            'min':float(values.min()), 'max':float(values.max()),
            **dict(zip(('p25','median','p75','p95'), map(float,np.quantile(values,[.25,.5,.75,.95]))))}


def sampling_summary(positive, negative, head, scores, weights, entities, relations):
    keys = (negative[:,:,0]*relations+negative[:,:,1])*entities+negative[:,:,2]
    order = np.argsort(keys, axis=1, kind='stable')
    sorted_keys = np.take_along_axis(keys,order,axis=1)
    sorted_weights = np.take_along_axis(weights,order,axis=1).astype(np.float64)
    starts = np.ones_like(sorted_keys,dtype=bool)
    starts[:,1:] = sorted_keys[:,1:] != sorted_keys[:,:-1]
    unique = starts.sum(1)
    group = np.cumsum(starts,axis=1)-1
    merged = np.zeros_like(sorted_weights)
    np.add.at(merged,(np.arange(len(positive))[:,None],group),sorted_weights)
    entropy = -(weights*np.log(np.maximum(weights,np.finfo(weights.dtype).tiny))).sum(1)
    return {'positive_count':len(positive), 'negative_count':int(keys.size),
            'head_count':describe(head.sum(1)), 'unique_count':describe(unique),
            'duplicate_fraction':describe(1-unique/keys.shape[1]),
            'head_weight_mass':describe((weights*head).sum(1)),
            'position_ess':describe(1/np.square(weights.astype(np.float64)).sum(1)),
            'unique_ess':describe(1/np.square(merged).sum(1)),
            'weight_entropy':describe(entropy),
            'head_scores':describe(scores[head]), 'tail_scores':describe(scores[~head])}


def gradient_observation(model, positive, negative, head, adv=2.0, chunk=250, counterfactual=False):
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    labels = ['positive','head','tail','total'] + (['counterfactual'] if counterfactual else [])
    accumulated = {name:[torch.zeros_like(p,dtype=torch.float64) for p in parameters] for name in labels}
    arrays = {'scores':[], 'weights':[]}
    losses = {name:0.0 for name in labels}
    for start in range(0,len(positive),chunk):
        pos, neg, mask = positive[start:start+chunk], negative[start:start+chunk], head[start:start+chunk]
        score_pos = model(*pos.unbind(-1))[:,None]
        score_neg = model(*neg.unbind(-1))
        weights = F.softmax(-score_neg*adv,dim=-1).detach()
        terms = weights*F.logsigmoid(score_neg)
        components = {'positive':-.5*F.logsigmoid(-score_pos).sum(),
                      'head':-.5*(terms*mask).sum(), 'tail':-.5*(terms*~mask).sum(),
                      'total':core.adversarial_loss(score_pos,score_neg,adv)}
        if counterfactual:
            if not (mask.any(1) & (~mask).any(1)).all().item():
                raise ValueError('Counterfactual requires both directions for every positive')
            logits = -score_neg*adv
            separate = .5*(F.softmax(logits.masked_fill(~mask,-torch.inf),-1)+
                           F.softmax(logits.masked_fill(mask,-torch.inf),-1)).detach()
            components['counterfactual'] = components['positive']-.5*(separate*F.logsigmoid(score_neg)).sum()
        for index,name in enumerate(labels):
            gradients = torch.autograd.grad(components[name],parameters,retain_graph=index<len(labels)-1)
            for dest,source in zip(accumulated[name],gradients):
                dest.add_(source.detach().double())
            losses[name] += float(components[name].detach().item())
        arrays['scores'].append(score_neg.detach().cpu().numpy())
        arrays['weights'].append(weights.cpu().numpy())
    accumulated['negative']=[h+t for h,t in zip(accumulated['head'],accumulated['tail'])]
    if counterfactual:
        accumulated['counterfactual_negative']=[total-pos for total,pos in zip(accumulated['counterfactual'],accumulated['positive'])]
    norms = {name:math.sqrt(sum(float(g.square().sum().item()) for g in gradients))
             for name,gradients in accumulated.items()}
    residual = [total-pos-h-t for total,pos,h,t in zip(
        accumulated['total'],accumulated['positive'],accumulated['head'],accumulated['tail'])]
    error = math.sqrt(sum(float(g.square().sum().item()) for g in residual)) / max(norms['total'],1e-30)
    result = {'losses':losses, 'gradient_norms':norms,
              'gradient_reconstruction_relative_error':error,
              'gradient_cosines':{f'{a}/{b}':cosine(accumulated[a],accumulated[b])
                                  for a,b in (('head','tail'),('head','total'),('tail','total'),('negative','total'))},
              'undefined_cosine_reason':'zero-norm component' }
    if counterfactual:
        result['gradient_cosines']['counterfactual/total'] = cosine(accumulated['counterfactual'],accumulated['total'])
        result['gradient_cosines']['counterfactual_negative/negative'] = cosine(accumulated['counterfactual_negative'],accumulated['negative'])
    return result,{key:np.concatenate(value) for key,value in arrays.items()}


def historical_observation(batch, entities, negatives, known):
    summary = {'positive_count':len(batch), 'candidate_count':0, 'retry_calls':0,
               'fallback_positive_count':0, 'fallback_true_count':0,
               'cross_retry_duplicate_count':0, 'direction_switch_count':0}
    previous_direction = {}
    def observe(positive, round_number, direction, triples, accepted, fallback):
        summary['retry_calls'] += 1
        summary['candidate_count'] += len(triples)
        summary['fallback_positive_count'] += int(fallback)
        summary['fallback_true_count'] += sum(row in known or row==positive for row in triples) if fallback else 0
        admitted = triples if fallback else triples-known
        summary['cross_retry_duplicate_count'] += len(set(accepted)&admitted)
        if round_number > 1:
            summary['direction_switch_count'] += int(previous_direction[positive] != direction)
        previous_direction[positive] = direction
    sampler = core.CPUSampler(entities,negatives,known,observer=observe)
    return sampler.generate(batch),summary
