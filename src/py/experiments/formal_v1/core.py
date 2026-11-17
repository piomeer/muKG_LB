"""Historical TransE and explicit, non-equivalent negative-sampling policies."""
from collections import defaultdict
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class TransE(nn.Module):
    def __init__(self, entities, relations, dim=250, margin=7.0):
        super().__init__()
        self.ent_embeddings=nn.Embedding(entities,dim)
        self.rel_embeddings=nn.Embedding(relations,dim)
        self.margin=nn.Parameter(torch.tensor([float(margin)]),requires_grad=False)
        self.embedding_range=nn.Parameter(torch.tensor([(margin+2.0)/dim]),requires_grad=False)
        nn.init.uniform_(self.ent_embeddings.weight,-self.embedding_range.item(),self.embedding_range.item())
        nn.init.uniform_(self.rel_embeddings.weight,-self.embedding_range.item(),self.embedding_range.item())
    def forward(self,h,r,t):
        return torch.norm(self.ent_embeddings(h)+self.rel_embeddings(r)-self.ent_embeddings(t),p=1,dim=-1)-self.margin

def adversarial_loss(pos,neg,adv=2.0):
    weights=F.softmax(-neg*adv,dim=-1).detach()
    return -(torch.sum(F.logsigmoid(-pos))+torch.sum(weights*F.logsigmoid(neg)))/2

class CPUSampler:
    """fdfd880 algorithm: direction per retry, random.sample, set filtering.

    Last retry intentionally retains the historical unfiltered fallback.
    This is different from the GPU rejection policy, including duplicate rules.
    """
    def __init__(self,entities,negatives,known,max_try=10,observer=None):
        if not 0<negatives<=entities:raise ValueError('CPU random.sample requires 0<K<=entity count')
        self.entities=list(range(entities));self.k=negatives;self.known=set(known);self.max_try=max_try
        self.observer=observer
    def generate(self,batch):
        output=[]
        for head,relation,tail in batch.tolist():
            negatives=[];remaining=self.k
            for i in range(self.max_try):
                corrupt=np.random.binomial(1,.5)
                candidates=random.sample(self.entities,remaining)
                triples={(e,relation,tail) for e in candidates} if corrupt else {(head,relation,e) for e in candidates}
                if self.observer is not None:
                    self.observer((head,relation,tail),i+1,bool(corrupt),triples,negatives,
                                  i==self.max_try-1)
                if i==self.max_try-1:
                    negatives+=list(triples);break
                negatives+=list(triples-self.known)
                if len(negatives)==self.k:break
                remaining=self.k-len(negatives)
            if len(negatives)!=self.k:raise RuntimeError('Historical sampler count mismatch')
            output.append(negatives)
        return torch.tensor(output,dtype=torch.long).reshape(len(batch),self.k,3)

class TensorSampler:
    """Uniform-with-replacement tensor sampling on CPU or GPU; train rejection."""
    def __init__(self,entities,relations,negatives,known,mode='balanced',device='cuda',max_rounds=64):
        if mode not in ('balanced','tail') or negatives<=0 or (mode=='balanced' and negatives%2):raise ValueError('Balanced sampling requires positive even K')
        if entities*entities*relations>=2**63:raise ValueError('Triple key exceeds int64')
        self.n,self.r,self.k=entities,relations,negatives
        self.mode,self.device,self.max_rounds=mode,device,max_rounds
        self.keys=torch.tensor(sorted({(h*relations+r)*entities+t for h,r,t in known}),dtype=torch.long,device=device)
    def generate(self,batch):
        pos=batch.to(self.device);out=pos[:,None,:].expand(-1,self.k,-1).clone()
        if not len(pos):return out
        head=torch.arange(self.k,device=self.device)<(self.k//2 if self.mode=='balanced' else 0)
        pending=torch.ones(out.shape[:2],dtype=torch.bool,device=self.device)
        for _ in range(self.max_rounds):
            candidates=torch.randint(self.n,out.shape[:2],device=self.device)
            out[:,:,0]=torch.where(pending & head[None,:],candidates,out[:,:,0])
            out[:,:,2]=torch.where(pending & ~head[None,:],candidates,out[:,:,2])
            keys=(out[:,:,0]*self.r+out[:,:,1])*self.n+out[:,:,2]
            collision=torch.zeros_like(pending)
            if self.keys.numel():
                idx=torch.searchsorted(self.keys,keys)
                collision=(idx<self.keys.numel()) & (self.keys[idx.clamp_max(self.keys.numel()-1)]==keys)
            pending=collision | (out==pos[:,None,:]).all(dim=-1)
            if not pending.any().item():return out
        raise RuntimeError('Tensor sampler exhausted rejection rounds; no positive fallback allowed')

@torch.no_grad()
def evaluate(model,triples,known,entities,device='cuda',query_chunk=32,entity_chunk=2048):
    if not triples or query_chunk<=0 or entity_chunk<=0:raise ValueError('Nonempty queries and positive chunk sizes required')
    model.eval();indices={'head':defaultdict(set),'tail':defaultdict(set)}
    for h,r,t in known:indices['head'][r,t].add(h);indices['tail'][h,r].add(t)
    ranks={}
    for mode in ('head','tail'):
        ranks[mode]=[]
        for start in range(0,len(triples),query_chunk):
            rows=triples[start:start+query_chunk];q=torch.tensor(rows,dtype=torch.long,device=device)
            h,r,t=q.unbind(1);target=h if mode=='head' else t
            truth=model(h,r,t);better=torch.zeros(len(q),device=device,dtype=torch.long);ties=better.clone()
            for lo in range(0,entities,entity_chunk):
                ids=torch.arange(lo,min(lo+entity_chunk,entities),device=device)
                score=model(ids[None,:],r[:,None],t[:,None]) if mode=='head' else model(h[:,None],r[:,None],ids[None,:])
                allowed=(ids[None,:]!=target[:,None]).expand(len(q),-1).clone()
                for i,(hh,rr,tt) in enumerate(rows):
                    blocked=[e-lo for e in indices[mode][(rr,tt) if mode=='head' else (hh,rr)] if lo<=e<lo+len(ids)]
                    if blocked:allowed[i,torch.tensor(blocked,device=device)]=False
                better+=((score<truth[:,None]) & allowed).sum(1)
                ties+=((score==truth[:,None]) & allowed).sum(1)
            ranks[mode].extend((1+better.double()+.5*ties.double()).cpu().tolist())
    def metrics(values):
        return {'mrr':sum(1/x for x in values)/len(values),**{f'hits@{k}':sum(x<=k for x in values)/len(values) for k in (1,3,10)}}
    return {'head':metrics(ranks['head']),'tail':metrics(ranks['tail']),'overall':metrics(ranks['head']+ranks['tail']),'ranks':ranks}
