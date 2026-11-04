"""Official integer-ID data; diagnostic paths never read the test split."""
from pathlib import Path
import torch

def read_triples(path,entities,relations):
    with Path(path).open() as f:
        count=int(f.readline());rows=[]
        for line in f:
            if not line.strip():continue
            h,t,r=map(int,line.split())
            if not (0<=h<entities and 0<=t<entities and 0<=r<relations):raise ValueError('Triple ID out of range')
            rows.append((h,r,t))
    if len(rows)!=count or len(set(rows))!=count:raise ValueError('Split length or duplicate mismatch')
    return rows

def load(path,include_test=False):
    path=Path(path)
    with (path/'entity2id.txt').open() as f:n=int(f.readline())
    with (path/'relation2id.txt').open() as f:r=int(f.readline())
    result={'entities':n,'relations':r}
    for name in ('train','valid','test') if include_test else ('train','valid'):
        result[name]=read_triples(path/f'{name}2id.txt',n,r)
    if path.name.upper()=='FB15K237':
        expected={'train':272115,'valid':17535,'test':20466}
        if (n,r)!=(14541,237) or any(len(result[k])!=expected[k] for k in expected if k in result):raise ValueError('FB15K237 official counts mismatch')
    return result

def batches(triples,batch_size,seed,epoch):
    generator=torch.Generator().manual_seed(seed*100003+epoch)
    order=torch.randperm(len(triples),generator=generator)
    values=torch.tensor(triples,dtype=torch.long)
    for start in range(0,len(order),batch_size):yield values[order[start:start+batch_size]]
