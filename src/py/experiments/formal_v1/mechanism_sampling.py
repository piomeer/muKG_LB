"""Explicit two-factor sampling policies with independent candidate/direction streams."""
import torch

GROUPS = ('M00', 'M10', 'M01', 'M11')


class PolicySampler:
    def __init__(self, entities, relations, negatives, known, group, seed,
                 device='cuda', max_rounds=64, observe=False):
        if group not in GROUPS or entities <= 0 or relations <= 0 or negatives <= 0:
            raise ValueError('Invalid mechanism sampling configuration')
        if negatives % 2 or max_rounds <= 0 or entities * entities * relations >= 2**63:
            raise ValueError('Even K, positive retry budget and int64 triple keys required')
        self.n, self.r, self.k = entities, relations, negatives
        self.group, self.device = group, torch.device(device)
        self.max_rounds, self.observe = max_rounds, observe
        self.unique = group in ('M01', 'M11')
        self.single = group in ('M10', 'M11')
        self.candidate_generator = torch.Generator(device=self.device).manual_seed(seed*100003+1)
        self.direction_generator = torch.Generator(device=self.device).manual_seed(seed*100003+2)
        self.keys = torch.tensor(sorted({(h*relations+r)*entities+t for h,r,t in known}),
                                 dtype=torch.long, device=self.device)
        self.last_observation = None
        if self.unique:
            heads = self.keys // (relations*entities)
            rels = (self.keys // entities) % relations
            tails = self.keys % entities
            self.head_pairs, self.head_counts = torch.unique(rels*entities+tails, return_counts=True)
            self.tail_pairs, self.tail_counts = torch.unique(heads*relations+rels, return_counts=True)

    def _contains(self, keys):
        if not self.keys.numel():
            return torch.zeros_like(keys, dtype=torch.bool)
        indices = torch.searchsorted(self.keys, keys)
        return (indices < self.keys.numel()) & (self.keys[indices.clamp_max(self.keys.numel()-1)] == keys)

    @staticmethod
    def _counts(keys, counts, queries):
        if not keys.numel():
            return torch.zeros_like(queries)
        indices = torch.searchsorted(keys, queries)
        safe = indices.clamp_max(keys.numel()-1)
        return torch.where((indices < keys.numel()) & (keys[safe] == queries), counts[safe], 0)

    def _check_capacity(self, pos, directions):
        h, r, t = pos.unbind(-1)
        own = self._contains((h*self.r+r)*self.n+t)
        head_available = self.n-self._counts(self.head_pairs, self.head_counts, r*self.n+t)-(~own).long()
        tail_available = self.n-self._counts(self.tail_pairs, self.tail_counts, h*self.r+r)-(~own).long()
        head_need = directions.sum(1)
        if ((head_need > head_available) | (self.k-head_need > tail_available)).any().item():
            raise ValueError('Insufficient distinct legal candidates for this positive/direction')

    @staticmethod
    def _duplicates(keys, pending):
        # Accepted entries win over new proposals. Within either class, first position wins.
        priority = torch.argsort(pending.long(), dim=1, stable=True)
        by_key = torch.argsort(keys.gather(1, priority), dim=1, stable=True)
        order = priority.gather(1, by_key)
        sorted_keys = keys.gather(1, order)
        repeated = torch.zeros_like(pending)
        repeated[:,1:] = sorted_keys[:,1:] == sorted_keys[:,:-1]
        result = torch.zeros_like(pending)
        result.scatter_(1, order, repeated)
        return result

    def generate(self, batch):
        if batch.dtype != torch.long or batch.ndim != 2 or batch.shape[1] != 3:
            raise ValueError('Expected long [B,3] positive triples')
        pos = batch.to(self.device)
        if len(pos) and ((pos < 0).any() | (pos[:,0] >= self.n).any() |
                         (pos[:,2] >= self.n).any() | (pos[:,1] >= self.r).any()).item():
            raise ValueError('Triple ID outside entity/relation range')
        out = pos[:,None,:].expand(-1,self.k,-1).clone()
        if self.single:
            head = torch.randint(2, (len(pos),), device=self.device,
                                 generator=self.direction_generator).bool()[:,None].expand(-1,self.k)
        else:
            head = (torch.arange(self.k, device=self.device) < self.k//2)[None,:].expand(len(pos),-1)
        if not len(pos):
            if self.observe:
                self.last_observation = {'directions':head, 'rounds':[], 'proposals_drawn':0}
            return out
        if self.unique:
            self._check_capacity(pos, head)
        pending = torch.ones(out.shape[:2], dtype=torch.bool, device=self.device)
        records = []
        for round_index in range(self.max_rounds):
            candidates = torch.randint(self.n, out.shape[:2], device=self.device,
                                       generator=self.candidate_generator)
            out[:,:,0] = torch.where(pending & head, candidates, out[:,:,0])
            out[:,:,2] = torch.where(pending & ~head, candidates, out[:,:,2])
            keys = (out[:,:,0]*self.r+out[:,:,1])*self.n+out[:,:,2]
            collision = self._contains(keys)
            unchanged = (out == pos[:,None,:]).all(-1)
            duplicates = self._duplicates(keys, pending) if self.unique else torch.zeros_like(pending)
            if self.observe:
                records.append({'pending':int(pending.sum().item()),
                                'original_rejections':int((pending & unchanged).sum().item()),
                                'known_rejections':int((pending & collision & ~unchanged).sum().item()),
                                'duplicate_rejections':int((pending & duplicates & ~collision & ~unchanged).sum().item())})
            pending = collision | unchanged | duplicates
            if not pending.any().item():
                if self.observe:
                    self.last_observation = {'directions':head.clone(), 'rounds':records,
                                             'proposals_drawn':(round_index+1)*len(pos)*self.k}
                return out
        raise RuntimeError('Mechanism sampler exhausted 64/configured rejection rounds')
