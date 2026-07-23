"""
Phase 8 – GPU Negative Sampler (Vectorized v2)
Fast vectorized approach: randint + isin against batch pos_tails.
Useful as drop-in replacement for CPU negative sampling.
"""
import torch
from typing import List, Tuple


class GPUNegativeSampler:
    def __init__(
        self,
        n_entities: int,
        neg_num: int,
        oversample_factor: float = 1.5,
        device: str = 'cuda'
    ):
        self.n_entities = n_entities
        self.neg_num = neg_num
        self.oversample_factor = oversample_factor
        self.device = device

    def generate(
        self,
        batch_triples: List[Tuple[int, int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized GPU negative sampling (tail corruption).
        Returns:
            neg_heads: (batch_size * neg_num,) — original heads, repeated
            neg_tails: (batch_size * neg_num,) — corrupted tails
        """
        batch_size = len(batch_triples)
        total_needed = batch_size * self.neg_num
        num_candidates = int(total_needed * self.oversample_factor)

        pos_heads = torch.tensor(
            [t[0] for t in batch_triples], dtype=torch.long, device=self.device
        )
        pos_tails = torch.tensor(
            [t[2] for t in batch_triples], dtype=torch.long, device=self.device
        )

        valid_tails = torch.empty(0, dtype=torch.long, device=self.device)
        while valid_tails.size(0) < total_needed:
            candidates = torch.randint(
                0, self.n_entities, (num_candidates,), device=self.device
            )
            mask = ~torch.isin(candidates, pos_tails)
            filtered = candidates[mask]
            valid_tails = torch.cat([valid_tails, filtered])

        valid_tails = valid_tails[:total_needed]
        neg_heads = pos_heads.repeat_interleave(self.neg_num)
        return neg_heads, valid_tails