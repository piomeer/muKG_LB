import itertools
import unittest
from collections import Counter
from unittest.mock import patch

import torch

from src.py.experiments.formal_v1 import core


class SamplingTests(unittest.TestCase):
    def sampler(self, group, n=300, k=150, known=(), seed=401, rounds=64):
        from src.py.experiments.formal_v1.mechanism_sampling import PolicySampler
        return PolicySampler(n, 2, k, known, group, seed, 'cpu', max_rounds=rounds)

    def test_m00_bridges_historical_candidate_rng(self):
        known={(0, 0, 1), (2, 1, 3)}
        batch=torch.tensor(list(known))
        old=core.TensorSampler(300, 2, 150, known, device='cpu')
        new=self.sampler('M00', known=known)
        torch.manual_seed(123)
        new.candidate_generator.set_state(torch.get_rng_state())
        expected=old.generate(batch)
        expected_state=torch.get_rng_state()
        self.assertTrue(torch.equal(expected, new.generate(batch)))
        self.assertTrue(torch.equal(expected_state, new.candidate_generator.get_state()))

    def test_layout_filter_and_uniqueness(self):
        batch=torch.tensor([[0, 0, 1], [2, 1, 3], [4, 0, 5]])
        known={tuple(row) for row in batch.tolist()} | {(6, 0, 1), (2, 1, 7)}
        for group in ('M00', 'M10', 'M01', 'M11'):
            sampler=self.sampler(group, known=known)
            result=sampler.generate(batch)
            self.assertEqual(result.shape, (3, 150, 3))
            for positive, negatives in zip(batch.tolist(), result.tolist()):
                heads=sum(row[0] != positive[0] for row in negatives)
                self.assertEqual(heads, 75 if group in ('M00', 'M01') else heads)
                if group in ('M10', 'M11'):
                    self.assertIn(heads, (0, 150))
                self.assertTrue(all(tuple(row) not in known for row in negatives))
                self.assertTrue(all(row[1] == positive[1] for row in negatives))
                self.assertTrue(all(0 <= row[0] < 300 and 0 <= row[2] < 300 for row in negatives))
                if group in ('M01', 'M11'):
                    self.assertEqual(len({tuple(row) for row in negatives}), 150)

    def test_direction_stream_is_independent_of_rejection_consumption(self):
        a=self.sampler('M10', n=200)
        b=self.sampler('M11', n=200)
        batch=torch.tensor([[0, 0, 0]] * 10)
        for _ in range(3):
            left=a.generate(batch)
            right=b.generate(batch)
            self.assertTrue(torch.equal((left[:,:,0] != 0).all(1), (right[:,:,0] != 0).all(1)))

    def test_empty_input_and_short_batch(self):
        for group in ('M00', 'M10', 'M01', 'M11'):
            sampler=self.sampler(group)
            self.assertEqual(sampler.generate(torch.empty((0, 3), dtype=torch.long)).shape, (0, 150, 3))
            self.assertEqual(sampler.generate(torch.tensor([[0, 0, 1]])).shape, (1, 150, 3))

    def test_capacity_and_invalid_ids_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.sampler('M11', n=4, k=150).generate(torch.tensor([[0, 0, 0]]))
        with self.assertRaises(ValueError):
            self.sampler('M00').generate(torch.tensor([[999, 0, 0]]))
        with self.assertRaises(ValueError):
            self.sampler('unknown')

    def test_rejection_exhaustion_has_no_true_fallback(self):
        sampler=self.sampler('M00', n=4, k=2, known={(0,0,0)}, rounds=1)
        with patch('torch.randint', return_value=torch.zeros((1,2), dtype=torch.long)):
            with self.assertRaises(RuntimeError):
                sampler.generate(torch.tensor([[0,0,0]]))

    def test_unique_sampler_finite_distribution_is_uniform_on_success(self):
        counts=Counter()
        failures=0
        for sequence in itertools.product(range(4), repeat=4):
            sampler=self.sampler('M11', n=4, k=2, known={(0,0,0)}, rounds=2)
            proposals=iter((sequence[:2], sequence[2:]))
            def draw(high, size, **kwargs):
                if len(size)==1:
                    return torch.ones(size, dtype=torch.long)
                return torch.tensor([next(proposals)])
            with patch('torch.randint', side_effect=draw):
                try:
                    result=sampler.generate(torch.tensor([[0,0,0]]))
                    counts[tuple(sorted(result[0,:,0].tolist()))]+=1
                except RuntimeError:
                    failures+=1
        self.assertEqual(counts, {(1,2):58, (1,3):58, (2,3):58})
        self.assertEqual(failures, 82)


if __name__=='__main__':
    unittest.main()
