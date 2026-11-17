import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.py.experiments.formal_v1 import confirmation as c, confirmation_jobs as jobs
from src.py.experiments.formal_v1 import confirmation_report as report
from src.py.experiments.formal_v1 import core, runtime


class ProtocolTests(unittest.TestCase):
    def test_catalog_balanced(self):
        tasks = c.catalog()
        self.assertEqual(len({task['id'] for task in tasks}), 204)
        for phase, count in zip('STQE', (72, 36, 48, 48)):
            rows = [row for row in tasks if row['kind'] == phase]
            self.assertEqual(len(rows), count)
            groups = c.GROUPS if phase in 'ST' else c.QUALITY
            for position in range(len(groups)):
                self.assertCountEqual([rows[i]['group'] for i in range(position, count, len(groups))],
                                      list(groups) * (count // len(groups)**2))

    def test_test_copy_requires_all_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'required'):
                c.open_gate(root, {'tasks': c.catalog()}, {})
            self.assertFalse((root/'official_test').exists())

    def test_report_no_decision_for_missing_seed(self):
        result = report.comparison({501: 0.3}, {501: 0.2}, range(501, 513), .95)
        self.assertEqual(result['decision'], 'incomplete')
        self.assertEqual(len(result['missing_seeds']), 11)

    def test_rank_metrics_and_invalid_rank(self):
        values = {'head': [1., 2.5], 'tail': [2., 1.]}
        result = report.metrics_from_ranks(values, 2, 4)
        self.assertAlmostEqual(result['overall']['mrr'], .725)
        self.assertEqual(result['head']['hits@1'], .5)
        values['head'][0] = 1.25
        with self.assertRaises(ValueError):
            report.metrics_from_ranks(values, 2, 4)


class AcceptanceTests(unittest.TestCase):
    def exercise(self, device):
        ds = {'entities': 256, 'relations': 2, 'train': [(0, 0, 1), (2, 1, 3), (1, 0, 1), (0, 0, 0)]}
        batch = torch.tensor(ds['train'][:2])
        cfg = c.config(5000)
        identities = []
        for group in c.GROUPS:
            model, opt, sampler = jobs.build(ds, cfg, group, 550, device)
            identities.append(jobs.weights_identity(model))
            negatives = sampler.generate(batch).cpu()
            if group != 'A':
                for pos, row in zip(batch.tolist(), negatives.tolist()):
                    self.assertTrue(all(tuple(triple) not in ds['train'] for triple in row))
                    self.assertTrue(all(h == pos[0] or t == pos[2] for h, r, t in row))
                    heads = sum(h != pos[0] for h, _, _ in row)
                    self.assertIn(heads, (75,) if group == 'M00' else (0, 150))
                    if group.startswith('M11'):
                        self.assertEqual(len(set(map(tuple, row))), 150)
                self.assertEqual(sampler.generate(batch[:0]).shape, (0, 150, 3))
                with patch('torch.randint', side_effect=lambda high, size, **kw: torch.zeros(size, dtype=torch.long, device=kw['device'])):
                    with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                        sampler.generate(batch[:1])
            clone, reference_opt, reference_sampler = (
                runtime.build(ds, cfg, 'A', 550, device) if group == 'A'
                else jobs.build(ds, cfg, group, 550, device))
            positive = batch.to(device)
            negative = reference_sampler.generate(batch).to(device)
            pos_scores = clone(*positive.unbind(-1))[:, None]
            neg_scores = clone(*negative.unbind(-1))
            if group == 'A':
                runtime.seed_all(550)
            ref_loss = core.adversarial_loss(pos_scores, neg_scores)
            ref_loss.backward()
            reference_opt.step()
            if group != 'A':
                _, _, sampler = jobs.build(ds, cfg, group, 550, device)
            loss, parts = runtime.step(model, opt, sampler, batch, cfg, device,
                                       trace=True, split_sampling=True)
            torch.testing.assert_close(loss, ref_loss)
            for actual, expected in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(actual, expected)
                if actual.requires_grad:
                    torch.testing.assert_close(actual.grad, expected.grad)
            self.assertEqual(sum(parts[key] for key in jobs.COMPONENTS), parts['total_step_ns'])
            self.assertTrue(math.isfinite(loss.item()))
        self.assertTrue(all(item == identities[0] for item in identities))

    def test_cpu(self):
        self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_actual_cuda(self):
        self.exercise('cuda')

class RecoveryTests(unittest.TestCase):
    def test_anchored_second_attempt_ignores_earlier_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = c.catalog()[0]
            path = root/'jobs'/task['id']/'attempt-01'
            c.d.write(path/'status.json', {'status': 'failed', 'error_type': 'InfrastructureError'})
            anchored = {'status': 'complete', 'path': f'jobs/{task["id"]}/attempt-02'}
            with patch.object(report, 'verify_task') as verify:
                self.assertEqual(c.run_one(root, {}, task, anchored, False, None), anchored)
            verify.assert_called_once_with(root, {}, task, anchored)

    def test_unsealed_completion_requires_resume_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = c.catalog()[0]
            path = root/'jobs'/task['id']/'attempt-01'
            c.d.write(path/'status.json', {'status': 'complete'})
            with self.assertRaises(InterruptedError):
                c.run_one(root, {}, task, None, False, None)
            c.d.verify_seal(path)
            self.assertEqual(c.d.read(path/'status.json')['error_type'], 'UnsealedCompletion')
            with patch.object(c, 'dispatch', return_value={'status': 'failed'}) as dispatch:
                c.run_one(root, {}, task, None, True, None)
            self.assertEqual(dispatch.call_args.args[3].name, 'attempt-02')

    def test_earliest_best_on_equal_mrr(self):
        self.assertEqual(jobs.earliest_best({100: .3, 200: .4, 300: .4}), 200)

    def test_invalid_gate_never_stages_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = c.catalog()
            outcomes = {task['id']: {'status': 'complete'} for task in tasks if task['kind'] != 'E'}
            with patch.object(report, 'verify_task', side_effect=ValueError('Corrupt ranking')):
                with self.assertRaisesRegex(ValueError, 'Corrupt ranking'):
                    c.open_gate(root, {'tasks': tasks}, outcomes)
            self.assertFalse((root/'official_test').exists())
