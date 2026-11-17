import csv
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from src.py.experiments.formal_v1 import diagnostics as d, protocol as p, runtime


def fixture(root):
    root.mkdir()
    (root / 'entity2id.txt').write_text('6\n')
    (root / 'relation2id.txt').write_text('1\n')
    (root / 'train2id.txt').write_text('3\n0 1 0\n2 3 0\n4 5 0\n')
    (root / 'valid2id.txt').write_text('1\n0 2 0\n')
    (root / 'test2id.txt').write_text('1\n1 3 0\n')


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class OfficialTests(unittest.TestCase):
    def test_catalog_order_and_fixed_budget(self):
        from src.py.experiments.formal_v1 import formal
        tasks = formal.task_catalog()
        self.assertEqual(len(tasks), 27)
        self.assertEqual(len({task['id'] for task in tasks}), 27)
        self.assertEqual([task['id'] for task in tasks[:6]],
                         ['S-A-201', 'S-B-201', 'S-C-201', 'S-B-202', 'S-C-202', 'S-A-202'])
        self.assertEqual([task['id'] for task in tasks[15:18]], ['T-A-201', 'T-B-201', 'T-C-201'])
        self.assertEqual([task['id'] for task in tasks[-3:]], ['Q-C-103', 'Q-A-103', 'Q-B-103'])
        self.assertEqual(formal.config().max_epochs, 1000)
        self.assertEqual(formal.config().validation_every, 10)

    def test_formal_quality_uses_fixed_epochs_and_tests_earliest_best_once(self):
        from src.py.experiments.formal_v1 import core, data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / 'data')
            cfg = replace(p.Config(), dim=4, negatives=2, batch_size=2, max_epochs=3,
                          validation_every=1, min_epochs=1, patience=1)
            events = []
            real_evaluate = core.evaluate
            real_read = data.read_triples

            def read(path, *args):
                if path.name == 'test2id.txt':
                    self.assertEqual(len(csv_rows(root / 'run' / 'epochs.csv')), 3)
                    events.append('test_read')
                return real_read(path, *args)

            def evaluate(model, rows, known, *args):
                result = real_evaluate(model, rows, known, *args)
                if rows == [(1, 0, 3)]:
                    self.assertIn((0, 0, 2), known)
                    self.assertIn((1, 0, 3), known)
                    checkpoint=torch.load(root/'run/best.pt',weights_only=True)
                    for key,value in model.state_dict().items():
                        torch.testing.assert_close(value,checkpoint['model'][key],rtol=0,atol=0)
                    events.append('test_evaluate')
                else:
                    self.assertNotIn((1, 0, 3), known)
                    result['overall']['mrr'] = .5
                    events.append('valid')
                return result

            with patch.object(data, 'read_triples', side_effect=read), patch.object(core, 'evaluate', side_effect=evaluate):
                result = runtime.run_official_job(root/'run', root/'data', cfg, 'B', 101, 'quality', 'fixture', 'cpu')
            self.assertEqual(result['epochs_completed'], 3)
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(events, ['valid', 'valid', 'valid', 'test_read', 'test_evaluate'])
            self.assertEqual(d.read(root/'run/test_metrics.json')['checkpoint_epoch'], 1)

    def test_timing_warmup_retains_objects_and_is_excluded_from_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root/'data')
            cfg = replace(p.Config(), dim=4, negatives=2, batch_size=2)
            (root/'data/test2id.txt').unlink()
            for mode in ('throughput', 'trace'):
                with patch.object(runtime, 'build', wraps=runtime.build) as build:
                    result = runtime.run_official_job(root/mode, root/'data', cfg, 'B', 201, mode, 'fixture', 'cpu')
                self.assertEqual(build.call_count, 1)
                self.assertEqual(result['measured_epochs'], 5)
                rows = csv_rows(root/mode/'epochs.csv')
                self.assertEqual([row['phase'] for row in rows], ['warmup']+['measured']*5)
                self.assertTrue(all(int(row['examples']) == 3 for row in rows))
                self.assertAlmostEqual(result['training_seconds'], sum(int(r['epoch_time_ns']) for r in rows[1:])/1e9)
                self.assertFalse((root/mode/'test_metrics.json').exists())
            trace = csv_rows(root/'trace/steps.csv')
            self.assertEqual({int(row['epoch']) for row in trace}, {2, 3, 4, 5, 6})
            self.assertEqual(sum(row['is_partial'] == 'True' for row in trace), 5)
            self.assertFalse((root/'throughput/steps.csv').exists())

    def test_pair_statistics_and_incomplete_matrix(self):
        from src.py.experiments.formal_v1 import formal_report as report
        tasks = {}
        for seed in (101, 102, 103):
            for group, mrr in zip('ABC', (.5, .49, .48)):
                tasks[f'Q-{group}-{seed}'] = {'metrics': {side: {'mrr': mrr} for side in ('head', 'tail', 'overall')}}
        result = report.paired(tasks, 'quality')
        self.assertAlmostEqual(result['primary']['geometric_mean'], .96)
        self.assertTrue(result['primary']['interval_supports_target'])
        del tasks['Q-C-103']
        self.assertEqual(report.paired(tasks, 'quality')['status'], 'incomplete')
        self.assertNotIn('primary', report.paired(tasks, 'quality'))

    def test_freeze_requires_an_intact_catalog_and_test_file_identity(self):
        from src.py.experiments.formal_v1 import formal
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root/'data')
            original = formal.identity(root/'data')
            self.assertIn('test2id.txt', original['data'])
            (root/'data/test2id.txt').write_text('1\n1 4 0\n')
            self.assertNotEqual(original, formal.identity(root/'data'))
            run = root/'run'
            run.mkdir()
            tasks = formal.task_catalog()
            d.write(run/'task_catalog.json', tasks)
            payload = {'protocol': formal.PROTOCOL, 'tasks_sha256': d.sha(run/'task_catalog.json')}
            d.write(run/'freeze.json', {**payload, 'freeze_sha256': p.digest(payload)})
            tasks[0]['seed'] = 999
            d.write(run/'task_catalog.json', tasks)
            with self.assertRaises(ValueError):
                formal.read_freeze(run)

    def test_live_checks_reject_code_data_and_environment_changes(self):
        from src.py.experiments.formal_v1 import formal
        record={'data_dir':'fixture','provenance':{'sources':{'runtime.py':'old'},'data':{'test2id.txt':'old'}},
                'environment':{'torch':'old'}}
        for current in ({'sources':{'runtime.py':'new'},'data':{'test2id.txt':'old'}},
                        {'sources':{'runtime.py':'old'},'data':{'test2id.txt':'new'}}):
            with patch.object(formal,'identity',return_value=current), self.assertRaises(ValueError):
                formal.verify_live(record)
        with patch.object(formal,'identity',return_value=record['provenance']), patch.object(formal,'environment',return_value={'torch':'new'}):
            with self.assertRaises(ValueError):
                formal.verify_live(record)

    def test_gpu_lock_rejects_parallel_owner_and_busy_device(self):
        from src.py.experiments.formal_v1 import formal
        with patch.object(formal.subprocess,'check_output',return_value='GPU-formal-test-fixture\n'):
            with patch.object(d,'gpu_processes',return_value=[]):
                with formal.gpu_lock():
                    with self.assertRaises(RuntimeError), formal.gpu_lock():
                        self.fail('second owner acquired GPU lock')
            with patch.object(d,'gpu_processes',return_value=[42]):
                with self.assertRaises(RuntimeError), formal.gpu_lock():
                    self.fail('GPU with a compute process was accepted')

    def test_single_task_dispatch_and_no_automatic_retry(self):
        from src.py.experiments.formal_v1 import formal
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fail(run,output,request,**kwargs):
                self.assertEqual(request['id'],'S-A-201')
                self.assertEqual(request['kind'],'throughput')
                self.assertEqual(kwargs['worker_command'],'_formal-worker')
                self.assertFalse(kwargs['wait_for_gpu'])
                d.write(output/'status.json',{'status':'failed','error_type':'InfrastructureError'})
                return d.read(output/'status.json')
            with patch.object(formal,'read_freeze',return_value={'freeze_sha256':'fixture'}), \
                 patch.object(formal,'verify_live'), patch.object(formal,'gpu_lock',return_value=nullcontext()), \
                 patch.object(d,'launch',side_effect=fail) as launch:
                result=formal.run_task(root,'S-A-201')
                self.assertEqual(launch.call_count,1)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(set(d.read(root/'outcomes.json')),{'S-A-201'})
                with self.assertRaises(ValueError):
                    formal.run_task(root,'D-A-7')
                self.assertEqual(launch.call_count,1)

    def test_check_rejects_output_inside_closed_diagnostic(self):
        from src.py.experiments.formal_v1 import formal_checks
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            output=root/'nested-check'
            with self.assertRaises(ValueError):
                formal_checks.check(root,root/'receipt.json',output)
            self.assertFalse(output.exists())

    def test_explicit_restart_preserves_unsealed_attempt_and_uses_new_directory(self):
        from src.py.experiments.formal_v1 import formal
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            prior=root/'jobs/S-A-201/attempt-01/status.json'
            d.write(prior,{'status':'complete'})
            with patch.object(formal,'read_freeze',return_value={'freeze_sha256':'fixture'}), \
                 patch.object(formal,'verify_live'), patch.object(formal,'gpu_lock',return_value=nullcontext()), \
                 patch.object(d,'launch',return_value={'status':'failed'}) as launch:
                outcome=formal.run_task(root,'S-A-201')
            self.assertEqual(d.read(prior)['status'],'interrupted')
            self.assertEqual(outcome['path'],'jobs/S-A-201/attempt-02')
            self.assertEqual(launch.call_count,1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class OfficialCUDATests(unittest.TestCase):
    def test_tiny_abc_formal_trace_with_short_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root/'data')
            cfg=replace(p.Config(),dim=4,negatives=2,batch_size=2)
            initial=[]
            orders=[]
            for group in 'ABC':
                path=root/group
                result=runtime.run_official_job(path,root/'data',cfg,group,201,'trace','fixture')
                self.assertEqual(result['measured_epochs'],5)
                self.assertEqual(sum(row['is_partial']=='True' for row in csv_rows(path/'steps.csv')),5)
                initial.append(d.read(path/'initial_weights.json'))
                orders.append(d.read(path/'batch_order.json'))
            self.assertTrue(initial[0]==initial[1]==initial[2])
            self.assertTrue(orders[0]==orders[1]==orders[2])
