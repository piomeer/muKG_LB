import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class QueueTests(unittest.TestCase):
    def test_original_job_anchor_cannot_be_replaced_on_restart(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'jobs/D-A-7/attempt-01';path.mkdir(parents=True)
            (path/'x').write_text('original');d.seal(path)
            d.write(root/'outcomes.json',{'D-A-7':{'status':'complete','path':'jobs/D-A-7/attempt-01','artifacts_sha256':d.sha(path/'artifacts.json')}})
            d.check_anchor(root,'D-A-7',path)
            (path/'x').write_text('changed');d.seal(path)
            with self.assertRaises(ValueError):d.check_anchor(root,'D-A-7',path)

    def test_worker_parent_death_guard_releases_process(self):
        child_code="import os,time; from src.py.experiments.formal_v1.diagnostics import arm_parent_death; arm_parent_death(os.getppid()); print('armed',flush=True); time.sleep(60)"
        parent_code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[1]],stdout=subprocess.PIPE,text=True); print(str(p.pid)+' '+p.stdout.readline().strip(),flush=True); time.sleep(60)"
        parent=subprocess.Popen([sys.executable,'-B','-c',parent_code,child_code],stdout=subprocess.PIPE,text=True)
        child=None
        try:
            line=parent.stdout.readline();self.assertIn('armed',line);child=int(line.split()[0])
            parent.kill();parent.wait(timeout=5)
            running=True
            for _ in range(50):
                try:running=Path(f'/proc/{child}/stat').read_text().split(') ',1)[1][0] not in ('Z','X')
                except FileNotFoundError:running=False
                if not running:break
                time.sleep(.05)
            self.assertFalse(running,'worker survived controller death')
        finally:
            if parent.poll() is None:parent.kill();parent.wait()
            parent.stdout.close()
            if child:
                try:os.kill(child,signal.SIGKILL)
                except ProcessLookupError:pass

    def test_archive_is_verified_and_does_not_overwrite(self):
        from src.py.experiments.formal_v1.diagnostic_archive import archive_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();(root/'data').write_text('evidence')
            result=archive_run(root,Path(tmp)/'archive')
            self.assertEqual(result['verification_errors'],[])
            self.assertEqual(result['files'],2)
            with self.assertRaises(FileExistsError):archive_run(root,Path(tmp)/'archive')

    def test_native_signal_does_not_trigger_speculative_infrastructure_retry(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        for code in (-9,-11,-6,1):
            result=d.exit_state({'status':'running','epochs_completed':8},code)
            self.assertEqual(result['status'],'failed')
            self.assertFalse(d.retryable(result))
        self.assertFalse(d.retryable(d.exit_state({'status':'interrupted','error_type':'InterruptedError'},-15)))

    def test_analysis_of_missing_tasks_remains_incomplete(self):
        from src.py.experiments.formal_v1 import diagnostic_report as report
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d.write(root/'catalog.json',{})
            d.write(root/'outcomes.json',{'D-A-7':{'status':'failed','error_type':'ValueError'}})
            with patch.object(d,'validate_binding'):
                result=report.analyze(root)
            self.assertFalse(result['complete']);self.assertEqual(len(result['missing']),3)
            self.assertNotIn('budget',result)

    def test_budget_separates_observation_from_formal_projection(self):
        from src.py.experiments.formal_v1.diagnostic_report import budget_rows
        rows=[{'group':g,'training_seconds':1000.,'validation_seconds':200.,'wall_seconds':1300.,'epochs_completed':100,'validation_count':10} for g in ('A','B','C')]
        result=budget_rows(rows)
        self.assertEqual(result['candidate_epochs'],1000)
        self.assertFalse(result['formal_epochs_frozen'])
        self.assertEqual(result['per_group']['B']['observed_train_epoch_seconds'],10)
        self.assertEqual(result['per_group']['B']['projected_training_seconds'],10000)
        self.assertEqual(result['per_group']['B']['projected_validation_seconds'],2000)
        self.assertEqual(result['trace_seconds'],'unknown: synchronized instrumentation not measured')
        self.assertEqual(result['reserve_fraction'],.30)

    def test_transient_retry_is_bounded_and_all_attempts_are_retained(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'catalog.json').write_text('{}')
            def fail(root,task,attempt,catalog):
                out=root/'jobs'/task/f'attempt-{attempt:02d}'
                out.mkdir(parents=True)
                d.write(out/'status.json',{'status':'failed','error_type':'InfrastructureError'})
                return {'status':'failed','error_type':'InfrastructureError'}
            with patch.object(d,'dispatch',side_effect=fail):
                result=d.run_one(root,'D-B-7',{})
            self.assertEqual(result['status'],'failed')
            self.assertEqual(len(list((root/'jobs/D-B-7').glob('attempt-*'))),2)
            with patch.object(d,'dispatch',side_effect=AssertionError('third attempt')):
                self.assertEqual(d.run_one(root,'D-B-7',{})['status'],'failed')

    def test_deterministic_failure_is_not_retried_and_stop_never_dispatches(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fail(root,task,attempt,catalog):
                out=root/'jobs'/task/f'attempt-{attempt:02d}';out.mkdir(parents=True)
                d.write(out/'status.json',{'status':'failed','error_type':'ValueError'})
                return {'status':'failed','error_type':'ValueError'}
            with patch.object(d,'dispatch',side_effect=fail):
                d.run_one(root,'D-A-7',{})
            self.assertEqual(len(list((root/'jobs/D-A-7').glob('attempt-*'))),1)
            (root/'STOP').touch()
            with patch.object(d,'dispatch',side_effect=AssertionError('stopped dispatch')):
                with self.assertRaises(InterruptedError):d.run_one(root,'D-C-7',{})

    def test_unknown_or_formal_task_is_rejected(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        for task in ('Q-A-101','T-C-201','D-tail-7','../D-A-7'):
            with self.assertRaises(ValueError):d.run_one(Path('/tmp'),task,{})

    def test_inventory_verification_detects_modified_and_extra_files(self):
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x').write_text('a')
            d.seal(root);d.verify_seal(root)
            (root/'x').write_text('b')
            with self.assertRaises(ValueError):d.verify_seal(root)
            (root/'x').write_text('a');(root/'extra').write_text('x')
            with self.assertRaises(ValueError):d.verify_seal(root)
