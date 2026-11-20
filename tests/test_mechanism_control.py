import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class ControlTests(unittest.TestCase):
    def test_catalog_has_only_two_observations_and_twelve_validation_tasks(self):
        from src.py.experiments.formal_v1 import mechanism as m
        tasks=m.task_catalog()
        self.assertEqual([t['id'] for t in tasks[:2]],['O-A','O-TENSOR'])
        self.assertEqual(len(tasks),14)
        self.assertEqual([t['group'] for t in tasks[2:]],
                         ['M00','M10','M11','M01','M10','M01','M00','M11','M01','M11','M10','M00'])
        self.assertEqual({t['kind'] for t in tasks},{'observe_a','observe','validation'})
        self.assertEqual(m.config().max_epochs,1000)

    def test_status_import_is_standard_library_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'status.json').write_text('{"status":"pending"}')
            code='''import sys
class Deny:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','numpy','scipy'):
   raise AssertionError('heavy import '+fullname)
sys.meta_path.insert(0,Deny())
from src.py.experiments.formal_v1.mechanism import status
print(status(sys.argv[1]))
'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(root)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('pending',result.stdout)

    def test_matrix_analysis_and_candidate_selection(self):
        from src.py.experiments.formal_v1.mechanism_report import matrix_statistics
        matrix={str(seed):{'M00':.25,'M10':.275,'M01':.25,'M11':.275} for seed in (401,402,403)}
        report=matrix_statistics(matrix)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['candidate'],'M10')
        self.assertAlmostEqual(report['comparisons']['M10/M00']['geometric_mean'],1.1)
        self.assertAlmostEqual(report['effects']['interaction']['geometric_mean'],1.0)
        del matrix['403']['M11']
        report=matrix_statistics(matrix)
        self.assertEqual(report['status'],'incomplete')
        self.assertIsNone(report['candidate'])

    def test_failed_results_are_not_silently_retried_or_replaced(self):
        from src.py.experiments.formal_v1 import mechanism as m
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task={'id':'O-A','kind':'observe_a','group':'A','seed':601}
            calls=[]
            def dispatch(run, record, item, output):
                calls.append(output.name)
                output.mkdir(parents=True)
                state={'status':'failed','error_type':'InfrastructureError' if len(calls)==1 else 'ValueError'}
                d.write(output/'status.json',state)
                return state
            with patch.object(m,'dispatch',side_effect=dispatch):
                result=m.run_one(root,{},task)
                self.assertEqual(calls,['attempt-01','attempt-02'])
                self.assertEqual(result['status'],'failed')
                m.run_one(root,{},task)
                self.assertEqual(len(calls),2)
            self.assertEqual(d.read(root/'jobs/O-A/attempt-01/status.json')['error_type'],'InfrastructureError')

    def test_user_stop_prevents_new_attempt(self):
        from src.py.experiments.formal_v1 import mechanism as m
        from src.py.experiments.formal_v1 import diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'STOP').write_text('stop')
            with patch.object(m,'dispatch') as dispatch:
                with self.assertRaises(InterruptedError):
                    m.run_one(root,{}, {'id':'O-A','kind':'observe_a','group':'A','seed':601})
                self.assertFalse(dispatch.called)

    def test_worker_rejects_unbound_inputs_before_environment_or_execution(self):
        from src.py.experiments.formal_v1 import mechanism as m
        task=m.task_catalog()[0]
        record={'freeze_sha256':'fixture','config':m.config().to_dict(),
                'provenance':{'frozen':True},'environment':{'frozen':True},'anchors':[]}
        request={'controller_pid':1,'config':m.config().to_dict(),'phase':'frozen',
                 'run':'/tmp/mechanism-worker-test','task':task,'freeze_sha256':'fixture',
                 'output':'/tmp/mechanism-worker-test/jobs/O-A/attempt-01',
                 'data_dir':'/tmp/unbound-inputs','expected_provenance':{'other':True},
                 'expected_environment':{'other':True},'binding':{}}
        with patch.object(m.d,'arm_parent_death'),patch.object(m,'read_freeze',return_value=record), \
             patch.object(m,'verify_live') as live:
            with self.assertRaisesRegex(ValueError,'binding'):
                m.worker(request)
            self.assertFalse(live.called)

    def test_frozen_task_cards_are_hash_bound(self):
        from src.py.experiments.formal_v1 import mechanism as m, diagnostics as d, protocol as p
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            body={'protocol':m.PROTOCOL,'config':m.config().to_dict()}
            for name,field in (('source','source_inventory_sha256'),('inputs','input_inventory_sha256'),
                               ('preparation','preparation_inventory_sha256'),('documents','documents_inventory_sha256')):
                (root/name).mkdir()
                body[field]=d.seal(root/name)
            d.write(root/'task_catalog.json',m.task_catalog())
            body['tasks_sha256']=d.sha(root/'task_catalog.json')
            (root/'TASKS.md').write_text('original cards')
            body['task_cards_sha256']=d.sha(root/'TASKS.md')
            d.write(root/'freeze.json',{**body,'freeze_sha256':p.digest(body)})
            self.assertEqual(m.read_freeze(root)['protocol'],m.PROTOCOL)
            (root/'TASKS.md').write_text('changed cards')
            with self.assertRaises(ValueError):m.read_freeze(root)

    def test_interrupted_task_stops_queue_until_explicit_resume(self):
        from src.py.experiments.formal_v1 import mechanism as m
        tasks=[{'id':'first','dependencies':[]},{'id':'second','dependencies':[]}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(m,'read_freeze',return_value={}),patch.object(m,'task_catalog',return_value=tasks), \
                 patch.object(m,'run_one',return_value={'status':'interrupted','error_type':'InterruptedError'}) as run:
                with self.assertRaises(InterruptedError):m.run(root)
                self.assertEqual(run.call_count,1)

    def test_unknown_worker_exit_can_resume_only_when_explicit(self):
        from src.py.experiments.formal_v1 import mechanism as m, diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            previous=root/'jobs/O-A/attempt-01'
            d.write(previous/'status.json',{'status':'failed','error_type':'UnclassifiedWorkerExit'})
            task={'id':'O-A','kind':'observe_a','group':'A','seed':601}
            def dispatch(run,record,item,output):
                output.mkdir(parents=True)
                state={'status':'failed','error_type':'InfrastructureError'}
                d.write(output/'status.json',state)
                return state
            with patch.object(m,'dispatch',side_effect=dispatch) as launch:
                m.run_one(root,{},task)
                self.assertEqual(launch.call_count,0)
                m.run_one(root,{},task,resume=True)
                self.assertEqual(launch.call_count,2)

    def test_sidecar_only_reserved_attempt_is_preserved_and_explicitly_resumed(self):
        from src.py.experiments.formal_v1 import mechanism as m, diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory=root/'jobs/O-A'
            d.write(directory/'attempt-01.request.json',{'controller_pid':99999999})
            original=(directory/'attempt-01.request.json').read_bytes()
            d.write(root/'active.json',{'output':str(directory/'attempt-01'),'controller_pid':99999999})
            task={'id':'O-A','kind':'observe_a','group':'A','seed':601}
            def dispatch(run,record,item,output):
                if output.with_name(output.name+'.request.json').exists():raise FileExistsError(output)
                output.mkdir(parents=True)
                state={'status':'failed','error_type':'ValueError'}
                d.write(output/'status.json',state)
                return state
            with patch.object(m,'dispatch',side_effect=dispatch) as launch:
                outcome=m.run_one(root,{},task)
                self.assertEqual(outcome['status'],'interrupted')
                self.assertFalse(launch.called)
                m.run_one(root,{},task,resume=True)
                self.assertEqual(launch.call_args.args[-1].name,'attempt-02')
            self.assertEqual((directory/'attempt-01.request.json').read_bytes(),original)

    def test_unsealed_completion_is_an_interruption_not_a_valid_result(self):
        from src.py.experiments.formal_v1 import mechanism as m, diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            previous=root/'jobs/O-A/attempt-01'
            d.write(previous/'status.json',{'status':'complete','observation_calls':15})
            with patch.object(m,'dispatch') as launch:
                result=m.run_one(root,{}, {'id':'O-A','kind':'observe_a','group':'A','seed':601})
                self.assertEqual(result['status'],'interrupted')
                self.assertEqual(result['error_type'],'UnsealedCompletion')
                self.assertFalse(launch.called)

    def test_acceptance_saves_the_test_sources(self):
        from src.py.experiments.formal_v1 import mechanism as m
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'checks'
            with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
                result=m.check(output)
            source=m.d.ROOT/'tests/test_mechanism_control.py'
            self.assertEqual((output/'test_sources'/source.name).read_bytes(),source.read_bytes())
            self.assertEqual(result['test_sources'][source.name],m.d.sha(source))

    def test_closed_run_rejects_dispatch_and_stop_mutation(self):
        from src.py.experiments.formal_v1 import mechanism as m
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'artifacts.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.run(root)
            m.main(['stop','--run',str(root)])
            self.assertFalse((root/'STOP').exists())


if __name__=='__main__':
    unittest.main()
