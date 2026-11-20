from dataclasses import replace
from pathlib import Path
import json,tempfile,unittest
import torch
from src.py.experiments.formal_v1.protocol import Config
from src.py.experiments.formal_v1 import runtime as r
class RuntimeTests(unittest.TestCase):
 def dataset(self,root):
  root.mkdir();(root/'entity2id.txt').write_text('6\n');(root/'relation2id.txt').write_text('1\n')
  (root/'train2id.txt').write_text('3\n0 1 0\n2 3 0\n4 5 0\n');(root/'valid2id.txt').write_text('1\n0 2 0\n')
 def test_diagnostic_runs_without_test_file_and_preserves_partial_batch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.dataset(root/'data')
   cfg=replace(Config(),dim=4,negatives=2,batch_size=2,max_epochs=2,validation_every=1,min_epochs=2)
   result=r.run_job(root/'run',root/'data',cfg,'balanced',7,'diagnostic',device='cpu')
   self.assertEqual(result['epochs_completed'],2);self.assertEqual(result['validation_count'],2)
   self.assertFalse((root/'run/test_metrics.json').exists());self.assertTrue((root/'run/best.pt').exists())
   self.assertEqual(json.loads((root/'run/status.json').read_text())['status'],'complete')
   with self.assertRaises(FileExistsError):r.run_job(root/'run',root/'data',cfg,'balanced',7,'diagnostic',device='cpu')
 def test_trace_and_throughput_use_same_coverage_but_separate_outputs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.dataset(root/'data');cfg=replace(Config(),dim=4,negatives=2,batch_size=2)
   for mode in ('throughput','trace'):
    result=r.run_job(root/mode,root/'data',cfg,'balanced',201,mode,device='cpu')
    self.assertEqual(result['epochs_completed'],5)
   import csv
   with (root/'trace/steps.csv').open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),10);self.assertEqual(sum(x['is_partial']=='True' for x in rows),5)
   self.assertFalse((root/'throughput/steps.csv').exists())
 def test_quality_requires_freeze_binding(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):r.run_job(Path(d)/'run',Path(d)/'data',Config(),'cpu',101,'quality',device='cpu')
if __name__=='__main__':unittest.main()
