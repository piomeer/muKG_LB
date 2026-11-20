import unittest,tempfile,json
from pathlib import Path
from src.py.experiments.formal_v1 import protocol as p
class ProtocolTests(unittest.TestCase):
 def test_best_checkpoint_is_independent_of_patience_delta(self):
  s=p.EarlyStopping(min_epoch=30,patience=2,delta=.0001)
  self.assertEqual(s.observe(10,.1),(True,False))
  self.assertEqual(s.observe(20,.10001),(True,False))
  self.assertEqual(s.observe(30,.10002),(True,True))
  self.assertEqual(s.best_epoch,30)
 def test_freeze_requires_complete_diagnostics_and_sufficient_budget(self):
  with tempfile.TemporaryDirectory() as d:
   from test_formal_v1_integrity import diagnostic
   root=Path(d);run,prov=diagnostic(root)
   with self.assertRaises(ValueError):p.freeze(run,root/'bad.json',0,prov)
   result=p.freeze(run,root/'frozen.json',100,prov)
   self.assertGreater(result['budget']['required_hours_with_reserve'],0)
   p.validate_frozen(root/'frozen.json',prov)
   with self.assertRaises(ValueError):p.validate_frozen(root/'frozen.json',{'sources':{'core':'changed'},'data':{'train':'def'}})
   with self.assertRaises(FileExistsError):p.freeze(run,root/'frozen.json',100,prov)
 def test_new_output_is_not_overwritten_and_failure_is_retained(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'run';p.new_run(root,{'a':1})
   with self.assertRaises(FileExistsError):p.new_run(root,{'a':2})
   self.assertEqual(json.loads((root/'status.json').read_text())['status'],'running')
 def test_pair_summary_does_not_treat_epochs_as_independent_runs(self):
  result=p.paired_summary([2,4,8])
  self.assertAlmostEqual(result['geometric_mean'],4)
  self.assertEqual(result['n_pairs'],3)
  self.assertEqual(p.paired_summary([2])['ci95'],None)
if __name__=='__main__':unittest.main()
