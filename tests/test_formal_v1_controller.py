import unittest,tempfile,json
from pathlib import Path
from src.py.experiments.formal_v1 import controller as c
from src.py.experiments.formal_v1.protocol import Config,environment
class ControllerTests(unittest.TestCase):
 def test_quality_matrix_is_counterbalanced_and_disjoint(self):
  rows=c.matrix('quality')
  self.assertEqual(rows,[('cpu',101,'quality'),('balanced',101,'quality'),('balanced',102,'quality'),('cpu',102,'quality'),('cpu',103,'quality'),('balanced',103,'quality')])
  self.assertEqual(len(c.matrix('timing')),20)
 def test_preflight_binding_rejects_source_and_data_changes(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'preflight.json';f.write_text(json.dumps({'status':'passed','config':Config().to_dict(),'provenance':{'a':1},'environment':environment()}))
   self.assertEqual(c.read_preflight(f,{'a':1}).batch_size,5000)
   with self.assertRaises(ValueError):c.read_preflight(f,{'a':2})
 def test_partial_analysis_never_claims_target_met(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'suite.json').write_text(json.dumps({'status':'failed','kind':'quality','jobs':[]}))
   result=c.analyze(root)
   self.assertEqual(result['verdict'],'INCOMPLETE');self.assertFalse(result['quality_target_supported'])
if __name__=='__main__':unittest.main()
