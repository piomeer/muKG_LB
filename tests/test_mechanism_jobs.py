import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.py.experiments.formal_v1 import core, protocol as p


def fixture(root):
    root.mkdir()
    (root/'entity2id.txt').write_text('6\n')
    (root/'relation2id.txt').write_text('1\n')
    (root/'train2id.txt').write_text('3\n0 1 0\n2 3 0\n4 5 0\n')
    (root/'valid2id.txt').write_text('1\n0 2 0\n')


class JobTests(unittest.TestCase):
    def test_validation_job_finishes_without_test_and_keeps_first_best(self):
        from src.py.experiments.formal_v1.mechanism_jobs import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root/'data')
            cfg=p.Config(dim=4, negatives=2, batch_size=2, max_epochs=3, validation_every=1)
            scores={'ranks':{'head':[1.0],'tail':[2.0]}}
            scores.update({side:{'mrr':.75,'hits@1':.5,'hits@3':1.,'hits@10':1.} for side in ('head','tail','overall')})
            with patch.object(core,'evaluate',return_value=scores):
                result=execute(root/'out',root/'data',cfg,{'id':'V-M00-401','kind':'validation','group':'M00','seed':401},{},'cpu')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['epochs_completed'],3)
            self.assertEqual(result['validation_count'],3)
            self.assertEqual(result['best_epoch'],1)
            self.assertEqual(torch.load(root/'out/last.pt',weights_only=True)['epoch'],3)
            self.assertFalse((root/'out/test_metrics.json').exists())
            self.assertTrue((root/'out/artifacts.json').exists())
            with self.assertRaises(FileExistsError):
                execute(root/'out',root/'data',cfg,{'id':'V-M00-401','kind':'validation','group':'M00','seed':401},{},'cpu')

    def test_paired_initial_weights_and_positive_order(self):
        from src.py.experiments.formal_v1.mechanism_jobs import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root/'data')
            cfg=p.Config(dim=4,negatives=2,batch_size=2,max_epochs=1,validation_every=1)
            identities=[]
            for group in ('M00','M10','M01','M11'):
                execute(root/group,root/'data',cfg,{'id':group,'kind':'validation','group':group,'seed':401},{},'cpu')
                identities.append(tuple(json.loads((root/group/name).read_text()) for name in ('initial_weights.json','batch_order.json')))
            self.assertTrue(all(item==identities[0] for item in identities))

    def test_gradient_chunking_reconstructs_and_keeps_parameters(self):
        from src.py.experiments.formal_v1.mechanism_observe import gradient_observation
        torch.manual_seed(42)
        model=core.TransE(8,1,4).double()
        positive=torch.tensor([[0,0,1],[2,0,3],[4,0,5]])
        negative=positive[:,None,:].expand(-1,2,-1).clone()
        negative[:,0,0]=6
        negative[:,1,2]=7
        head=torch.tensor([[True,False]]*3)
        before={k:v.clone() for k,v in model.state_dict().items()}
        full, full_arrays=gradient_observation(model,positive,negative,head,chunk=3,counterfactual=True)
        chunks, chunk_arrays=gradient_observation(model,positive,negative,head,chunk=1,counterfactual=True)
        self.assertLess(full['gradient_reconstruction_relative_error'],1e-10)
        self.assertIn('negative',full['gradient_norms'])
        self.assertIn('counterfactual_negative/negative',full['gradient_cosines'])
        for key in full['gradient_norms']:
            self.assertAlmostEqual(full['gradient_norms'][key],chunks['gradient_norms'][key],places=10)
        np.testing.assert_allclose(full_arrays['weights'],chunk_arrays['weights'],rtol=1e-12,atol=1e-12)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(param.grad is None for param in model.parameters()))

    def test_observed_historical_sampler_preserves_output_and_rng(self):
        from src.py.experiments.formal_v1.mechanism_observe import historical_observation
        positive=torch.tensor([[0,0,1],[2,0,3]])
        known={(0,0,1),(2,0,3)}
        random.seed(10);np.random.seed(10)
        expected=core.CPUSampler(6,2,known).generate(positive)
        expected_python=random.getstate();expected_numpy=np.random.get_state()
        random.seed(10);np.random.seed(10)
        actual,summary=historical_observation(positive,6,2,known)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(random.getstate(),expected_python)
        self.assertTrue(np.array_equal(np.random.get_state()[1],expected_numpy[1]))
        self.assertEqual(summary['positive_count'],2)
        self.assertGreaterEqual(summary['candidate_count'],4)

    def test_duplicate_weight_mass_and_zero_gradient_cosine(self):
        from src.py.experiments.formal_v1.mechanism_observe import sampling_summary, cosine
        pos=np.array([[0,0,1]])
        neg=np.array([[[2,0,1],[2,0,1],[0,0,3],[0,0,4]]])
        weights=np.array([[.4,.3,.2,.1]])
        score=np.array([[1.,1.,2.,3.]])
        result=sampling_summary(pos,neg,np.array([[True,True,False,False]]),score,weights,10,1)
        self.assertAlmostEqual(result['unique_count']['mean'],3)
        self.assertAlmostEqual(result['head_weight_mass']['mean'],.7)
        self.assertAlmostEqual(result['unique_ess']['mean'],1/(.7**2+.2**2+.1**2))
        self.assertIsNone(cosine([torch.zeros(2)],[torch.ones(2)]))

    def test_fixed_observation_jobs_save_reconstructable_materials(self):
        from src.py.experiments.formal_v1.mechanism_jobs import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root/'data')
            cfg=p.Config(dim=4,negatives=2,batch_size=2)
            task={'id':'O-A','kind':'observe_a','group':'A','seed':601}
            result=execute(root/'historical',root/'data',cfg,task,{},'cpu')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['observation_calls'],6)
            task={'id':'O-TENSOR','kind':'observe','group':'tensor','seed':601}
            result=execute(root/'tensor',root/'data',cfg,task,{'anchors':[]},'cpu')
            self.assertEqual(result['observation_calls'],24)
            self.assertEqual(len(list((root/'tensor/arrays').glob('*.npz'))),24)


if __name__=='__main__':
    unittest.main()
