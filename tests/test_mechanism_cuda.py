from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.py.experiments.formal_v1 import core, protocol as p, runtime
from src.py.experiments.formal_v1.mechanism_sampling import PolicySampler


@unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
class CUDATests(unittest.TestCase):
    def test_cuda_bridge_scores_gradients_and_adam(self):
        runtime.seed_all(23)
        batch=torch.tensor([[0,0,1],[2,1,3],[4,0,5]])
        known={tuple(row) for row in batch.tolist()}
        model_a=core.TransE(256,2).cuda()
        model_b=core.TransE(256,2).cuda()
        model_b.load_state_dict(model_a.state_dict())
        old=core.TensorSampler(256,2,150,known,device='cuda')
        new=PolicySampler(256,2,150,known,'M00',23,'cuda')
        new.candidate_generator.set_state(torch.cuda.get_rng_state())
        negatives_a=old.generate(batch)
        negatives_b=new.generate(batch)
        self.assertTrue(torch.equal(negatives_a,negatives_b))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(),new.candidate_generator.get_state()))
        positive=batch.cuda()
        losses=[]
        for model,negative in ((model_a,negatives_a),(model_b,negatives_b)):
            loss=core.adversarial_loss(model(*positive.unbind(-1))[:,None],model(*negative.unbind(-1)))
            loss.backward();losses.append(loss)
        self.assertTrue(torch.equal(losses[0],losses[1]))
        for a,b in zip(model_a.parameters(),model_b.parameters()):
            if a.requires_grad:self.assertTrue(torch.equal(a.grad,b.grad))
        for model in (model_a,model_b):torch.optim.Adam(model.parameters(),lr=.0005).step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model_a.parameters(),model_b.parameters())))

    def test_four_cuda_policies_handle_short_and_empty_batches(self):
        batch=torch.tensor([[0,0,1],[2,1,3],[4,0,5]])
        known={tuple(row) for row in batch.tolist()}
        for group in ('M00','M10','M01','M11'):
            sampler=PolicySampler(256,2,150,known,group,401,'cuda')
            output=sampler.generate(batch)
            self.assertEqual(output.shape,(3,150,3))
            for positive,row in zip(batch.tolist(),output.cpu().tolist()):
                self.assertTrue(all(tuple(triple) not in known for triple in row))
                heads=sum(triple[0]!=positive[0] for triple in row)
                self.assertIn(heads,(0,150) if group in ('M10','M11') else (75,))
                if group in ('M01','M11'):self.assertEqual(len(set(map(tuple,row))),150)
            self.assertEqual(sampler.generate(torch.empty((0,3),dtype=torch.long)).shape,(0,150,3))

    def test_cuda_gradient_observation_matches_chunking(self):
        from src.py.experiments.formal_v1.mechanism_observe import gradient_observation
        runtime.seed_all(42)
        model=core.TransE(256,2,16).cuda()
        positive=torch.tensor([[0,0,1],[2,1,3],[4,0,5]],device='cuda')
        sampler=PolicySampler(256,2,150,set(map(tuple,positive.cpu().tolist())),'M00',401,'cuda',observe=True)
        negative=sampler.generate(positive)
        direction=sampler.last_observation['directions']
        a,_=gradient_observation(model,positive,negative,direction,chunk=3,counterfactual=True)
        b,_=gradient_observation(model,positive,negative,direction,chunk=1,counterfactual=True)
        self.assertLess(a['gradient_reconstruction_relative_error'],1e-5)
        for key,value in a['gradient_norms'].items():self.assertAlmostEqual(value/b['gradient_norms'][key],1.,places=5)

    def test_tiny_cuda_validation_without_test(self):
        from src.py.experiments.formal_v1.mechanism_jobs import execute
        from test_mechanism_jobs import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'data')
            cfg=replace(p.Config(),dim=4,negatives=2,batch_size=2,max_epochs=2,validation_every=1)
            identities=[]
            for group in ('M00','M10','M01','M11'):
                result=execute(root/group,root/'data',cfg,
                               {'id':group,'kind':'validation','group':group,'seed':401},{},'cuda')
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['validation_count'],2)
                self.assertFalse((root/group/'test_metrics.json').exists())
                identities.append((root/group/'initial_weights.json').read_text())
            self.assertTrue(all(value==identities[0] for value in identities))

    def test_preflight_memory_gate_includes_validation_peak(self):
        from src.py.experiments.formal_v1.mechanism_jobs import execute
        from test_mechanism_jobs import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'data')
            cfg=replace(p.Config(),dim=4,negatives=2,batch_size=2)
            original=core.evaluate
            evaluated=False
            def evaluate(*args,**kwargs):
                nonlocal evaluated
                result=original(*args,**kwargs)
                evaluated=True
                return result
            total=torch.cuda.get_device_properties(0).total_memory
            def reserved():return int(.95*total) if evaluated else 1024
            with patch.object(core,'evaluate',side_effect=evaluate),patch.object(torch.cuda,'max_memory_reserved',side_effect=reserved):
                with self.assertRaisesRegex(MemoryError,'90%'):
                    execute(root/'probe',root/'data',cfg,{'id':'P-M00-900','kind':'probe','group':'M00','seed':900},{},'cuda')


if __name__=='__main__':unittest.main()
