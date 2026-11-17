import csv
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from src.py.experiments.formal_v1 import core, runtime, protocol


def fixture(root):
    root.mkdir()
    (root/'entity2id.txt').write_text('6\n')
    (root/'relation2id.txt').write_text('1\n')
    (root/'train2id.txt').write_text('3\n0 1 0\n2 3 0\n4 5 0\n')
    (root/'valid2id.txt').write_text('1\n0 2 0\n')


class ABCBehaviorTests(unittest.TestCase):
    def test_b_sampler_is_cpu_tensor_and_pair_initialization_is_identical(self):
        ds={'entities':200,'relations':2,'train':[(0,0,1),(2,1,3)]}
        cfg=protocol.Config()
        built=[runtime.build(ds,cfg,g,7,'cpu') for g in ('A','B','C')]
        self.assertIsInstance(built[0][2],core.CPUSampler)
        self.assertIsInstance(built[1][2],core.TensorSampler)
        self.assertEqual(str(built[1][2].device),'cpu')
        for model,_,sampler in built[1:]:
            for a,b in zip(built[0][0].parameters(),model.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        out=built[1][2].generate(torch.tensor(ds['train']))
        self.assertEqual(out.shape,(2,150,3))
        self.assertTrue(torch.equal(out[:,:75,2],torch.tensor([1,3])[:,None].expand(2,75)))
        self.assertTrue(torch.equal(out[:,75:,0],torch.tensor([0,2])[:,None].expand(2,75)))
        self.assertFalse(set(map(tuple,out.reshape(-1,3).tolist())) & set(ds['train']))

    def test_diagnostic_never_stops_for_patience_and_saves_ranks_without_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'data')
            cfg=replace(protocol.Config(),dim=4,negatives=2,batch_size=2,max_epochs=3,validation_every=1,min_epochs=1,patience=1)
            real=core.evaluate
            def constant(*args,**kwargs):
                result=real(*args,**kwargs)
                result['overall']['mrr']=.25
                return result
            with patch.object(core,'evaluate',side_effect=constant):
                result=runtime.run_job(root/'run',root/'data',cfg,'B',7,'diagnostic','cpu')
            self.assertEqual(result['epochs_completed'],3)
            self.assertEqual(result['best_epoch'],1)
            self.assertEqual(len(list((root/'run/validation').glob('*.json'))),3)
            self.assertEqual(len(json.loads((root/'run/batch_order.json').read_text())),3)
            self.assertFalse((root/'run/test_metrics.json').exists())
            with (root/'run/validation.csv').open() as f:
                self.assertIn('head_hits@10',next(csv.DictReader(f)))
            with self.assertRaises(FileExistsError):
                runtime.run_job(root/'run',root/'data',cfg,'B',7,'diagnostic','cpu')

    def test_abc_worker_rejects_formal_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('quality','throughput','trace'):
                with self.assertRaises(ValueError):
                    runtime.run_job(Path(tmp)/kind,Path(tmp),protocol.Config(),'B',7,kind,'cpu',freeze_hash='x')

    def test_status_cli_is_read_only_and_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'status.json').write_text('{"status":"awaiting_review"}')
            guard=root/'sitecustomize.py'
            guard.write_text("import sys\nclass Guard:\n def find_spec(self,fullname,*args):\n  if fullname=='torch': raise RuntimeError('status imported torch')\nsys.meta_path.insert(0,Guard())\n")
            before=set(root.iterdir())
            p=subprocess.run([sys.executable,'-B','-m','src.py.experiments.formal_v1','status','--run',str(root)],env={**os.environ,'PYTHONPATH':str(root),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['status'],'awaiting_review')
            self.assertEqual(before,set(root.iterdir()))
