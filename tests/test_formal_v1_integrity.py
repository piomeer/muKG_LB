import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from src.py.experiments.formal_v1 import protocol as p,runtime,controller as c
from scripts import research_archive as archive

def dataset(root):
    root.mkdir();(root/'entity2id.txt').write_text('6\n');(root/'relation2id.txt').write_text('1\n')
    (root/'train2id.txt').write_text('3\n0 1 0\n2 3 0\n4 5 0\n');(root/'valid2id.txt').write_text('1\n0 2 0\n')
    return replace(p.Config(),dim=4,negatives=2,batch_size=2,max_epochs=1,validation_every=1)

def diagnostic(root):
    cfg=dataset(root/'data');run=root/'diagnostic';run.mkdir();prov=p.provenance(root/'data');jobs=[]
    for group in ('cpu','balanced','tail'):
        path=run/group;state=runtime.run_job(path,root/'data',cfg,group,7,'diagnostic',device='cpu');state['path']=group;jobs.append(state)
    value={'status':'complete','kind':'diagnostic','config':cfg.to_dict(),'provenance':prov,'environment':p.environment(),'jobs':jobs}
    p.write_json(run/'suite.json',value)
    return run,prov

class IntegrityTests(unittest.TestCase):
    def test_freeze_rejects_summary_only_without_verified_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,prov=diagnostic(root)
            (run/'cpu/artifacts.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):p.freeze(run,root/'frozen.json',100,prov)
    def test_freeze_rejects_modified_trace_and_summary_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,prov=diagnostic(root)
            summary=json.loads((run/'suite.json').read_text());summary['jobs'][0]['training_seconds']=.000001;p.write_json(run/'suite.json',summary)
            with self.assertRaises(ValueError):p.freeze(run,root/'frozen.json',100,prov)
    def test_archive_verification_covers_external_and_git_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();(src/'x').write_text('a');dest=root/'archive';archive.snapshot(src,dest)
            (dest/'history.bundle').write_text('git data');(dest/'seal.json').write_text(json.dumps({'metadata_hashes':{'history.bundle':archive.sha(dest/'history.bundle'),'manifest.json':archive.sha(dest/'manifest.json')}}))
            self.assertEqual(archive.verify(dest),[]);(dest/'history.bundle').write_text('corrupted');self.assertTrue(archive.verify(dest))
    def test_quality_analysis_rejects_wrong_freeze_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dataset(root/'data');(root/'data/test2id.txt').write_text('1\n0 3 0\n');run=root/'quality';run.mkdir();jobs=[]
            for group,seed,mode in c.matrix('quality'):
                name=f'{group}_{seed}';state=runtime.run_job(run/name,root/'data',cfg,group,seed,mode,device='cpu',freeze_hash='wrong');state['path']=name;jobs.append(state)
            p.write_json(run/'suite.json',{'status':'complete','kind':'quality','config':cfg.to_dict(),'provenance':p.provenance(root/'data'),'environment':p.environment(),'jobs':jobs,'frozen':{'freeze_sha256':'expected'}})
            with self.assertRaises(ValueError):c.analyze(run)
    def test_job_rejects_empty_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.write_json(root/'artifacts.json',{});p.write_json(root/'status.json',{'status':'complete'})
            with self.assertRaises(ValueError):c.verify_job(root)
    def test_archive_rejects_unlisted_additions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();(src/'x').write_text('a');dest=root/'archive';archive.snapshot(src,dest)
            (dest/'snapshot/unlisted').write_text('new');self.assertTrue(archive.verify(dest))
    def test_archive_rejects_linked_evidence_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();e=root/'evidence';e.mkdir();(e/'raw.csv').write_text('evidence');(src/'linked').symlink_to(e,target_is_directory=True)
            with self.assertRaises(ValueError):archive.snapshot(src,root/'archive')
if __name__=='__main__':unittest.main()
