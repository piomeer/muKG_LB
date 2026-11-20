import unittest,tempfile,json
from pathlib import Path
from scripts import research_archive as a
class ArchiveTests(unittest.TestCase):
 def test_roundtrip_ignored_evidence_and_tamper_detection(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'repo'; r.mkdir(); (r/'raw.csv').write_text('untracked evidence'); (r/'code.py').write_text('x=1')
   (r/'.aws').mkdir(); (r/'.aws'/'credentials').write_text('secret')
   (r/'node_modules').mkdir(); (r/'node_modules'/'x').write_text('cache')
   dest=Path(d)/'archive'; a.snapshot(r,dest)
   self.assertEqual(a.verify(dest),[])
   self.assertEqual((dest/'snapshot/raw.csv').read_text(),'untracked evidence')
   self.assertFalse((dest/'snapshot/.aws').exists())
   restored=Path(d)/'restored'; a.restore(dest,restored)
   self.assertEqual((restored/'code.py').read_text(),'x=1')
   (dest/'snapshot/raw.csv').write_text('changed'); self.assertTrue(a.verify(dest))
 def test_refuses_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'r';r.mkdir(); dest=Path(d)/'a';a.snapshot(r,dest)
   with self.assertRaises(FileExistsError):a.snapshot(r,dest)
if __name__=='__main__':unittest.main()
