import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import run_formal_v1_campaign as campaign
from src.py.experiments.formal_v1 import protocol as p

class CampaignTests(unittest.TestCase):
 def test_budget_failure_prevents_all_formal_jobs_and_retains_status(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'campaign'
   with patch.object(campaign.c,'preflight'),patch.object(campaign.c,'run_suite') as run,patch.object(campaign.c,'export_analysis'),patch.object(campaign.p,'provenance',return_value={}),patch.object(campaign.p,'environment',return_value={}),patch.object(campaign.p,'freeze',side_effect=ValueError('budget exceeded')):
    with self.assertRaises(ValueError):campaign.run(root,Path(d),720)
   self.assertEqual([call.args[2] for call in run.call_args_list],['diagnostic'])
   import json
   self.assertEqual(json.loads((root/'status.json').read_text())['status'],'failed')
   with self.assertRaises(FileExistsError):campaign.run(root,Path(d),720)
