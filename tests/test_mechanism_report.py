from pathlib import Path
import tempfile
import unittest


class ReportTests(unittest.TestCase):
    def test_observation_aggregation_weights_query_metrics_by_positive_count(self):
        from src.py.experiments.formal_v1.mechanism_report import observation_groups
        rows=[]
        for n,mass in ((2,.2),(1,.8)):
            rows.append({'anchor':'initial-401','group':'M00',
                         'summary':{'positive_count':n,**{name:{'mean':mass} for name in
                         ('head_weight_mass','duplicate_fraction','position_ess','unique_ess','weight_entropy')}},
                         'gradients':{'gradient_norms':{'head':2.,'tail':3.,'negative':4.,'total':5.},
                                      'gradient_cosines':{'head/tail':.1}}})
        result=observation_groups(rows)
        self.assertAlmostEqual(result[0]['head_weight_mass'],.4)
        self.assertEqual(result[0]['cells'],2)
        self.assertEqual(result[0]['positive_count'],3)

    def test_static_outputs_reproduce_byte_identically(self):
        from src.py.experiments.formal_v1.mechanism_report import matrix_statistics,render
        matrix={str(seed):{'M00':.25,'M10':.27,'M01':.25,'M11':.26} for seed in (401,402,403)}
        summary={'status':'complete','best':matrix_statistics(matrix),'last':matrix_statistics(matrix)}
        records={}
        for seed in (401,402,403):
            for group,value in matrix[str(seed)].items():
                best={'overall_mrr':value,'head_mrr':value-.02,'tail_mrr':value+.02}
                validation=[{'epoch':epoch,'training_seconds':epoch*3,'wall_seconds':epoch*4,**best} for epoch in (10,20,30)]
                records[f'V-{group}-{seed}']={'state':{'group':group,'seed':seed,'best_epoch':10,
                                'training_seconds':90,'validation_seconds':30,'wall_seconds':120},
                                'best':best,'last':best,'validation':validation}
        observation_rows=[]
        for anchor in ('initial-401','A-101','C-101'):
            for group in ('M00','M10','M01','M11'):
                observation_rows.append({'anchor':anchor,'group':group,
                    'summary':{'positive_count':2,**{name:{'mean':.5} for name in
                    ('head_weight_mass','duplicate_fraction','position_ess','unique_ess','weight_entropy')}},
                    'gradients':{'gradient_norms':{'head':2.,'tail':3.,'negative':4.,'total':5.},
                                 'gradient_cosines':{'head/tail':.1}}})
        records['O-TENSOR']={'observations':observation_rows}
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            a.mkdir();b.mkdir()
            render(a,summary,records);render(b,summary,records)
            self.assertTrue((a/'validation_vs_epoch.svg').exists())
            self.assertTrue((a/'mechanism_weight_mass.svg').exists())
            self.assertEqual({p.name:p.read_bytes() for p in a.iterdir()},
                             {p.name:p.read_bytes() for p in b.iterdir()})


if __name__=='__main__':unittest.main()
