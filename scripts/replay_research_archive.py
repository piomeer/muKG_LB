import csv,json,math,statistics
from pathlib import Path
from scipy.stats import t

def replay(root):
 root=Path(root);r=root/'output/results/c1_r1_combined_rerun';epochs={};sd={}
 read=lambda p:list(csv.DictReader(p.open()))
 for seed in range(42,48):
  for group in ('BL','GPU'):
   d=r/'jobs'/f'throughput_{group}_seed{seed}'
   if seed==45:d=r/'reruns/throughput_seed45_attempt2/jobs'/f'throughput_{group}_seed{seed}'
   epochs[group,seed]=statistics.mean(int(x['epoch_time_ns']) for x in read(d/'per_epoch.csv'))
   full=[x for x in read(r/'jobs'/f'trace_{group}_seed{seed}'/'per_step.csv') if x['is_partial'].lower()=='false' and int(x['batch_size_actual'])==5000]
   sd[group,seed]=statistics.mean(statistics.pstdev(int(x['neg_time_ns']) for x in full if x['epoch']==e) for e in sorted({x['epoch'] for x in full}))
 def interval(v):
  logs=list(map(math.log,v));m=statistics.mean(logs);d=t.ppf(.975,len(v)-1)*statistics.stdev(logs)/math.sqrt(len(v))
  return {'estimate':math.exp(m),'ci95':[math.exp(m-d),math.exp(m+d)]}
 out={'epoch_speedup':interval([epochs['BL',s]/epochs['GPU',s] for s in range(42,48)]),'sd_compression':interval([sd['BL',s]/sd['GPU',s] for s in range(42,48)]),'quality_seed42':{}}
 for group in ('bl','gpu'):
  data=json.loads((root/f'output/results/c1_q1_quality_formal/jobs/{group}_seed42/test_ranks.json').read_text())
  out['quality_seed42'][group]={side:statistics.mean(1/r for r in data['filtered_ranks_'+side]) for side in ('head','tail')}
 return out
if __name__=='__main__':
 import sys
 print(json.dumps(replay(sys.argv[1]),indent=2))
