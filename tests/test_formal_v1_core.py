import ast,random,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.py.experiments.formal_v1 import core as c
REF=Path(__file__).resolve().parents[1]/'src/py/experiments/formal_v1/reference'
def historical():
 class Base(torch.nn.Module):
  def __init__(self,args,kgs):
   super().__init__();self.args=args;self.ent_tot=kgs.entities_num;self.rel_tot=kgs.relations_num
 env={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'BasicModel':Base,'random':random,'np':np}
 for file,symbol in [('transe.py','TransE'),('losses.py','logistic_adv_loss_torch')]:
  node=next(n for n in ast.parse((REF/file).read_text()).body if getattr(n,'name',None)==symbol)
  exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),env)
 cls=next(n for n in ast.parse((REF/'dataloader.py').read_text()).body if getattr(n,'name',None)=='PyTorchTrainDataset')
 node=next(n for n in cls.body if getattr(n,'name',None)=='generate_neg_triples_fast')
 exec(compile(ast.Module(body=[node],type_ignores=[]),'cpu_reference','exec'),env)
 return env
class CoreTests(unittest.TestCase):
 def test_historical_multi_positive_150_negative_layout(self):
  env=historical();args=SimpleNamespace(dim=250,ent_l2_norm=False,margin=7,loss='logistic_adv')
  torch.manual_seed(17);old=env['TransE'](SimpleNamespace(entities_num=200,relations_num=2),args)
  torch.manual_seed(17);new=c.TransE(200,2,dim=250)
  pos=torch.tensor([[0,0,1],[2,1,3],[4,0,5]])
  neg=pos[:,None,:].expand(3,150,3).clone();neg[:,:,2]=torch.arange(6,156)[None,:]
  def old_score(x):
   h,r,t=x.unbind(-1);return old({'batch_h':h,'batch_r':r,'batch_t':t})
  a=old_score(pos)[:,None];b=new(*pos.unbind(-1))[:,None]
  an=old_score(neg.reshape(-1,3)).reshape(3,150);bn=new(*neg.unbind(-1))
  torch.testing.assert_close(a,b);torch.testing.assert_close(an,bn)
  la=env['logistic_adv_loss_torch'](a,an,2);lb=c.adversarial_loss(b,bn,2)
  torch.testing.assert_close(la,lb);la.backward();lb.backward()
  for x,y in zip(old.parameters(),new.parameters()):
   if x.requires_grad:torch.testing.assert_close(x.grad,y.grad)
  torch.optim.Adam(old.parameters(),lr=.0005).step();torch.optim.Adam(new.parameters(),lr=.0005).step()
  for x,y in zip(old.parameters(),new.parameters()):torch.testing.assert_close(x,y)
 def test_historical_scores_loss_gradients_and_adam_step(self):
  env=historical();args=SimpleNamespace(dim=8,ent_l2_norm=False,margin=7,loss='logistic_adv')
  torch.manual_seed(4);old=env['TransE'](SimpleNamespace(entities_num=9,relations_num=2),args)
  torch.manual_seed(4);new=c.TransE(9,2,dim=8)
  h=torch.tensor([0,1,2,3]);r=torch.tensor([0,1,0,1]);t=torch.tensor([4,5,6,7])
  a=old({'batch_h':h,'batch_r':r,'batch_t':t});b=new(h,r,t)
  torch.testing.assert_close(a,b,rtol=0,atol=0)
  la=env['logistic_adv_loss_torch'](a[:2,None],a[2:,None],2);lb=c.adversarial_loss(b[:2,None],b[2:,None],2)
  torch.testing.assert_close(la,lb,rtol=0,atol=0)
  la.backward();lb.backward()
  for x,y in zip(old.parameters(),new.parameters()):
   if x.requires_grad:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
  torch.optim.Adam(old.parameters(),lr=.0005).step();torch.optim.Adam(new.parameters(),lr=.0005).step()
  for x,y in zip(old.parameters(),new.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
 def test_cpu_sampler_matches_actual_historical_function(self):
  rows=[(0,0,1),(1,0,2)];known=set(rows);env=historical()
  random.seed(7);np.random.seed(7);old=env['generate_neg_triples_fast'](None,rows,known,list(range(8)),4)
  random.seed(7);np.random.seed(7);new=c.CPUSampler(8,4,known).generate(torch.tensor(rows))
  self.assertEqual([tuple(x) for x in new.reshape(-1,3).tolist()],old)
 def test_gpu_policy_counts_filtering_empty_and_impossible(self):
  pos=torch.tensor([[0,0,1],[2,0,3]]);known={(0,0,1),(0,0,2),(2,0,3),(4,0,3)}
  s=c.TensorSampler(6,1,4,known,mode='balanced',device='cpu');out=s.generate(pos)
  self.assertEqual(tuple(out.shape),(2,4,3))
  self.assertTrue(torch.equal(out[:,:2,2],pos[:,2,None].expand(-1,2)))
  self.assertTrue(torch.equal(out[:,2:,0],pos[:,0,None].expand(-1,2)))
  self.assertTrue(all(tuple(x) not in known for x in out.reshape(-1,3).tolist()))
  self.assertEqual(tuple(s.generate(pos[:0]).shape),(0,4,3))
  impossible={(0,0,i) for i in range(2)}
  with self.assertRaises(RuntimeError):c.TensorSampler(2,1,2,impossible,mode='tail',device='cpu',max_rounds=2).generate(torch.tensor([[0,0,0]]))
 def test_average_ties_filtering_and_chunk_invariance(self):
  class Constant(torch.nn.Module):
   def forward(self,h,r,t):return torch.zeros(torch.broadcast_shapes(h.shape,r.shape,t.shape))
  rows=[(0,0,1)];known={(0,0,1),(0,0,2)}
  a=c.evaluate(Constant(),rows,known,4,device='cpu',query_chunk=1,entity_chunk=1)
  b=c.evaluate(Constant(),rows,known,4,device='cpu',query_chunk=8,entity_chunk=4)
  self.assertEqual(a,b);self.assertEqual(a['ranks']['tail'],[2.0]);self.assertEqual(a['ranks']['head'],[2.5])
  self.assertAlmostEqual(a['overall']['mrr'],.45)
 def test_evaluator_matches_independent_bruteforce(self):
  torch.manual_seed(9);m=c.TransE(7,2,dim=4);rows=[(0,0,1),(2,1,3)];known=set(rows)|{(0,0,2),(4,1,3)}
  a=c.evaluate(m,rows,known,7,device='cpu',query_chunk=2,entity_chunk=3)
  for mode,col in [('head',0),('tail',2)]:
   expected=[]
   for triple in rows:
    def score(x):return float(m(*(torch.tensor([v]) for v in x)).detach())
    target=score(triple);better=equal=0
    for e in range(7):
     x=list(triple);x[col]=e
     if tuple(x)==triple or tuple(x) in known:continue
     v=score(x);better+=v<target;equal+=v==target
    expected.append(1+better+.5*equal)
   self.assertEqual(a['ranks'][mode],expected)
if __name__=='__main__':unittest.main()

@unittest.skipUnless(torch.cuda.is_available(),'CUDA not exposed to this process')
class CUDATests(unittest.TestCase):
 def test_actual_abc_sampling_devices_and_150_negatives(self):
  from src.py.experiments.formal_v1 import runtime,protocol
  cfg=protocol.Config();ds={'entities':200,'relations':2,'train':[(0,0,1),(2,1,3)]}
  initial=[]
  for group in ('A','B','C'):
   model,opt,sampler=runtime.build(ds,cfg,group,7,'cuda')
   self.assertEqual(model.ent_embeddings.weight.device.type,'cuda')
   initial.append(model.ent_embeddings.weight.detach().cpu())
   if group=='B':self.assertEqual(str(sampler.device),'cpu')
   if group=='C':self.assertEqual(str(sampler.device),'cuda')
   out=sampler.generate(torch.tensor(ds['train']))
   self.assertEqual(out.shape,(2,150,3))
   if group in ('B','C'):
    values=out.cpu()
    self.assertFalse(set(map(tuple,values.reshape(-1,3).tolist()))&set(ds['train']))
    self.assertTrue(torch.equal(values[:,:75,2],torch.tensor([1,3])[:,None].expand(2,75)))
    self.assertTrue(torch.equal(values[:,75:,0],torch.tensor([0,2])[:,None].expand(2,75)))
   del model,opt,sampler
  for item in initial[1:]:torch.testing.assert_close(initial[0],item,rtol=0,atol=0)
 def test_cuda_sampling_and_evaluation_match_independent_cpu_results(self):
  rows=[(0,0,1),(1,1,2),(3,0,4)];known=set(rows)|{(0,0,2),(2,1,2)}
  batch=torch.tensor(rows)
  for mode in ('balanced','tail'):
   out=c.TensorSampler(8,2,6,known,mode=mode,device='cuda').generate(batch).cpu()
   for i,positive in enumerate(rows):
    values=[tuple(x) for x in out[i].tolist()]
    self.assertFalse(set(values)&known);self.assertNotIn(positive,values)
    self.assertEqual(sum(x[0]!=positive[0] for x in values),3 if mode=='balanced' else 0)
    self.assertEqual(sum(x[2]!=positive[2] for x in values),3 if mode=='balanced' else 6)
  torch.manual_seed(13);cpu=c.TransE(8,2,dim=8);gpu=c.TransE(8,2,dim=8).cuda();gpu.load_state_dict(cpu.state_dict())
  self.assertEqual(c.evaluate(cpu,rows,known,8,'cpu',2,3),c.evaluate(gpu,rows,known,8,'cuda',2,3))
