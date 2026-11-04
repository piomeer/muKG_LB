"""Versioned configuration, provenance and compute-budget freeze."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math,statistics,sys,time,platform,subprocess
import torch
from scipy.stats import t

@dataclass(frozen=True)
class Config:
    protocol: str='formal-v1'
    dim: int=250
    margin: float=7.0
    adv: float=2.0
    learning_rate: float=.0005
    negatives: int=150
    batch_size: int=5000
    max_epochs: int=2000
    validation_every: int=10
    min_epochs: int=100
    patience: int=5
    min_delta: float=.0001
    query_chunk: int=32
    entity_chunk: int=2048
    max_rejection_rounds: int=64
    def to_dict(self):return asdict(self)

class EarlyStopping:
    def __init__(self,min_epoch=100,patience=5,delta=.0001):
        self.min_epoch,self.patience,self.delta=min_epoch,patience,delta
        self.best=float('-inf');self.anchor=float('-inf');self.bad=0;self.best_epoch=None
    def observe(self,epoch,value):
        if not math.isfinite(value):raise ValueError('Nonfinite validation metric')
        improved=value>self.best
        if improved:self.best=value;self.best_epoch=epoch
        if value-self.anchor>=self.delta:self.anchor=value;self.bad=0
        else:self.bad+=1
        return improved,epoch>=self.min_epoch and self.bad>=self.patience

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def new_run(root,metadata):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'manifest.json',metadata);write_json(root/'status.json',{'status':'running','started_at_unix':time.time()})
def provenance(data_dir):
    source=Path(__file__).parent
    return {'sources':{p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*.py'))},
            'data':{name:sha(Path(data_dir)/name) for name in ('train2id.txt','valid2id.txt','entity2id.txt','relation2id.txt')}}
def environment():
    available=torch.cuda.is_available();hardware=None
    if available:
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        hardware=result.stdout.strip()
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if available else None,'hardware':hardware,
            'cuda_available':available,'torch_threads':torch.get_num_threads()}
def paired_summary(values):
    if not values or any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('Positive finite paired ratios required')
    logs=list(map(math.log,values));mean=statistics.mean(logs);interval=None
    if len(logs)>1:
        d=t.ppf(.975,len(logs)-1)*statistics.stdev(logs)/math.sqrt(len(logs));interval=[math.exp(mean-d),math.exp(mean+d)]
    return {'n_pairs':len(values),'paired_ratios':values,'geometric_mean':math.exp(mean),'ci95':interval}
def budget(summary):
    jobs={j['group']:j for j in summary['jobs']}
    if set(jobs)!={'cpu','balanced','tail'} or any(j['status']!='complete' for j in jobs.values()):raise ValueError('Three complete diagnostic jobs required')
    cfg=Config(**summary['config']);quality=timing=0
    for group in ('cpu','balanced'):
        j=jobs[group]
        if j['epochs_completed']<=0 or j['validation_count']<=0:raise ValueError('Diagnostic training and full validation are required')
        if any(not math.isfinite(j[k]) or j[k]<=0 for k in ('training_seconds','validation_seconds')):raise ValueError('Positive finite measured durations required')
        epoch=j['training_seconds']/j['epochs_completed'];evaluation=j['validation_seconds']/j['validation_count']
        quality+=3*(cfg.max_epochs*epoch+(cfg.max_epochs//cfg.validation_every+20466/17535)*evaluation)
        timing+=50*epoch+20*epoch
    return {'quality_hours_upper_projection':quality/3600,'timing_hours_projection':timing/3600,
            'reserve_fraction':.30,'required_hours_with_reserve':1.3*(quality+timing)/3600,
            'assumption':'full epoch cap, observed mean diagnostic cost, test/validation query-count ratio; no early-stop savings assumed'}
def freeze(diagnostic,output,available_hours,current_provenance):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads((Path(diagnostic)/'suite.json').read_text())
    if summary['status']!='complete' or summary['provenance']!=current_provenance:raise ValueError('Complete current-source diagnostics required')
    from .controller import analyze
    if summary.get('kind')!='diagnostic' or summary.get('environment')!=environment():raise ValueError('Current-environment diagnostic suite required')
    verified=analyze(diagnostic)
    if verified['verdict']!='DESCRIPTIVE':raise ValueError('Verified diagnostic matrix required')
    estimate=verified['budget']
    if not math.isfinite(available_hours) or available_hours<estimate['required_hours_with_reserve']:raise ValueError('Insufficient confirmed GPU hours including 30% reserve')
    payload={'protocol':'formal-v1','config':summary['config'],'environment':summary['environment'],'provenance':current_provenance,'diagnostic_sha256':sha(Path(diagnostic)/'suite.json'),
             'available_hours':available_hours,'budget':estimate,'quality_seeds':[101,102,103],'timing_seeds':[201,202,203,204,205],
             'groups':['cpu','balanced'],'quality_ratio_target':.95}
    payload['freeze_sha256']=digest(payload);write_json(output,payload);return payload

def validate_frozen(path,current_provenance):
    value=json.loads(Path(path).read_text());stored=value.pop('freeze_sha256')
    if digest(value)!=stored or value['provenance']!=current_provenance or value.get('environment')!=environment():raise ValueError('Frozen configuration/source/data binding mismatch')
    value['freeze_sha256']=stored;return value
