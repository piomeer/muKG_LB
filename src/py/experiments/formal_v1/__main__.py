import argparse,json,signal,subprocess,sys
from pathlib import Path
from . import controller as c,protocol as p,runtime

def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
def main():
    parser=argparse.ArgumentParser(description='Formal v1 research runtime')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','diagnostic','formal'):
        cmd=sub.add_parser(name);cmd.add_argument('--output',type=Path,required=True);cmd.add_argument('--data',type=Path,default=c.ROOT/'src/py/data/FB15K237')
        if name=='diagnostic':cmd.add_argument('--preflight',type=Path,required=True)
        if name=='formal':cmd.add_argument('--frozen',type=Path,required=True);cmd.add_argument('--kind',choices=['quality','timing'],required=True)
    sub.add_parser('check')
    cmd=sub.add_parser('freeze');cmd.add_argument('--diagnostic',type=Path,required=True);cmd.add_argument('--output',type=Path,required=True);cmd.add_argument('--available-hours',type=float,required=True);cmd.add_argument('--data',type=Path,default=c.ROOT/'src/py/data/FB15K237')
    cmd=sub.add_parser('analyze');cmd.add_argument('--run',type=Path,required=True);cmd.add_argument('--output',type=Path,required=True)
    cmd=sub.add_parser('_worker');cmd.add_argument('--request',type=Path,required=True)
    a=parser.parse_args();signal.signal(signal.SIGTERM,interrupted)
    if a.command=='check':return subprocess.call([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_formal_v1*.py'],cwd=c.ROOT)
    if a.command=='preflight':result=c.preflight(a.output,a.data)
    elif a.command=='diagnostic':result=c.run_suite(a.output,a.data,'diagnostic',preflight_path=a.preflight)
    elif a.command=='formal':result=c.run_suite(a.output,a.data,a.kind,frozen_path=a.frozen)
    elif a.command=='freeze':result=p.freeze(a.diagnostic,a.output,a.available_hours,p.provenance(a.data))
    elif a.command=='analyze':result=c.export_analysis(a.run,a.output)
    else:
        r=json.loads(a.request.read_text())
        if r['kind']=='probe':result=c.probe(r)
        else:result=runtime.run_job(r['root'],r['data'],p.Config(**r['config']),r['group'],r['seed'],r['kind'],freeze_hash=r['freeze_sha256'])
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
