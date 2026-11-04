"""Budget-gated, fail-fast serial campaign. No automatic retries or scheme changes."""
import argparse,json,signal,time,traceback
from pathlib import Path
from src.py.experiments.formal_v1 import controller as c,protocol as p

def run(root,data_dir,available_hours):
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    p.new_run(root,{'provenance':p.provenance(data_dir),'environment':p.environment(),'available_hours':available_hours,
                    'controller_sha256':p.sha(__file__),'policy':'diagnostic then budget freeze then quality and independent throughput/trace; stop on any failure'})
    state={'status':'running','stage':'preflight','started_at_unix':time.time()}
    def stage(name):
        state['stage']=name;p.write_json(root/'status.json',state);print(json.dumps(state),flush=True)
    try:
        stage('preflight');c.preflight(root/'preflight',data_dir)
        stage('diagnostic');c.run_suite(root/'diagnostic',data_dir,'diagnostic',preflight_path=root/'preflight/preflight.json')
        stage('diagnostic_analysis');c.export_analysis(root/'diagnostic',root/'diagnostic_analysis')
        stage('budget_freeze');remaining_hours=available_hours-(time.time()-state['started_at_unix'])/3600
        frozen=p.freeze(root/'diagnostic',root/'frozen.json',remaining_hours,p.provenance(data_dir))
        for kind in ('quality','timing'):
            stage(kind);c.run_suite(root/kind,data_dir,kind,frozen_path=root/'frozen.json')
            stage(kind+'_analysis');c.export_analysis(root/kind,root/(kind+'_analysis'))
        quality=json.loads((root/'quality_analysis/summary.json').read_text());timing=json.loads((root/'timing_analysis/summary.json').read_text())
        result={'status':'complete','frozen_sha256':frozen['freeze_sha256'],'quality_ratio':quality['primary'],'epoch_speedup':timing['epoch_speedup'],
                'point_estimates_meet_target':quality['quality_target_point_estimate'] and timing['epoch_speedup']['geometric_mean']>1,
                'both_ci_support_target':quality['quality_target_supported'] and timing['epoch_speedup']['ci95'][0]>1,
                'interpretation':'Three quality pairs do not guarantee sufficient power; completed negative or inconclusive results remain valid.'}
        p.write_json(root/'core_results.json',result)
        (root/'REPORT.md').write_text('# Formal v1 core experiment report\n\n'
            f"Paired GPU/CPU MRR: {result['quality_ratio']['geometric_mean']:.6f}; 95% CI {result['quality_ratio']['ci95']}.\n\n"
            f"Paired CPU/GPU epoch time: {result['epoch_speedup']['geometric_mean']:.6f}; 95% CI {result['epoch_speedup']['ci95']}.\n\n"
            f"Point targets met: {result['point_estimates_meet_target']}. Both confidence intervals support targets: {result['both_ci_support_target']}.\n\n"
            'Directional metrics and raw ranks: quality_analysis/summary.json and quality/jobs/. Timing breakdown: timing_analysis/summary.json. Validation-time plots are in both analysis directories.\n\n'
            'Limitations: one GPU, one model and dataset; sampling semantics differ; n=3 quality pairs; boundary GPU telemetry cannot exclude every transient interference. Sensitivity and second dataset are deferred until core review.\n')
        state.update(status='complete',stage='complete')
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',error=repr(exc))
        (root/'error.log').write_text(traceback.format_exc());raise
    finally:
        state['updated_at_unix']=time.time();p.write_json(root/'status.json',state)
    return state

def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,default=c.ROOT/'src/py/data/FB15K237');parser.add_argument('--available-hours',type=float,required=True)
    args=parser.parse_args();signal.signal(signal.SIGTERM,interrupted)
    run(args.output,args.data,args.available_hours)
