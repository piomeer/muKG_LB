"""Verify formal artifacts and compute only complete paired comparisons."""
import csv
import math
from pathlib import Path
from statistics import mean

from . import diagnostics as d, formal
from .protocol import paired_summary


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_test_metrics(metrics):
    sides = metrics['ranks']
    if any(len(sides[side]) != 20466 for side in ('head', 'tail')):
        raise ValueError('Expected complete official test ranks')
    for side, ranks in [('head', sides['head']), ('tail', sides['tail']),
                        ('overall', sides['head'] + sides['tail'])]:
        if any(not (1 <= rank <= 14541 and rank * 2 == int(rank * 2)) for rank in ranks):
            raise ValueError('Invalid averaged rank')
        actual = {'mrr': math.fsum(1/rank for rank in ranks) / len(ranks)}
        actual.update({f'hits@{k}':sum(rank<=k for rank in ranks)/len(ranks) for k in (1,3,10)})
        if any(not math.isclose(value, metrics[side][name], rel_tol=0, abs_tol=1e-12)
               for name, value in actual.items()):
            raise ValueError(f'Test metric differs from raw ranks: {side}')


def verify_task(root, task, outcome, record):
    path = (root / outcome['path']).resolve()
    if path.parent != root/'jobs'/task['id'] or not path.name.startswith('attempt-'):
        raise ValueError('Invalid result path for formal task')
    if d.sha(path/'artifacts.json') != outcome['artifacts_sha256']:
        raise ValueError('Formal task artifact anchor changed')
    d.verify_seal(path)
    for name in ('request.json', 'stdout.log', 'stderr.log'):
        if d.sha(path.with_name(path.name+'.'+name)) != outcome['sidecars'][name]:
            raise ValueError('Formal task request or log changed')
    state = d.read(path/'status.json')
    if state['status'] != 'complete' or any(outcome[key] != value for key,value in state.items()):
        raise ValueError('Outcome differs from complete worker state')
    manifest = d.read(path/'manifest.json')
    expected = {key: task[key] for key in ('group','seed','kind')}
    expected.update({key: record[key] for key in ('config','provenance','environment','freeze_sha256')})
    if manifest != expected:
        raise ValueError('Formal task binding mismatch')

    kind = task['kind']
    count = 1000 if kind == 'quality' else 6
    epochs = rows(path/'epochs.csv')
    if state['epochs_completed'] != count or [int(row['epoch']) for row in epochs] != list(range(1,count+1)):
        raise ValueError('Incorrect epoch coverage')
    if any(int(row['examples']) != 272115 or int(row['steps']) != 55 for row in epochs):
        raise ValueError('Incorrect full epoch or short-batch coverage')
    order = d.read(path/'batch_order.json')
    if set(order) != {str(epoch) for epoch in range(1,count+1)}:
        raise ValueError('Missing positive batch identity')
    result = {'initial_weights':d.read(path/'initial_weights.json'), 'batch_order':order,
              'wall_seconds':state['wall_seconds'], 'training_seconds':state['training_seconds'],
              'peak_allocated_bytes':max(int(row['peak_allocated_bytes']) for row in epochs),
              'peak_reserved_bytes':max(int(row['peak_reserved_bytes']) for row in epochs)}
    if kind == 'quality':
        if state['validation_count'] != 100 or state['test_sha256'] != record['provenance']['data']['test2id.txt']:
            raise ValueError('Missing complete validation or matching test identity')
        validation = rows(path/'validation.csv')
        if [int(row['epoch']) for row in validation] != list(range(10,1001,10)):
            raise ValueError('Incorrect validation cadence')
        if not (path/'best.pt').is_file() or any(
            not (path/'validation'/f'epoch-{epoch:04d}.json').is_file() for epoch in range(10,1001,10)
        ):
            raise ValueError('Missing checkpoint or validation rank artifact')
        best = max(validation, key=lambda row: float(row['mrr']))
        metrics = d.read(path/'test_metrics.json')
        if (metrics['checkpoint_epoch'] != state['best_epoch'] or int(best['epoch']) != state['best_epoch']
                or float(best['mrr']) != state['best_validation_mrr']):
            raise ValueError('Test checkpoint does not match earliest best validation')
        verify_test_metrics(metrics)
        result.update(metrics={side:metrics[side] for side in ('head','tail','overall')},
                      best_validation=best, last_validation=validation[-1],
                      validation_seconds=state['validation_seconds'], test_seconds=state['test_seconds'])
    else:
        if state['measured_epochs'] != 5 or [row['phase'] for row in epochs] != ['warmup']+['measured']*5:
            raise ValueError('Incorrect timing warmup or measurement coverage')
        if (path/'test_metrics.json').exists() or state['validation_count'] != 0 or state['test_seconds'] != 0:
            raise ValueError('Timing pass contains quality evaluation')
        measured = [int(row['epoch_time_ns'])/1e9 for row in epochs[1:]]
        if not math.isclose(math.fsum(measured), state['training_seconds'], rel_tol=0, abs_tol=1e-9):
            raise ValueError('Measured time differs from epoch rows')
        result['mean_epoch_seconds'] = mean(measured)
        if kind == 'trace':
            trace = rows(path/'steps.csv')
            expected = [(epoch, step, 2115 if step==54 else 5000)
                        for epoch in range(2,7) for step in range(55)]
            actual = [(int(row['epoch']), int(row['step']), int(row['batch_size'])) for row in trace]
            if actual != expected or any((row['is_partial']=='True') != (int(row['batch_size'])==2115) for row in trace):
                raise ValueError('Incorrect complete/short batch trace coverage')
            components = [key for key in trace[0] if key.endswith('_ns')]
            result['components'] = {}
            for name, partial in (('full', 'False'), ('short', 'True')):
                subset = [row for row in trace if row['is_partial']==partial]
                result['components'][name] = {
                    'count':len(subset),
                    'mean_ms':{key.removesuffix('_ns'):mean(int(row[key]) for row in subset)/1e6 for key in components},
                }
        elif (path/'steps.csv').exists():
            raise ValueError('Throughput pass contains component probes')
    return result


def paired(records, kind):
    tasks = [task for task in formal.task_catalog() if task['kind']==kind]
    missing = [task['id'] for task in tasks if task['id'] not in records]
    if missing:
        return {'status':'incomplete', 'missing':missing}
    seeds = sorted({task['seed'] for task in tasks})
    prefix = tasks[0]['id'][0]
    if kind == 'quality':
        comparisons = [('C','A'), ('B','A'), ('C','B')]
        scopes = ('overall','head','tail')
        target = formal.RULES['quality_mrr_ratio']
    else:
        comparisons = [('A','C'), ('A','B'), ('B','C')]
        scopes = ('epoch',)
        target = formal.RULES['speed_time_ratio']
    result = {}
    for left, right in comparisons:
        key = f'{left}/{right}'
        result[key] = {}
        for side in scopes:
            ratios = []
            for seed in seeds:
                a,b = (records[f'{prefix}-{group}-{seed}'] for group in (left,right))
                ratio = a['metrics'][side]['mrr']/b['metrics'][side]['mrr'] if kind=='quality' else a['mean_epoch_seconds']/b['mean_epoch_seconds']
                ratios.append(ratio)
            result[key][side] = {**paired_summary(ratios), 'seeds':seeds}
    left, right = comparisons[0]
    primary = result[f'{left}/{right}'][scopes[0]]
    meets = (lambda value: value>=target) if kind=='quality' else (lambda value: value>target)
    primary.update(target=target, point_estimate_meets_target=meets(primary['geometric_mean']),
                   interval_supports_target=meets(primary['ci95'][0]))
    return {'status':'complete', 'primary':primary, 'comparisons':result}


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(root/name) for name in ('source','checks','jobs')):
        raise ValueError('Analysis output must be outside sealed inputs')
    record = formal.read_freeze(root)
    for name in ('formal_report.py','protocol.py','formal.py','diagnostics.py'):
        if d.sha(Path(__file__).with_name(name)) != record['provenance']['sources'][name]:
            raise ValueError(f'Analysis source changed; run the sealed source version: {name}')
    outcomes = d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
    records, missing, pairs = {}, {}, {}
    for task in formal.task_catalog():
        task_id = task['id']
        outcome = outcomes.get(task_id)
        if not outcome or outcome['status'] != 'complete':
            missing[task_id] = outcome or {'status':'pending'}
            continue
        result = verify_task(root, task, outcome, record)
        key = task['kind'],task['seed']
        control = result.pop('initial_weights'),result.pop('batch_order')
        if key in pairs and pairs[key] != control:
            raise ValueError(f'Paired initial weights or positive batch order differ: {key}')
        pairs[key] = control
        records[task_id] = {**result,'path':outcome['path']}
    result = {
        'status':'complete' if not missing else 'incomplete',
        'freeze_sha256':record['freeze_sha256'], 'records':records, 'missing':missing,
        'quality':paired(records,'quality'), 'throughput':paired(records,'throughput'),
        'trace':{task_id:result['components'] for task_id,result in records.items() if task_id.startswith('T-')},
    }
    output.mkdir(parents=True)
    d.write(output/'summary.json', result)
    lines = ['# A/B/C 正式实验分析', '', f'状态为 {result["status"]}。有效任务 {len(records)}/27。', '']
    for label, key in (('质量 C/A', 'quality'), ('吞吐 A/C', 'throughput')):
        section = result[key]
        if section['status']=='incomplete':
            lines.append(f'- {label} 配对待完成，缺失 {len(section["missing"])} 项。')
        else:
            primary = section['primary']
            lo, hi = primary['ci95']
            lines.append(f'- {label} 几何均值 {primary["geometric_mean"]:.6f}，95% 区间 [{lo:.6f}, {hi:.6f}]。'
                         f'点估计达标 {primary["point_estimate_meets_target"]}，区间支持 {primary["interval_supports_target"]}。')
    lines += ['', '逐种子配对值、头尾指标、尝试路径与缺失原因见 summary.json。Trace 按单种子和批次类型描述。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    d.seal(output)
    return result
