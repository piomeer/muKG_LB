"""Recompute confirmation results from sealed ranks and timing observations."""
import csv
import math
from pathlib import Path
import shutil
import statistics
import time

from . import confirmation as c, diagnostics as d

METRICS = ('mrr', 'hits@1', 'hits@3', 'hits@10')


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def metrics_from_ranks(ranks, expected, entities):
    if set(ranks) != {'head', 'tail'} or any(len(side) != expected for side in ranks.values()):
        raise ValueError('Incorrect bidirectional ranking cardinality')
    for side in ranks.values():
        if any(not isinstance(rank, (float, int)) or not math.isfinite(rank)
               or not 1 <= rank <= entities or not (2*float(rank)).is_integer() for rank in side):
            raise ValueError('Rank outside average-tie range')
    output = {}
    for key, values in {**ranks, 'overall': ranks['head']+ranks['tail']}.items():
        output[key] = {'mrr': math.fsum(1/rank for rank in values)/len(values),
                       **{f'hits@{k}': sum(rank <= k for rank in values)/len(values) for k in (1, 3, 10)}}
    return output


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f'Independent recomputation mismatch: {actual} versus {expected}')


def rank_file(path, expected):
    raw = d.read(path)
    metrics = metrics_from_ranks(raw['ranks'], expected, 14541)
    for side, values in metrics.items():
        for key, value in values.items():
            close(raw[side][key], value)
    return metrics


def checkpoint(path, cfg, epoch, mrr):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['epoch'] != epoch:
        raise ValueError('Checkpoint epoch changed')
    close(state['validation_mrr'], mrr)
    dim = cfg['dim']
    shapes = {'margin': (1,), 'embedding_range': (1,),
              'ent_embeddings.weight': (14541, dim), 'rel_embeddings.weight': (237, dim)}
    if set(state['model']) != set(shapes):
        raise ValueError('Checkpoint model identity mismatch')
    for key, value in state['model'].items():
        if tuple(value.shape) != shapes[key] or not torch.isfinite(value).all().item():
            raise ValueError(f'Invalid checkpoint tensor: {key}')
    close(state['model']['margin'].item(), cfg['margin'])
    return d.sha(path)


def verify_epochs(path, task, state, cfg):
    phase = task['kind']
    epochs = rows(path/'epochs.csv')
    expected = cfg['max_epochs'] if phase == 'Q' else 6
    batch_size = cfg['batch_size']
    steps = math.ceil(272115/batch_size)
    if [int(row['epoch']) for row in epochs] != list(range(1, expected+1)) or state['epochs_completed'] != expected:
        raise ValueError('Missing complete epochs')
    for index, row in enumerate(epochs):
        expected_mode = ('training' if phase == 'Q' else 'warmup' if not index
                         else 'trace' if phase == 'T' else 'measured')
        allocated, reserved = int(row['peak_allocated_bytes']), int(row['peak_reserved_bytes'])
        if (row['phase'] != expected_mode or int(row['examples']) != 272115 or int(row['steps']) != steps
                or int(row['epoch_time_ns']) <= 0 or not math.isfinite(float(row['mean_batch_loss']))
                or not 0 < allocated <= reserved or len(row['batch_order_sha256']) != 64):
            raise ValueError('Invalid epoch accounting')
        close(row['examples_per_second'], 272115*1e9/int(row['epoch_time_ns']))
    measured = epochs if phase == 'Q' else epochs[1:]
    elapsed = sum(int(row['epoch_time_ns']) for row in measured)/1e9
    close(elapsed, state['training_seconds'])
    result = {'training_seconds': elapsed, 'epochs': epochs,
              'epoch_mean_seconds': elapsed/len(measured),
              'order': [row['batch_order_sha256'] for row in epochs],
              'initial_weights': d.read(path/'initial_weights.json')}
    if phase != 'Q':
        close(epochs[0]['epoch_time_ns'], state['warmup_seconds']*1e9)
    if phase == 'T':
        from .confirmation_jobs import COMPONENTS
        step_rows = rows(path/'steps.csv')
        if len(step_rows) != 5*steps:
            raise ValueError('Missing T steps')
        by_batch = {'full': [], 'short': []}
        for index, row in enumerate(step_rows):
            epoch, step = divmod(index, steps)
            size = min(batch_size, 272115-step*batch_size)
            values = {key: int(row[key]) for key in (*COMPONENTS, 'total_step_ns')}
            if (int(row['epoch']) != epoch+2 or int(row['step']) != step or int(row['batch_size']) != size
                    or row['is_partial'] != str(size < batch_size) or any(value < 0 for value in values.values())
                    or not values['total_step_ns'] or sum(values[k] for k in COMPONENTS) != values['total_step_ns']):
                raise ValueError('Invalid T step accounting')
            by_batch['short' if size < batch_size else 'full'].append(values)
        for offset, epoch in enumerate(measured):
            if sum(int(row['total_step_ns']) for row in step_rows[offset*steps:(offset+1)*steps]) > int(epoch['epoch_time_ns']):
                raise ValueError('T step durations exceed the enclosing epoch')
        result['stages'] = {kind: {'count': len(batch), **{
            key: statistics.mean(row[key] for row in batch)/1e9 for key in (*COMPONENTS, 'total_step_ns')}}
                            for kind, batch in by_batch.items()}
    return result


def verify_task(root, record, task, outcome):
    relative = Path(outcome['path'])
    if relative.parent != Path('jobs')/task['id'] or not relative.name.startswith('attempt-'):
        raise ValueError('Outcome points outside its task')
    path = root/relative
    if (d.sha(path/'artifacts.json') != outcome['artifacts_sha256']
            or c.outcome(root, path, d.read(path/'status.json')) != outcome):
        raise ValueError('Outcome anchor changed')
    d.verify_seal(path)
    manifest = d.read(path/'manifest.json')
    binding = c.binding(root, record)
    phase = task['kind']
    input_hashes = record['training' if phase in 'ST' else 'validation']
    selection = None
    if phase == 'E':
        gate_path = root/'official_test/selection.json'
        gate = d.read(gate_path)
        binding['gate_sha256'] = d.sha(gate_path)
        selection = gate['selection']['Q'+task['id'][1:]]
        input_hashes = {**input_hashes, c.TEST_FILE: record['test_sha256']}
    cfg = record['config']
    if manifest != {'task': task, 'config': cfg, 'binding': binding,
                    'input_hashes': input_hashes, 'selection': selection}:
        raise ValueError('Task manifest differs from freeze or test selection')
    if outcome['status'] != 'complete' or outcome['task'] != task['id'] or outcome['wall_seconds'] <= 0:
        raise ValueError('Invalid completed attempt')
    result = {'task': task, 'wall_seconds': outcome['wall_seconds']}
    if phase == 'E':
        if outcome['test_seconds'] <= 0:
            raise ValueError('Invalid official test duration')
        result.update(metrics=rank_file(path/'test_metrics.json', 20466), test_seconds=outcome['test_seconds'])
        return result
    result.update(verify_epochs(path, task, outcome, cfg))
    if phase == 'Q':
        history = rows(path/'validation.csv')
        expected = list(range(cfg['validation_every'], cfg['max_epochs']+1, cfg['validation_every']))
        if [int(row['epoch']) for row in history] != expected:
            raise ValueError('Incomplete full validation sequence')
        values = {}
        for epoch, row in zip(expected, history):
            value = rank_file(path/'validation'/f'epoch-{epoch:04d}.json', 17535)['overall']['mrr']
            close(value, row['mrr'])
            if float(row['evaluation_seconds']) <= 0:
                raise ValueError('Invalid validation duration')
            close(row['training_seconds'], sum(int(item['epoch_time_ns']) for item in result['epochs'][:epoch])/1e9)
            values[epoch] = value
        best = max(values, key=lambda epoch: (values[epoch], -epoch))
        if best != outcome['best_epoch']:
            raise ValueError('Selected epoch is not the earliest maximum verified validation MRR')
        mrr = float(history[expected.index(best)]['mrr'])
        close(outcome['best_validation_mrr'], mrr)
        best_hash = checkpoint(path/'best.pt', cfg, best, mrr)
        checkpoint(path/'last.pt', cfg, expected[-1], values[expected[-1]])
        seconds = math.fsum(float(row['evaluation_seconds']) for row in history)
        close(outcome['validation_seconds'], seconds)
        result.update(selection={'checkpoint': (path/'best.pt').relative_to(root).as_posix(), 'checkpoint_sha256': best_hash,
                                 'epoch': best, 'mrr': mrr}, history=history, validation_seconds=seconds)
    return result


def verify_pairing(results):
    seen = {}
    for row in results:
        task = row['task']
        if task['kind'] == 'E':
            continue
        key = task['kind'], task['seed']
        identity = row['initial_weights'], row['order']
        if seen.setdefault(key, identity) != identity:
            raise ValueError(f'Initial weights or ordered positives differ for pair {key}')


def comparison(numerator, denominator, seeds, threshold=None, *, strict=False):
    from .protocol import paired_summary
    seeds = list(seeds)
    paired = [seed for seed in seeds if seed in numerator and seed in denominator]
    missing = sorted(set(seeds)-set(paired))
    result = paired_summary([numerator[seed]/denominator[seed] for seed in paired]) if paired else {
        'n_pairs': 0, 'paired_ratios': [], 'geometric_mean': None, 'ci95': None}
    if paired:
        from scipy.stats import t
        values = [math.log(numerator[seed])-math.log(denominator[seed]) for seed in paired]
        mean = math.fsum(values)/len(values)
        close(result['geometric_mean'], math.exp(mean))
        if len(values) > 1:
            delta = t.isf(.025, len(values)-1)*math.sqrt(math.fsum((x-mean)**2 for x in values)
                                                        /((len(values)-1)*len(values)))
            for actual, expected in zip(result['ci95'], (math.exp(mean-delta), math.exp(mean+delta))):
                close(actual, expected)
    decision = 'incomplete' if missing else 'secondary'
    if not missing and threshold is not None:
        lower, upper = result['ci95']
        if (lower > threshold if strict else lower >= threshold):
            decision = 'supported'
        elif upper < threshold or strict and upper == threshold:
            decision = 'below_threshold'
        else:
            decision = 'indeterminate'
    return {**result, 'seeds': paired, 'missing_seeds': missing, 'decision': decision}


def summary(results, missing):
    def values(phase, group, metric):
        return {row['task']['seed']: metric(row) for row in results
                if row['task']['kind'] == phase and row['task']['group'] == group}
    quality = {}
    for numerator, denominator in (('M11-GPU', 'A'), ('M10-GPU', 'A'), ('M10-GPU', 'M00'),
                                    ('M11-GPU', 'M00'), ('M11-GPU', 'M10-GPU')):
        quality[f'{numerator}/{denominator}'] = comparison(
            values('E', numerator, lambda x: x['metrics']['overall']['mrr']),
            values('E', denominator, lambda x: x['metrics']['overall']['mrr']), range(501, 513),
            .95 if (numerator, denominator) == ('M11-GPU', 'A') else None)
    speed = {}
    for numerator, denominator in (('A', 'M11-GPU'), ('M10-CPU', 'M10-GPU'), ('M11-CPU', 'M11-GPU')):
        speed[f'{numerator}/{denominator}'] = comparison(
            values('S', numerator, lambda x: x['epoch_mean_seconds']),
            values('S', denominator, lambda x: x['epoch_mean_seconds']), range(521, 533),
            1 if numerator == 'A' else None, strict=True)
    return {'quality': quality, 'speed': speed, 'missing': missing, 'verified_tasks': len(results),
            'joint_supported': quality['M11-GPU/A']['decision'] == speed['A/M11-GPU']['decision'] == 'supported'}


def plots(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for phase, file_name in (('E', 'head-tail-metrics'), ('Q', 'validation-over-time'), ('T', 'step-stages')):
        selected = [row for row in results if row['task']['kind'] == phase]
        if not selected:
            continue
        groups = {group: [row for row in selected if row['task']['group'] == group]
                  for group in c.GROUPS}
        groups = {group: subset for group, subset in groups.items() if subset}
        if phase == 'E':
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, side in zip(axes, ('head', 'tail')):
                for index, (group, subset) in enumerate(groups.items()):
                    vals = [statistics.mean(row['metrics'][side][key] for row in subset) for key in METRICS]
                    ax.bar(np.arange(4)+.18*index, vals, width=.18, label=group)
                ax.set(title=side, xticks=np.arange(4)+.27, xticklabels=METRICS, ylim=(0, 1))
            axes[1].legend()
        elif phase == 'Q':
            figure, ax = plt.subplots(figsize=(8, 5))
            for group, subset in groups.items():
                for index, row in enumerate(subset):
                    ax.plot([float(x['wall_seconds'])/3600 for x in row['history']],
                            [float(x['mrr']) for x in row['history']], color=f'C{list(groups).index(group)}',
                            label=group if not index else None, alpha=.5)
            ax.set(xlabel='Wall time (h)', ylabel='Validation MRR')
            ax.legend()
        else:
            from .confirmation_jobs import COMPONENTS
            figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            for ax, kind in zip(axes, ('full', 'short')):
                bottom = np.zeros(len(groups))
                for key in COMPONENTS:
                    vals = np.array([statistics.mean(row['stages'][kind][key] for row in subset)*1000
                                     for subset in groups.values()])
                    ax.bar(list(groups), vals, bottom=bottom, label=key.removesuffix('_ns'))
                    bottom += vals
                ax.set(title=kind, ylabel='Milliseconds per step')
                ax.tick_params(axis='x', rotation=25)
            axes[1].legend(fontsize='small')
        figure.tight_layout()
        figure.savefig(output/f'{file_name}.png', dpi=160)
        plt.close(figure)


def render(output, results, report):
    labels = {'incomplete': '固定设计尚缺配对', 'secondary': '次要对照', 'supported': '区间支持',
              'below_threshold': '区间低于门槛', 'indeterminate': '证据未定'}
    lines = ['# muKG 确认实验', '', f'已独立核验 {len(results)}/204 项。', '',
             'FB15k-237 官方测试集在旧 Q 阶段已使用过。', '']
    for name, section in (('质量', 'quality'), ('速度', 'speed')):
        lines.extend([f'## {name}', '', '| 比值 | 配对数 | 几何均值 | 95% 区间 | 判断 |',
                      '|---|---:|---:|---|---|'])
        for label, item in report[section].items():
            mean, ci = item['geometric_mean'], item['ci95']
            interval = f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '—'
            lines.append(f'| {label} | {item["n_pairs"]} | {format(mean, ".4f") if mean else "—"} | '
                         f'{interval} | {labels[item["decision"]]} |')
        lines.append('')
    if report['joint_supported']:
        lines.extend(['M11 的测试质量与整体训练加速均获得区间支持。', ''])
    elif len(results) == len(c.catalog()):
        lines.extend(['联合达成条件未获区间支持。', ''])
    quality = [row for row in results if row['task']['kind'] == 'E']
    if quality:
        lines.extend(['## 官方测试指标（各种子算术均值）', '',
                      '| 组 | 查询侧 | MRR | Hits@1 | Hits@3 | Hits@10 |', '|---|---|---:|---:|---:|---:|'])
        for group in c.QUALITY:
            subset = [row for row in quality if row['task']['group'] == group]
            for side in ('overall', 'head', 'tail'):
                if subset:
                    metrics = [statistics.mean(row['metrics'][side][key] for row in subset) for key in METRICS]
                    lines.append(f'| {group} | {side} | ' + ' | '.join(f'{value:.4f}' for value in metrics)+' |')
        lines.append('')
    timings = [row for row in results if row['task']['kind'] == 'T']
    if timings:
        from .confirmation_jobs import COMPONENTS
        keys = (*COMPONENTS, 'total_step_ns')
        lines.extend(['## T 阶段耗时（每步毫秒）', '',
                      '| 组 | 批 | ' + ' | '.join(key.removesuffix('_ns') for key in keys)+' |',
                      '|---|---|'+'---:|'*len(keys)])
        for group in c.GROUPS:
            subset = [row for row in timings if row['task']['group'] == group]
            for kind in ('full', 'short'):
                if subset:
                    seconds = [statistics.mean(row['stages'][kind][key] for row in subset) for key in keys]
                    lines.append(f'| {group} | {kind} | '+' | '.join(f'{x*1000:.3f}' for x in seconds)+' |')
        lines.append('')
    for phase, title, keys in (('S', '每轮耗时', ('epoch_mean_seconds',)),
                               ('Q', '质量训练与验证墙钟', ('training_seconds', 'validation_seconds', 'wall_seconds')),
                               ('E', '官方测试耗时', ('test_seconds',))):
        lines.extend([f'## {title}', ''])
        for group in c.GROUPS:
            subset = [row for row in results if row['task']['kind'] == phase and row['task']['group'] == group]
            if subset:
                lines.append(f'- {group}（{len(subset)} 项），' + '，'.join(
                    f'{key} {statistics.mean(row[key] for row in subset):.3f} s' for key in keys))
        lines.append('')
    lines.extend(['## 缺失或终止任务', ''])
    lines.extend(f'- {name} · {reason}' for name, reason in report['missing'].items())
    if report['missing']:
        lines.extend(['', '缺失配对的比较仅展示已完成配对的区间，正式判断仍待完整配对。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def analyze(root, output):
    c.guard_output(output, root)
    with d.lock(root):
        if (root/'active.json').exists():
            active = d.read(root/'active.json')
            if any(item and d.process_identity(item['pid']) == item
                   for item in (active.get('controller_identity'), active.get('worker_identity'))):
                raise RuntimeError('An experiment process is still active')
        record = c.read_freeze(root)
        outcomes = d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
        results, missing = [], {}
        for task in record['tasks']:
            state = outcomes.get(task['id'], {})
            if state.get('status') == 'complete':
                results.append(verify_task(root, record, task, state))
            else:
                missing[task['id']] = state.get('status', 'pending')
        verify_pairing(results)
        if (root/'official_test').exists():
            d.verify_seal(root/'official_test')
            gate = d.read(root/'official_test/selection.json')
            if gate != c.gate_payload(root, record, outcomes, results):
                raise ValueError('Recomputed pre-test outcomes or selections differ from the opened gate')
        report = summary(results, missing)
        output.mkdir(parents=True, exist_ok=False)
        render(output, results, report)
        plots(output, results)
        d.write(output/'summary.json', report)
        d.write(output/'recomputed_tasks.json', results)
        if not missing:
            for manifest in sorted((root/'jobs').glob('*/*/artifacts.json')):
                d.verify_seal(manifest.parent)
            if (root/'artifacts.json').exists():
                d.verify_seal(root)
            else:
                d.seal(root)
            archive = output/'archive'
            shutil.copytree(root, archive)
            d.verify_seal(archive)
            report['archive_sha256'] = d.sha(archive/'artifacts.json')
            d.write(output/'summary.json', report)
        d.seal(output)
        d.verify_seal(output)
    return report
