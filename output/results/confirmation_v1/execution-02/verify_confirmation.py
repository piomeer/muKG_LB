"""Independent read-only audit of the archived confirmation run."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from scipy.stats import t


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_manifest(root):
    records = read(root/'artifacts.json')
    observed = {file.relative_to(root).as_posix() for file in root.rglob('*')
                if file.is_file() and file != root/'artifacts.json'}
    if set(records) != observed:
        raise ValueError(f'Archive inventory mismatch: {root}')
    for name, item in records.items():
        file = root/name
        if file.is_symlink() or file.stat().st_size != item['bytes'] or sha(file) != item['sha256']:
            raise ValueError(f'Archive digest mismatch: {file}')
    return len(records)


def close(value, reference):
    if not math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'Independent comparison failed: {value} vs {reference}')


def metrics(path, length):
    document = read(path)
    bins = {side: Counter(document['ranks'][side]) for side in ('head', 'tail')}
    if any(sum(hist.values()) != length for hist in bins.values()):
        raise ValueError(f'Ranking cardinality differs: {path}')
    bins['overall'] = bins['head']+bins['tail']
    result = {}
    for side, hist in bins.items():
        count = sum(hist.values())
        values = {'mrr': math.fsum(n/rank for rank, n in hist.items())/count}
        values.update({f'hits@{k}': sum(n for rank, n in hist.items() if rank <= k)/count
                       for k in (1, 3, 10)})
        for key, value in values.items():
            close(value, document[side][key])
        result[side] = values
    return result


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def audit(output):
    archive = output/'archive'
    file_count = verify_manifest(output)
    verify_manifest(archive)
    summary = read(output/'summary.json')
    if summary['archive_sha256'] != sha(archive/'artifacts.json'):
        raise ValueError('Archive root hash changed')
    expected = read(archive/'freeze.json')['tasks']
    outcomes = read(archive/'outcomes.json')
    recomputed = {row['task']['id']: row for row in read(output/'recomputed_tasks.json')}
    if len(expected) != 204 or len(recomputed) != 204 or len(outcomes) != 204:
        raise ValueError('Formal task count mismatch')
    gate = read(archive/'official_test/selection.json')
    if gate['freeze_sha256'] != sha(archive/'freeze.json') or gate['test_sha256'] != sha(
            archive/'official_test/FB15K237/test2id.txt'):
        raise ValueError('Test gate input hash mismatch')
    time_values, quality = {}, {}
    for task in expected:
        task_id, phase = task['id'], task['kind']
        path = archive/outcomes[task_id]['path']
        group, seed = task['group'], task['seed']
        result = recomputed[task_id]
        if phase == 'E':
            raw = metrics(path/'test_metrics.json', 20466)
            for side, values in raw.items():
                for key, value in values.items():
                    close(value, result['metrics'][side][key])
            quality[group, seed] = raw['overall']['mrr']
            continue
        if gate['outcomes'][task_id] != outcomes[task_id]:
            raise ValueError(f'Frozen pre-test outcome differs: {task_id}')
        epochs = csv_rows(path/'epochs.csv')
        if phase in 'ST':
            if len(epochs) != 6:
                raise ValueError(f'Wrong warmup plus measurement coverage: {task_id}')
            duration = sum(int(row['epoch_time_ns']) for row in epochs[1:])/5e9
            close(duration, result['epoch_mean_seconds'])
            if phase == 'S':
                time_values[group, seed] = duration
            else:
                steps = csv_rows(path/'steps.csv')
                for batch, partial in (('full', 'False'), ('short', 'True')):
                    subset = [row for row in steps if row['is_partial'] == partial]
                    if len(subset) != result['stages'][batch]['count']:
                        raise ValueError(f'Wrong T step count: {task_id}')
                    for key, value in result['stages'][batch].items():
                        if key != 'count':
                            close(sum(int(row[key]) for row in subset)/len(subset)/1e9, value)
        else:
            history = csv_rows(path/'validation.csv')
            if len(epochs) != 1000 or [int(row['epoch']) for row in history] != list(range(10, 1001, 10)):
                raise ValueError(f'Wrong Q epoch coverage: {task_id}')
            for row in history:
                epoch = int(row['epoch'])
                close(metrics(path/'validation'/f'epoch-{epoch:04d}.json', 17535)['overall']['mrr'], float(row['mrr']))
            best = max(history, key=lambda row: (float(row['mrr']), -int(row['epoch'])))
            selected = gate['selection'][task_id]
            if int(best['epoch']) != selected['epoch'] or sha(archive/selected['checkpoint']) != selected['checkpoint_sha256']:
                raise ValueError(f'Wrong best checkpoint: {task_id}')
            close(float(best['mrr']), selected['mrr'])
            close(sum(float(row['evaluation_seconds']) for row in history), result['validation_seconds'])
    for label, inputs in (('quality', quality), ('speed', time_values)):
        for name, saved in summary[label].items():
            numerator, denominator = name.split('/')
            logs = [math.log(inputs[numerator, seed]/inputs[denominator, seed]) for seed in saved['seeds']]
            if len(logs) != 12:
                raise ValueError(f'Wrong paired sample count: {name}')
            center = math.fsum(logs)/12
            variance = math.fsum((item-center)**2 for item in logs)/11
            margin = t.isf(.025, 11)*math.sqrt(variance/12)
            close(math.exp(center), saved['geometric_mean'])
            close(math.exp(center-margin), saved['ci95'][0])
            close(math.exp(center+margin), saved['ci95'][1])
    return {'status': 'verified', 'tasks': len(expected), 'files': file_count,
            'analysis_sha256': sha(output/'artifacts.json'), 'archive_sha256': sha(archive/'artifacts.json')}


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1]).resolve()), indent=2))
