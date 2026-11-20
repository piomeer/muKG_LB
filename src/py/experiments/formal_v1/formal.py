"""Prepare an immutable ABC protocol and explicitly run one frozen task."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time

from . import diagnostics as d

PROTOCOL = 'formal-abc-v2'
DATA = d.ROOT / 'src/py/data/FB15K237'
DOCUMENT = d.ROOT / 'docs/formal_v1/FORMAL_ABC_V2_PROTOCOL.md'
ARCHIVE_RECEIPT = Path('/home/hma/muKG_LB/.research-archive/2026-11-06-formal-abc-v2-diagnostic-01/receipt.json')
RULES = {
    'early_stopping': False,
    'warmup_epochs': 1,
    'measured_epochs': 5,
    'quality_mrr_ratio': 0.95,
    'speed_time_ratio': 1.0,
    'confidence_interval': 'two-sided 95% t interval on paired log ratios',
    'checkpoint_selection': 'highest overall validation MRR, earliest tie',
    'validation_filter': 'train+valid',
    'test_filter': 'train+valid+test',
    'rank_ties': 'average',
    'test_policy': 'once after fixed training, best validation checkpoint',
    'attempt_policy': 'first valid complete attempt, explicit new launch after failure',
    'threads': 10,
}


def config():
    from .protocol import Config
    return Config(protocol=PROTOCOL, dim=250, margin=7, adv=2, learning_rate=.0005,
                  negatives=150, batch_size=5000, max_epochs=1000, validation_every=10,
                  query_chunk=32, entity_chunk=2048, max_rejection_rounds=64)


def guard_output(output, *inputs):
    if any(output.is_relative_to(path) for path in (*inputs, DATA, Path(__file__).parent)):
        raise ValueError('Output must be outside diagnostic, acceptance, source and data inputs')


def task_catalog():
    tasks = []
    for prefix, kind, seeds, orders in (
        ('S', 'throughput', range(201, 206), ('ABC', 'BCA', 'CAB', 'CBA', 'BAC')),
        ('T', 'trace', (201,), ('ABC',)),
        ('Q', 'quality', (101, 102, 103), ('ABC', 'BCA', 'CAB')),
    ):
        for seed, order in zip(seeds, orders):
            for group in order:
                tasks.append({'id': f'{prefix}-{group}-{seed}', 'kind': kind,
                              'group': group, 'seed': seed})
    return tasks


def identity(data_dir):
    result = d.source_identity(data_dir)
    result['data']['test2id.txt'] = d.sha(Path(data_dir) / 'test2id.txt')
    return result


def environment():
    import torch
    from .protocol import environment as collect
    torch.set_num_threads(RULES['threads'])
    result = collect()
    if not result['cuda_available'] or result['gpu'] != 'NVIDIA GeForce RTX 3070':
        raise RuntimeError('Formal ABC requires the validated RTX 3070 environment')
    if not result['hardware'] or len(result['hardware'].splitlines()) != 1:
        raise RuntimeError('Formal ABC requires a single visible GPU')
    return result


@contextmanager
def gpu_lock():
    uuid = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
    ).strip()
    if not uuid.startswith('GPU-') or not all(c.isalnum() or c == '-' for c in uuid):
        raise RuntimeError('Expected one NVIDIA GPU UUID')
    directory = Path('/tmp') / f'mukg-formal-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / f'{uuid}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another formal task or preflight holds this GPU') from exc
        if d.gpu_processes():
            raise RuntimeError('GPU already has a compute process')
        yield


def read_freeze(root):
    from .protocol import digest
    root = Path(root)
    record = d.read(root / 'freeze.json')
    body = {key: value for key, value in record.items() if key != 'freeze_sha256'}
    if record.get('protocol') != PROTOCOL or record.get('freeze_sha256') != digest(body):
        raise ValueError('Invalid formal freeze hash or version')
    catalog = root / 'task_catalog.json'
    if d.sha(catalog) != record['tasks_sha256'] or d.read(catalog) != task_catalog():
        raise ValueError('Frozen task catalog changed')
    if record['config'] != config().to_dict() or record['rules'] != RULES:
        raise ValueError('Frozen configuration or rules differ from this implementation')
    d.verify_seal(root / 'source')
    if d.sha(root/'source/artifacts.json') != record['source_inventory_sha256']:
        raise ValueError('Frozen source inventory changed')
    inventory = d.read(root/'source/artifacts.json')
    if {name: item['sha256'] for name, item in inventory.items()} != record['provenance']['sources']:
        raise ValueError('Frozen source snapshot does not match source identity')
    d.verify_seal(root / 'checks')
    if d.sha(root/'checks/artifacts.json') != record['checks_inventory_sha256']:
        raise ValueError('Frozen acceptance evidence changed')
    for name, expected in record['documents'].items():
        if d.sha(root/name) != expected:
            raise ValueError(f'Frozen document changed: {name}')
    return record


def verify_live(record):
    if identity(record['data_dir']) != record['provenance']:
        raise ValueError('Current code or dataset differs from formal freeze')
    if environment() != record['environment']:
        raise ValueError('Current environment differs from formal freeze')


def write_task_cards(path, root, budget):
    lines = ['# 正式任务卡', '', '状态为已冻结，待启动。按清单顺序逐项调用，保留实际执行时段。', '',
             f'在工作树根目录运行。工作目录为 `{d.ROOT}`。', '']
    for task in task_catalog():
        per_group = budget['per_group'][task['group']]
        estimate_key = {'quality': 'quality_job_seconds',
                        'throughput': 'throughput_job_lower_projection_seconds'}.get(task['kind'])
        if estimate_key:
            seconds = per_group[estimate_key]
            estimate = f'外推 {seconds/60:.2f} 分钟，含30%余量 {1.3*seconds/60:.2f} 分钟。'
        else:
            estimate = '耗时待首个计划内 trace 实测。'
        goal = {'quality': '固定1000轮训练、100次验证、最佳 checkpoint 官方测试',
                'throughput': '完整预热1轮与独立测量5轮',
                'trace': '完整预热1轮与分阶段记录5轮'}[task['kind']]
        lines.extend([f'## {task["id"]}', '', f'组别 {task["group"]}，种子 {task["seed"]}，协议 {PROTOCOL}。',
                      f'完成条件为{goal}。{estimate}',
                      '依赖已封存协议、匹配的代码数据环境与空闲 GPU。',
                      f'结果写入 jobs/{task["id"]}/attempt-NN。', '',
                      '```bash',
                      f'{shlex.quote(sys.executable)} -B -m src.py.experiments.formal_v1 run-formal-task --run {shlex.quote(str(root))} --task {task["id"]}',
                      '```', ''])
    path.write_text('\n'.join(lines))


def freeze(diagnostic, checks, output):
    from .protocol import digest
    from .formal_checks import diagnostic_identity
    diagnostic, checks, output = (Path(path).resolve() for path in (diagnostic, checks, output))
    guard_output(output, diagnostic, checks)
    if output.exists():
        raise FileExistsError(output)
    d.verify_seal(checks)
    receipt = d.read(checks/'verification.json')
    diag = diagnostic_identity(diagnostic, Path(receipt['diagnostic']['archive_receipt']))
    if receipt['status'] != 'passed' or diag != receipt['diagnostic']:
        raise ValueError('Acceptance or diagnostic identity mismatch')
    if not receipt['reanalysis_byte_identical'] or not receipt['shared_science_verified']:
        raise ValueError('Missing diagnostic replay or shared-path verification')
    if receipt['test_runs'] != {'cpu': 'passed', 'cuda': 'passed'}:
        raise ValueError('Both CPU and actual CUDA checks are required')
    if receipt['config'] != config().to_dict():
        raise ValueError('Acceptance config differs from formal config')
    verify_live(receipt)
    if [item['group'] for item in receipt['preflight']] != list('ABC'):
        raise ValueError('Three preflights are required')
    for item in receipt['preflight']:
        if item['status'] != 'passed' or item['batch_size'] != 5000 or item['peak_reserved_bytes'] >= .9 * item['total_bytes']:
            raise ValueError('Preflight failed to establish common batch size 5000')
        probe = checks / 'preflight' / item['group']
        d.verify_seal(probe)
        if d.sha(probe/'artifacts.json') != item['inventory_sha256']:
            raise ValueError('Preflight evidence changed')
        request = d.read(probe/'request.json')
        if request['config'] != receipt['config'] or d.read(probe/'environment.json') != receipt['environment']:
            raise ValueError('Preflight config/environment binding mismatch')

    output.mkdir(parents=True)
    source = Path(__file__).parent
    for name in receipt['provenance']['sources']:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    source_hash = d.seal(output/'source')
    shutil.copytree(checks, output/'checks')
    shutil.copy2(DOCUMENT, output/'PROTOCOL.md')
    d.write(output/'task_catalog.json', task_catalog())
    observed_budget = d.read(diagnostic/'analysis/summary.json')['budget']
    budget = {key: value for key,value in observed_budget.items()
              if key not in ('candidate_epochs', 'formal_epochs_frozen')}
    budget['training_epochs'] = config().max_epochs
    write_task_cards(output/'TASKS.md', output, budget)
    body = {
        'protocol': PROTOCOL, 'config': receipt['config'], 'rules': RULES,
        'environment': receipt['environment'], 'provenance': receipt['provenance'],
        'data_dir': receipt['data_dir'], 'diagnostic': diag, 'budget': budget,
        'source_inventory_sha256': source_hash,
        'checks_inventory_sha256': d.sha(output/'checks/artifacts.json'),
        'tasks_sha256': d.sha(output/'task_catalog.json'),
        'documents': {name: d.sha(output/name) for name in ('PROTOCOL.md', 'TASKS.md')},
        'created_unix': time.time(),
    }
    d.write(output/'freeze.json', {**body, 'freeze_sha256': digest(body)})
    record = read_freeze(output)
    d.write(output/'status.json', {'status': 'frozen_awaiting_start', 'formal_tasks': 27, 'completed': 0,
                                  'freeze_sha256': record['freeze_sha256']})
    return record


def recover_active(root):
    active_file = root/'active.json'
    if not active_file.exists():
        return
    active = d.read(active_file)
    path = Path(active['output']).resolve()
    if not path.is_relative_to(root/'jobs'):
        raise ValueError('Active worker path is outside formal jobs')
    for field in ('worker_identity', 'controller_identity'):
        item = active.get(field)
        if item and d.process_identity(item['pid']) == item:
            raise RuntimeError('Previous worker or controller is still alive')
    state_file = path/'status.json'
    if not state_file.exists() or d.read(state_file)['status'] == 'running':
        previous = d.read(state_file) if state_file.exists() else {}
        d.write(state_file, {**previous, 'status': 'interrupted', 'error_type': 'UnclassifiedControllerExit'})
    d.event(root, 'recovered_formal_attempt', output=str(path))
    active_file.unlink()


def run_task(root, task_id):
    from .formal_report import verify_task
    root = Path(root).resolve()
    task = next((entry for entry in task_catalog() if entry['id'] == task_id), None)
    if task is None:
        raise ValueError(f'Unknown formal task {task_id}')
    record = read_freeze(root)
    with d.lock(root), gpu_lock():
        recover_active(root)
        verify_live(record)
        outcomes_path = root/'outcomes.json'
        outcomes = d.read(outcomes_path) if outcomes_path.exists() else {}
        if outcomes.get(task_id, {}).get('status') == 'complete':
            verify_task(root, task, outcomes[task_id], record)
            raise FileExistsError(f'Task {task_id} already has a valid result')
        task_root = root/'jobs'/task_id
        attempt = 1
        while True:
            output = task_root/f'attempt-{attempt:02d}'
            if (output/'status.json').exists():
                prior = d.read(output/'status.json')
                if prior['status'] == 'complete' and not (output/'artifacts.json').is_file():
                    d.write(output/'status.json', {**prior, 'status':'interrupted',
                                                   'error_type':'UnsealedCompletion'})
                    d.event(root, 'unsealed_formal_attempt', output=str(output))
                elif prior['status'] == 'complete':
                    recovered = complete_outcome(root, output, prior)
                    verify_task(root, task, recovered, record)
                    outcomes[task_id] = recovered
                    d.write(outcomes_path, outcomes)
                    d.write(root/'status.json', {'status':'task_finished', 'task':task_id,
                                                'outcome':'complete',
                                                'completed':sum(item['status']=='complete' for item in outcomes.values())})
                    return recovered
            if not output.exists() and not output.with_suffix('.request.json').exists():
                break
            attempt += 1
        request = {**task, 'run': str(root), 'output': str(output),
                   'freeze_sha256': record['freeze_sha256']}
        d.write(root/'status.json', {'status':'running', 'task':task_id, 'attempt':attempt, 'pid':os.getpid()})
        try:
            state = d.launch(root, output, request, worker_command='_formal-worker', wait_for_gpu=False)
            outcome = {**state, 'path': output.relative_to(root).as_posix()}
            if state['status'] == 'complete':
                outcome = complete_outcome(root, output, state)
                verify_task(root, task, outcome, record)
            outcomes[task_id] = outcome
            d.write(outcomes_path, outcomes)
            d.write(root/'status.json', {'status':'task_finished', 'task':task_id,
                                        'outcome':state['status'],
                                        'completed':sum(item['status']=='complete' for item in outcomes.values())})
            return outcome
        except BaseException as exc:
            interrupted = isinstance(exc, (InterruptedError, KeyboardInterrupt))
            d.write(root/'status.json', {'status':'interrupted' if interrupted else 'task_error',
                                        'task':task_id, 'attempt':attempt, 'error':str(exc)})
            raise


def complete_outcome(root, output, state):
    files = ('request.json', 'stdout.log', 'stderr.log')
    return {**state, 'path':output.relative_to(root).as_posix(),
            'artifacts_sha256':d.sha(output/'artifacts.json'),
            'sidecars':{name:d.sha(output.with_name(output.name+'.'+name)) for name in files}}


def worker(request):
    d.arm_parent_death(request['controller_pid'])
    from . import runtime
    root = Path(request['run']).resolve()
    record = read_freeze(root)
    task = next((entry for entry in task_catalog() if entry['id'] == request['id']), None)
    if task is None or any(request[k] != v for k, v in task.items()) or request['freeze_sha256'] != record['freeze_sha256']:
        raise ValueError('Worker request does not match frozen task')
    output = Path(request['output']).resolve()
    if output.parent != root/'jobs'/task['id'] or not output.name.startswith('attempt-'):
        raise ValueError('Worker result path does not match task')
    verify_live(record)
    return runtime.run_official_job(output, record['data_dir'], config(), task['group'], task['seed'],
                                    task['kind'], record['freeze_sha256'], provenance=record['provenance'])


def main(argv):
    parser = argparse.ArgumentParser(description='Frozen ABC formal protocol')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check-formal', 'freeze-formal'):
        subparser = sub.add_parser(command)
        subparser.add_argument('--diagnostic', type=Path, required=True)
        subparser.add_argument('--output', type=Path, required=True)
        if command == 'freeze-formal':
            subparser.add_argument('--checks', type=Path, required=True)
        else:
            subparser.add_argument('--archive-receipt', type=Path, default=ARCHIVE_RECEIPT)
    launch = sub.add_parser('run-formal-task')
    launch.add_argument('--run', type=Path, required=True)
    launch.add_argument('--task', choices=[item['id'] for item in task_catalog()], required=True)
    analyze = sub.add_parser('analyze-formal')
    analyze.add_argument('--run', type=Path, required=True)
    analyze.add_argument('--output', type=Path, required=True)
    internal = sub.add_parser('_formal-worker')
    internal.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupt(signum, frame):
        d.STOPPED = True
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.command == 'check-formal':
        from .formal_checks import check
        result = check(args.diagnostic, args.archive_receipt, args.output)
    elif args.command == 'freeze-formal':
        result = freeze(args.diagnostic, args.checks, args.output)
    elif args.command == 'run-formal-task':
        result = run_task(args.run, args.task)
    elif args.command == '_formal-worker':
        result = worker(d.read(args.request))
    else:
        from .formal_report import analyze as export
        result = export(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get('status') in ('failed', 'interrupted'))
