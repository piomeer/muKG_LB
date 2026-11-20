"""Frozen confirmation protocol and persistent single-GPU queue."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from . import diagnostics as d

PROTOCOL = 'confirmation-v1'
DATA = d.ROOT/'src/py/data/FB15K237'
OUTPUT = d.ROOT/'output/results/confirmation_v1'
DOC = d.ROOT/'docs/formal_v1/CONFIRMATION.md'
QUALITY = ('A', 'M00', 'M10-GPU', 'M11-GPU')
GROUPS = (*QUALITY, 'M10-CPU', 'M11-CPU')
BATCH_SIZES = (5000, 2500, 1000)
TRAIN_FILES = ('entity2id.txt', 'relation2id.txt', 'train2id.txt')
VALID_FILES = (*TRAIN_FILES, 'valid2id.txt')
TEST_FILE = 'test2id.txt'
# The same official files were bound by formal-abc-v2.
OFFICIAL = {
    'entity2id.txt': '2df2382d68595f17c638a13557265226d06beaac04b2f4b31c9646d8c3ffe16b',
    'relation2id.txt': 'ecdf2f828ce7ddfdfd4c24d0615638948c28f5f94c3589f33dbf35da762d32f5',
    'train2id.txt': '5f44223a02b39b8e398e77a787feb4f06e9ffccf1ae87047bbf38fcc8cc08bd2',
    'valid2id.txt': '7d21e778de4369d5348f5cff23a08dc847ab5fce4b5f07c80113297d2dc60a86',
    TEST_FILE: '6027e649c1716ac9c1ef2f24672a416aa277aff3e14850ef915ac5a21df70417',
}
PHASES = {'S': (521, 12, GROUPS), 'T': (541, 6, GROUPS),
          'Q': (501, 12, QUALITY), 'E': (501, 12, QUALITY)}


def config(batch_size):
    from .protocol import Config
    return Config(protocol=PROTOCOL, max_epochs=1000, batch_size=batch_size, validation_every=10)


def catalog():
    tasks = []
    for phase, (first, count, groups) in PHASES.items():
        n = len(groups)
        order = [0] + [((i+1)//2 if i % 2 else -(i//2)) % n for i in range(1, n)]
        for offset in range(count):
            for position in order:
                group = groups[(position+offset) % n]
                seed = first+offset
                tasks.append({'id': f'{phase}-{group}-{seed}', 'kind': phase, 'group': group, 'seed': seed})
    return tasks


def source():
    path = Path(__file__).parent
    return {p.relative_to(path).as_posix(): d.sha(p) for p in sorted(path.rglob('*.py'))}


def environment():
    import torch
    from .formal import environment as collect
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return {**collect(), 'interop_threads': torch.get_num_interop_threads(),
            'cpu_affinity': sorted(os.sched_getaffinity(0))}


def hashes(path, names):
    return {name: d.sha(path/name) for name in names}


def guard_output(output, *inputs):
    for path in (*inputs, DATA, Path(__file__).parent):
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output overlaps a protected input or source directory')


def copy_inputs(origin, target, identities):
    target.mkdir(parents=True, exist_ok=False)
    for name in identities:
        shutil.copy2(origin/name, target/name)
    if hashes(target, identities) != identities:
        raise ValueError('Input changed during copy')


def check(output):
    guard_output(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = source()
    hardware = environment()
    commands = []
    (output/'test_sources').mkdir()
    for pattern in ('test_formal_v1*.py', 'test_mechanism_*.py', 'test_confirmation*.py'):
        for test in sorted((d.ROOT/'tests').glob(pattern)):
            shutil.copy2(test, output/'test_sources'/test.name)
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern]
        name = pattern.replace('*', 'all')+'.log'
        with gpu_slot(output), (output/name).open('w') as stream:
            result = subprocess.run(command, cwd=d.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS': '10', 'MKL_NUM_THREADS': '10'})
        commands.append({'command': command, 'log': name, 'returncode': result.returncode})
        log = (output/name).read_text()
        if result.returncode or not re.search(r'Ran [1-9]\d* tests? in', log) or 'skipped=' in log:
            raise RuntimeError(f'Confirmation checks failed, see {output/name}')
    if identity != source() or any(hashes(d.ROOT/'tests', [p.name]) != {p.name: d.sha(p)}
                                  for p in (output/'test_sources').glob('*.py')):
        raise ValueError('Source or tests changed during correctness checks')
    receipt = {'status': 'passed', 'source': identity, 'environment': hardware, 'commands': commands}
    d.write(output/'verification.json', receipt)
    d.seal(output)
    return receipt


@contextmanager
def gpu_slot(root):
    from contextlib import ExitStack
    from .formal import gpu_lock
    with ExitStack() as stack:
        while True:
            d.wait_gpu(root)
            try:
                stack.enter_context(gpu_lock())
                break
            except RuntimeError as exc:
                if str(exc) not in ('GPU already has a compute process', 'Another formal task or preflight holds this GPU'):
                    raise
                d.event(root, 'waiting_resource_race', reason=str(exc))
                time.sleep(10)
        yield


def launch(root, output, task, cfg, binding, inputs, input_hashes, selection=None):
    request = {**task, 'task': task, 'output': str(output), 'data_dir': str(inputs),
               'config': cfg, 'input_hashes': input_hashes, 'selection': selection, 'binding': binding,
               'run': str(root)}
    with gpu_slot(root):
        return d.launch(root, output, request, worker_command='_confirmation-worker', wait_for_gpu=False)


def preflight(checks, output):
    guard_output(output, checks)
    d.verify_seal(checks)
    hardware, identity = environment(), source()
    receipt = d.read(checks/'verification.json')
    if (receipt['status'], receipt['environment'], receipt['source']) != ('passed', hardware, identity):
        raise ValueError('Current environment and source require a fresh correctness check')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checks, output/'checks')
    training = hashes(DATA, TRAIN_FILES)
    validation = hashes(DATA, VALID_FILES)
    if validation != {name: OFFICIAL[name] for name in VALID_FILES}:
        raise ValueError('Official FB15k-237 training or validation files differ from the historical identity')
    copy_inputs(DATA, output/'train/FB15K237', training)
    copy_inputs(DATA, output/'valid/FB15K237', validation)
    binding = {'source': identity, 'environment': hardware}
    candidates = {}
    for batch in BATCH_SIZES:
        rows = {}
        candidates[str(batch)] = rows
        for group in GROUPS:
            quality = group in QUALITY
            task = {'id': f'P-{batch}-{group}-550', 'kind': 'P', 'group': group, 'seed': 550}
            output_path = output/'attempts'/task['id']
            result = launch(output, output_path, task, config(batch).to_dict(), binding,
                            output/('valid' if quality else 'train')/'FB15K237',
                            validation if quality else training)
            if result['status'] not in ('complete', 'oom', 'memory_margin_exceeded'):
                raise RuntimeError(f'Preflight stopped at {task["id"]}: {result}')
            d.verify_seal(output_path)
            rows[group] = result
            d.write(output/'candidates.json', candidates)
        if all(row['status'] == 'complete' for row in rows.values()):
            record = {'status': 'passed', 'batch_size': batch, 'source': identity, 'environment': hardware,
                      'training': training, 'validation': validation, 'candidates': candidates}
            d.write(output/'preflight.json', record)
            d.seal(output)
            return record
    raise RuntimeError('No common preflight batch size passed')


def estimate(task, measured):
    phase = task['kind']
    if phase in 'ST':
        seconds = measured['warmup_seconds']+5*measured['trace_seconds' if phase == 'T' else 'measured_seconds']
        basis = 'seed 550 complete warmup and five times one complete measured epoch'
    elif phase == 'Q':
        seconds = 1000*measured['measured_seconds']+100*measured['validation_seconds']
        basis = 'seed 550 complete training epoch ×1000 and full validation ×100'
    else:
        seconds = measured['validation_seconds']*20466/17535
        basis = 'seed 550 full validation scaled by official test/validation query count'
    return {'seconds': seconds, 'scheduled_seconds': 1.30*seconds, 'basis': basis}


def freeze(preparation, root):
    guard_output(root, preparation)
    d.verify_seal(preparation)
    prep = d.read(preparation/'preflight.json')
    batch = prep['batch_size']
    if prep['status'] != 'passed' or prep['source'] != source() or prep['environment'] != environment():
        raise ValueError('Freeze requires a passing current preflight')
    if prep['validation'] != hashes(DATA, VALID_FILES):
        raise ValueError('Official training or validation identity changed')
    if batch != next((size for size in BATCH_SIZES if all(
            prep['candidates'][str(size)][group]['status'] == 'complete' for group in GROUPS)), None):
        raise ValueError('Selected batch differs from preflight')
    test_identity = d.sha(DATA/TEST_FILE)
    if test_identity != OFFICIAL[TEST_FILE]:
        raise ValueError('Official test file differs from the historical identity')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(preparation, root/'preflight')
    copy_inputs(preparation/'train/FB15K237', root/'inputs/train/FB15K237', prep['training'])
    copy_inputs(preparation/'valid/FB15K237', root/'inputs/valid/FB15K237', prep['validation'])
    for name in prep['source']:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, target)
    if hashes(root/'source', prep['source']) != prep['source']:
        raise ValueError('Live source changed during freeze')
    shutil.copy2(DOC, root/'PROTOCOL.md')
    cards = []
    for task in catalog():
        command = [sys.executable, '-B', '-m', 'src.py.experiments.formal_v1', 'confirmation',
                   'run-task', '--run', str(root), '--task', task['id']]
        card = {**task, 'command': command, 'config': config(batch).to_dict(),
                'estimate': estimate(task, prep['candidates'][str(batch)][task['group']]),
                'prior_phases': list('STQE'[:'STQE'.index(task['kind'])])}
        d.write(root/'cards'/f'{task["id"]}.json', card)
        cards.append(card)
    (root/'TASKS.md').write_text('# 204 张独立任务卡\n\n' + '\n'.join(
        f'- {card["id"]} · {card["estimate"]["scheduled_seconds"]/3600:.3f} 小时（含 30%）· `{shlex.join(card["command"])}`'
        for card in cards)+'\n')
    record = {'protocol': PROTOCOL, 'source': prep['source'], 'environment': prep['environment'],
              'config': config(batch).to_dict(), 'training': prep['training'], 'validation': prep['validation'],
              'test_sha256': test_identity, 'tasks': catalog(),
              'protocol_sha256': d.sha(root/'PROTOCOL.md'), 'cards_sha256': d.seal(root/'cards'),
              'source_sha256': d.seal(root/'source'), 'preflight_sha256': d.sha(root/'preflight/artifacts.json'),
              'task_index_sha256': d.sha(root/'TASKS.md'),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip(),
              'created_unix': time.time()}
    d.write(root/'freeze.json', record)
    d.write(root/'freeze_seal.json', {'sha256': d.sha(root/'freeze.json')})
    d.write(root/'status.json', {'status': 'frozen', 'completed': 0, 'tasks': len(cards)})
    return {'status': 'frozen', 'batch_size': batch, 'tasks': len(cards),
            'scheduled_hours': sum(card['estimate']['scheduled_seconds'] for card in cards)/3600}


def read_freeze(root):
    record = d.read(root/'freeze.json')
    if (d.read(root/'freeze_seal.json')['sha256'] != d.sha(root/'freeze.json')
            or record['protocol'] != PROTOCOL or record['tasks'] != catalog()
            or record['config']['batch_size'] not in BATCH_SIZES):
        raise ValueError('Frozen protocol or task order changed')
    for name, key in (('PROTOCOL.md', 'protocol_sha256'), ('TASKS.md', 'task_index_sha256'),
                      ('cards/artifacts.json', 'cards_sha256'), ('source/artifacts.json', 'source_sha256'),
                      ('preflight/artifacts.json', 'preflight_sha256')):
        if d.sha(root/name) != record[key]:
            raise ValueError(f'Frozen material changed: {name}')
    for name in ('cards', 'source', 'preflight'):
        d.verify_seal(root/name)
    return record


def outcome(root, path, state):
    sidecars = {suffix: d.sha(path.with_name(path.name+'.'+suffix))
                for suffix in ('request.json', 'stdout.log', 'stderr.log')}
    return {**state, 'path': path.relative_to(root).as_posix(), 'sidecars': sidecars,
            'artifacts_sha256': d.sha(path/'artifacts.json')}


def gate_payload(root, record, outcomes, verified):
    required = {task['id'] for task in record['tasks'] if task['kind'] != 'E'}
    pretest = {row['task']['id']: row for row in verified if row['task']['kind'] != 'E'}
    if set(pretest) != required:
        raise ValueError('All S, T and Q tasks are required before official test access')
    return {'freeze_sha256': d.sha(root/'freeze.json'), 'test_sha256': record['test_sha256'],
            'outcomes': {key: outcomes[key] for key in required},
            'selection': {key: row['selection'] for key, row in pretest.items() if row['task']['kind'] == 'Q'}}


def open_gate(root, record, outcomes):
    from .confirmation_report import verify_task, verify_pairing
    required = [task for task in record['tasks'] if task['kind'] != 'E']
    if any(outcomes.get(task['id'], {}).get('status') != 'complete' for task in required):
        raise ValueError('All S, T and Q tasks are required before official test access')
    verified = [verify_task(root, record, task, outcomes[task['id']]) for task in required]
    verify_pairing(verified)
    gate = gate_payload(root, record, outcomes, verified)
    if d.sha(DATA/TEST_FILE) != record['test_sha256']:
        raise ValueError('Official test identity differs from freeze')
    gate_dir = root/'official_test'
    if gate_dir.exists():
        d.verify_seal(gate_dir)
        if d.read(gate_dir/'selection.json') != gate:
            raise ValueError('Selection or pre-test outcomes changed after gate opening')
    else:
        staging = root/f'.gate-{os.getpid()}-{time.time_ns()}'
        identities = {**record['validation'], TEST_FILE: record['test_sha256']}
        dataset = staging/'FB15K237'
        copy_inputs(root/'inputs/valid/FB15K237', dataset, record['validation'])
        shutil.copy2(DATA/TEST_FILE, dataset/TEST_FILE)
        if hashes(dataset, identities) != identities:
            raise ValueError('Gated dataset changed during copy')
        d.write(staging/'selection.json', gate)
        d.seal(staging)
        staging.rename(gate_dir)
    return gate


def binding(root, record):
    return {'source': record['source'], 'environment': record['environment'],
            'freeze_sha256': d.sha(root/'freeze.json')}


def dispatch(root, record, task, output, gate):
    phase = task['kind']
    inputs = root/'inputs'/('train' if phase in 'ST' else 'valid')/'FB15K237'
    input_hashes = record['training' if phase in 'ST' else 'validation']
    selection = None
    bound = binding(root, record)
    if phase == 'E':
        if gate is None:
            raise ValueError('Official test gate is required')
        inputs = root/'official_test/FB15K237'
        input_hashes = {**input_hashes, TEST_FILE: record['test_sha256']}
        selection = gate['selection']['Q'+task['id'][1:]]
        bound['gate_sha256'] = d.sha(root/'official_test/selection.json')
    return launch(root, output, task, record['config'], bound, inputs, input_hashes, selection)


def interrupted(state):
    return state['status'] == 'interrupted' or state.get('error_type') == 'UnclassifiedWorkerExit'


def run_one(root, record, task, prior, resume, gate):
    from .confirmation_report import verify_task
    directory = root/'jobs'/task['id']
    numbers = sorted({int(path.name.split('.')[0].split('-')[1]) for path in directory.glob('attempt-*')})
    if prior and prior['status'] == 'complete':
        verify_task(root, record, task, prior)
        return prior
    infra = 0
    for index in numbers:
        path = directory/f'attempt-{index:02d}'
        state = d.read(path/'status.json') if (path/'status.json').exists() else {'status': 'interrupted'}
        if state['status'] == 'complete':
            if (path/'artifacts.json').is_file():
                result = outcome(root, path, state)
                verify_task(root, record, task, result)
                return result
            d.write(path/'recovery.json', {'prior_state': state, 'recovered_unix': time.time()})
            state = {**state, 'status': 'interrupted', 'error_type': 'UnsealedCompletion'}
        if state['status'] == 'running':
            state = d.recover_running(root, path, state)
        if not (path/'artifacts.json').exists():
            d.write(path/'status.json', state)
            d.seal(path)
        d.verify_seal(path)
        if interrupted(state):
            if not resume:
                raise InterruptedError(f'{task["id"]} requires explicit --resume')
        elif state.get('error_type') == 'InfrastructureError':
            infra += 1
            if infra >= 2:
                return {**state, 'path': path.relative_to(root).as_posix()}
        else:
            return {**state, 'path': path.relative_to(root).as_posix()}
    index = max(numbers, default=0)+1
    while True:
        d.stopped(root)
        path = directory/f'attempt-{index:02d}'
        state = dispatch(root, record, task, path, gate)
        if state['status'] == 'complete':
            result = outcome(root, path, state)
            verify_task(root, record, task, result)
            return result
        if state.get('error_type') != 'InfrastructureError' or infra:
            return {**state, 'path': path.relative_to(root).as_posix()}
        infra += 1
        d.event(root, 'retry_infrastructure', task=task['id'], previous_attempt=index)
        index += 1


def queue(root, task_id=None, phase=None, resume=False):
    if (root/'artifacts.json').exists():
        raise FileExistsError('Closed confirmation run is immutable')
    record = read_freeze(root)
    if source() != record['source'] or environment() != record['environment']:
        raise ValueError('Live source or environment changed after freeze')
    tasks = [task for task in record['tasks'] if (not task_id or task['id'] == task_id)
             and (not phase or task['kind'] == phase)]
    if not tasks:
        raise ValueError('No matching frozen tasks')
    with d.lock(root):
        if (root/'active.json').exists():
            active = d.read(root/'active.json')
            for key in ('worker_identity', 'controller_identity'):
                identity = active.get(key)
                if identity and d.process_identity(identity['pid']) == identity:
                    raise RuntimeError('Previous worker or controller remains alive')
        if resume:
            (root/'STOP').unlink(missing_ok=True)
        outcomes = d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
        gate = None
        try:
            for task in tasks:
                d.stopped(root)
                if outcomes.get(task['id'], {}).get('status') != 'complete':
                    next_task = next(item for item in record['tasks']
                                     if outcomes.get(item['id'], {}).get('status') != 'complete')
                    if next_task != task:
                        raise ValueError(f'Frozen order requires {next_task["id"]} before {task["id"]}')
                if task['kind'] == 'E' and gate is None:
                    gate = open_gate(root, record, outcomes)
                d.write(root/'status.json', {'status': 'running', 'task': task['id'],
                                           'completed': sum(x['status'] == 'complete' for x in outcomes.values()),
                                           'tasks': len(record['tasks'])})
                result = run_one(root, record, task, outcomes.get(task['id']), resume, gate)
                outcomes[task['id']] = result
                d.write(root/'outcomes.json', outcomes)
                if interrupted(result):
                    raise InterruptedError(f'Explicit resume required after {task["id"]}')
                if result['status'] != 'complete':
                    raise RuntimeError(f'Dispatch halted at {task["id"]}: {result.get("error")}')
            count = sum(x['status'] == 'complete' for x in outcomes.values())
            result = {'status': 'complete' if count == len(record['tasks']) else 'selection_complete',
                      'completed': count, 'tasks': len(record['tasks'])}
            d.write(root/'status.json', result)
            return result
        except BaseException as exc:
            d.write(root/'status.json', {'status': 'interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                                         else 'failed', 'error': str(exc)})
            raise


def worker(request_path):
    request = d.read(request_path)
    d.arm_parent_death(request['controller_pid'])
    root = Path(request['run']).resolve()
    active = d.read(root/'active.json')
    if (active['controller_pid'] != os.getppid() or Path(active['request']) != request_path
            or active['output'] != request['output']):
        raise ValueError('Worker requires the active controller launch record')
    from .confirmation_jobs import execute
    task = request['task']
    cfg = config(request['config']['batch_size'])
    if request['config'] != cfg.to_dict() or cfg.batch_size not in BATCH_SIZES:
        raise ValueError('Worker config outside confirmation protocol')
    bound = request['binding']
    if bound['source'] != source() or bound['environment'] != environment():
        raise ValueError('Worker source or environment changed')
    phase = task['kind']
    if phase == 'P':
        expected = {'id': f'P-{cfg.batch_size}-{task["group"]}-550', 'kind': 'P',
                    'group': task['group'], 'seed': 550}
        if task != expected or task['group'] not in GROUPS or Path(request['output']) != root/'attempts'/task['id']:
            raise ValueError('Invalid independent preflight task')
        label = 'valid' if task['group'] in QUALITY else 'train'
        inputs = root/label/'FB15K237'
        expected_hashes = {name: OFFICIAL[name] for name in (VALID_FILES if label == 'valid' else TRAIN_FILES)}
    else:
        record = read_freeze(root)
        if (task not in record['tasks'] or record['config'] != request['config']
                or Path(request['output']).parent != root/'jobs'/task['id']):
            raise ValueError('Worker task outside frozen catalog')
        expected_binding = binding(root, record)
        label = 'train' if phase in 'ST' else 'valid'
        inputs = root/'inputs'/label/'FB15K237'
        expected_hashes = record['training' if label == 'train' else 'validation']
        if phase == 'E':
            gate_dir = root/'official_test'
            d.verify_seal(gate_dir)
            gate_path = gate_dir/'selection.json'
            gate = d.read(gate_path)
            expected_binding['gate_sha256'] = d.sha(gate_path)
            required = {item['id'] for item in record['tasks'] if item['kind'] != 'E'}
            if (set(gate['outcomes']) != required
                    or any(item['status'] != 'complete' for item in gate['outcomes'].values())
                    or gate['selection'].get('Q'+task['id'][1:]) != request['selection']
                    or gate['freeze_sha256'] != expected_binding['freeze_sha256']
                    or gate['test_sha256'] != record['test_sha256']):
                raise ValueError('Test request differs from fixed selection')
            inputs = gate_dir/'FB15K237'
            expected_hashes = {**expected_hashes, TEST_FILE: record['test_sha256']}
        if bound != expected_binding:
            raise ValueError('Worker binding differs from freeze')
    if phase != 'E' and request['selection'] is not None:
        raise ValueError('Only an E worker may receive a test selection')
    if Path(request['data_dir']) != inputs or request['input_hashes'] != expected_hashes:
        raise ValueError('Invalid worker input paths or identities')
    return execute(request, cfg)


def main(argv):
    parser = argparse.ArgumentParser(description='muKG final confirmation experiment')
    commands = parser.add_subparsers(dest='command', required=True)
    defaults = {'check': 'checks', 'preflight': 'preflight', 'freeze': 'run', 'analyze': 'analysis'}
    for name, default in defaults.items():
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, default=OUTPUT/default)
        if name == 'preflight':
            command.add_argument('--checks', type=Path, default=OUTPUT/'checks')
        if name == 'freeze':
            command.add_argument('--preflight', type=Path, default=OUTPUT/'preflight')
        if name == 'analyze':
            command.add_argument('--run', type=Path, default=OUTPUT/'run')
    for name in ('run-task', 'run-queue', 'status', 'stop'):
        command = commands.add_parser(name)
        command.add_argument('--run', type=Path, default=OUTPUT/'run')
        if name in ('run-task', 'run-queue'):
            command.add_argument('--resume', action='store_true')
        if name == 'run-task':
            command.add_argument('--task', required=True)
        if name == 'run-queue':
            command.add_argument('--phase', choices=tuple(PHASES))
    command = commands.add_parser('_worker')
    command.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupt(signum, frame):
        d.STOPPED = True
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    cmd = args.command
    if cmd == 'status':
        result = d.status(args.run)
    elif cmd == 'stop':
        root = args.run.resolve()
        if not (root/'status.json').is_file():
            raise ValueError('Unknown confirmation run')
        if (root/'artifacts.json').exists():
            result = {'status': 'already_closed'}
        else:
            d.write(root/'STOP', {'requested_unix': time.time()})
            result = {'status': 'stop_requested'}
    elif cmd == 'check':
        result = check(args.output.resolve())
    elif cmd == 'preflight':
        result = preflight(args.checks.resolve(), args.output.resolve())
    elif cmd == 'freeze':
        result = freeze(args.preflight.resolve(), args.output.resolve())
    elif cmd in ('run-task', 'run-queue'):
        result = queue(args.run.resolve(), getattr(args, 'task', None), getattr(args, 'phase', None), args.resume)
    elif cmd == '_worker':
        path = args.request.resolve()
        try:
            result = worker(path)
        except Exception as exc:
            # The reserved attempt is named by its controller-created sidecar.
            target = path.with_name(path.name.removesuffix('.request.json'))
            if path.name.endswith('.request.json') and not target.exists():
                import traceback
                target.mkdir()
                (target/'error.log').write_text(traceback.format_exc())
                d.write(target/'status.json', {'status': 'interrupted' if isinstance(exc, InterruptedError)
                                               else 'failed', 'error_type': type(exc).__name__,
                                               'error': str(exc)})
                d.seal(target)
            raise
    else:
        from .confirmation_report import analyze
        result = analyze(args.run.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
