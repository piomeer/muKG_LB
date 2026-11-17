"""One explicitly authorized manual recovery, followed by the frozen queue."""
import json
import signal
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))

from src.py.experiments.formal_v1 import confirmation as c, confirmation_report as report, diagnostics as d

root = c.OUTPUT/'run-01'
execution = c.OUTPUT/'execution-02'
task_id = 'Q-A-504'


def main():
    record = c.read_freeze(root)
    if c.source() != record['source'] or c.environment() != record['environment']:
        raise ValueError('Current source or environment differs from freeze')
    task = next(item for item in record['tasks'] if item['id'] == task_id)
    with d.lock(root):
        if (root/'STOP').exists():
            raise InterruptedError('A user stop is pending')
        if (root/'active.json').exists():
            active = d.read(root/'active.json')
            for key in ('worker_identity', 'controller_identity'):
                identity = active.get(key)
                if identity and d.process_identity(identity['pid']) == identity:
                    raise RuntimeError('Previous worker or controller remains alive')
        outcomes = d.read(root/'outcomes.json')
        next_task = next(item for item in record['tasks'] if outcomes.get(item['id'], {}).get('status') != 'complete')
        if next_task != task:
            raise ValueError('Next incomplete task differs from authorized recovery')
        verified = []
        for item in record['tasks']:
            state = outcomes.get(item['id'], {})
            if state.get('status') == 'complete':
                verified.append(report.verify_task(root, record, item, state))
        report.verify_pairing(verified)
        previous = {}
        for number in (1, 2):
            path = root/'jobs'/task_id/f'attempt-{number:02d}'
            d.verify_seal(path)
            state = d.read(path/'status.json')
            if state.get('error_type') != 'InfrastructureError':
                raise ValueError('Manual recovery is restricted to documented infrastructure failures')
            previous[path.relative_to(root).as_posix()] = d.sha(path/'artifacts.json')
        attempt = root/'jobs'/task_id/'attempt-03'
        if attempt.exists() or attempt.with_name(attempt.name+'.request.json').exists():
            raise FileExistsError('Manual recovery attempt already exists')
        receipt = {
            'authorization': 'User requested continuation after being informed that the automatic retry was exhausted.',
            'user_message': '能否继续任务。',
            'scope': 'One manual fresh attempt for Q-A-504, then continue the frozen queue.',
            'execution_policy_exception': 'A third attempt is explicitly scheduled after two infrastructure failures.',
            'freeze_sha256': d.sha(root/'freeze.json'),
            'previous_attempt_artifacts': previous,
            'previous_outcomes_sha256': d.sha(root/'outcomes.json'),
            'verified_completed_tasks': len(verified),
            'task': task,
            'new_attempt': attempt.relative_to(root).as_posix(),
            'started_unix': time.time(),
        }
        receipt_path = root/'recovery/manual-20261111.json'
        if receipt_path.exists():
            raise FileExistsError(receipt_path)
        d.write(receipt_path, receipt)
        d.event(root, 'authorized_manual_recovery', task=task_id, attempt=3,
                receipt=receipt_path.relative_to(root).as_posix())
        d.write(root/'status.json', {'status': 'running', 'task': task_id, 'attempt': 3,
                                   'completed': len(verified), 'tasks': len(record['tasks'])})
        result = c.dispatch(root, record, task, attempt, None)
        if result['status'] != 'complete':
            outcomes[task_id] = {**result, 'path': attempt.relative_to(root).as_posix()}
            d.write(root/'outcomes.json', outcomes)
            raise RuntimeError(f'Manual recovery stopped: {result}')
        outcome = c.outcome(root, attempt, result)
        report.verify_task(root, record, task, outcome)
        outcomes[task_id] = outcome
        d.write(root/'outcomes.json', outcomes)
        d.event(root, 'manual_recovery_complete', task=task_id, attempt=3)
    result = c.queue(root, resume=True)
    if result['status'] != 'complete':
        raise RuntimeError(f'Frozen queue did not complete: {result}')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    def interrupt(signum, frame):
        d.STOPPED = True
        raise InterruptedError(f'User signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed'
        d.write(root/'status.json', {'status': status, 'error_type': type(exc).__name__, 'error': str(exc)})
        d.write(execution/'driver-status.json', {'status': status, 'error': str(exc), 'finished_unix': time.time()})
        raise
