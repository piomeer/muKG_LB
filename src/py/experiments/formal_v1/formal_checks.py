"""Acceptance evidence for formal ABC; production data is used only in two-step probes."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from . import diagnostics as d, formal

REPORT_FILES = (
    'summary.json', 'diagnostic_summary.csv', 'REPORT.md',
    'validation_vs_training_time.png', 'validation_vs_wall_time.png', 'head_tail_by_epoch.png',
)


def diagnostic_identity(root, archive_receipt):
    root, archive_receipt = Path(root).resolve(), Path(archive_receipt).resolve()
    d.verify_seal(root)
    receipt = d.read(archive_receipt)
    archive = Path(receipt['archive'])
    if (Path(receipt['source_run']).resolve() != root or receipt['verification_errors']
            or archive.stat().st_size != receipt['bytes'] or d.sha(archive) != receipt['sha256']):
        raise ValueError('Diagnostic archive receipt or archive content mismatch')
    with tarfile.open(archive, 'r:gz') as bundle:
        if bundle.extractfile('artifacts.json').read() != (root/'artifacts.json').read_bytes():
            raise ValueError('Live diagnostic inventory differs from closed archive')
    catalog = d.read(root/'catalog.json')
    summary = d.read(root/'analysis/summary.json')
    outcomes = d.read(root/'outcomes.json')
    if (catalog['protocol'] != 'formal-abc-v2-diagnostic'
            or d.read(root/'status.json')['status'] != 'awaiting_review'
            or not summary['complete'] or summary['test_evaluated']
            or set(outcomes) != set(d.TASKS)):
        raise ValueError('Expected a complete and closed validation-only ABC diagnostic')
    if summary['catalog_sha256'] != d.sha(root/'catalog.json'):
        raise ValueError('Diagnostic report does not match its catalog')
    for task in d.TASKS:
        outcome = outcomes[task]
        path = root/outcome['path']
        if d.sha(path/'artifacts.json') != outcome['artifacts_sha256']:
            raise ValueError('Diagnostic result anchor changed')
        d.verify_job(root, task, path, catalog)
    return {
        'root': str(root), 'inventory_sha256': d.sha(root/'artifacts.json'),
        'catalog_sha256': summary['catalog_sha256'],
        'summary_sha256': d.sha(root/'analysis/summary.json'),
        'archive_receipt': str(archive_receipt),
        'archive_receipt_sha256': d.sha(archive_receipt),
        'archive_sha256': receipt['sha256'],
    }


def verify_shared_science(root, provenance):
    catalog = d.read(root/'catalog.json')
    archived = catalog['provenance']
    if any(provenance['data'][name] != value for name, value in archived['data'].items()):
        raise ValueError('Formal training/validation data or mapping differs from diagnostic')
    history = d.read(d.ROOT/'docs/formal_v1/archive_manifest.json')
    test = next(row for row in history['files'] if row['path']=='src/py/data/FB15K237/test2id.txt')
    if provenance['data']['test2id.txt'] != test['sha256']:
        raise ValueError('Test bytes differ from the archived official split')
    for name, value in archived['sources'].items():
        if (name in ('core.py', 'data.py') or name.startswith('reference/')) and provenance['sources'].get(name) != value:
            raise ValueError(f'Scientific source differs from diagnostic: {name}')
    old = ast.parse((root/'source/runtime.py').read_text())
    new = ast.parse(Path(__file__).with_name('runtime.py').read_text())
    for name in ('seed_all', 'build', 'step'):
        a, b = (next(node for node in tree.body if getattr(node, 'name', None) == name) for tree in (old, new))
        if ast.dump(a) != ast.dump(b):
            raise ValueError(f'Shared training function differs from diagnostic: {name}')


def replay_diagnostic(root, log):
    with tempfile.TemporaryDirectory(prefix='mukg-frozen-replay-') as directory:
        work = Path(directory)
        shutil.copytree(root/'source', work/'src/py/experiments/formal_v1')
        mirror = work/'diagnostic'
        mirror.mkdir()
        for name in ('catalog.json', 'outcomes.json'):
            shutil.copy2(root/name, mirror/name)
        for name in ('jobs', 'source'):
            (mirror/name).symlink_to(root/name, target_is_directory=True)
        subprocess_check(
            [sys.executable, '-B', '-c',
             'from pathlib import Path; from src.py.experiments.formal_v1.diagnostic_report import analyze; analyze(Path("diagnostic"))'],
            log, cwd=work, env={**os.environ, 'PYTHONPATH':str(work),
                               'CUDA_VISIBLE_DEVICES':'', 'MPLCONFIGDIR':str(work/'mpl')},
        )
        for name in REPORT_FILES:
            if d.sha(mirror/'analysis'/name) != d.sha(root/'analysis'/name):
                raise ValueError(f'Archived diagnostic replay differs: {name}')


def subprocess_check(command, log, *, cwd=d.ROOT, env=None):
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Check exited {result.returncode}; see {log}')


def check(diagnostic, archive_receipt, output):
    diagnostic, output = Path(diagnostic).resolve(), Path(output).resolve()
    formal.guard_output(output, diagnostic)
    output.mkdir(parents=True, exist_ok=False)
    d.write(output/'status.json', {'status': 'checking'})
    try:
        diag = diagnostic_identity(diagnostic, archive_receipt)
        provenance = formal.identity(formal.DATA)
        verify_shared_science(diagnostic, provenance)
        replay_diagnostic(diagnostic, output/'diagnostic-replay.log')
        test_sources = output/'test_sources'
        test_sources.mkdir()
        for path in (d.ROOT/'tests').glob('test_formal_v1*.py'):
            shutil.copy2(path, test_sources/path.name)
        env = {**os.environ, 'OMP_NUM_THREADS':'10', 'MKL_NUM_THREADS':'10',
               'PYTHONDONTWRITEBYTECODE':'1'}
        subprocess_check(
            [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_formal_v1*.py'],
            output/'cpu.log', env={**env, 'CUDA_VISIBLE_DEVICES':''},
        )
        with formal.gpu_lock():
            environment = formal.environment()
            subprocess_check(
                [sys.executable, '-B', '-m', 'unittest', 'test_formal_v1_core.CUDATests',
                 'test_formal_v1_official.OfficialCUDATests'],
                output/'cuda.log', env={**env, 'PYTHONPATH':str(d.ROOT/'tests')},
            )
            cuda_log = (output/'cuda.log').read_text()
            if 'skipped=' in cuda_log or 'Ran 0 tests' in cuda_log:
                raise ValueError('CUDA acceptance requires all actual CUDA tests')
            preflight = []
            cfg = formal.config().to_dict()
            for group in 'ABC':
                path = output/'preflight'/group
                request = {
                    'kind':'probe', 'group':group, 'seed':7, 'output':str(path),
                    'data_dir':str(formal.DATA), 'config':cfg,
                    'expected_provenance':d.source_identity(formal.DATA),
                    'expected_environment':environment,
                }
                status = d.launch(output, path, request, wait_for_gpu=False)
                if status['status'] != 'passed':
                    raise RuntimeError(f'Common batch size 5000 failed for {group}: {status}')
                d.verify_seal(path)
                preflight.append({**status, 'inventory_sha256':d.sha(path/'artifacts.json')})
        receipt = {
            'status':'passed', 'diagnostic':diag, 'reanalysis_byte_identical':True,
            'shared_science_verified':True, 'test_runs':{'cpu':'passed', 'cuda':'passed'},
            'config':cfg, 'environment':environment, 'data_dir':str(formal.DATA),
            'provenance':provenance, 'preflight':preflight,
        }
        formal.verify_live(receipt)
        d.write(output/'verification.json', receipt)
        d.write(output/'status.json', {'status':'passed'})
        d.seal(output)
        return receipt
    except BaseException as exc:
        d.write(output/'status.json', {'status':'failed', 'error_type':type(exc).__name__, 'error':str(exc)})
        raise
