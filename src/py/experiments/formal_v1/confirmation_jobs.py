"""Confirmation worker. Each invocation builds a fresh model and optimizer."""
import hashlib
import math
from pathlib import Path
import time
import traceback

import torch

from . import core, data, diagnostics as d, runtime
from .mechanism_jobs import weights_identity
from .mechanism_sampling import PolicySampler

COMPONENTS = ('zero_grad_ns', 'sampling_ns', 'negative_transfer_ns', 'positive_transfer_ns',
              'forward_loss_ns', 'backward_ns', 'optimizer_ns')


def build(dataset, cfg, group, seed, device):
    if group == 'A':
        return runtime.build(dataset, cfg, group, seed, device)
    runtime.seed_all(seed)
    model = core.TransE(dataset['entities'], dataset['relations'], cfg.dim, cfg.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    policy, _, location = group.partition('-')
    sampler = PolicySampler(dataset['entities'], dataset['relations'], cfg.negatives,
                            dataset['train'], policy, seed,
                            'cpu' if location == 'CPU' else device, cfg.max_rejection_rounds)
    return model, optimizer, sampler


def load_inputs(path, kind):
    if kind not in ('S', 'T'):
        return data.load(path, include_test=kind == 'E')
    with (path/'entity2id.txt').open() as f:
        entities = int(f.readline())
    with (path/'relation2id.txt').open() as f:
        relations = int(f.readline())
    train = data.read_triples(path/'train2id.txt', entities, relations)
    if (entities, relations, len(train)) != (14541, 237, 272115):
        raise ValueError('Invalid official training split')
    return {'entities': entities, 'relations': relations, 'train': train}


def update_epoch(root, model, opt, sampler, train, cfg, seed, epoch, mode):
    runtime.gpu_snapshot(root, epoch, f'before_{mode}')
    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses = torch.zeros((), device='cuda')
    steps = examples = 0
    trace = []
    runtime.sync('cuda')
    start = time.perf_counter_ns()
    for index, batch in enumerate(data.batches(train, cfg.batch_size, seed, epoch)):
        loss, parts = runtime.step(model, opt, sampler, batch, cfg, 'cuda',
                                   trace=mode == 'trace', split_sampling=mode == 'trace')
        losses += loss
        steps += 1
        examples += len(batch)
        if mode == 'trace':
            trace.append({'epoch': epoch, 'step': index, 'batch_size': len(batch),
                          'is_partial': len(batch) < cfg.batch_size, **parts})
    runtime.sync('cuda')
    elapsed = time.perf_counter_ns() - start
    loss = float(losses.cpu()) / steps
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    runtime.gpu_snapshot(root, epoch, f'after_{mode}')
    for row in trace:
        runtime.append(root/'steps.csv', row)
    order = hashlib.sha256()
    for batch in data.batches(train, cfg.batch_size, seed, epoch):
        order.update(batch.numpy().tobytes())
    runtime.append(root/'epochs.csv', {'epoch': epoch, 'phase': mode, 'epoch_time_ns': elapsed,
                   'examples': examples, 'steps': steps, 'examples_per_second': examples*1e9/elapsed,
                   'mean_batch_loss': loss, 'peak_allocated_bytes': allocated,
                   'peak_reserved_bytes': reserved, 'batch_order_sha256': order.hexdigest()})
    return elapsed/1e9, reserved


def evaluate(root, model, dataset, cfg, split, epoch):
    runtime.gpu_snapshot(root, epoch, f'before_{split}')
    torch.cuda.reset_peak_memory_stats()
    runtime.sync('cuda')
    start = time.perf_counter()
    known = set(dataset['train']) | set(dataset['valid'])
    if split == 'test':
        known.update(dataset['test'])
    metrics = core.evaluate(model, dataset[split], known, dataset['entities'], 'cuda',
                            cfg.query_chunk, cfg.entity_chunk)
    runtime.sync('cuda')
    seconds = time.perf_counter() - start
    # Canonical summation makes validation selection independent of rank ordering.
    ranks = metrics['ranks']['head'] + metrics['ranks']['tail']
    metrics['overall']['mrr'] = math.fsum(1/rank for rank in ranks)/len(ranks)
    reserved = torch.cuda.max_memory_reserved()
    runtime.gpu_snapshot(root, epoch, f'after_{split}')
    return metrics, seconds, reserved


def earliest_best(values):
    return max(values, key=lambda epoch: (values[epoch], -epoch))


def training(root, dataset, cfg, task, state):
    group, seed, kind = task['group'], task['seed'], task['kind']
    model, opt, sampler = build(dataset, cfg, group, seed, 'cuda')
    d.write(root/'initial_weights.json', weights_identity(model))
    values = {}
    start = time.perf_counter()
    epochs = cfg.max_epochs if kind == 'Q' else 6
    for epoch in range(1, epochs+1):
        mode = 'training' if kind == 'Q' else 'warmup' if epoch == 1 else 'trace' if kind == 'T' else 'measured'
        seconds, reserved = update_epoch(root, model, opt, sampler, dataset['train'], cfg, seed, epoch, mode)
        time_key = 'warmup_seconds' if mode == 'warmup' else 'training_seconds'
        state[time_key] = state.get(time_key, 0) + seconds
        state['epochs_completed'] = epoch
        state['peak_reserved_bytes'] = max(state.get('peak_reserved_bytes', 0), reserved)
        if kind == 'Q' and epoch % cfg.validation_every == 0:
            metrics, seconds, reserved = evaluate(root, model, dataset, cfg, 'valid', epoch)
            value = metrics['overall']['mrr']
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite validation MRR')
            values[epoch] = value
            state['validation_seconds'] = state.get('validation_seconds', 0) + seconds
            state['peak_reserved_bytes'] = max(state['peak_reserved_bytes'], reserved)
            d.write(root/'validation'/f'epoch-{epoch:04d}.json', metrics)
            runtime.append(root/'validation.csv', {'epoch': epoch, 'mrr': value,
                           'evaluation_seconds': seconds, 'training_seconds': state['training_seconds'],
                           'wall_seconds': time.perf_counter()-start})
            checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'validation_mrr': value}
            if earliest_best(values) == epoch:
                torch.save(checkpoint, root/'best.pt')
                state.update(best_epoch=epoch, best_validation_mrr=value)
            if epoch == epochs:
                torch.save(checkpoint, root/'last.pt')
            print(task['id'], epoch, value, flush=True)
        d.write(root/'status.json', state)


def probe(root, dataset, cfg, task, state):
    group, seed = task['group'], task['seed']
    model, opt, sampler = build(dataset, cfg, group, seed, 'cuda')
    rows = list(data.batches(dataset['train'], cfg.batch_size, seed, 0))
    if len(rows[0]) != cfg.batch_size or len(rows[-1]) >= cfg.batch_size:
        raise ValueError('Preflight must exercise a complete and a short batch')
    runtime.gpu_snapshot(root, 0, 'before_preflight_updates')
    torch.cuda.reset_peak_memory_stats()
    for batch in (rows[0], rows[-1]):
        loss, _ = runtime.step(model, opt, sampler, batch, cfg, 'cuda', trace=True, split_sampling=True)
        if not math.isfinite(loss.item()):
            raise FloatingPointError('Nonfinite preflight update')
    total = torch.cuda.get_device_properties(0).total_memory
    state.update(total_bytes=total, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    def check_margin(reserved):
        state['peak_reserved_bytes'] = max(state['peak_reserved_bytes'], reserved)
        if reserved >= .9*total:
            raise MemoryError('Reserved CUDA memory reached 90% of device capacity')
    check_margin(state['peak_reserved_bytes'])
    runtime.gpu_snapshot(root, 0, 'after_preflight_updates')
    for epoch, mode in enumerate(('warmup', 'measured', 'trace'), 1):
        seconds, reserved = update_epoch(root, model, opt, sampler, dataset['train'], cfg, seed, epoch, mode)
        state[f'{mode}_seconds'] = seconds
        check_margin(reserved)
    if 'valid' in dataset:
        metrics, seconds, reserved = evaluate(root, model, dataset, cfg, 'valid', 3)
        state['validation_seconds'] = seconds
        d.write(root/'validation.json', metrics)
        check_margin(reserved)


def test(root, dataset, cfg, selection, state, run):
    relative = Path(selection['checkpoint'])
    checkpoint_path = run/relative
    if relative.is_absolute() or '..' in relative.parts or not checkpoint_path.resolve().is_relative_to(run):
        raise ValueError('Selected checkpoint is outside the frozen run')
    if d.sha(checkpoint_path) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint identity changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['epoch'], checkpoint['validation_mrr']) != (selection['epoch'], selection['mrr']):
        raise ValueError('Selected checkpoint metadata changed')
    model = core.TransE(dataset['entities'], dataset['relations'], cfg.dim, cfg.margin).cuda()
    model.load_state_dict(checkpoint['model'])
    metrics, seconds, reserved = evaluate(root, model, dataset, cfg, 'test', selection['epoch'])
    d.write(root/'test_metrics.json', metrics)
    state.update(test_seconds=seconds, peak_reserved_bytes=reserved)


def execute(request, cfg):
    root = Path(request['output'])
    root.mkdir(parents=True, exist_ok=False)
    task = request['task']
    state = {'status': 'running', 'task': task['id'], 'started_unix': time.time()}
    start = time.perf_counter()
    d.write(root/'manifest.json', {key: request[key] for key in
                                  ('task', 'config', 'binding', 'input_hashes', 'selection')})
    d.write(root/'status.json', state)
    try:
        path = Path(request['data_dir'])
        if set(p.name for p in path.iterdir()) != set(request['input_hashes']):
            raise ValueError('Unexpected split in worker input directory')
        for name, expected in request['input_hashes'].items():
            if (path/name).is_symlink() or d.sha(path/name) != expected:
                raise ValueError(f'Worker input identity changed: {name}')
        kind = task['kind']
        dataset = load_inputs(path, 'Q' if kind == 'P' and 'valid2id.txt' in request['input_hashes']
                              else 'S' if kind == 'P' else kind)
        if kind == 'E':
            test(root, dataset, cfg, request['selection'], state, Path(request['run']))
        elif kind == 'P':
            probe(root, dataset, cfg, task, state)
        else:
            training(root, dataset, cfg, task, state)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError))
                     else 'oom' if isinstance(exc, torch.cuda.OutOfMemoryError)
                     else 'memory_margin_exceeded' if isinstance(exc, MemoryError) and '90%' in str(exc)
                     else 'failed', error_type=type(exc).__name__, error=str(exc))
        (root/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        state.update(wall_seconds=time.perf_counter()-start, finished_unix=time.time())
        d.write(root/'status.json', state)
        d.seal(root)
    return state
