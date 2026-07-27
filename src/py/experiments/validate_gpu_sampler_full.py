"""Phase 8 Step 2.5: Verify GPU Sampler against original MuKG semantics."""
import sys, os, time, csv, random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.gpu_sampler import GPUNegativeSampler

# Load dataset
print("Loading FB15k-237 dataset...")
args_path = os.path.join(os.path.dirname(__file__), 'args_kge', 'transe_fb15k237_args.json')
cmd_args = load_args(args_path)
cmd_args.is_torch = True
kgs = read_kgs_from_folder('lp', cmd_args.training_data, cmd_args.dataset_division,
                            cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False)
train_triples = kgs.local_relation_triples_list
n_entities = kgs.entities_num
all_triples_set = set(train_triples)
print(f"Loaded {len(train_triples)} triples, {n_entities} entities")

# Parameters
BATCH_SIZE = 5000
NEG_NUM = 150
WARMUP = 3
BENCH = 10

random.seed(42)
sampled_triples = random.sample(train_triples, BATCH_SIZE)

gpu_sampler = GPUNegativeSampler(n_entities, NEG_NUM, all_triples_set)

print("Warming up GPU Sampler...")
for _ in range(WARMUP):
    _ = gpu_sampler.generate(sampled_triples)

print("Benchmarking GPU Sampler...")
gpu_times = []
for _ in range(BENCH):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    neg_h, neg_t, corrupt_mask = gpu_sampler.generate(sampled_triples)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_times.append((t1 - t0) * 1000)

avg_gpu = np.mean(gpu_times)
print(f"GPU Sampler avg time: {avg_gpu:.2f} ms")

print("Checking validity...")
pos_rels_np = np.array([t[1] for t in sampled_triples])
neg_h_np = neg_h.cpu().numpy()
neg_t_np = neg_t.cpu().numpy()
corrupt_mask_np = corrupt_mask.cpu().numpy()

invalid_count = 0
for i in range(BATCH_SIZE * NEG_NUM):
    h = int(neg_h_np[i])
    t = int(neg_t_np[i])
    r = int(pos_rels_np[i // NEG_NUM])
    if (h, r, t) in all_triples_set:
        invalid_count += 1

invalid_rate = invalid_count / (BATCH_SIZE * NEG_NUM)
print(f"Invalid negatives (in training set): {invalid_count} / {BATCH_SIZE*NEG_NUM} = {invalid_rate:.6f}")

head_corrupt_ratio = float(corrupt_mask_np.mean())
print(f"Head corruption ratio: {head_corrupt_ratio:.3f} (expected ~0.5)")

os.makedirs('output/results/gpu_sampler_full', exist_ok=True)
with open('output/results/gpu_sampler_full/validation.md', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'time_ms', 'invalid_rate', 'head_corrupt_ratio'])
    for i in range(BENCH):
        writer.writerow([i, gpu_times[i], invalid_rate, head_corrupt_ratio])

print("Validation results saved.")
if invalid_rate < 0.001 and abs(head_corrupt_ratio - 0.5) < 0.05:
    print("✅ GPU Sampler semantic alignment VERIFIED.")
else:
    print("⚠️ Alignment may be imperfect. Check statistics.")
