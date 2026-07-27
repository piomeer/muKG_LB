"""Validate GPU Sampler correctness and measure performance."""
import sys, os, time, csv
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from py.load.gpu_sampler import GPUNegativeSampler

# Parameters matching typical training batch
BATCH_SIZE = 5000
NEG_NUM = 150
N_ENTITIES = 14541  # FB15k-237
WARMUP_STEPS = 3
BENCH_STEPS = 10

# Create a dummy batch of triples
triples = []
for i in range(BATCH_SIZE):
    h = np.random.randint(0, N_ENTITIES)
    r = np.random.randint(0, 237)
    t = np.random.randint(0, N_ENTITIES)
    triples.append((h, r, t))

sampler = GPUNegativeSampler(N_ENTITIES, NEG_NUM)

print("Warming up GPU Sampler...")
for _ in range(WARMUP_STEPS):
    _ = sampler.generate(triples)

print("Benchmarking...")
times = []
for _ in range(BENCH_STEPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    neg_h, neg_t = sampler.generate(triples)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)
print(f"GPU Sampler avg time: {avg_time:.2f} ms ± {std_time:.2f} ms")

# Validate output shapes
assert neg_h.shape[0] == BATCH_SIZE * NEG_NUM, "Wrong head count"
assert neg_t.shape[0] == BATCH_SIZE * NEG_NUM, "Wrong tail count"
assert neg_h.device.type == 'cuda', "Heads not on GPU"
assert neg_t.device.type == 'cuda', "Tails not on GPU"
print("Output shapes correct.")

# Save to CSV
os.makedirs('output/results/gpu_sampler', exist_ok=True)
with open('output/results/gpu_sampler/validation.md', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'time_ms'])
    for i, t in enumerate(times):
        writer.writerow([i, t])

print("Validation CSV saved.")
