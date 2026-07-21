import time, numpy as np, torch, csv, os, sys

N_vals = [75000, 150000, 300000, 750000, 1500000, 3000000, 7500000]
batch_size = 5000   # 正样本数，影响 isin 查找的 keys 数量
n_entities = 14541
repeat = 10
device = 'cuda'

# 预先生成正样本尾实体（固定，模拟一次 batch 的 true tails）
pos_tails_np = np.random.randint(0, n_entities, size=batch_size, dtype=np.int64)
pos_tails_gpu = torch.from_numpy(pos_tails_np).to(device)

def measure_gpu(N):
    # 预热
    _ = torch.randint(0, n_entities, (1000,), device=device)
    torch.cuda.synchronize()
    times_total = []
    times_rand = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        candidates = torch.randint(0, n_entities, (N,), device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mask = ~torch.isin(candidates, pos_tails_gpu)
        # 取有效数量不操作实际取值，只是让 mask 计算完成
        valid = candidates[mask]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times_rand.append((t1 - t0) * 1000)
        times_total.append((t2 - t0) * 1000)
    # 返回中位数
    return np.median(times_rand), np.median(times_total)

def measure_cpu(N):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = np.random.randint(0, n_entities, size=N, dtype=np.int64)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

out_dir = 'output/results/gpu_cost_model'
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'benchmark.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'cpu_time_ms', 'gpu_randint_ms', 'gpu_total_ms'])
    for N in N_vals:
        rand_t, total_t = measure_gpu(N)
        cpu_t = measure_cpu(N)
        writer.writerow([N, cpu_t, rand_t, total_t])
        print(f"N={N:>8} | CPU={cpu_t:.3f}ms | GPU randint={rand_t:.3f}ms | GPU total={total_t:.3f}ms")

print(f"Benchmark saved to {csv_path}")
