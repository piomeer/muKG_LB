import pandas as pd, numpy as np

df = pd.read_csv('output/results/gpu_cost_model/benchmark.md')

# 线性拟合 CPU: T = a_cpu * N + b_cpu
coef_cpu = np.polyfit(df['N'], df['cpu_time_ms'], 1)
a_cpu, b_cpu = coef_cpu
coef_gpu_total = np.polyfit(df['N'], df['gpu_total_ms'], 1)
a_gpu, b_gpu = coef_gpu_total

print("=== Cost Models ===")
print(f"CPU: T = {a_cpu:.9f} * N + {b_cpu:.4f}  ms")
print(f"GPU total: T = {a_gpu:.9f} * N + {b_gpu:.4f}  ms")

# 固定开销占 GPU 的比例（在 N=750k）
N_today = 750000
gpu_fixed = b_gpu
gpu_at_N = a_gpu * N_today + b_gpu
print(f"\nGPU fixed overhead (intercept): {b_gpu:.4f} ms = {b_gpu/gpu_at_N*100:.1f}% of total at N=750k")

# 交点 N*
if abs(a_cpu - a_gpu) > 1e-12:
    N_star = (b_gpu - b_cpu) / (a_cpu - a_gpu)
    print(f"Break-even N* = {int(N_star):,}")
else:
    print("Slopes equal, no intersection.")
    N_star = None

# 在当前 N=750k 的比较
t_cpu = a_cpu * N_today + b_cpu
t_gpu = a_gpu * N_today + b_gpu
ratio = t_cpu / t_gpu if t_gpu > 0 else float('inf')
print(f"\nAt N=750,000 (current batch):")
print(f"  CPU={t_cpu:.2f}ms, GPU={t_gpu:.2f}ms, speedup={ratio:.2f}x")
print(f"  GPU randint only: {df[df['N']==N_today]['gpu_randint_ms'].values[0]:.3f}ms")
print(f"  GPU isin filter overhead: {t_gpu - df[df['N']==N_today]['gpu_randint_ms'].values[0]:.3f}ms")

# 所有 N 值的 GPU 加速比
print(f"\n=== Speedup at each N ===")
for _, row in df.iterrows():
    sp = row['cpu_time_ms'] / row['gpu_total_ms'] if row['gpu_total_ms'] > 0 else float('inf')
    print(f"N={int(row['N']):>8} | CPU={row['cpu_time_ms']:.3f}ms | GPU={row['gpu_total_ms']:.3f}ms | {sp:.2f}x")

# 写入摘要文件
with open('output/results/gpu_cost_model/cost_model_summary.md', 'w') as f:
    f.write(f"CPU: T = {a_cpu:.9f}*N + {b_cpu:.4f} ms\n")
    f.write(f"GPU: T = {a_gpu:.9f}*N + {b_gpu:.4f} ms\n")
    f.write(f"Break-even N: {int(N_star):,}\n")
    f.write(f"At N=750k: CPU={t_cpu:.2f}ms, GPU={t_gpu:.2f}ms, GPU speedup={ratio:.2f}x\n")
    f.write(f"GPU fixed overhead: {b_gpu:.4f}ms ({b_gpu/gpu_at_N*100:.1f}% at N=750k)\n")
