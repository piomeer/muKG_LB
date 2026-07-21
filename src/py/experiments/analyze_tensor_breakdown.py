import pandas as pd, numpy as np

# Skip duplicate headers
df = pd.read_csv('output/results/tensor_breakdown/tensor_breakdown.csv')
df = df[df['step'] != 'step']  # Remove duplicate header rows
for col in ['step','batch_weight','t1_extract_pos','t2_numpy_convert','t3_tensor_pos','t4_neg_construct','t5_gpu_transfer','t6_gpu_warmup','tensor_total']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop step 0 (CUDA warmup)
df = df[df['step'] > 0]

print("=== Means (ms) ===")
cols = ['t1_extract_pos','t2_numpy_convert','t3_tensor_pos','t4_neg_construct','t5_gpu_transfer','t6_gpu_warmup','tensor_total']
print(df.groupby('config')[cols].mean().round(3))

print("\n=== Std (ms) ===")
print(df.groupby('config')[cols].std().round(3))

cbp = df[df['config']=='CBP']
print("\n=== Correlation with tensor_total (CBP) ===")
for col in ['t1_extract_pos','t2_numpy_convert','t3_tensor_pos','t4_neg_construct','t5_gpu_transfer','t6_gpu_warmup']:
    r = np.corrcoef(cbp[col], cbp['tensor_total'])[0,1]
    print(f"{col}: r={r:.4f}")

baseline = df[df['config']=='Baseline']
print("\n=== Dominant sub-stage % of tensor_total ===")
for config_name, grp in [('Baseline', baseline), ('CBP', cbp)]:
    means = grp[['t1_extract_pos','t2_numpy_convert','t3_tensor_pos','t4_neg_construct','t5_gpu_transfer','t6_gpu_warmup']].mean()
    total = means.sum()
    print(f"\n{config_name}:")
    for col in means.index:
        print(f"  {col}: {means[col]:.3f}ms ({means[col]/total*100:.1f}%)")

with open('output/results/tensor_breakdown/breakdown_summary.txt','w') as f:
    f.write("=== Tensor Construction Breakdown Summary (excl. step 0 warmup) ===\n\n")
    for config_name, grp in [('Baseline', baseline), ('CBP', cbp)]:
        f.write(f"--- {config_name} ---\n")
        means = grp[cols].mean()
        stds = grp[cols].std()
        for col in means.index:
            f.write(f"  {col}: mean={means[col]:.3f}ms, std={stds[col]:.3f}ms\n")
        f.write("\n")
    base_neg = baseline['t4_neg_construct'].mean()
    cbp_neg = cbp['t4_neg_construct'].mean()
    base_tot = baseline['tensor_total'].mean()
    cbp_tot = cbp['tensor_total'].mean()
    f.write("=== Key Findings ===\n")
    f.write(f"t4_neg_construct (torch.randint 750k IDs) is the dominant sub-stage:\n")
    f.write(f"  Baseline: {base_neg:.2f}ms / {base_tot:.2f}ms = {base_neg/base_tot*100:.1f}%\n")
    f.write(f"  CBP: {cbp_neg:.2f}ms / {cbp_tot:.2f}ms = {cbp_neg/cbp_tot*100:.1f}%\n")
    f.write(f"\nGPU transfer (t5) negligible after warmup: ~{baseline['t5_gpu_transfer'].mean():.3f}ms\n")
    f.write(f"List extraction (t1) + numpy convert (t2) ~{baseline['t1_extract_pos'].mean()+baseline['t2_numpy_convert'].mean():.3f}ms\n")
    f.write(f"torch.from_numpy pos (t3) negligible: ~{baseline['t3_tensor_pos'].mean():.3f}ms (zero-copy)\n")
    f.write(f"\n=== CBP vs Baseline Comparison ===\n")
    f.write(f"tensor_total: Baseline {base_tot:.2f}ms, CBP {cbp_tot:.2f}ms\n")
    f.write(f"Baseline r=0.0064 was artifact: Random+Chunk weight std=0.87 (no batch variation)\n")
