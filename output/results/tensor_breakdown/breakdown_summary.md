=== Tensor Construction Breakdown Summary (excl. step 0 warmup) ===

--- Baseline ---
  t1_extract_pos: mean=0.518ms, std=0.050ms
  t2_numpy_convert: mean=0.337ms, std=0.028ms
  t3_tensor_pos: mean=0.007ms, std=0.001ms
  t4_neg_construct: mean=11.540ms, std=0.258ms
  t5_gpu_transfer: mean=1.144ms, std=0.015ms
  t6_gpu_warmup: mean=0.031ms, std=0.001ms
  tensor_total: mean=13.576ms, std=0.267ms

--- CBP ---
  t1_extract_pos: mean=0.526ms, std=0.050ms
  t2_numpy_convert: mean=0.336ms, std=0.027ms
  t3_tensor_pos: mean=0.006ms, std=0.000ms
  t4_neg_construct: mean=11.492ms, std=0.013ms
  t5_gpu_transfer: mean=1.154ms, std=0.028ms
  t6_gpu_warmup: mean=0.030ms, std=0.001ms
  tensor_total: mean=13.545ms, std=0.079ms

=== Key Findings ===
t4_neg_construct (torch.randint 750k IDs) is the dominant sub-stage:
  Baseline: 11.54ms / 13.58ms = 85.0%
  CBP: 11.49ms / 13.54ms = 84.8%

GPU transfer (t5) negligible after warmup: ~1.144ms
List extraction (t1) + numpy convert (t2) ~0.855ms
torch.from_numpy pos (t3) negligible: ~0.007ms (zero-copy)

=== CBP vs Baseline Comparison ===
tensor_total: Baseline 13.58ms, CBP 13.54ms
Baseline r=0.0064 was artifact: Random+Chunk weight std=0.87 (no batch variation)
