============================================================
RUNTIME ATTRIBUTION ANALYSIS REPORT
============================================================
Dataset: FB15K237
Batch size: 1000, Neg num: 150
Baseline: RandomSorter + ChunkPacker
CBP: CostSorter + FFDPacker

--- Baseline ---
  Batch weight: mean=515.8152, std=0.8679
  Neg sampling: mean=62.2ms, std=15.5ms
  Tensor build: mean=24.7ms, std=1.5ms
  Forward:      mean=7.7ms, std=0.4ms
  Total step:   mean=94.6ms, std=16.0ms

--- CBP ---
  Batch weight: mean=514.3472, std=37.4118
  Neg sampling: mean=62.6ms, std=3.4ms
  Tensor build: mean=25.9ms, std=2.0ms
  Forward:      mean=7.7ms, std=0.4ms
  Total step:   mean=96.3ms, std=5.5ms

Correlation                    Baseline r      CBP r
----------------------------------------------------
Weight vs Neg Sampling             0.0064     0.7124
Weight vs Tensor                  -0.1670     0.5527
Weight vs Forward                 -0.1561     0.7320
Weight vs Total                   -0.0136     0.6952
Neg vs Total                       0.9933     0.9763
Tensor vs Total                    0.2717     0.9066
Forward vs Total                   0.3451     0.9146

============================================================
Causal Chain Analysis
============================================================

1. Weight → Neg Sampling: Baseline r=0.0064, CBP r=0.7124
   ❌ CBP did NOT weaken the Weight→NegSampling link.

2. Neg Sampling → Total: Baseline r=0.9933, CBP r=0.9763
   Interpretation: If r > 0.9, neg sampling dominates total time.
   → Neg sampling IS the dominant factor in step time.

============================================================
CONCLUSION
============================================================
CBP did NOT reduce Weight→NegSampling correlation.
→ Cost model may not match actual runtime cost.
→ Investigate cost_table vs actual neg sampling time mapping.
