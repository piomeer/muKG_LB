============================================================
RUNTIME ATTRIBUTION - 深度解读 (Corrected Analysis)
============================================================

## 原始数据摘要

| 指标 | Baseline | CBP | 改善 |
|---|---|---|---|
| Batch weight mean | 515.8 ± 0.9 | 514.3 ± 37.4 | std +43x (有差异了!) |
| Neg sampling time | 62.2 ± 15.5 ms | 62.6 ± 3.4 ms | std -78% ✅ |
| Tensor build | 24.7 ± 1.5 ms | 25.9 ± 2.0 ms | std +33% |
| Forward | 7.7 ± 0.4 ms | 7.7 ± 0.4 ms | 不变 |
| Total step time | 94.6 ± 16.0 ms | 96.3 ± 5.5 ms | std -66% ✅ |

## 相关性矩阵

| Correlation | Baseline r | CBP r | 解读 |
|---|---|---|---|
| Weight vs Neg Sampling | 0.0064 | 0.7124 | Baseline 无差异→r=0; CBP 激活了相关性 ✅ |
| Weight vs Total | -0.0136 | 0.6952 | 同上 |
| Neg vs Total | 0.9933 | 0.9763 | 负采样始终主导总时间 |

## 核心结论

### 1. Cost Model 是准确的
CBP 下 Weight vs Neg Sampling r=0.71，说明 cost_table 预测的 cost 与实际负采样执行时间高度正相关。✅

### 2. CBP 显著均衡了 batch 间负采样时间
- Baseline: neg sampling std = 15.5ms (CV=24.9%)
- CBP: neg sampling std = 3.4ms (CV=5.4%)
- 减少: 78%

### 3. 总时间方差降低 66%
- Baseline: 16.0ms → CBP: 5.5ms
- 但 neg sampling 占比 65% + Tensor 27% = CPU 占 92%
- 瓶颈转移到了 Tensor 构建 (CBP 下 r=0.91)

### 4. Baseline r=0.0064 是假阴性
Random+Chunk 下大数定理使每个 batch 的 avg cost ≈ 全局均值 (std=0.87)
不是 CBP 失效，而是 Baseline 没有 batch 间 cost 差异可言。

## 推荐方向
下一步: GPU 负采样，同时解决 neg sampling (65%) + tensor (27%) 的 CPU→GPU 瓶颈。
