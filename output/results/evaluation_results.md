# MuKG GPU Negative Sampler — 评估对照表

## 实验设置

| 参数 | 值 |
|------|-----|
| 环境 | server_node4 (RTX 3070 8GB) |
| 数据集 | FB15k-237 (310,116 triples, 14,541 entities, 237 relations) |
| batch_size | 5000 |
| neg_num | 150 |
| Scheduler | CostSorter + FFDPacker |
| 模型 | Simple TransE (dim=400, margin=1.0) |
| Optimizer | Adam (lr=1e-3) |

## CPU vs GPU 负采样性能对比

| 指标 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| neg_time 均值 | 596 ms | 3.0 ms | **198x** |
| fwd_time 均值 | 37.5 ms | 37.5 ms | 1.0x |
| bwd_time 均值 | 38.2 ms | 38.2 ms | 1.0x |
| total_step 均值 | 674 ms | 79.7 ms | **8.5x** |
| epoch 耗时 | 37.06 s | 4.78 s | **7.7x** |
| Loss Epoch 0 | 1.029 | 0.949 | — |
| Loss Epoch 1 | 0.666 | 0.689 | — |

## Runtime 分解 (GPU 模式)

| 阶段 | 耗时 (ms) | 占比 |
|------|-----------|------|
| neg_time | 3.0 | 3.8% |
| fwd_time | 37.5 | 47.0% |
| bwd_time | 38.2 | 48.0% |
| opt_time | 1.2 | 1.5% |
| **Total** | **79.7** | **100%** |

## GPU Sampler 技术参数

| 参数 | 值 |
|------|-----|
| 实现方式 | 全向量化 (`torch.randint` + `torch.isin`) |
| 碰撞策略 | batch-level pos_tails_set |
| 损坏策略 | tail corruption |
| oversample_factor | 1.5 |
| 平均耗时 | ~2.17 ms (单次) |
| 重试逻辑 | while 循环 oversample 直到满足数量 |

## Loss 收敛曲线 (GPU 5 epochs)

| Epoch | Avg Loss | Time (s) |
|-------|----------|----------|
| 0 | 0.949 | 4.90 |
| 1 | 0.689 | 4.78 |
| 2 | 0.536 | 4.78 |
| 3 | 0.440 | 4.78 |
| 4 | 0.375 | 4.78 |

## 已知差异 (vs 原始实现)

| 维度 | 原始实现 | GPU 实现 (当前) | 影响 |
|------|---------|----------------|------|
| 损坏头/尾 | Bernoulli(0.5) head OR tail | 仅 tail corruption | MRR/Hits@K 可能偏高 |
| 碰撞检查 | 全局 all_triples_set (272k) | batch-level pos_tails_set | ~1% 碰撞污染 |
| 实现语言 | Python for 循环 | CUDA 向量化 | 速度优势 |

## 结论

GPU 负采样器在 FB15k-237 上实现了 **198x 负采样加速**，端到端 **7.7x epoch 加速**。代价是负采样语义与原始实现存在偏差（tail-only + batch-level 碰撞），对于训练收敛影响有限（Loss 下降正常），但对 MRR/Hits@K 的数值可比性可能需要验证。