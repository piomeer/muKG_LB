# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 6 - Node 4: Runtime Attribution Causal Chain 验证

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构：FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦**  *(自动映射自 L1 宪法)*
- **Scheduler 策略组合：Scheduler(CostSorter, FFDPacker) CBP | Scheduler(RandomSorter, ChunkPacker) Baseline**  *(自动映射自 L1 宪法)*
- **BatchProvider 零侵入注入：Adapter 模式，DataLoader 感知不到调度层存在**  *(自动映射自 L1 宪法)*
- **§6.2 实验真值：所有 GPU 实验结果必须有真实 stdout/CSV/日志支撑，严禁推断**  *(自动映射自 L1 宪法)*
- **Pearson r 在自变量方差过低时不可靠，应同时检查 std 和可视化分布**  *(自动映射自 L1 宪法)*
- **batch_size=1000 规避 OOM (原 5000 失败)**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 6 - Runtime Attribution 完成 [2026-07-20]
  ✅ Baseline 273 batches + CBP 273 batches (batch_size=1000, neg_num=150)
  ✅ Cost Model 准确: CBP 下 Weight vs Neg Sampling r=0.71
  ✅ Neg Sampling std 降 78%: 15.5ms → 3.4ms
  ✅ Total step std 降 66%: 16.0ms → 5.5ms
  ✅ Neg Sampling 主导总时间 (r=0.98), CPU 占 92%
  ✅ Baseline r=0.0064 假阴性 (Random+Chunk 无 batch 间差异)
  ✅ CBP 下瓶颈转移至 Tensor 构建 (r=0.91)
⚠️ blocker: 大 batch (5000) OOM, 改用 1000 完成验证
⚠️ blocker: Neg Sampling CPU 占比 65% 是总时间方差最终瓶颈
⚠️ blocker: Tensor 构建 (27%) 在 CBP 均衡后成为第二瓶颈
⚠️ blocker: FFDPacker 性能瓶颈待优化
⚠️ blocker: CBP 实验 5/10 epochs 未完成

## 4. 卡点 (Blockers)
[近] GPU 负采样：解决 neg sampling (65%) + tensor (27%) 的 CPU→GPU 瓶颈
[近] 优化 FFDPacker → round-robin 分配，降低 overhead
[中] 重启 CBP 实验完成剩余 5 epochs, 验证 MRR/Hits@K 稳定性
[中] 清理 run_cbp_evaluation.py 中的 DataLoader 死代码 import

## 5. 下一步计划 (Next Steps)
1. [近] GPU 负采样：解决 neg sampling (65%) + tensor (27%) 的 CPU→GPU 瓶颈
2. [近] 优化 FFDPacker → round-robin 分配，降低 overhead
3. [中] 重启 CBP 实验完成剩余 epochs, 验证 MRR/Hits@K 稳定性
4. [中] 清理 run_cbp_evaluation.py 中的 DataLoader 死代码 import
