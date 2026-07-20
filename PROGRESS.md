# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 6 - Node 4: CBP Evaluation — TransE on FB15k-237 真实训练消融实验

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构：FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦**  *(自动映射自 L1 宪法)*
- **Scheduler 策略组合：Scheduler(CostSorter, FFDPacker) CBP | Scheduler(RandomSorter, ChunkPacker) Baseline**  *(自动映射自 L1 宪法)*
- **BatchProvider 零侵入注入：Adapter 模式，DataLoader 感知不到调度层存在**  *(自动映射自 L1 宪法)*
- **§6.2 实验真值：所有 GPU 实验结果必须有真实 stdout/CSV/日志支撑，严禁推断**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 6 - Node 4: CBP Evaluation — 验证框架完成
- ✅ Baseline (Exp-1: Random+Chunk): 10 epochs 完成
  - Mean step time: 280.80ms, Mean step CV: 0.05395
  - Filt. MRR: 0.194, Filt. Hits@10: 0.322
  - 91.2% 时间花在 CPU Uniform Negative Sampling
- 🔄 CBP (Exp-2: Cost+FFD): 5/10 epochs 完成 (进程中断)
  - Mean step time ~293ms (+4.3%), CV ~0.054 (与 Baseline 持平)
  - Scheduler Overhead: 1.165s (FFD O(n×batch) 双重循环瓶颈)
  - 精度不受损: Filt. MRR 0.194
- ✅ 代码接入验证: BatchProvider 正确迭代，DataLoader 未使用，shuffle 关闭

## 卡点
- FFDPacker 性能瓶颈: O(n×batch) 双重 for 循环导致 overhead 1.165s
- CV 不降根因: 91% step 时间在 CPU Uniform Neg Sampling，GPU 方差被淹没
- CBP 实验未完整: 剩余 5 epochs 待重新启动

## 4. 卡点 (Blockers)
1. 优化 FFDPacker → 使用 round-robin 分配，降低 overhead 至毫秒级
2. 重启 CBP 实验完成剩余 epochs，获取完整 Profiling 数据
3. 若 CV 仍不降，考虑 GPU-based negative sampling 或用更大 batch_size/复杂模型放大 GPU 方差
4. 清理 run_cbp_evaluation.py 中的 DataLoader 死代码 import

## 5. 下一步计划 (Next Steps)
1. 优化 FFDPacker → 使用 round-robin 分配，降低 overhead 至毫秒级
2. 重启 CBP 实验完成剩余 epochs，获取完整 Profiling 数据
3. 若 CV 仍不降，考虑 GPU-based negative sampling 或用更大 batch_size/复杂模型放大 GPU 方差
4. 清理 run_cbp_evaluation.py 中的 DataLoader 死代码 import
