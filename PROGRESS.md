# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 6 - Node 3: Runtime Framework Integration — Feature/Cost 解耦 + Sort/Pack 拆分 + BatchProvider Adapter + 探针日志

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构：FeatureExtractor → CostModel → Scheduler → BatchProvider 四层解耦**  *(自动映射自 L1 宪法)*
- **Scheduler 架构：Scheduler(CostSorter, FFDPacker) 组合，支持 4 种策略变体**  *(自动映射自 L1 宪法)*
- **BatchProvider 零侵入注入：Adapter 模式，DataLoader 感知不到调度层存在**  *(自动映射自 L1 宪法)*
- **CBP 四层架构：FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter)**  *(自动映射自 L1 宪法)*
- **Scheduler 策略组合：Scheduler(CostSorter, FFDPacker) 支持 4 种变体，Scheduler(RandomSorter, ChunkPacker) 为 baseline**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 6 - Node 3: Runtime Framework Integration — Stage A~D 全部完成
- ✅ Stage A: FeatureExtractor + CostModel 解耦 (features.py + cost_model.py)
- ✅ Stage B: Sort/Pack 策略拆分 (schedulers.py: Scheduler(sorter, packer))
- ✅ Stage C: BatchProvider Adapter (batch_provider.py: 零侵入注入)
- ✅ Stage D: 验证探针日志 (batch_weight_distribution.csv + scheduler_overhead.csv)

端到端验证: 55k triples, 11 batches, scheduler_overhead=28.9ms, Weight CV=0.0125

## 4. 下一步计划 (Next Steps)
Node 4: Evaluation — node4 单卡消融实验 + node6 DDP Benchmark
