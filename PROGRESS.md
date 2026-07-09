# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 6 - Node 1 & Node 2: Cost-aware Runtime Framework Skeleton — Offline Cost Estimator + Scheduler Polymorphism

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **Per-batch profiling 数据（profiling_summary.csv, hub_analysis.csv）通过每 epoch 写入方式防丢失**  *(自动映射自 L1 宪法)*
- **OOM 保护：验证前执行 torch.cuda.empty_cache()**  *(自动映射自 L1 宪法)*
- **time.perf_counter() 用于微秒级计时，比 time.time() 精度更高**  *(自动映射自 L1 宪法)*
- **DataLoader collate_fn 内部需自包含 reset 逻辑，不可依赖外部调用**  *(自动映射自 L1 宪法)*
- **avg_entity_degree 存储数据类型不匹配（numpy int64 写入 csv 后读取为 0），需统一为 Python int**  *(自动映射自 L1 宪法)*
- **每次建议必须基于 env_identity.json 能力字段动态决策，不硬编码机器名**  *(自动映射自 L1 宪法)*
- **修改源码后必须提醒同步（sync_required == true 时强制执行）**  *(自动映射自 L1 宪法)*
- **离线环境不得执行 Push/GitHub/DeepSeek 等联网操作**  *(自动映射自 L1 宪法)*
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构：机制-策略分离，CostEstimator 离线预计算 + Scheduler 多态策略**  *(自动映射自 L1 宪法)*
- **CostEstimator 缓存机制：cost_table.npy + neighbor_dict.pkl 持久化，Zero-Runtime-Overhead**  *(自动映射自 L1 宪法)*
- **Scheduler 多态：BaseScheduler → RandomScheduler (baseline) / FFDScheduler (CBP 核心)**  *(自动映射自 L1 宪法)*
- **CostEstimator 缓存机制：cost_table.npy + neighbor_dict.pkl 持久化到 output/results/，首次构建后 Zero-Runtime-Overhead**  *(自动映射自 L1 宪法)*
- **Scheduler 多态架构：BaseScheduler → RandomScheduler (baseline) / FFDScheduler (CBP core)**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 6 - Node 1 & Node 2: Cost-aware Runtime Framework Skeleton — 完成
- ✅ Node 1: src/py/load/cost_estimator.py — Offline CostEstimator (neighbor_dict 14505 entities, cost_table 512.84ms mean, 强制缓存)
- ✅ Node 2: src/py/load/schedulers.py — Scheduler 多态架构 (BaseScheduler → RandomScheduler / FFDScheduler, 工厂函数)

## 4. 下一步计划 (Next Steps)
1. Node 3: Framework Integration — 将 CostEstimator + Scheduler 注入 PyTorchTrainDataLoader
2. Node 4: Evaluation — node4 单卡验证 + node6 DDP 多卡 Benchmark
