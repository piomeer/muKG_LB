# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 5.5: Algorithm Validation（算法假设验证）— 验证 DDBP 的 Weight vs Runtime 相关性假设，修正权重公式

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
- **内存降级：无法访问 Memory Server 时保留 Payload 等待后续环境处理**  *(自动映射自 L1 宪法)*
- **server_pc_cluster 无 GPU，禁止宣称任何 GPU 实验结果**  *(自动映射自 L1 宪法)*
- **Memory Server 统一命名为 Academic Memory MCP Server，不是 server-memory 或 npm 包名**  *(自动映射自 L1 宪法)*
- **离线判断改用 internet 字段（不是 network），Memory Server 是本地 JSON 不依赖互联网**  *(自动映射自 L1 宪法)*
- **能力优先原则：capability 字段 > 硬件约束，避免未来硬编码新机器名称**  *(自动映射自 L1 宪法)*
- **无 GPU 环境严禁假设训练/显存/利用率/MRR/Profiling 数据**  *(自动映射自 L1 宪法)*
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **DDBP Weight 公式修正：权重应基于候选池大小而非 degree/candidate_size 比值，核心驱动变量为 candidate_size（R=0.9008）**  *(自动映射自 L1 宪法)*
- **DDBP 权重公式修正：batch_weight = sum(1/(1 - N_neg/candidate_size(entity)))，而非 d/c_size 比值**  *(自动映射自 L1 宪法)*
- **验证实验关键发现：avg_candidate_size vs actual_time 的 Pearson R=0.9008 — 候选池大小是成本预测的核心变量**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 5.5: Algorithm Validation — 验证完成
- ✅ 编写 scripts/validate_weight_assumption.py 并在 node4 执行 400 batch
- ✅ 采集 output/results/weight_validation.csv + weight_validation_summary.txt
- ✅ 后验分析 scripts/analyze_weight_validation.py

验证结果：
- 原始假设 R=0.1657 ❌（d/c_size 公式扁平化）
- 候选池大小 candidate_size vs time: R=0.9008 ✅
- 修正后的 DDBP 权重公式: batch_weight = sum(1/(1-N_neg/candidate_size(entity)))
- DDBP 假设本质上被验证通过，修正公式后推进 Phase 6

## 4. 下一步计划 (Next Steps)
进入 Phase 6 — DDBP 原型编码:
- 使用修正权重公式: batch_weight = sum(1/(1-N_neg/candidate_size(entity)))
- Step 1: DegreeTracker + BinPackingScheduler 实现 (ddbp_sampler.py)
- Step 2: batch.py + pytorch_dataloader.py 修改
- Step 3: node4 验证 → node6 Benchmark
