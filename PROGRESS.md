# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 5 - Step 5: Implementation Planning（技术落地规划）— 完成 DDBP 在 muKG_LB 项目结构中的代码落地规划

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
- **MCP Tool 描述修正：引用 Academic Memory MCP Server 提供的图谱工具（read_graph、search_nodes、create_entities 等），不引用包名或假设存在 academic_memory Tool**  *(自动映射自 L1 宪法)*
- **CPU Development Environment 泛化：server_pc_cluster 只是典型实例，使用 capability 字段描述（can_modify=true, can_train=false），未来新增无 GPU 节点无需修改规则**  *(自动映射自 L1 宪法)*
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **负采样成本服从双模结构（Dual-Regime Cost Law）：全候选池（candidate_size>=5000）为常数 295.7ms，窄化池（neighbor dict）下随 candidate_size 和 collision_rate 缩放**  *(自动映射自 L1 宪法)*
- **B3 Collision Check（~52ms）不随候选池缩小而降速，是窄化池下的隐含瓶颈**  *(自动映射自 L1 宪法)*
- **DDBP 实施计划确定：新增 ddbp_sampler.py (~250行)，修改 5 文件 (~60行)，总计 ~310 行**  *(自动映射自 L1 宪法)*
- **Phase 5 设计阶段共 5 个 Step 全部完成，正式进入 Phase 6 原型编码**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 5 设计阶段全部完成 — 共 5 个 Step
- ✅ Step 1: Design Space Exploration → 6 条优化路径评估 (`algorithm_candidates.md`)
- ✅ Step 2: Runtime Cost Model → 500 batch 探测 + 双模成本定律 (`cost_model.md`)
- ✅ Step 3: Algorithm Design → 三种候选算法详细设计 (`algorithm_design.md`)
- ✅ Step 4: Algorithm Selection → DDBP 最终选型 (`algorithm_selection.md`)
- ✅ Step 5: Implementation Planning → DDBP 代码落地规划 (`implementation_plan.md`)

无阻塞。Phase 5 设计阶段已完整结束, 准备进入 Phase 6 原型编码。

## 4. 下一步计划 (Next Steps)
1. 等待 Human Review 确认 `docs/implementation_plan.md` 实施计划
2. 进入 Phase 6 — DDBP 原型编码:
   - Step 1: DegreeTracker + BinPackingScheduler 实现 (ddbp_sampler.py)
   - Step 2: batch.py + pytorch_dataloader.py 修改
   - Step 3: main_FB15K237.py 装配 + args 配置
   - Step 4: node4 单卡验证 → node6 多卡 Benchmark
