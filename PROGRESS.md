# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
增量 Patch .clinerules 第6轮修正：修正 MCP Tool 描述 / CPU Development Environment 泛化 / 新增 §0.6 Artifact Truth Source

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

## 3. 当前进度与卡点 (Current Progress & Blockers)
已完成全部 3 项增量 Patch：(1) §3 修正 MCP Tool 描述 — 引用 Server 提供图谱工具而非 academic_memory Tool (2) §2 泛化 server_pc_cluster 为 CPU Development Environment + capability 描述 (3) §0.5→§1 之间新增 §0.6 Artifact Truth Source。无阻塞。

## 4. 下一步计划 (Next Steps)
无后续任务，本次 Patch 完成。等待用户确认。
