# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X — Evidence Audit Part 1 v1.2 收尾完成。已冻结 28 条候选 Claim 注册表、GPU 路径定位与 Parts 2–7 审计协议；下一步先讨论 Part 2 范围，不自动进入实验或代码修改。

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。batch_size=5000 OOM，安全使用 1000。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。GPU 负采样必须全向量化。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构层定义暂未冻结：story freeze 为四层，Method draft 与 runtime spec 各给出不同五层边界；C2.1 保持 HOLD，Part 3 须统一论文正式定义并区分已实现模块与设计概念。**
- **Scheduler 策略组合：Scheduler(CostSorter, FFDPacker) CBP | Scheduler(RandomSorter, ChunkPacker) Baseline**  *(自动映射自 L1 宪法)*
- **BatchProvider 零侵入注入：Adapter 模式，DataLoader 感知不到调度层存在**  *(自动映射自 L1 宪法)*
- **§6.2 实验真值：所有 GPU 实验结果必须有真实 stdout/CSV/日志支撑，严禁推断**  *(自动映射自 L1 宪法)*
- **GPU 负采样器定位 A：这是重新设计的 GPU-native sampler（tail-only + batch-level pos_tails filter），不是原 CPU Bernoulli/global-collision sampler 的语义等价移植。**
- **环境路由 (Environment Routing)：server_node4 (RTX3070 8GB), network=offline, can_train=true**
- **基线冻结 (Baseline Freeze)：四组实验组别锁定 — BL (Random+Chunk+CPU) / CBP (Cost+FFD+CPU) / GPU (Random+Chunk+GPU) / CBP+GPU (Cost+FFD+GPU)**  *(自动映射自 L1 宪法)*
- **无新增约束。所有数据严格基于 Phase 2 原始 Profiling，不使用 Phase 9 优化后的数据。**  *(自动映射自 L1 宪法)*
- **所有文档/PPT/代码注释中必须精确区分 muKG 原论文 vs muKG_CBP（曾用名 muKG_LB）**  *(自动映射自 L1 宪法)*
- **禁止使用模糊表述如'你的实验'、'你的代码'、'我们的方法'**  *(自动映射自 L1 宪法)*
- **无新增约束。本次操作为纯规划类，未修改代码或数据。**  *(自动映射自 L1 宪法)*
- **batch_size=10000 OOM on RTX 3070 8GB，论文中需标注8GB显存上限**  *(自动映射自 L1 宪法)*
- **Cost Model 的 R²=0.90 与 Phase 10 R²=0.38 使用不同目标；后者不能验证前者，二者均须在 Part 4 追溯后才能写入论文。**
- **Phase 10 CSV 在统计前已四舍五入到一位小数，出现 std=0 与 [nan,nan] CI；不能据此宣称运行时方差为零。**
- **Part 1 的 ACTIVE/HOLD/RETRACTED 是工作流状态，不是 A/B/C/D 可信度结论。**
- **Method 章节不含具体实验结果数值（留给 Experiments 章节），仅提及设计预期和定性关系**  *(自动映射自 L1 宪法)*
- **算法伪代码使用 algorithm/algorithmic 环境，便于后续 LaTeX 转换**  *(自动映射自 L1 宪法)*
- **Figure X 框架架构图尚未生成，标注为占位符**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
已完成: `docs/evidence_audit_part1_claim_inventory.md` v1.2 与 `docs/evidence_audit_template.md` v1.1。Part 1 共保留 28 个可追溯 ID：16 ACTIVE、10 HOLD、2 RETRACTED。已确认 `gpu_cost_microbench.py` 存在；撤回 C2.6 “0.5ms overhead” 与 C3.5 `hub_entity_count R≈0.816`；将 C1.5/C1.8 质量等价表述置于 HOLD。审核复核后将 C2.1 架构定义与 C4.7 within-batch cost CV 置于 HOLD，并修正 Fig.2、Fig.3、Fig.6 的 Claim 证据映射。

## 4. 卡点 (Blockers)
无执行阻塞。进入 Part 2 前需与用户讨论并冻结：C1 审计顺序、性能与质量 Claim 的验收标准、是否允许补跑实验。C2.1 架构层边界留待 Part 3 正式冻结；Fig.2/Fig.3 的指标血缘与重绘方案留待 Part 4/Part 5 处理。

## 5. 下一步计划 (Next Steps)
1. 与用户讨论 Phase X Part 2 — C1 GPU Runtime Evidence Audit 的执行边界。
2. 先做只读证据追溯与独立重算，再决定是否需要补实验。
3. 未经审计，不把 Part 1 的候选数字直接写入论文。
