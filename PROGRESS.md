# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X — Evidence Audit Part 3（C2 Unified Runtime Framework）已完成。C2 正式冻结为“双阶段、五个已实现角色”，审计结论为 4 A、1 B、0 C、1 D；下一步进入 Part 4（C3 Offline Cost Model）。

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：batch_size=10000、neg_num=150 在 RTX 3070 8GB 上仍为已知 OOM 配置。C1-R1 preflight 证明 batch_size=5000、neg_num=150 的完整 BL/GPU step 在当前环境峰值 reserved 约 44%，但其他代码路径仍须独立预检。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。GPU 负采样必须全向量化。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **C2 canonical architecture 已冻结：离线控制面 FeatureExtractor → CostModel → Cost Table；在线每 epoch 路径 Scheduler → BatchProvider；训练循环在框架外显式选择 CPU/GPU backend。RuntimePolicy/GPUExecution 仅为 Future Extensions。**
- **Scheduler 策略组合接口存在，但冻结 fixture 已证明当前 FFDPacker 与 ChunkPacker 对有序输入等价；这是 Part 5 blocker，不得在修复/复核前主张两种 packer 产生不同布局。**
- **BatchProvider 是正三元组 batch iterator，不是 PyTorch DataLoader，不 yield sampler，也不自动注入 GPU backend；每次 iterate() 都重新调度。**
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
- **C1 严格 A 级门槛：未舍入原始观测、统一且对称的 estimand、至少 3 次独立重复、重复间不确定性、有效代码与协议限定措辞必须同时通过。**
- **Phase 8 的 CPU comparator 是 synthetic validation sampler（同时替换 head/tail、无全局碰撞检查），不是原始 CPU Bernoulli/global-collision sampler；198×/8.5× 已退出论文证据。**
- **Phase 9 的 25.1s/4.4s 与 142× 已被 C1-R1 替换，不再作为论文定稿值。新值为 6.013× [5.944, 6.084] 与 87.88× [72.92, 105.91] standard-deviation compression。**
- **C1 不主张 CPU/GPU 质量等价或 non-inferiority；C1.5/C1.8 均退出论文正文。**
- **Method 章节不含具体实验结果数值（留给 Experiments 章节），仅提及设计预期和定性关系**  *(自动映射自 L1 宪法)*
- **算法伪代码使用 algorithm/algorithmic 环境，便于后续 LaTeX 转换**  *(自动映射自 L1 宪法)*
- **Figure X 框架架构图尚未生成，标注为占位符**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
已完成 `docs/unified_runtime_architecture_freeze.md`、`docs/evidence_audit_part3_c2_framework.md` 与 CPU-only 审计脚本 `scripts/audit_c2_framework.py`。C2.1-R1/C2.3/C2.4/C2.5 为 A，C2.2 因 Phase 9 Step 2 写 `summary.md`、聚合读 `summary.csv` 的 artifact lineage 不自洽而为 B，C2.6 保持 RETRACTED/D。Phase 6 scheduler overhead 复核为 64.757ms（Random+Chunk）和 1165ms（Cost+FFD）；C1-R1 Random+Chunk scheduler mean 为 BL 73.0879968667ms、GPU 66.8442073333ms，占平均 epoch 0.279904% / 1.529412%，仅作协议限定描述，不创建 C2.6-R1。审计输出含 source manifest、architecture mapping、recomputed metrics 和 machine checks。

## 4. 卡点 (Blockers)
C2 架构定义不再阻塞，但 C2.2 artifact lineage 尚待 Part 7/数据整理修复；旧 C2.6 “~0.5ms” 必须继续删除。当前 FFDPacker 与 ChunkPacker 在冻结 fixture 上等价，是 Part 5 的算法语义 blocker。论文 Method、story freeze、旧 runtime spec 与 Figure 尚未按 canonical architecture 修订，这些修改明确留给 Part 7。

## 5. 下一步计划 (Next Steps)
1. 进入 Part 4 — C3 Offline Cost Model 审计，优先追溯 `candidate_size`、R²=0.9008、r=0.7124 与预测目标/样本单位。
2. Part 5 审计 C4 时，把 FFDPacker==ChunkPacker 作为首要实现 blocker，区分 sorter 效果、packer 效果与历史实验命名。
3. Part 7 按 canonical freeze 修订 Method、story/architecture wording、Figure，并修复或披露 C2.2 `.md`→`.csv` artifact lineage。
