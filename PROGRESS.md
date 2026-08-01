# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X — Evidence Audit Part 2（C1 GPU Runtime）已完成只读证据追溯、独立重算与 A/B/C/D 定级。下一步由用户决定：先执行 C1 补实验，或继续 Part 3（C2 架构审计）。

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
- **C1 严格 A 级门槛：未舍入原始观测、统一且对称的 estimand、至少 3 次独立重复、重复间不确定性、有效代码与协议限定措辞必须同时通过。**
- **Phase 8 的 CPU comparator 是 synthetic validation sampler（同时替换 head/tail、无全局碰撞检查），不是原始 CPU Bernoulli/global-collision sampler；198×/8.5× 已退出论文证据。**
- **Phase 9 的 25.1s/4.4s 可从舍入 summary 重现为 5.7045×，但当前定级为 C，未经合格补跑不得作为已验证 headline。**
- **C1 不主张 CPU/GPU 质量等价或 non-inferiority；C1.5/C1.8 均退出论文正文。**
- **Method 章节不含具体实验结果数值（留给 Experiments 章节），仅提及设计预期和定性关系**  *(自动映射自 L1 宪法)*
- **算法伪代码使用 algorithm/algorithmic 环境，便于后续 LaTeX 转换**  *(自动映射自 L1 宪法)*
- **Figure X 框架架构图尚未生成，标注为占位符**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
已完成 `docs/evidence_audit_part2_c1_gpu_runtime.md` v1.0 与可复算审计包。C1.1–C1.9 最终定级为 0 A、2 B（C1.1/C1.4）、7 C、0 D。审计脚本核对 29 个来源并生成 26 条派生指标与 20 项机器检查；连续两次运行输出哈希一致。确认 Phase 8 的 198×/8.5× comparator 无效；Phase 9 5.7× 缺少未舍入合格重复；142× 是舍入后的单 epoch within-run dispersion 比值；sampler-only VRAM 与统一口径 bottleneck shift 均无现成证据。

## 4. 卡点 (Blockers)
无文档执行阻塞。论文的 C1 性能 headline 当前被 C 级证据阻塞；升级到 A 需要 matched BL/GPU、未舍入逐 step/epoch 数据、明确 warm-up/短 batch 规则与至少 3 次独立重复。任何补实验仍需用户另行批准。C2.1 架构层边界留待 Part 3 正式冻结。

## 5. 下一步计划 (Next Steps)
1. 与用户复核 Part 2 的 0 A / 2 B / 7 C 结论。
2. 决定先设计并执行 C1.2/C1.3/C1.7 合并补跑协议，还是继续 Part 3 — C2 Unified Runtime Framework 审计。
3. 未经合格补跑，不把 5.7×、142×或 sampler-only memory 写成已验证论文结论。
