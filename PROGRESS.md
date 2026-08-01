# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X — Evidence Audit Part 2（C1 GPU Runtime）与 C1-R1 v1.1 合并补跑均已完成。C1.2-R1、C1.3-R1、C1.7-R1 已升为 A；下一步进入 Part 3（C2 架构审计）。

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：batch_size=10000、neg_num=150 在 RTX 3070 8GB 上仍为已知 OOM 配置。C1-R1 preflight 证明 batch_size=5000、neg_num=150 的完整 BL/GPU step 在当前环境峰值 reserved 约 44%，但其他代码路径仍须独立预检。
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
- **Phase 9 的 25.1s/4.4s 与 142× 已被 C1-R1 替换，不再作为论文定稿值。新值为 6.013× [5.944, 6.084] 与 87.88× [72.92, 105.91] standard-deviation compression。**
- **C1 不主张 CPU/GPU 质量等价或 non-inferiority；C1.5/C1.8 均退出论文正文。**
- **Method 章节不含具体实验结果数值（留给 Experiments 章节），仅提及设计预期和定性关系**  *(自动映射自 L1 宪法)*
- **算法伪代码使用 algorithm/algorithmic 环境，便于后续 LaTeX 转换**  *(自动映射自 L1 宪法)*
- **Figure X 框架架构图尚未生成，标注为占位符**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
已完成 `docs/evidence_audit_part2_c1_gpu_runtime.md` v1.1 与 C1-R1 原始证据包。六个配对 seeds、throughput/trace 独立进程、24 个主 jobs、6 个 compute-only 诊断均完成。C1.2-R1 端到端 speedup=6.013× [5.944, 6.084]；C1.3-R1 full-batch within-epoch standard-deviation compression=87.88× [72.92, 105.91]；C1.7-R1 GPU neg time=3.0026ms（sample SD=0.0229ms，95% CI [2.9786, 3.0266]）。Part 2 最终为 3 A、2 B、4 C、0 D。seed45 throughput GPU attempt1 因 warm-up 前 thermal 标志被排除，完整 GPU→BL pair 按协议仅补跑一次并通过；失败 artifact 保留。

## 4. 卡点 (Blockers)
C1 性能 headline 不再被证据等级阻塞，但必须披露 CPU Bernoulli/global-collision 与 GPU tail-only/batch-tail-filter 的语义差异，且不得推导质量等价。C1.6 sampler-only memory 与 C1.9 统一瓶颈占比仍为 C。C2.1 架构层边界留待 Part 3 正式冻结。

## 5. 下一步计划 (Next Steps)
1. 进入 Part 3 — C2 Unified Runtime Framework 审计，先冻结四层/五层正式边界。
2. 从 C1-R1 派生 CSV 重新生成论文 Fig.5/Fig.6 与表格；不得复用 hardcoded 5.7×/142×。
3. C1.6 与 C1.9 仅在论文确有需要时另行设计隔离测量；C1.5/C1.8 继续退出正文。
