# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X X0（RQ、scope、contribution 与 estimand freeze）已完成。论文采用方案 A：C1 是唯一主实证贡献，C2 是支持性实现架构，C3/C4 在 Part 4/5 通过前保持条件性或探索性。下一步进入 X1.5 文献与新颖性审计。

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
- **X0 canonical freeze：论文最小可行主线不依赖 C3/C4；默认按二者审计失败仍可完整成稿**  *(自动映射自 L1 宪法)*
- **外部有效性边界：现有性能证据仅覆盖 muKG、SimpleTransE、FB15k-237、RTX 3070、batch_size=5000、neg_num=150**  *(自动映射自 L1 宪法)*
- **当前无论文级 batch-size/neg-num sensitivity；Phase 10 舍入数据不得替代；跨模型、跨数据集、跨 GPU 型号单卡复现与多 GPU scaling 必须分别注册新 Claim/协议**  *(自动映射自 L1 宪法)*
- **RQ1/RQ2 是 post-result paper-level formalization 下的 primary RQ/estimand；C1-R1-v1.1 replacement protocol 在补跑前冻结，不得把 X0 称为前瞻性预注册**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
已批准并冻结 docs/phase_x_x0_research_freeze.md。RQ1 对应 E1 六 paired seeds 的 epoch speedup 6.013× [5.944, 6.084]；RQ2 对应 E2 full-batch within-epoch SD compression 87.88× [72.92, 105.91]；E3 为 GPU full-batch neg time 3.0026ms [2.9786, 3.0266]；RQ3/E4 限于实现事实。C3 需 Part 4 恢复 target provenance、排除 leakage 并建立 out-of-sample estimand；C4 受 FFDPacker==ChunkPacker 阻塞。三单外部有效性仍是投稿风险，但不通过无证据措辞扩张 scope。

当前卡点：X1.5 文献与新颖性审计尚未执行；C3/C4 尚未完成 Part 4/5 裁决；跨模型、跨数据集、跨 GPU 型号单卡复现与多 GPU scaling 仍是可选 gap-closing 分支。

## 4. 下一步计划 (Next Steps)
1. 执行 X1.5 文献与新颖性审计，建立 KGE runtime systems / GPU negative sampling / modular reproducibility 的 systematic mapping 与 novelty matrix。
2. Part 4 审计 C3，决定 CostModel 是贡献、实现细节还是撤回。
3. Part 5 审计 C4，并按最坏情况骨架决定 CBP 进入正文、附录或删除。
4. 只有完成前三项贡献裁决后，才设计可选的跨模型、跨数据集、跨 GPU 型号或多 GPU gap-closing experiments。
