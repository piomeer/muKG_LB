# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase X X8 — C1-R1 clean-room is BLOCKED_ENVIRONMENT before preflight; X1.5 remains governance-frozen

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
- **X1.5 人工裁决必须通过 output/results/evidence_audit_x1_5/manual_adjudications.json 可重放；未知 record_id/DOI/标题目标必须形成 blocker。**  *(自动映射自 L1 宪法)*
- **REMOTE_LOCATED 仅表示已验证远程全文入口，不得伪造本地 SHA-256；C1 在剩余人工队列或检索不完整时保持 UNRESOLVED。**  *(自动映射自 L1 宪法)*
- **自动主题筛选只能排除无 KGE/runtime 信号的明显无关元数据；UNKNOWN/UNCERTAIN 不得静默排除。**  *(自动映射自 L1 宪法)*
- **X1.5 Part 3 的 MQ facet 是基于显式编码和保守元数据信号的 mapping，不是对论文结论或新颖性的自动推断。**  *(自动映射自 L1 宪法)*
- **Part 3 只读 overlay manual_adjudications.json 中 Part 2 records.csv 未保存的机制字段；不得把缺失字段当作 false。**  *(自动映射自 L1 宪法)*
- **novelty_evidence_matrix.csv 只能继承 C1 gate 的 UNRESOLVED/阻塞状态，不能单独释放 RETAIN、NARROW、REFRAME 或 DROP。**  *(自动映射自 L1 宪法)*
- **Part 4 closure audit 只判断是否满足最终人工 novelty decision 的前置条件，不自动选择 RETAIN/NARROW/REFRAME/DROP。**  *(自动映射自 L1 宪法)*
- **Part 3 matrix 的全局 blocker 不得复制为已核验候选的字段错误；closure audit 必须按当前 candidate status 重新计算 peer-review、全文、locator 和检索阻塞。**  *(自动映射自 L1 宪法)*
- **C1 gate 只有在人工队列清空、检索完成/关闭、直接候选证据完整后才能进入 READY_FOR_HUMAN_DECISION。**  *(自动映射自 L1 宪法)*
- **DBLP 重试必须按每批不超过 3 条、单查询 3–5 秒抖动和后续 batch 600 秒等待标记执行；本轮只完成 round 1 batch 0，剩余批次不得绕过等待直接运行。**  *(自动映射自 L1 宪法)*
- **G1/G2 snowball 记录必须与主 C1 corpus 隔离，先去重、自动筛选和人工裁决，未完成前不得写入 novelty verdict。**  *(自动映射自 L1 宪法)*
- **OpenAlex forward 结果存在 5 个 parent 的分页截断；聚合 snowball status 必须保持 PARTIAL，不得将第一页返回当作完整 citation coverage。**  *(自动映射自 L1 宪法)*
- **DBLP retries use persistent query identity, at most one request per query per round, batches of at most three, deterministic 3-5 second jitter, and a minimum 600-second inter-batch interval.**  *(自动映射自 L1 宪法)*
- **retrieval_cutoff.json uses OPEN, COMPLETE, CLOSED_WITH_FALLBACK, or CLOSED_BLOCKED; qualified fallback is advisory for C1, while uncovered gaps remain hard blockers.**  *(自动映射自 L1 宪法)*
- **NOT_DUE retry checks are read-only: no network request and no artifact mutation before the 600-second interval elapses.**  *(自动映射自 L1 宪法)*
- **The first persistent next batch was attempted once for KBGAN, LibKGE, and Marius; all three recorded DNS transport failures and must not receive an immediate same-round retry.**  *(自动映射自 L1 宪法)*
- **The next DBLP batch is not due before 2026-08-08T06:50:59Z; preserve the 600-second gate even when the prior transport failed.**  *(自动映射自 L1 宪法)*
- **X1.5 governance freeze:** retrieval remains `OPEN`, C1 closure remains `UNRESOLVED`, and the 14-query retry universe (3 recovered, 11 pending, 0 completed rounds) is preserved without further network calls until X5.5/X6 and X6.5 completion or waiver.
- **X4 C3 audit:** strict read-only audit; no GPU, training, network, runtime-code, paper-body, or Part 1 changes. The primary rescue estimand is held-out complete-batch CPU negative-sampling time.
- **X5 C4 audit:** strict read-only audit; current FFDPacker is behaviorally equivalent to ChunkPacker, so historical CBP effects cannot be attributed to packing. Composite CBP contribution gate is FAIL; sorter-only rescue is forwarded to X5.5.
- **C4 historical reanalysis:** Phase6 all-row SD 15.5295→3.4086 ms is warm-up/partial-sensitive; complete interior SD is 1.0509→1.1285 ms. Phase9 Step4.5 complete-batch per-epoch SD mean is 9.2381→2.4537 ms, descriptive only.
- **X6.5 C4 candidate:** only after X5.5 approval; CPU-sampler full-training context, 2×2 sorter×distinct GreedyLeastLoad packer, seeds 42–47, five measured epochs, 10%+CI+≤5% mean-time gate.

## 3. 当前进度与卡点 (Current Progress & Blockers)
X0.5 legacy-narrative quarantine has been replayed on production: the register, reverse Claim mapping, historical-data policies, Safe Writing Sources, and five document headers pass the quarantine checker. X1.5 retry/fallback machinery is implemented and its snapshot is frozen for governance; retrieval remains OPEN and C1 remains UNRESOLVED. X4 C3 and X5 C4 audits are complete with deterministic outputs; predictive C3 is not eligible and the composite CBP gate is FAIL.

X5.5 contribution triage is complete: C1.2-R1/C1.3-R1 are the only primary Claims; C1.7-R1, C2.1-R1, and C2.3 are supporting; C2.2/C2.4/C2.5/C3.3/C3.6 and negative C4 reanalyses are Appendix-only; all other inventory/replacement Claims are removed from manuscript authority. C3 and C4 X6.5 promotion branches are formally `WAIVED`.

X6a consumed the finalized triage and is `X6A_COMPLETE_X6B_PENDING`; X6b is `COMPLETE_X6B_WAIVED`. The statistical overlay remains `ANALYZED` because no clean-room rerun has occurred. X1.5 remains frozen and is not automatically resumed.

X8 C1-R1 clean-room preparation at executor commit `ed403bd` is
`BLOCKED_ENVIRONMENT`: the offline capsule and Conda clone were constructed,
but `nvidia-smi` exited 9 (driver communication failure) and the cloned PyTorch
environment reports `torch.cuda.is_available()==False`. No preflight, matrix
job, seal, independent analysis, comparison, or E1/E2/E3 value was produced.
`docs/phase_x_x8_c1_r1_clean_room_report.md` and the deterministic blocked
closure preserve the available lineage. The isolated-worktree X0.5 checker also
remains baseline-blocked by absent historical `output/results/phase9_step4_5`;
the shared checkout's untracked historical evidence was not used.

## 4. 下一步计划 (Next Steps)
Repair the NVIDIA driver/runtime first. Then create a fresh X8 root, repeat
`prepare`, run `status` and `preflight`, and execute the frozen clean-room
matrix only if preflight passes. Do not run `--retry-dblp-next`, Snowball, or
any external retrieval while X1.5 is `FROZEN_DEFERRED`; X1.5 may only resume
through its explicit release gate after X5.5, X6, and X6.5 waiver closure.
