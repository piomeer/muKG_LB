# 项目总览与研究索引（2026-11-04）

主线：性能分析 → GPU 采样实现 → 效率与质量评估 → 局限讨论。目标是一个月内优先完成核心实验，扩展以实际算力预算为条件。

## 存档与恢复

本地存档：`/home/hma/muKG_LB/.research-archive/2026-11-04-pre-formal-v1/`。
包含工作区快照（含未提交文件）、完整 Git bundle、原始数据/日志/checkpoint、外部 CUDA 审计证据、环境记录与哈希。账号配置与可重建缓存排除。原始文件未搬动；这是同机恢复副本，不是异机备份。

`archive_receipt.json` 保存验收及元数据哈希，`archive_manifest.json` 给出每个快照文件的大小和哈希。

恢复：`python scripts/research_archive.py restore ARCHIVE NEW_EMPTY_PATH`。
验证：`python scripts/research_archive.py verify ARCHIVE`。
重算：`python scripts/replay_research_archive.py RESTORED_PATH`。
Git 历史：`git clone ARCHIVE/history.bundle NEW_CLONE`，随后按快照恢复工作区，以获得当时未提交内容。

## 实验索引

所有相对路径均指存档的 snapshot 根目录；运行状态不等于结论有效。

| 工作与研究问题 | 实际状态 | 代码／配置与数据来源 | 结论及限制 | 用途／重跑 |
|---|---|---|---|---|
| 早期 profiling：训练时间消耗在哪里 | 已执行 | 原版改造的 kge_trainer、pytorch_dataloader；output/results/profiling_summary.md、training_time_breakdown.md、negative_sampling_breakdown.md；src/py/experiments/log | 发现 CPU 数据准备开销；嵌套区域、同步和不同阶段分母需复核 | 动机；正式基线上重测统一计时 |
| Hub 与成本模型：图特征能预测耗时吗 | 已执行并审计 | scripts/fit_cost_model.py、validate_weight_assumption.py；output/results/evidence_audit_part4 | R²=0.9008 已撤回；合成相关性不是实测预测验证 | 探索与负结果；不挽救该分支 |
| CBP：排序打包能平衡时间吗 | 已执行并审计 | src/py/load/schedulers.py；output/results/evidence_audit_part5 | FFD 等同顺序切块；78% 历史结果受预热与短 batch 影响 | 探索与局限；不作主要成功贡献 |
| Phase8/9 GPU 原型 | 已执行，部分结论被替代 | run_unified_runtime_validation.py、phase9_step*；output/results/unified_runtime、phase9_step* | 198×、8.5×、142× 不进入新结论；旧质量评估存在错误或范围不足 | 原型历史；保留代码和结果 |
| C1-R1 六种子运行时 | 已完成，原始数据已重算 | c1_r1_combined_rerun.py；output/results/c1_r1_combined_rerun；protocol.json | 6.01339× epoch speedup、87.8771× SD compression；仅旧 SimpleTransE／采样协议，CPU 身份需明确 | 可用旧协议证据；新方案独立重测 |
| C1-Q1 正式划分质量 | 用户中止，1/6 配对完成 | 未提交的 c1_q1_quality.py、filtered_link_prediction.py；output/results/c1_q1_quality_formal | seed42 MRR CPU/GPU=.096050/.065141；头预测差异突出；n=1，未满足旧收敛门槛 | 诊断；不能声称非劣效；新协议重测 |
| 证据审计与贡献取舍 | 已完成相关阶段 | docs/evidence_audit_part*、phase_x_x5_5_contribution_triage.md | C1 为主要旧证据，C2 支撑；C3/C4 放弃晋级 | 复用结论边界，不沿用复杂执行依赖 |
| 文献检索 | 冻结、未完成 | output/results/evidence_audit_x1_5 | 检索和人工裁决未关闭，不能宣称检索穷尽或全局新颖性 | 相关工作候选；以后按最终方法缩减 |
| X8 隔离复现 | 旧执行受环境阻塞，未产生新结果 | scripts/run_x8_c1_r1_clean_room.py；output/results/x8_c1_r1_clean_room* | 旧失败不等于新主机驱动故障；本次沙箱外可识别 GPU | 失败历史；不把复现工具当实验证据 |
| CUDA randint 源码审计 | 独立诊断报告 | CUDA_randint_source_audit.md；external/ 中原 /tmp 材料 | 解释指定版本与调用的 kernel；不证明全训练加速 | 机制背景，不能扩成历史性能结论 |

## 原版—旧实验—正式路径差异

| 项目 | 历史 fdfd880 | 旧 SimpleTransE 实验 | formal-v1 |
|---|---|---|---|
| 模型／距离 | TransE，L1 | SimpleTransE，L2 | 历史 L1 模型，固定输入核对 |
| 维度／初始化 | 250；uniform ±(margin+2)/dim | 400；Embedding 默认初始化 | 250，历史初始化，无归一化 |
| 损失／学习率 | logistic_adv；margin7、adv2；Adam .0005 | margin loss1；Adam .001 | 保留历史模型和损失语义 |
| CPU 采样 | 每次重试 Bernoulli；random.sample；集合过滤；最后一轮不过滤 | 每个负样本独立 randint 和 Bernoulli | 原历史函数行为，含原有 fallback；接口显式 |
| GPU 采样 | 无本项目优化 | 仅尾，排除整个 batch 的正尾实体 | 75头+75尾；有放回；过滤训练集完整三元组；64轮失败显式报错 |
| 数据 | 官方数据接口 | R1 从 train 留出5000；Q1使用官方划分 | 官方划分；诊断不读取 test |
| 训练器 | 原框架入口，可多 worker | 独立脚本＋固定顺序 Scheduler | 共同精简循环；相同种子/epoch 顺序；CPU 单进程 |
| 评估 | 历史 evaluator | 有错误旧版及 strict-lower 修正版 | 平均并列名次；完整双向；独立穷举检查 |
| 测量 | 后续大量探针混入训练循环 | R1 独立 throughput／trace | 延续分离测量；统一计时边界；保留短 batch |

formal-v1 是“基于历史原版的配置化基线”，不是对作者完整运行环境或论文结果的复制。CPU 与 GPU 采样语义不完全相同；优化归因于声明的整条路径。

## 撤回与禁止引用登记

完整历史编号及传播关系保存在快照的 `docs/phase_x_x0_5_legacy_narrative_quarantine.md`（Q-01至Q-11）与 `docs/evidence_audit_part1_claim_inventory.md`；Part2–5 的最终审计优先于早期 ACTIVE/HOLD 标签。以下均不得当作新正式实验结论：

| 原说法 | 当前处置与原因 |
|---|---|
| 596ms→3.0ms（约198×）组件加速、674ms→79.7ms（约8.5×）step加速 | 永久排除旧数字；比较的计时边界／执行协议不匹配（Q-01/02，Part2） |
| 25.1s→4.4s epoch、28.5ms→0.2ms波动（约142×） | 被C1-R1重测替代；不得混合不同阶段数据（Q-03/04） |
| 成本模型R²=.9008、455个实测样本、解释90%方差 | 撤回；来源与采样单位不符，实际合成材料的Pearson r≈.90079、r²≈.81143，不是样本外预测验证（C3.1-L） |
| hub_entity_count的Pearson R≈.816 | 撤回；变量只有两个不同值，主要编码最后短batch（C3.5） |
| 调度开销约.5ms／epoch、端到端开销可忽略 | 撤回；历史实测baseline64.757ms、CostSorter+FFDPacker1165ms，不能从O(1)查表推导整轮开销（C2.6/Q-06） |
| FFD／装箱是独立贡献，CBP降低78%波动 | 不成立；FFD行为等价顺序切块；排除首尾batch后该Phase6数据不显示波动降低（Q-07/Part5） |
| 已实现四／五层自动透明GPU后端 | 被实现事实替代；运行循环显式选后端，部分层只是未来设计（Q-08/11） |
| 保持质量、已经收敛、非劣效 | 禁止；旧短训练、抽样训练留出评估及未完成Q1不能支持这些结论（Q-09） |
| 采样器独占显存、瓶颈已转移、DDP-ready、普遍适用或SOTA | 禁止／范围外；分别缺隔离显存、统一计时、真实分布式或泛化比较证据（Q-10） |

6.01339× 是可重算的旧协议端到端结果；该数字本身未撤回，但不用于预测 formal-v1 的速度或质量。
