# muKG 本轮研究结果与材料交接索引

日期：2026-11-20。交付状态：主实验结果定稿，进入完整报告、论文写作和四次组会说明。当前以 M10 为主要结果、M11 为补充，原冻结协议及原主判断身份保留。本索引记录已存在材料与访问方式，不启动训练，也不将同机存档称为异机备份。

## 1 接收者先读什么

| 顺序 | 材料 | 用途 |
|---|---|---|
| 1 | [完整研究报告](FINAL_RESEARCH_REPORT.md) | 六章连续正文及摘要，当前论文写作底稿 |
| 2 | [四次组会汇报说明稿](GROUP_MEETING_SERIES.md) | 八张说明卡、24 页提纲和讲述稿、16 组问答；每次约 10 分钟，尚未实际试讲 |
| 3 | [论点与证据对应表](CLAIM_EVIDENCE_MAP_2026-11-17.md) | 数值、对照职责、统计身份、最近工作及解释边界 |
| 4 | [确认实验结果总结](CONFIRMATION_RESULTS_2026-11-17.md) | 全部组、逐种子测试与选模、成本、失败和恢复 |
| 5 | [确认协议与命令](CONFIRMATION.md) | 冻结设计和历史运行方式；本轮整理不授权新训练 |

四次汇报按“探索与取舍→A/B/C 质量问题→方向／去重机制→最终结果与收敛”展开。报告与说明稿统一采用 M10/A 质量比 0.9987、整体训练轮约 8.44 倍、匹配执行约 3.55 倍的当前结果，比较对象、区间及统计角色见正文。完整 Q worker 墙钟均值之比约 4.86，仅为描述性补充。

原确认协议以 M11 为主判断，M10 为预设省时备选；当前突出 M10 属于结果后的写作选择。正式结果、图表和讲述均不把它追溯为预设主判断。旧 `paper/draft/method.md` 是标记为非权威的历史稿，不作为当前正文入口。

## 2 工作区、源码和环境身份

- 当前实验 worktree：`/home/hma/.codex/worktrees/formal-v1/muKG_LB`，分支 `codex/formal-v1`。
- GitHub：[当前分支](https://github.com/piomeer/muKG_LB/tree/codex/formal-v1)。接收时记录实际 commit；分支可继续有文档提交，不以浮动分支名替代实验源码身份。
- 冻结源码位于 `output/results/confirmation_v1/run-01/source/`，冻结 `git_head` 为 `212b238169812ae9ecdac03abd35ccb46547d06e`。最终身份以冻结的源码文件哈希、`source/artifacts.json` 和源码快照为准，不能仅用当时的 Git HEAD 代替。
- 当前实现入口：[confirmation.py](../../src/py/experiments/formal_v1/confirmation.py)；训练与评估：[confirmation_jobs.py](../../src/py/experiments/formal_v1/confirmation_jobs.py)；采样：[mechanism_sampling.py](../../src/py/experiments/formal_v1/mechanism_sampling.py)。

[freeze.json](../../output/results/confirmation_v1/run-01/freeze.json)记录的实验环境为 Python 3.13.12、PyTorch 2.7.1+cu118、CUDA runtime 11.8、驱动 550.67、RTX 3070 8,192 MiB；torch CPU 线程 10、interop 线程 1。分析依赖记录为 NumPy 2.4.4、SciPy 1.17.1、Matplotlib 3.11.1。这里描述原运行环境，不表示已在其他操作系统或硬件验证兼容性。

## 3 文件位置与 GitHub 保存范围

下表相对路径均以当前 worktree 根目录为起点。2026-11-20 已核对列出的目录和代表文件存在；“Git 保存范围”区分精选文件和本机完整产物，不表示第三方账号已获得主机访问权限。

| 材料 | 相对位置／代表文件 | Git 保存范围与接收用途 |
|---|---|---|
| 当前报告、说明稿及索引 | `docs/formal_v1/` | 本轮文档进入 Git，作为阅读入口 |
| 冻结协议与任务卡 | `output/results/confirmation_v1/run-01/freeze.json`、`PROTOCOL.md`、`TASKS.md`、`cards/` | 冻结文件及任务卡已进入 Git；只读保留 |
| 精选分析与逐任务复算 | `output/results/confirmation_v1/analysis-01/summary.json`、`recomputed_tasks.json`、`REPORT.md`、三个 PNG | 这些分析文件已进入 Git，可从汇总重算比值、查阅验证历史和重绘曲线 |
| 完整运行与全部尝试 | `output/results/confirmation_v1/run-01/` | `jobs/` 完整内容主要在本机；包含成功与失败尝试，不能以精选文件替代 |
| 最佳与末轮 checkpoint | `run-01/jobs/Q-M10-GPU-501/attempt-01/best.pt`、`last.pt` 等 | 相对于上述 confirmation_v1 根；本机完整产物，不在 Git 中发布全部模型 |
| 验证原始排名 | 同一 Q 尝试下 `validation/epoch-0010.json` 等 100 份文件 | 本机；核对完整验证排名、指标和最早最佳选模 |
| 测试原始排名 | `run-01/jobs/E-M10-GPU-501/attempt-01/test_metrics.json` 等 | 本机；各头尾查询排名与测试指标，不只有平均分 |
| 输入与测试选模清单 | `run-01/inputs/train/`、`inputs/valid/`、`official_test/FB15K237/`、`official_test/selection.json` | 完整冻结输入及选模清单在本机；本轮不修改测试材料 |
| 源码、环境及正确性材料 | `run-01/source/`、`preflight/`、冻结文件和逐尝试 manifest | 完整快照与验收材料在本机，仓库保存当前代码及精选核验文件 |
| 完整同机副本 | `output/results/confirmation_v1/analysis-01/archive/` | 包含原运行的完整复制；Git 保存清单，未发布全部内容 |
| 失败与人工恢复 | `run-01/recovery/manual-20261111.json` 和 Q-A-504 的三个尝试 | 恢复清单已进入 Git，失败原始材料保存在运行／同机副本中 |
| 独立核验程序及回执 | `output/results/confirmation_v1/execution-02/verify_confirmation.py`、`verification.json` | 已进入 Git；完整核验需要本机完整 archive |
| 汇报图与复现脚本 | `docs/formal_v1/figures/` | 历史三图原字节副本、最终验证三横轴图、脚本及来源清单已进入 Git |

上表中从 `run-01/` 开始的缩写统一相对于 `output/results/confirmation_v1/`。完整 archive 的清单包含 7,195 项，分析总清单包含 7,202 项；按清单字节求和约 3.56 GB，不含分析清单文件自身。清单存在不等于原始大文件已经进入 Git。

### 历史阶段

| 阶段 | 本机完整位置 | 阅读入口与范围 |
|---|---|---|
| A/B/C | `output/results/formal_abc_v2/formal-20261106-01/` | [旧结果](FINAL_RESULTS_2026-11-08.md)；质量 3 对、速度 5 对、T 单种子，不与最终确认混合 |
| A/B/C 同机存档 | `/home/hma/muKG_LB/.research-archive/2026-11-08-formal-abc-v2-complete-01/` | 已核对目录存在；旧阶段的 receipt 保存自身身份 |
| 机制研究 | `output/results/mechanism_v1/mechanism-20261108-01/` | [机制结果](MECHANISM_RESULTS_2026-11-09.md)；2 个观测、12 个三种子验证训练 |
| 机制同机存档 | `/home/hma/muKG_LB/.research-archive/2026-11-09-mechanism-v1-complete-01/` | 已核对目录存在；完整原始观测、模型及对应回执 |
| 更早探索与撤回 | [历史研究索引](RESEARCH_INVENTORY.md)及其存档位置 | 用于历史追踪，不把已撤回指标重新引入当前成果 |

## 4 关键 SHA-256 身份

| 对象 | SHA-256 |
|---|---|
| 确认 freeze.json | `e695ba6f688dcc8428da9023bc3ce20f2df4f9cbb4456672e5b4b9237818d855` |
| 冻结 source/artifacts.json | `a86bd5f9d8bddd2766fb064aad300731a41a42bba99d43c8f2177ee1a46cdd92` |
| analysis-01/summary.json | `8f5472ab291b007699e7ee3321115cfb20ec42541fb44840f52feb778b789ebd` |
| analysis-01/recomputed_tasks.json | `9f5d28ed4b398cad935fa5b6e138d7b9946f07509bc19f90a4ff9b12d2aa1fb8` |
| analysis-01/artifacts.json | `acf144b4aad6b82e31368d4773cd29460906960ddb248d5e3e0a7cb857bf6b4f` |
| run-01/artifacts.json 与 archive/artifacts.json | `2201c903a9e4d259a4192743eb2838f5ff73b6c8c7e266eaaa40068438a8bdf4` |
| 官方 test2id.txt | `6027e649c1716ac9c1ef2f24672a416aa277aff3e14850ef915ac5a21df70417` |

实体／关系映射和 train、valid 文件各自的 SHA-256 位于冻结文件的 `training`、`validation` 字段。已有核验回执记录 204 项任务、7,202 个清单条目通过；本次做文档、位置和关键身份核对，不将旧回执改写为今天重新完成原始排名与全部大文件的核验。

## 5 接收后的分层检查

### 5.1 从 Git 中的精选记录阅读与重绘

在当前 worktree 中进入以下位置；其他机器使用实际 clone 根目录。先记录分支与 commit，再核对精选分析文件身份：

```bash
cd /home/hma/.codex/worktrees/formal-v1/muKG_LB
git branch --show-current
git rev-parse HEAD
sha256sum output/results/confirmation_v1/run-01/freeze.json \
  output/results/confirmation_v1/analysis-01/summary.json \
  output/results/confirmation_v1/analysis-01/recomputed_tasks.json
```

下列复现只需汇总记录与 Matplotlib，不使用 CUDA，也不需要原始 checkpoint；它会在新临时目录写出图及其 JSON 来源清单，拒绝覆盖已有输出：

```bash
mukg_handover_replot_dir=$(mktemp -d /tmp/mukg-handover-replot.XXXXXX)
python3 -B docs/formal_v1/figures/plot_confirmation_validation.py \
  --input output/results/confirmation_v1/analysis-01/recomputed_tasks.json \
  --output "$mukg_handover_replot_dir/validation.png"
```

通过此步骤说明已保存的验证汇总能够重绘，不证明原始训练、所有排名或 checkpoint 已完整移交。输入身份不匹配时停止检查，不修改固定哈希来绕过失败。

### 5.2 拿到完整归档后的原始结果核验

确认 `analysis-01/archive/` 的全部内容已到位后，使用已有只读核验程序：

```bash
python3 -B output/results/confirmation_v1/execution-02/verify_confirmation.py \
  output/results/confirmation_v1/analysis-01
```

预期输出 `status: verified`、`tasks: 204`、`files: 7202`，分析与 archive 清单哈希应与上表一致。该程序读取完整归档，核对文件、原始验证／测试排名、选模和统计，需要本机完整材料及 SciPy；不启动 GPU worker，不重新训练，也不重新运行模型推理测试。缺少归档时应报告材料缺失，不将 Git 中的精选记录当成完整替代。

### 5.3 完整训练复现属于后续独立事项

本轮没有在新环境重训。若未来要做训练复现，应先核对冻结源码、输入、环境、GPU 资源和预算，在新输出目录另立执行记录。不能在已封存目录重跑、覆盖或只依据最新文档分支反推原实验环境。当前交接不包含自动执行该事项。

## 6 完成状态与未确认事项

| 项目 | 当前状态 |
|---|---|
| 主实验与分析 | 完成；结果封存，204/204 有效任务，无缺失配对 |
| 当前研究报告 | 已形成六章正文底稿，最终指本轮实验结果，不代表已经通过发表评议 |
| 四次口头汇报 | 逐页说明稿与备查材料已准备；实际试讲、PPT 和组会日期未安排 |
| 原始证据可追溯 | 既有核验与哈希清单保留，本次核对关键身份和路径 |
| 本机材料 | 当前 worktree 与同机副本已定位；两份都在本机，不构成异机备份 |
| 异机备份与接收者访问 | 尚未确认目的地、访问权限和恢复成功记录；不声称已完成外部移交 |
| 算法新颖性／外部适用性 | 保留定向核对和单配置边界；旧 C1 门槛仍为 UNRESOLVED |
| 新实验 | 当前不启动。仅在具体缺口影响核心结论或决定扩大主张时另行讨论与冻结 |

完成本轮整理后，每项独立改动按用户约定检查、commit 并 push。接收者可以从当前分支阅读精选材料；如需完整归档，应同时确认文件清单、接收位置及访问方式，并在接收端运行相应只读核验。未确定外部目的地前，不将“已有同机副本”登记为“已经异机恢复”。
