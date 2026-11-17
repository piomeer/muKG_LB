# Formal v1 使用与实验协议

> **2026-11-17 当前研究主线。论文以 M10 为主要结果，M11 为去重补充对照。** M10/A 测试 MRR 配对比值为 0.9987，95% 区间 [0.9958, 1.0017]；完整训练轮相对 A 加速 8.4368 倍，同规则 CPU/GPU 训练轮加速 3.5532 倍。结论限定于当前模型、数据、硬件和配置。

导师讨论入口为[研究故事与论文框架](RESEARCH_STORY_2026-11-17.md)及[论点、证据与最近工作对应表](CLAIM_EVIDENCE_MAP_2026-11-17.md)，附已有 4,800 次验证记录的三种横轴曲线、速度定义及补实验触发条件。当前整理不启动新训练。

论文角色调整是在结果出来后作出的；原冻结协议仍以 M11 为主判断、M10 为预设省时备选。M10 质量保留次要比较身份，A/M10 速度保留事后补充分析身份，不改写冻结协议或原主判断。

> **2026-11-17 冻结确认协议结果。204/204 项已完成，原始排名重算、配对区间及同机存档校验通过。** M11/A 测试 MRR 比值为 1.0003，双侧 95% 区间为 [0.9980, 1.0027]，支持 0.95 质量门槛。完整训练轮加速为 6.9167 倍，95% 区间为 [6.8513, 6.9827]，两项主判断均获得区间支持。

原确认协议与封存结果见[确认实验结果总结](CONFIRMATION_RESULTS_2026-11-17.md)、[确认实验协议与命令](CONFIRMATION.md)和[机器可读结果汇总](../../output/results/confirmation_v1/analysis-01/summary.json)。M10 在原协议中按预先声明的省时备选报告，去重相对 M10 的额外质量收益仍未获得区间支持。总结记录了测试集历史使用、两次 GPU 干扰失败和一次人工恢复。

GitHub 保存源码、测试、协议、任务卡、精选结果、图表和核验清单。完整原始排名、checkpoint 和同机存档按 SHA-256 清单保存在实验主机，本地路径见结果总结。

上一阶段材料见[机制结果](MECHANISM_RESULTS_2026-11-09.md)、[结果解释与研究故事补充](MECHANISM_INTERPRETATION_2026-11-09.md)、[实现使用说明](MECHANISM_USAGE.md)和[执行记录](MECHANISM_EXECUTION_LEDGER.md)。历史正式 A/B/C 结果见[正式结果](FINAL_RESULTS_2026-11-08.md)。

以下正文保留2026-11-04路径及执行历史，不能作为当前协议或运行授权。该阶段结果见[本次记录](records/2026-11-04-campaign-01/RESULTS.md)，独立任务草案见[任务清单](records/2026-11-04-campaign-01/TASKS.md)。后续阶段以各自冻结协议及用户授权为准。

代码位于 `src/py/experiments/formal_v1/`。历史原版源文件保留于 reference/，核心测试直接抽取历史类／函数比较分数、损失、梯度、Adam 更新及 CPU 采样结果。正式实现不依赖 Ray、CBP、成本缓存或旧治理工具。

## 操作顺序

在独立 worktree 根目录执行，使用现有 `/home/hma/miniconda3/bin/python3`。无需安装依赖。沙箱工具需要 GPU 设备权限；普通主机终端使用本机可见 CUDA 环境。

```bash
python3 -B -m src.py.experiments.formal_v1 check
python3 -B -m src.py.experiments.formal_v1 preflight --output output/results/formal_v1/preflight-01
python3 -B -m src.py.experiments.formal_v1 diagnostic --preflight output/results/formal_v1/preflight-01/preflight.json --output output/results/formal_v1/diagnostic-01
python3 -B -m src.py.experiments.formal_v1 analyze --run output/results/formal_v1/diagnostic-01 --output output/results/formal_v1/diagnostic-analysis-01
```

诊断完成后，查看 suite.json 的 budget。先确认实际 GPU 可用小时数，再运行 freeze：

```bash
python3 -B -m src.py.experiments.formal_v1 freeze --diagnostic output/results/formal_v1/diagnostic-01 --output output/results/formal_v1/frozen-01.json --available-hours HOURS
python3 -B -m src.py.experiments.formal_v1 formal --frozen output/results/formal_v1/frozen-01.json --kind quality --output output/results/formal_v1/quality-01
python3 -B -m src.py.experiments.formal_v1 formal --frozen output/results/formal_v1/frozen-01.json --kind timing --output output/results/formal_v1/timing-01
```

`HOURS` 必须替换为已确认的可用小时数，不能用估计值充当授权。所有输出目录拒绝覆盖；源码或 train/valid/ID 映射变化后，旧预检／冻结文件失效。修改协议另开版本与输出目录。

## 固定设计

TransE dim250、L1、原 uniform 初始化、不归一化；logistic_adv margin7、adv2；Adam .0005；150 negatives。预检按5000、2500、1000选择三组均通过且峰值 reserved <90% 的最大 batch。正式组 cpu／balanced；tail 仅诊断。

诊断 seed7 最多100 epoch，不读取测试集。质量 seeds101–103，交替 CPU/GPU 顺序，每任务独立进程，最多2000 epoch。每10 epoch验证，至少100 epoch后连续五次相对耐心锚点改善不足.0001时早停；最佳 checkpoint 另按严格最大 validation MRR选择，同值取最早。达到2000不标记为收敛。诊断任务限时6小时，正式单任务限时24小时；超时／信号中止保留日志和状态，不自动重跑。

计时 seeds201–205，throughput／trace 各一组独立进程，三次一次性预热后重新初始化模型及随机数，再测五个 epoch。epoch包含顺序构造、采样、传输、前后向和更新；评估单独计时。trace保存实际 batch 大小，完整 batch 标准差分析排除短 batch，端到端 epoch保留短 batch。

质量为完整双向 filtered MRR、Hits@1/3/10，平均并列名次，另报头尾结果。validation过滤train+valid，test过滤三划分。测试只在正式任务的验证最优 checkpoint 上执行一次，保存每个查询排名。

质量目标 GPU/CPU MRR≥.95。主报告配对几何平均与双侧95% log-t CI；点估计达标不代表统计证据支持，CI下界≥.95才标记支持该门槛。仅三个配对可能结论不确定。速度和质量分开统计，二者均需完整报告。实验完成不要求优化成功。

## 一个月范围

第一周归档、基线和正确性；第二周诊断及预算冻结；第三周核心质量与速度；第四周分析和有限扩展。预算按2000 epoch上限及实测完整评估耗时估算，再留30%余量。资源不足先取消 WN18RR 与扩展敏感性，不静默减少核心种子。

敏感性与 WN18RR 不由本版核心命令自动启动：核心完成后另立配置与输出，改变一个因素，不混入主结果。CUDA源码审计只用于机制解释。存档索引见 RESEARCH_INVENTORY.md。

## 历史连续执行方式（已停用）

先前运行曾按 RTX 3070 持续独占、30天720小时设置上限；该连续执行安排已被用户本次指令撤销。历史算法为：冻结时扣除已耗用的预检／诊断时间，剩余预算仍须覆盖核心任务与30%余量。

```bash
python3 -B -m scripts.run_formal_v1_campaign --output output/results/formal_v1/campaign-01 --available-hours 720
```

流程依次执行预检、三个 seed7 诊断、诊断分析、预算冻结、六个正式质量任务、20个独立吞吐／trace任务及分析。任一步失败即停止，保留状态与原始材料；不自动重试。状态见 campaign 根目录 status.json，当前任务状态见各 suite 的 jobs/*/status.json。对主进程发送 SIGTERM 会中止当前子任务并保留中断记录。进程意外消失而状态仍为 running 时，应标记执行中断，不视为成功，不覆盖原输出重启。

环境绑定包括 Python、PyTorch/CUDA、驱动、GPU UUID、CPU线程数。每个epoch前后记录GPU进程、温度、功率及频率，发现其他计算进程即停止；这不能排除边界采样之间的短暂干扰。GPU整轮峰值 allocated/reserved 包含整个训练任务，不解释为采样器独占显存。trace 的 sampling_ns 包含负样本生成及其设备传输；positive_transfer_ns 单独记录正样本传输。

冻结前核对完整任务矩阵、强制材料清单、逐文件哈希、配置、环境、初始权重和耗时原始记录。正式结果核对同一冻结协议身份。未完成、被修改或缺失的记录不能通过冻结／正式统计。

## 本次执行位置

实现提交 `e4035f1`；已结束的主流程保存在 `output/results/formal_v1/campaign-01/`。最新预检已通过三组 batch5000。该目录的 status.json 是实时状态；VALIDATION.json 仅为交付时刻的检查快照，不能当作后续任务完成证明。

CPU／GPU组都在GPU上训练同一个模型，名称区分负采样所在设备及规则。启动入口的 check 运行24项正式路径测试；完整测试集159项另包含旧研究的测试，并使用从存档恢复的历史材料。全量测试日志保存在 output/results/formal_v1/implementation-validation/。新正式路径不依赖这些旧输出文件。

运行中的正式 Python 源码不要修改；代码／数据变化会使后续任务或预算冻结拒绝继续。保留当前输出用于解释中断原因，另立版本和结果目录，不覆盖重跑。
