# mechanism-v1 运行说明

用户授权范围为实现、检查、O-A/O-TENSOR、12个验证训练任务、分析及存档。任务清单和实际配置以预检后冻结目录为准。当前检查状态见MECHANISM_EXECUTION_LEDGER.md。

在当前worktree根目录使用现有Python运行。

```bash
python3 -B -m src.py.experiments.formal_v1 mechanism check --output output/results/mechanism_v1/checks-01
python3 -B -m src.py.experiments.formal_v1 mechanism prepare --checks output/results/mechanism_v1/checks-01 --output output/results/mechanism_v1/preparation-01
python3 -B -m src.py.experiments.formal_v1 mechanism freeze --preparation output/results/mechanism_v1/preparation-01 --output output/results/mechanism_v1/mechanism-20261108-01
```

冻结后，单任务执行或串行队列均使用同一任务清单和校验。

```bash
python3 -B -m src.py.experiments.formal_v1 mechanism run-task --run output/results/mechanism_v1/mechanism-20261108-01 --task O-A
python3 -B -m src.py.experiments.formal_v1 mechanism run-queue --run output/results/mechanism_v1/mechanism-20261108-01
python3 -B -m src.py.experiments.formal_v1 mechanism status --run output/results/mechanism_v1/mechanism-20261108-01
python3 -B -m src.py.experiments.formal_v1 mechanism stop --run output/results/mechanism_v1/mechanism-20261108-01
```

status只读且不导入torch。stop写入停止请求，由控制器停止自有worker和后续派发。显式恢复使用run-queue或run-task的`--resume`，从新尝试开始中断的完整任务。基础设施故障自动重试最多一次，已完成输出先核验后跳过。存在的结果目录拒绝覆盖。源码、配置、冻结输入或环境不匹配时拒绝运行。

```bash
python3 -B -m src.py.experiments.formal_v1 mechanism run-queue --run output/results/mechanism_v1/mechanism-20261108-01 --resume
python3 -B -m src.py.experiments.formal_v1 mechanism analyze --run output/results/mechanism_v1/mechanism-20261108-01 --output output/results/mechanism_v1/mechanism-20261108-01/analysis-01
```

每个V任务为一个组和一个种子的完整1000轮，仅验证，保存best.pt和last.pt。O-A及O-TENSOR为两个独立观测任务。完整任务卡含每项实测外推时长及30%排期余量。主阶段完成后形成机制报告，新的损失控制、测试集确认和匹配CPU速度任务另行讨论。

运行中查看active.json对应输出的status.json、validation.csv和stdout侧边日志。输出路径为jobs/任务编号/attempt-NN，request/stdout/stderr与attempt目录并列保存。没有新结果时，根目录冻结状态只表示等待启动。

核心实验材料完成后使用冻结源码重算，再由执行记录注明实际本地存档位置及哈希。
