# A/B/C 诊断运行与结果

本入口只运行三个seed7、100epoch诊断，结束后等待用户讨论。不会调用旧campaign、正式quality、throughput、trace或freeze入口。模型在GPU，A/B采样在CPU，C采样在GPU；线程10。

从当前工作树根目录执行，RUN为一个尚不存在的新目录；执行器恢复时用相同目录：

```bash
python -B -m src.py.experiments.formal_v1 run-diagnostics --output "$RUN"
python -B -m src.py.experiments.formal_v1 status --run "$RUN"
```

只执行一个诊断：

```bash
python -B -m src.py.experiments.formal_v1 run-task --output "$RUN" --task D-B-7
```

首次执行先做三组共同batch预检；单任务入口也不绕过该步骤。已有完整结果须配置、源码、环境、清单及锚定哈希一致才能跳过；已有结果不覆盖。单任务报告可能显示INCOMPLETE，因为其他组尚未执行，这是预期状态。

状态读取使用标准库，不导入torch、不探测CUDA。`active.progress`展示epoch、验证计数、耗时、最佳验证值和状态；输出路径含尝试编号。日志在对应attempt目录旁的stdout/stderr文件。每个epoch更新status，验证每10epoch输出。

停止：前台Ctrl-C或向控制进程发送SIGTERM；也可在RUN下创建STOP文件。控制器停止当前自有worker并退出，不自动重试用户中断。不要仅杀worker来表达停止整个队列。未分类信号退出不自动重试；仅明确记录的InfrastructureError可最多重试一次。无完整断点恢复，重试从头训练。

异常恢复先检查status/events/active及实际PID；存活的旧控制器或worker不会被新队列重复调度。Linux父进程退出保护使worker收到SIGTERM，释放自有GPU资源。无法判定的异常保留为interrupted，不擅自分类为可重试基础设施故障。

诊断结果：jobs/D-组别-7/attempt-编号/。其中epochs.csv、validation.csv、validation/epoch-*.json、batch_order.json、initial_weights.json、best.pt、gpu_telemetry.jsonl、status.json及artifacts.json逐项留存。验证JSON含完整头尾排名和指标。best.pt非完整续训点。

analysis包含CSV、REPORT.md、summary.json、训练/墙钟时间曲线、头尾曲线。完整时状态为awaiting_review；缺失时为incomplete，不声称目标达标。预算只外推1000epoch候选，test耗时按查询数量估计，trace未知，不冻结正式轮数。

封存另生成本地tar.gz和SHA-256清单，逐成员流式校验，不解压或运行其中代码。源数据由已记录哈希引用历史存档，不宣称异机容灾。
