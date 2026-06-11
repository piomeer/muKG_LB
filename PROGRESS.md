# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
构建 MuKG Single-GPU Performance Profiling 框架并完成 16 epochs 实验分析

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **Per-batch profiling 数据（profiling_summary.csv, hub_analysis.csv）通过每 epoch 写入方式防丢失**  *(自动映射自 L1 宪法)*
- **OOM 保护：验证前执行 torch.cuda.empty_cache()**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
完成 16/20 epochs 训练（TransE + FB15k-237, batch=5000, neg=150）。

核心发现：
- Negative Sampling 占总训练时间 60.0%（最大瓶颈）
- Collate（ID Mapping）占 21.3%
- GPU Forward + Backward + Optimizer 仅占 13.3%
- 每 step 平均约 542ms，其中负采样 344ms、ID Mapping 122ms

已实现：
- Stage A-F 计时框架
- GPU 资源监控（max_memory_allocated, memory_reserved）
- Hub Entity 分析（degree 预计算 + 每 batch degree 统计）
- Retry Count 追踪
- 增量式 CSV 写入（每 epoch 落地一次）

阻碍：
- RTX 3070 8GB 约 16 epoch 后验证阶段 OOM（已通过 cache clearing 缓解）
- 未完成 20 epoch 完整实验
- hub_analysis.csv 因程序崩溃未保存，batch-level Hub/Neg Sampling 相关性未分析

## 4. 下一步计划 (Next Steps)
[1] 降低 batch_size（如 3000）或 neg_triple_num（如 100）重新运行 20 epochs 以生成完整 profiling_summary.csv 和 hub_analysis.csv
[2] 基于 batch-level 数据进行 Pearson 相关性分析（Hub count vs Neg Sampling, Retry vs Neg Sampling）
[3] 实现 GPU 端负采样优化突破 CPU 瓶颈
[4] 将 profiling 框架泛化到其他模型（RotatE, ConvE）
