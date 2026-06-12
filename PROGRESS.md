# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
MuKG Negative Sampling Deep Profiling (Phase 2) — 构建 Neg Sampling Cost Model 并回答 5 个研究问题

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **Per-batch profiling 数据（profiling_summary.csv, hub_analysis.csv）通过每 epoch 写入方式防丢失**  *(自动映射自 L1 宪法)*
- **OOM 保护：验证前执行 torch.cuda.empty_cache()**  *(自动映射自 L1 宪法)*
- **time.perf_counter() 用于微秒级计时，比 time.time() 精度更高**  *(自动映射自 L1 宪法)*
- **DataLoader collate_fn 内部需自包含 reset 逻辑，不可依赖外部调用**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
完成 Phase 2 实验。

数据收集：5 epochs (455 steps), batch_size=3000, neg=150, TransE+FB15k-237

核心发现：
- B1: Sampling (42.3%) 是负采样最大瓶颈（random.sample + 伯努利采样）
- B2: Candidate Build (23.0%) — set comprehension 构建候选三元组
- B3: Collision Check (14.2%) — set difference 过滤已有三元组
- B4: Retry (0.2%) 和 B5: Output Build (1.1%) 几乎可忽略

相关性分析：
- Hub Entity vs Neg Sampling Time: Pearson R=0.7014（强正相关）
- Collision Check vs Neg Sampling Time: Pearson R=0.8640（强正相关）
- avg_retry vs Neg Sampling Time: R=0.0254（无关）

avg_entity_degree 为 NaN（数据类型映射问题），需修复。

## 4. 下一步计划 (Next Steps)
[1] 修复 entity degree 数据映射问题（int vs numpy int64）
[2] 针对 B1: Sampling 模块优化（使用 GPU 端统一采样）
[3] 针对 Collision Check 优化（使用 Bloom Filter 或双缓冲 set）
[4] 扩展实验到更多 epoch（需解决 OOM）
