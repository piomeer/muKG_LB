# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
MuKG Negative Sampling Phase 3 — Hub Entity Correlation Deep Dive

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。
- **Per-batch profiling 数据（profiling_summary.csv, hub_analysis.csv）通过每 epoch 写入方式防丢失**  *(自动映射自 L1 宪法)*
- **OOM 保护：验证前执行 torch.cuda.empty_cache()**  *(自动映射自 L1 宪法)*
- **time.perf_counter() 用于微秒级计时，比 time.time() 精度更高**  *(自动映射自 L1 宪法)*
- **DataLoader collate_fn 内部需自包含 reset 逻辑，不可依赖外部调用**  *(自动映射自 L1 宪法)*
- **avg_entity_degree 存储数据类型不匹配（numpy int64 写入 csv 后读取为 0），需统一为 Python int**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
Phase 3 完成。

Hub Entity 影响路径完全解析：

Hub Count vs B1-B5:
- vs Sampling (B1):    R = 0.8163 (强正相关) ← 主要影响路径
- vs Retry (B4):       R = 0.7359 (强正相关)
- vs Output (B5):      R = 0.6192 (中等正相关)
- vs Collision (B3):   R = 0.5404 (中等正相关)
- vs Candidate (B2):   R = 0.4170 (中等正相关)

Top 20 Slowest Batches: 全部为 hub_count=6000，表明 Hub 是决定批次速度的首要因素。

可行预测模型：Total_NS_Time = a * hub_count + b，R² ≈ 0.49

## 4. 下一步计划 (Next Steps)
[1] 修复 avg_entity_degree 数据类型问题
[2] 实现 GPU 端 Sampling（替换 random.sample）
[3] 实现 Bloom Filter 加速 Collision Check
[4] 构建线性回归预测模型
