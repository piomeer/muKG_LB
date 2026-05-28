# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
重构 L1/L2/L3 融合流转机制 — 新 L3 切面状态机 + Bouncer v4

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。

## 3. 当前进度与卡点 (Current Progress & Blockers)
完成 Bouncer v4 重写。L3 从永久锚点模式切换为纯切面状态机（4 固定板块：## 1.-## 4.）。新增 _replace_section 函数用正则精确定位替换。new_constraints 支持追加到 ## 2. 原有约束下方（L1→L3 动态映射）。Payload Schema 改为 active_task/new_constraints/progress_and_blockers/next_steps/l2_graph_updates。干跑测试验证通过：2 条 L2 更新正确追加到 mukg-memory.json，4 板块替换正确，2 条新约束成功追加到 ## 2.
等待下个任务分配新目标

## 4. 下一步计划 (Next Steps)
[待分配]
