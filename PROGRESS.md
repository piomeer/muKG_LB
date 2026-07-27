# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
研究回顾文档生成 — RESEARCH_HANDOVER_20260727_1739.md 完成

## 2. 活跃约束提醒 (Active Constraints)
- **显存红线**：严格控制 batch_size 与 neg_triple_num 的乘积，防止 OOM。batch_size=5000 OOM，安全使用 1000。
- **性能红线**：重构代码时，严禁在 DataLoader 的高频循环中使用纯 Python 的 O(n) 操作（如 for 循环装配列表、重复构建 set）。GPU 负采样必须全向量化。
- **§0.6 Artifact Truth Source：GPU 实验的唯一可信来源为 stdout/stderr/TensorBoard/WandB/CSV/JSON/实验日志/checkpoint/用户返回等真实 Artifact**  *(自动映射自 L1 宪法)*
- **CBP 架构：FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦**  *(自动映射自 L1 宪法)*
- **Scheduler 策略组合：Scheduler(CostSorter, FFDPacker) CBP | Scheduler(RandomSorter, ChunkPacker) Baseline**  *(自动映射自 L1 宪法)*
- **BatchProvider 零侵入注入：Adapter 模式，DataLoader 感知不到调度层存在**  *(自动映射自 L1 宪法)*
- **§6.2 实验真值：所有 GPU 实验结果必须有真实 stdout/CSV/日志支撑，严禁推断**  *(自动映射自 L1 宪法)*
- **GPU 负采样器 (GPUNegativeSampler)：向量化 randint + isin，tail corruption + batch-level pos_tails 碰撞检查。统一运行时验证通过。**
- **环境路由 (Environment Routing)：server_node4 (RTX3070 8GB), network=offline, can_train=true**
- **基线冻结 (Baseline Freeze)：四组实验组别锁定 — BL (Random+Chunk+CPU) / CBP (Cost+FFD+CPU) / GPU (Random+Chunk+GPU) / CBP+GPU (Cost+FFD+GPU)**  *(自动映射自 L1 宪法)*

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 9 Step 2 四组对比基准完成 (BL/CBP/GPU/CBP+GPU, 5 epochs)
  ✅ GPU epoch 4.4s (5.7x vs BL)
  ✅ CBP+GPU epoch 4.7s (5.4x)
  ✅ CBP zero overhead on GPU

blocker: MCP Memory Server JSON 解析错误
blocker: server_node4 offline 不能 git push
blocker: Phase 9 Step 3 (10 epoch) 未执行

## 4. 卡点 (Blockers)
1. [近] rsync 到 pc-cluster: rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/
2. [近] 在 pc-cluster 上执行 memory_bouncer.py 完成 L2/L3 同步 + git push
3. [中] Phase 9 Step 3: 10 epoch 收敛验证 + 全量 MRR/Hits@10 评估
4. [中] 完善论文结果表 (BL/CBP/GPU/CBP+GPU 四组对比)

## 5. 下一步计划 (Next Steps)
1. [近] 同步代码到 pc-cluster: `rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/`
2. [近] 在 pc-cluster 上 `git push origin production`
3. [近] MCP Server 在 pc-cluster 运行: 同步 L2 实体和关系
4. [中] 完善论文结果表 (BL/CBP/GPU/CBP+GPU 四组对比)
5. [远] Phase 9 Step 3: 完整 10 epoch 实验验证收敛稳定性
