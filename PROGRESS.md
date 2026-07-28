# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 9 Step 3 — 消融实验 (10 epochs × 4 configs) — 完成

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
✅ Phase 7, 8, 9 Step 1-2 全部完成
✅ Phase 9 Step 3 消融实验完成 (2026-07-28)

BL (Random+Chunk+CPU): loss 1.037→0.404, MRR 0.0064→0.0252, Hits@10 0.0075→0.060, epoch 24.9s, neg_std 28.5ms, step_std 34.3ms
CBP (Cost+FFD+CPU): loss 1.042→0.405, MRR 0.0077→0.0249, Hits@10 0.0125→0.058, epoch 25.4s, neg_std 28.5ms, step_std 34.3ms
GPU (Random+Chunk+GPUv2): loss 0.976→0.223, MRR 0.0059→0.0227, Hits@10 0.005→0.058, epoch 4.4s, neg_std 0.2ms, step_std 6.0ms
CBP+GPU (Cost+FFD+GPUv2): loss 0.986→0.223, MRR 0.0063→0.0125, epoch 4.7s, neg_std 0.2ms, step_std 6.0ms

核心发现: GPU 将 neg_std 从 28.5ms 降到 0.2ms (142x), step_std 从 34.3ms 降到 6.0ms (5.7x)。CBP 在 GPU 路径上边际收益减弱。

## 4. 卡点 (Blockers)
1. 同步代码到 pc-cluster 并 git push (当前 node4 internet=true)

## 5. 下一步计划 (Next Steps)
1. [近] 同步代码到 pc-cluster: `rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/`
2. [近] 在 pc-cluster 上 `git push origin production`
3. [近] MCP Server 在 pc-cluster 运行: 同步 L2 实体和关系
4. [中] 完善论文结果表 (BL/CBP/GPU/CBP+GPU 四组对比)
5. [远] Phase 9 Step 3: 完整 10 epoch 实验验证收敛稳定性
