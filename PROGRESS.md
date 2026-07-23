# MuKG & RAG 项目实验航海日志
*(Cline 指令: 开始任务前全文读取，任务阶段性结束后通过 memory_bouncer.py 更新)*

## 1. 当前活动目标 (Active Task)
Phase 8 - Node 4: GPU Negative Sampler & Unified Runtime Validation — 收尾完成

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

## 3. 当前进度与卡点 (Current Progress & Blockers)
✅ Phase 6 - Runtime Attribution 完成 [2026-07-20]
  ✅ Baseline 273 batches + CBP 273 batches (batch_size=1000, neg_num=150)
  ✅ Neg Sampling std 降 78%: 15.5ms → 3.4ms
  ✅ Total step std 降 66%: 16.0ms → 5.5ms

✅ Phase 7 - GPU 迁移可行性研究 完成 [2026-07-23]
  ✅ Step 1: Tensor Construction Deep Profiling
  ✅ Step 2: GPU Migration Feasibility Doc — docs/gpu_migration_feasibility.md
  ✅ Step 3: GPU Cost Model Microbench — break-even N*=264k
  ✅ Step 4: GPU Runtime Architecture Design — Route C
  ✅ Step 5: Runtime Framework Specification

✅ Phase 8 - GPU Negative Sampler 完成 [2026-07-23]
  ✅ Step 0: Architecture Freeze
  ✅ Step 1: GPU Sampler 原型 — ~2.17ms, 43x 加速 vs CPU
  ✅ Step 2: Unified Runtime Integration & Validation (batch_size=5000, neg_num=150)
    ✅ GPU 5 epochs: Loss 0.949 → 0.375, epoch avg 4.78s
    ✅ CPU 2 epochs: Loss 1.029 → 0.666, epoch avg 37.06s
    ✅ 加速比: neg_time 198x, epoch 7.7x
  ⚠️ 已知差异: GPU 版仅 tail corruption + batch-level 碰撞; 原始含 Bernoulli head/tail + 全局碰撞

⚠️ blocker: server_node4 (RTX3070 8GB VRAM) batch_size=5000 OOM
⚠️ blocker: network=offline — 无法 git push

## 4. 卡点 (Blockers)
1. [中] 验证 GPU Sampler 语义对齐差异对 MRR/Hits@K 的实际影响
2. [中] 重启 CBP 实验 (run_cbp_evaluation.py) 完成 10 epochs
3. [远] 规划 Phase 9: 全 GPU Pipeline (负采样 + Tensor 构建 + 训练全部 GPU 化)

## 5. 下一步计划 (Next Steps)
1. [近] 同步代码到 pc-cluster 开发环境，执行 Memory Bouncer + Git 提交
2. [中] 验证 GPU Sampler 语义对齐差异对 MRR/Hits@K 的实际影响
3. [中] 重启 run_cbp_evaluation.py 完成 10 epochs 验证 CBP 稳定性
4. [远] 规划 Phase 9: 全 GPU Pipeline (负采样 + Tensor 构建 + 训练全部 GPU 化)
