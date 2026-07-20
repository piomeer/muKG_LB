# 研究进度与实验台账 (Research Handover)
**生成时间**: 2026-07-17 13:38 JST (更新于 13:46 JST)  
**当前环境**: `server_node4` (RTX3070, 8GB VRAM, offline)  
**项目**: MuKG (Multi-source Knowledge Graph Embedding) + CBP (Cost-aware Batch Packing)  
**Git HEAD**: `d84f72c` (Phase 6 - Node 3)

---

## 一、实验台账总表

| 实验ID | 名称 | 状态 | 数据集 | 模型 | 命令/脚本 | 核心结果 | 备注 |
|--------|------|------|--------|------|-----------|----------|------|
| **Phase-0.0** | 项目 Fork 与初始化 | ✅ **完成** | — | — | — | 项目基础结构就绪 | 来自 nju-websoft/muKG |
| **Phase-0.1** | 基础模型移植 | ✅ **完成** | 多数据集 | 16+ KGE 模型 | — | Torch/TF2 双后端就绪 | TransE/ConvE/RotatE 等 |
| **Phase-1.0** | 四阶段拆分探针 | ✅ **成功** | FB15k-237 | TransE (dim=400) | `profile_transe.py` (v1) | Collate 46.6%, NegSamp 35.7% 占主导 | Git: `bf3ed94` |
| **Phase-1.1** | 单卡 Profiler 探针 | ⚠️ **中断** | FB15k-237 | TransE | Profiler 集成 | 6 个 trace (~7.4MB) | 命名规则混乱 |
| **Phase-1.2** | 满血参数基准 (Node4) | ✅ **成功** | FB15k-237 | TransE | `main_FB15K237.py` | **batch_size=5000, neg_num=150** 为最终配置 | Git: `2e66798` |
| **Phase-2.0** | 负采样 5 子阶段拆解 | ✅ **成功** | FB15k-237 | TransE | `analyze_neg_sampling.py` | B1=42.3%, B2=23.0%, B3=14.2% | 日志: `negative_sampling_cost.csv` |
| **Phase-3.0** | Hub Entity 相关性 | ✅ **成功** | FB15k-237 | TransE | `profile_transe.py` (v2) | Hub vs Sampling: R=**0.8163** | 核心发现驱动后续方向 |
| **Phase-3.1** | Hub Collision 深度分析 | ✅ **成功** | FB15k-237 | TransE | `analyze_hub_correlation.py` | Collision vs Sampling: R=**0.8640** | 重试 R=0.0254 (非瓶颈) |
| **Phase-3.2** | Hub Reuse 分析 | ✅ **成功** | FB15k-237 | 静态分析 | `analyze_hub_reuse.py` | Hub 分布 SVG 图 | `figs/entity_rank_vs_occurrence.svg` |
| **Phase-4** | 设计空间探索 | ✅ **完成** | — | 文档分析 | — | 6 Route (A~F)，选择 Route C (DDBP) | `docs/algorithm_candidates.md` |
| **Phase-5.0** | DDBP 算法设计 | ✅ **完成** | — | 文档设计 | — | DDBP 最终选择 | `docs/algorithm_design.md` |
| **Phase-5.1** | CBP 重构 (DDBP→CBP) | ✅ **完成** | — | 代码重构 | — | 术语统一，4层架构定稿 | Git: `263fd21` |
| **Phase-5.5** | Weight Assumption 验证 | ✅ **成功** | FB15k-237 | 统计分析 | `validate_weight_assumption.py` | R²=**0.9008** (candidate_size模型) | 原始 R²=0.1657 被拒绝 |
| **Phase-5.6** | Cost Model 拟合 | ✅ **成功** | FB15k-237 | 线性回归 | `fit_cost_model.py` | B3≈51.8ms | `docs/cost_model.md` |
| **Phase-6-N1** | Cost Estimator + Scheduler | ✅ **完成** | — | 代码实现 | — | `cost_estimator.py` + `schedulers.py` | Git: `a9d3f56` |
| **Phase-6-N2** | Feature/Cost 解耦 | ✅ **完成** | — | 代码实现 | — | `features.py` + `cost_model.py` 纯函数 | |
| **Phase-6-N3** | BatchProvider + Stage D | ✅ **完成** | FB15k-237 | 代码+验证 | — | 55k triples 验证, Weight CV=0.0125 | Git: `d84f72c` |
| **Phase-6-E1-v1** | Baseline (初版, 有Bug) | ❌ **失败** | FB15k-237 | TransE | `run_cbp_evaluation.py` (v1) | 6 epoch 后 Validation 崩溃 | Bug: `args.is_torch` 未设置 |
| **Phase-6-E1** | Baseline (Random+Chunk) | ✅ **成功** | FB15k-237 | TransE (dim=400) | `run_cbp_evaluation.py --sorter Random --packer Chunk --epochs 10` | Mean step=280.8ms, CV=0.05395, Filt.MRR=0.194 | 10 epoch 完整输出 |
| **Phase-6-E2** (初版) | CBP (初版, 有Bug) | ❌ **失败** | FB15k-237 | TransE | `run_cbp_evaluation.py` (v1, CBP) | 4 epoch 后 Validation 崩溃 | Bug: `args.is_torch` 未设置 |
| **Phase-6-E2** | CBP (Cost+FFD) | 🔄 **运行中** | FB15k-237 | TransE (dim=400) | `run_cbp_evaluation.py --sorter Cost --packer FFD --epochs 10` | Epoch 5/10, mean_step~293ms, CV~0.054, Filt.MRR=0.194 | Scheduler overhead=1.165s (18x) |

---

## 二、各阶段详细说明

### Phase 0: 项目初始化和基础模型移植

| 时间 | 事件 | 详情 |
|------|------|------|
| 2025 末 ~ 2026 初 | Fork muKG | 从 nju-websoft/muKG fork，初始化项目结构 |
| 2026 初 | 模型移植 | Torch: TransE, TransH, TransR, TransD, ConvE, DistMult, ComplEx, HolE, RESCAL, RotatE, SimplE, TuckER, Analogy (13个) |
| | 模型移植 | TF2: TransE, TransH, Analogy (3个) |
| | 数据集 | FB15k-237, WN18, WN18RR, FB15K, FB13, NELL-995, YAGO3-10 适配完成 |
| | 评估流水线 | Link Prediction: MRR, Hits@1/5/10/50, Mean Rank (Raw + Filt.) |

### Phase 1: 第一阶段 Profiling — 训练四阶段耗时分解

**时间**: 2026-04 ~ 2026-05 | **Git**: `bf3ed94` ~ `d4c9134` | **节点**: Node4 (RTX3070)

**目的**: 分解 TransE 训练的 4 个阶段的耗时占比。

**核心结果** (来自 `training_time_breakdown.csv`):

| 阶段 | 耗时占比 | 说明 |
|------|---------|------|
| Collate (ID Mapping) | **46.6%** | CPU 端实体 ID 映射 |
| Negative Sampling | **35.7%** | CPU 端负采样 |
| Tensor Construction | 10.7% | 张量构建 |
| Forward | 3.3% | GPU 前向传播 |
| Backward | 3.4% | GPU 反向传播 |
| Optimizer | 0.2% | GPU 优化器更新 |

**关键结论**: CPU 端操作占训练 93%，GPU 仅占 7%。瓶颈在数据加载而非计算。

**问题/经验**:
- `profile_transe.py` (v1) 第一版探针代码输出不完整
- PyTorch Profiler trace (12个JSON文件) 命名规则混乱，不便分析
- 满血参数 `batch_size=5000, neg_num=150, max_epoch=3` 在 8GB VRAM 上显存峰值 ~3,500MB

### Phase 2: 第二轮 Profiling — Negative Sampling 子阶段拆解

**时间**: 2026-05 | **Git**: `2230b24` | **节点**: Node4

**目的**: 将 Negative Sampling 细分为 B1~B5，找到具体瓶颈。

**核心结果** (来自 `negative_sampling_breakdown.csv`):

| 子阶段 | 组件 | 占比 | 瓶颈等级 |
|--------|------|------|---------|
| B1 | Random Sampling (随机数生成) | **42.3%** | 🔴 #1 瓶颈 |
| B2 | Candidate Build (候选构建) | 23.0% | 🟡 |
| B3 | Collision Check (碰撞检查) | **14.2%** | 🔴 #2 瓶颈 |
| B4 | Retry (重试) | 0.4% | 🟢 可忽略 |
| B5 | Output Build (输出构建) | 2.6% | 🟢 可忽略 |

### Phase 3: 第三轮 Profiling — Hub Entity 影响分析

**时间**: 2026-05 ~ 2026-06 | **Git**: `82f9658`, `99d4298`

**目的**: 分析 Hub entity 对负采样耗时的影响。

**核心发现**:
- **Hub Count vs Sampling Time**: Pearson **R = 0.8163** ← 核心结论，驱动后续论文方向
- **Collision Check vs 总时间**: **R = 0.8640** (最强波动解释因子)
- **avg_retry vs 总时间**: **R = 0.0254** (几乎无关 → 重试不是瓶颈)
- **candidate_size 模型 R² = 0.9008** → Cost Model 最终采用

**Phase 3 输出文件**:
| 文件 | 行数 | 说明 |
|------|------|------|
| `profiling_summary.csv` | 734 rows (12 June) | 完整 profiling 数据 (batch_size=3000) |
| `hub_analysis.csv` | 734 rows | Hub entity 分析 |
| `negative_sampling_cost.csv` | 734 rows | B1-B5 子阶段分解 |
| `negative_sampling_breakdown.csv` | — | 阶段汇总占比 |

### Phase 4: 设计空间探索与算法选择

**时间**: 2026-06 | **Git**: `99d4298`

- 提出 6 个优化 Route (A~F)
- 评估各 Route 的复杂度、收益、风险
- **最终选择 Route C (Degree-Driven Bin Packing)**，即后续的 CBP
- 详细分析: `docs/algorithm_candidates.md`

### Phase 5: 算法设计与验证

| 子阶段 | Git | 关键产出 |
|--------|-----|---------|
| Phase 5.0 DDBP 设计 | `d9a3c02` | `docs/algorithm_design.md` — DDBP 算法描述 |
| Phase 5.1 重构 (DDBP→CBP) | `263fd21` | 术语统一，4层架构定稿 |
| Phase 5.5 Weight 验证 | `765f707` | R²=0.9008 确认，原始 R²=0.1657 拒绝 |
| Phase 5.6 Cost Model 拟合 | — | B3≈51.8ms, `docs/cost_model.md` |

**Weight Assumption 验证核心结果**:
```
原始假设 R² = 0.0274 (拒绝)
candidate_size 模型 R² = 0.9008 (采用)
最终公式: E_retry * B3_const, E_retry = min(max_try, 1/(1 - N_neg/candidate_size))
```

### Phase 6: CBP Runtime Framework 实现与评估 (当前活动)

#### Node 1~3: 框架实现 (已完成)

| Node | 组件 | 文件 |
|------|------|------|
| Node 1 | Cost Estimator + Scheduler | `cost_estimator.py`, `schedulers.py` |
| Node 2 | Feature/Cost 解耦 | `features.py`, `cost_model.py` |
| Node 3 | BatchProvider + Stage D | `batch_provider.py` |
| 验证 | 端到端验证 | Weight CV=0.0125 (55k triples) |

#### Node 4: 评估实验

##### Exp-1: Baseline (RandomSorter + ChunkPacker) ✅ 已完成

| 指标 | 值 |
|------|-----|
| 启动时间 | 2026-07-17 13:11 |
| 完成时间 | 2026-07-17 13:27 |
| 总耗时 | 801.99 秒 (~13.4 min) |
| Mean Step Time | 280.80 ms (范围 278.6~284.3 ms) |
| Mean Step CV | 0.05395 (稳定，范围 0.05322~0.05617) |
| Mean Epoch Time | 80.20 s |
| Loss 衰减 | 52.55 (epoch 0) → 0.94 (epoch 9) |
| Filt. MRR | 0.194 |
| Filt. Hits@10 | 0.322 |
| 阶段耗时 | Neg 91.2%, Fwd 1.8%, Bwd 2.1%, Opt 0.4% |
| Scheduler Overhead | 64.8 ms |
| 输出文件 | `epoch_summary.csv`, `profiling_summary.csv` (2730 rows), `batch_runtime_variance.csv` (2730 rows) |

##### Exp-2: CBP (CostSorter + FFDPacker) 🔄 运行中 (5/10 epochs)

| 指标 | 值 (截至 epoch 4 平均) |
|------|----------------------|
| Mean Step Time | ~293 ms (比 Baseline 慢 ~4.3%) |
| Mean Step CV | ~0.054 (与 Baseline 持平) |
| Mean Epoch Time | ~84.4 s (比 Baseline 慢 ~5.2%) |
| Scheduler Overhead | 1165 ms (Baseline 的 18x) |
| Filt. MRR | 0.194 (与 Baseline 一致) |
| Filt. Hits@10 | 0.32 (与 Baseline 一致) |

---

## 三、失败实验详细分析

### 1. `args.is_torch` AttributeError (Phase 6-E1/E2 初版)

**错误信息**:
```
File "evaluation.py", line 245, in fomulate
    if self.args.is_torch:
AttributeError: 'ARGs' object has no attribute 'is_torch'
```

**原因**: 实验脚本 `run_cbp_evaluation.py` 绕过 `kge_models` 直接创建 Trainer，未设置 `args.is_torch = True`。

**修复**: 在 training loop 前添加 `args.is_torch = True`。

### 2. Phase-1.1 Profiler 探针中断

**问题**: PyTorch Profiler trace 文件混合了多个 GPU PID 的时间戳，12 个 JSON 文件命名不统一。

**根因**: Profiler 启动/停止逻辑未正确同步，多个 trace 覆盖写入了不同 PID 的文件。

### 3. Known: evaluation_metrics.csv 为空

Baseline 输出中 `evaluation_metrics.csv` 只有 header（41 bytes），`print_results()` 返回了空 dict。需检查 `LinkPredictionEvaluator` 的返回值处理。

---

## 四、当前最佳成果

| 指标 | Baseline (Random+Chunk) | CBP (Cost+FFD, 5/10) |
|------|------------------------|----------------------|
| Filt. MRR | **0.194** | **0.194** (持平) |
| Filt. Hits@10 | **0.322** | **0.32** (持平) |
| Filt. Hits@1 | **0.131** | **0.132** (持平) |
| Mean Step Time | **280.8 ms** | ~293 ms (慢 4.3%) |
| Mean Step CV | **0.05395** | ~0.054 (持平) |
| Mean Epoch Time | **80.2 s** | ~84.4 s (慢 5.2%) |
| Scheduler Overhead | **64.8 ms** | **1165 ms** (慢 18x) |

**模型权重路径**: `output/results/TransE/FB15K237//torch/` (由 `model.save()` 生成)

---

## 五、已知待解决问题清单

1. **[严重] CBP Scheduler 开销过高**: FFD Packer O(n×batch) 双重循环，1.165s overhead。需向量化或改用 round-robin。
2. **[中等] Validation 时 GPU 显存风险**: 全实体评分在 8GB VRAM 下接近极限。
3. **[中等] CV 不降**: Baseline 和 CBP 的 CV 几乎一致 (~0.054)，因 91% 时间在 CPU Negative Sampling。
4. **[低] 多环境 Memory 同步**: offline 环境无法 push，需手动 rsync。
5. **[低] evaluation_metrics.csv 为空**: `print_results()` 返回空 dict。
6. **[低] Phase-1.1 Trace 文件混乱**: 12 个 JSON Profiler trace 命名不规范。

---

## 六、对下一个接手者的 3 条最关键建议

### 🔥 建议 1: 先等待 CBP 实验完成再评估

PID `173881` 正在运行 CBP (`Cost+FFD`)，已完成 5/10 epochs。
```bash
tail -f output/results/exp_CBP/training.log
# 如果进程中断，重新启动:
nohup python3 -u src/py/experiments/run_cbp_evaluation.py \
  --sorter Cost --packer FFD --epochs 10 --exp-label CBP \
  > output/results/exp_CBP/training.log 2>&1 &
```

### 🔥 建议 2: 优先优化 FFD Packer 实现

CBP `scheduler_overhead` = 1.165s (Baseline 65ms)，差 18 倍。
- 方案: `FFDPacker.pack()` 中的双重 `for triple... for b_idx...` 改为 round-robin 分配
- 复杂度从 O(n×batch) 降到 O(n)

### 🔥 建议 3: 关注 CV 不降的问题

91% step 时间花在 CPU Negative Sampling。CBP 只影响 GPU 阶段。
- 需用 GPU-based negative sampling 重新实验
- 或改用更大的 batch_size/更复杂模型 (ConvE) 放大 GPU 方差
- 论文可论证: CBP **不损害精度** + 理论负载均衡能力 (Weight CV=0.0125)

---

## 七、关键文件索引

| 类别 | 文件 | 说明 |
|------|------|------|
| 实验入口 | `src/py/experiments/run_cbp_evaluation.py` | Phase 6 评估脚本 |
| 批量调度 | `scripts/run_cbp_benchmark.sh` | 串行 Baseline→CBP |
| Baseline 日志 | `output/results/exp_Baseline/training.log` | 10 epoch 完整日志 |
| CBP 日志 | `output/results/exp_CBP/training.log` | 运行中 (epoch 5/10) |
| Scheduler | `src/py/load/schedulers.py` | Sort+Pack 策略模式 |
| BatchProvider | `src/py/load/batch_provider.py` | 零侵入注入 Adapter |
| Cost Model | `src/py/load/cost_model.py` | Cost table (R²=0.9008) |
| Feature 提取 | `src/py/load/features.py` | 图拓扑特征提取 |
| 历史 Profiling | `output/results/profiling_summary.csv` | Phase 3 完整数据 |
| 历史 Neg 分析 | `output/results/negative_sampling_cost.csv` | Phase 2 B1-B5 |
| 算法设计文档 | `docs/algorithm_design.md` | DDBP/CBP 算法描述 |
| 算法候选 | `docs/algorithm_candidates.md` | 6 Route 评估 |
| Cost Model 文档 | `docs/cost_model.md` | 公式推导 |
| 实验航海日志 | `PROGRESS.md` | 项目状态跟踪 |
| 项目宪法 | `.clinerules` | 环境路由等约束 |
| Memory 图谱 | `mukg-memory.json` | L2 记忆存储 |
| Memory Bouncer | `utils/memory_bouncer.py` | 强制收尾脚本 |

---

## 八、环境信息

```json
{
  "env": "server_node4",
  "role": "gpu_compute",
  "network": "offline",
  "gpu": "RTX3070",
  "vram": "8GB",
  "can_train": true,
  "sync_source": "pc-cluster",
  "default_sync_command": "rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/"
}
```

**跨环境 Memory 同步提醒**: 此环境 offline，不能直接 push GitHub。修改代码后需提醒用户:
```bash
rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/
```
然后在 pc-cluster 上执行 `python3 utils/memory_bouncer.py` 完成 L2/L3 同步。

---

*本台账由 Cline 自动生成，覆盖 Phase 0~6 完整实验历史。CBP 实验完成后，请更新 PROGRESS.md 并执行 memory_bouncer.py 完成闭环。*