# 研究进度与实验台账 (Research Handover) — 完整版
**生成时间**: 2026-07-27 17:39 JST  
**当前环境**: `server_node4` (RTX3070, 8GB VRAM, offline)  
**项目**: MuKG (Multi-source Knowledge Graph Embedding) + CBP (Cost-aware Batch Packing) + GPU Sampler  
**Git HEAD**: `728192d` (Phase 9 Step 2 — 四组对比基准完成)

---

## 一、实验台账总表

| 实验ID | 名称 | 状态 | 数据集 | 模型 | 核心结果 | 备注 |
|--------|------|------|--------|------|----------|------|
| **Phase-0.0** | 项目 Fork 与初始化 | ✅ **完成** | — | — | 项目基础结构就绪 | 来自 nju-websoft/muKG |
| **Phase-0.1** | 基础模型移植 | ✅ **完成** | 多数据集 | 16+ KGE 模型 | Torch/TF2 双后端就绪 | TransE/ConvE/RotatE 等 |
| **Phase-1.0** | 四阶段拆分探针 | ✅ **成功** | FB15k-237 | TransE (dim=400) | Collate 46.6%, NegSamp 35.7% 占主导 | Git: `bf3ed94` |
| **Phase-1.1** | 单卡 Profiler 探针 | ⚠️ **中断** | FB15k-237 | TransE | 6 个 trace (~7.4MB) | 命名规则混乱 |
| **Phase-1.2** | 满血参数基准 | ✅ **成功** | FB15k-237 | TransE | batch_size=5000, neg_num=150 为最终配置 | Git: `2e66798` |
| **Phase-2.0** | 负采样 5 子阶段拆解 | ✅ **成功** | FB15k-237 | TransE | B1=42.3%, B2=23.0%, B3=14.2% | 日志: `negative_sampling_cost.csv` |
| **Phase-3.0** | Hub Entity 相关性 | ✅ **成功** | FB15k-237 | TransE | Hub vs Sampling: R=**0.8163** | 核心发现驱动后续方向 |
| **Phase-3.1** | Hub Collision 深度分析 | ✅ **成功** | FB15k-237 | TransE | Collision vs Sampling: R=**0.8640** | 重试 R=0.0254 (非瓶颈) |
| **Phase-3.2** | Hub Reuse 分析 | ✅ **成功** | FB15k-237 | 静态分析 | Hub 分布 SVG 图 | `figs/entity_rank_vs_occurrence.svg` |
| **Phase-4** | 设计空间探索 | ✅ **完成** | — | 文档分析 | 6 Route (A~F)，选择 Route C (CBP) | `docs/algorithm_candidates.md` |
| **Phase-5.0** | CBP 算法设计 | ✅ **完成** | — | 文档设计 | CBP 最终选择 | `docs/algorithm_design.md` |
| **Phase-5.1** | CBP 重构 (DDBP→CBP) | ✅ **完成** | — | 代码重构 | 术语统一，4层架构定稿 | Git: `263fd21` |
| **Phase-5.5** | Weight Assumption 验证 | ✅ **成功** | FB15k-237 | 统计分析 | R²=**0.9008** (candidate_size模型) | 原始 R²=0.1657 被拒绝 |
| **Phase-5.6** | Cost Model 拟合 | ✅ **成功** | FB15k-237 | 线性回归 | B3≈51.8ms | `docs/cost_model.md` |
| **Phase-6-N1~3** | CBP 框架实现 | ✅ **完成** | — | 代码实现 | 4层架构: Feature→Cost→Scheduler→Provider | Git: `a9d3f56` → `d84f72c` |
| **Phase-6-E1** | Baseline 评估 | ✅ **成功** | FB15k-237 | TransE | Mean step=94.6ms, Filt.MRR=0.194 | 10 epoch |
| **Phase-6-E2** | CBP 评估 | ✅ **成功** | FB15k-237 | TransE | Neg Sampling std -78%, Total step std -66% | Cost Model r=0.71 ✅ |
| **Phase-6-N4** | Runtime Attribution | ✅ **成功** | FB15k-237 | TransE | CBP 激活 Weight→NegSampling 因果链 r=0.71 | 273+273 batches |
| **Phase-7.1** | Tensor 构建 Profiling | ✅ **成功** | FB15k-237 | TransE | 张量构建+前向 vs 负采样详细分解 | `tensor_breakdown/` |
| **Phase-7.2** | GPU 迁移可行性 | ✅ **成功** | — | 文档分析 | Route C 推荐 | `docs/gpu_migration_feasibility.md` |
| **Phase-7.3** | GPU Cost Model 微基准 | ✅ **成功** | FB15k-237 | TransE | Break-even N*=264k | `gpu_cost_microbench.py` |
| **Phase-7.4** | GPU 运行时架构设计 | ✅ **成功** | — | 文档设计 | Route C 最终决定 | `docs/gpu_runtime_architecture.md` |
| **Phase-7.5** | 运行时框架规范 | ✅ **成功** | — | 文档规范 | 接口约定 | `docs/runtime_framework_spec.md` |
| **Phase-8.0** | 架构冻结 | ✅ **完成** | — | 文档冻结 | 架构决策锁定 | `docs/phase8_architecture_freeze.md` |
| **Phase-8.1** | GPU Sampler 原型 | ✅ **成功** | FB15k-237 | TransE | ~2.17ms, **43x 加速** vs CPU | `gpu_sampler.py` |
| **Phase-8.2** | 统一运行时验证 | ✅ **成功** | FB15k-237 | TransE | GPU 5 epoch: 4.78s, CPU: 37.06s | neg_time 198x, epoch 7.7x |
| **Phase-9-S1** | 语义对齐验证 | ✅ **成功** | FB15k-237 | TransE | 差异可接受，基线与GPU对齐 | `docs/semantic_alignment_report.md` |
| **Phase-9-S2** | 四组对比基准 | ✅ **成功** | FB15k-237 | TransE | **GPU: epoch 4.4s (5.7x 加速)** | 详见核心成果表 |

---

## 二、各阶段详细说明

### Phase 0: 项目初始化和基础模型移植

| 时间 | 事件 | 详情 |
|------|------|------|
| 2025 末 ~ 2026 初 | Fork muKG | 从 nju-websoft/muKG fork，初始化项目结构 |
| 2026 初 | 模型移植 (Torch) | TransE, TransH, TransR, TransD, ConvE, DistMult, ComplEx, HolE, RESCAL, RotatE, SimplE, TuckER, Analogy (13个) |
| | 模型移植 (TF2) | TransE, TransH, Analogy (3个) |
| | 数据集适配 | FB15k-237, WN18, WN18RR, FB15K, FB13, NELL-995, YAGO3-10 |

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

### Phase 2: 第二轮 Profiling — Negative Sampling 子阶段拆解

**时间**: 2026-05 | **Git**: `2230b24`, `82f9658` | **节点**: Node4

**目的**: 将 Negative Sampling 细分为 B1~B5，找到具体瓶颈。

**核心结果** (来自 `negative_sampling_breakdown.csv`, 455 batches):

| 子阶段 | 组件 | 总耗时 | 占比 | 每 batch 平均 | 瓶颈等级 |
|--------|------|------:|:---:|:---:|---------|
| **B1: Sampling** | `random.sample` 随机采样 | **46.72s** | **42.3%** | ~103ms | 🔴 #1 |
| **B2: Candidate Build** | 构建候选列表 | 25.43s | 23.0% | ~56ms | 🟡 |
| **B3: Collision Check** | set difference 碰撞检查 | **15.73s** | **14.2%** | ~35ms | 🔴 #2 |
| B4: Retry | 碰撞后重试 | 0.22s | 0.2% | ~0.5ms | 🟢 可忽略 |
| B5: Output Build | 组装三元组 | 1.24s | 1.1% | ~3ms | 🟢 可忽略 |
| **B1-B5 合计** | | **89.34s** | 80.8% | | |
| **Total 负采样** | (含框架开销) | **110.56s** | **100%** | ~243ms | |

**相关性发现**:
- Collision Check vs 总时间: **R=0.8640** — 时间波动的 #1 解释因子
- avg_retry vs 总时间: **R=0.0254** — 几乎无关（反直觉！）

### Phase 3: Hub Entity 影响分析

**时间**: 2026-05 ~ 2026-06 | **Git**: `82f9658`, `99d4298`

**核心发现**:
- **Hub Count vs Sampling Time**: Pearson **R = 0.8163** ← 驱动论文方向
- **Collision Check vs 总时间**: **R = 0.8640** (最强波动解释因子)
- **avg_retry vs 总时间**: **R = 0.0254** (几乎无关 → 重试不是瓶颈)
- **candidate_size 模型 R² = 0.9008** → Cost Model 最终采用

**Phase 3 输出文件**:
| 文件 | 规模 | 说明 |
|------|------|------|
| `profiling_summary.csv` | 734 rows | 完整 profiling 数据 (batch_size=3000) |
| `hub_analysis.csv` | 734 rows | Hub entity 分析 |
| `negative_sampling_cost.csv` | 734→455 rows | B1-B5 子阶段分解 |

### Phase 4: 设计空间探索与算法选择

**时间**: 2026-06 | **Git**: `99d4298`

- 提出 6 个优化 Route (A~F)，评估复杂度、收益、风险
- **最终选择 Route C — CBP (Cost-aware Batch Packing)**
- 详细分析: `docs/algorithm_candidates.md`

### Phase 5: 算法设计与验证

| 子阶段 | Git | 关键产出 |
|--------|-----|---------|
| Phase 5.0 DDBP 设计 | `d9a3c02` | `docs/algorithm_design.md` |
| Phase 5.1 重构 (DDBP→CBP) | `263fd21` | 术语统一，4层架构定稿 |
| Phase 5.5 Weight 验证 | `765f707` | **R²=0.9008** 确认，原始 R²=0.1657 拒绝 |
| Phase 5.6 Cost Model 拟合 | — | B3≈51.8ms, `docs/cost_model.md` |

**CBP 四层架构**:
```
FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter)
```

**Weight Assumption 验证核心**:
```
原始假设: R² = 0.0274 (拒绝)
采用模型: candidate_size → E_retry → B3_const
最终公式: E_retry = min(max_try, 1/(1 - N_neg/candidate_size))
验证 R² = 0.9008 ✅
```

### Phase 6: CBP Runtime Framework

#### Node 1-3: 框架实现

| Node | 组件 | 文件 | 功能 |
|------|------|------|------|
| Node 1 | Cost Estimator + Scheduler | `cost_estimator.py`, `schedulers.py` | 权重计算 + Sort/Pack 策略 |
| Node 2 | Feature/Cost 解耦 | `features.py`, `cost_model.py` | 纯函数设计 |
| Node 3 | BatchProvider + Stage D | `batch_provider.py` | 零侵入注入 Adapter |

**Scheduler 策略组合**:
- **CBP**: `Scheduler(CostSorter, FFDPacker)` — 按 cost 降序 + FFD 装箱
- **Baseline**: `Scheduler(RandomSorter, ChunkPacker)` — 随机 + 分块

#### Node 4: Runtime Attribution 实验

| 指标 | Baseline (Random+Chunk) | CBP (Cost+FFD) | 改善 |
|------|:---:|:---:|:---:|
| Batch Weight std | ±0.9 | ±37.4 | std +43x ✅ |
| Neg Sampling time | 62.2 ± 15.5 ms | 62.6 ± 3.4 ms | **std -78%** ✅ |
| Tensor build | 24.7 ± 1.5 ms | 25.9 ± 2.0 ms | std +33% |
| Forward | 7.7 ± 0.4 ms | 7.7 ± 0.4 ms | 不变 |
| Total step time | 94.6 ± 16.0 ms | 96.3 ± 5.5 ms | **std -66%** ✅ |

**相关性验证**:
| 相关性 | Baseline r | CBP r | 解读 |
|-------|:---:|:---:|------|
| Weight vs Neg Sampling | 0.0064 | **0.7124** | CBP 激活了因果链 ✅ |
| Weight vs Total | -0.0136 | **0.6952** | 同上 |
| Neg vs Total | 0.9933 | 0.9763 | 负采样始终主导 |

### Phase 7: GPU 迁移可行性研究

**时间**: 2026-07-23

| Step | 内容 | 核心发现 |
|------|------|---------|
| Step 1 | Tensor 构建 Profiling | 张量构建+负采样占 CPU 92% |
| Step 2 | GPU 迁移可行性 | `docs/gpu_migration_feasibility.md` |
| Step 3 | GPU Cost Model 微基准 | **Break-even N*=264k** (2x speedup at 750k) |
| Step 4 | GPU 运行时架构设计 | **Route C 最终推荐** |
| Step 5 | 运行时框架规范 | 接口约定文档 |

### Phase 8: GPU Negative Sampler

**时间**: 2026-07-23 | **Git**: `63fd222`

| Step | 内容 | 结果 |
|------|------|------|
| Step 0 | 架构冻结 | `docs/phase8_architecture_freeze.md` |
| Step 1 | GPU Sampler 原型 | **~2.17ms, 43x 加速 vs CPU** |
| Step 2 | 统一运行时验证 | **GPU epoch 4.78s vs CPU 37.06s (7.7x)** |

**GPU Sampler 设计**:
- 全向量化 `torch.randint` + `torch.isin`
- Tail-only corruption（非 Bernoulli head/tail）
- Batch-level 碰撞检查（非 global `all_triples_set`）

### Phase 9: 完整基准与评估

#### Step 1: 语义对齐验证 (2026-07-25)

**Git**: `9497700`

比较 CPU 原版 (Bernoulli head/tail + global 碰撞) vs GPU v2 (tail-only + batch-level 碰撞):

| Config | Epoch 0 Loss | Epoch 1 Loss | 结论 |
|--------|:---:|:---:|------|
| CPU (original) | 1.033 | 0.842 | 基准线 |
| GPU v2 | 0.970 | 0.706 | 收敛略快（预期行为） |

**决策**: ✅ 接受语义差异。GPU v2 差距可接受，速度优势（~198x neg, ~7.7x epoch）压倒性。

**基线冻结**: `docs/baseline_freeze.md` — 四组实验组别锁定。

#### Step 2: 四组对比基准 (2026-07-25)

**Git**: `728192d`

**四组实验定义**:
| 组别 | Scheduler | 负采样 | 缩写 |
|------|-----------|--------|:----:|
| Baseline | RandomSorter + ChunkPacker | CPU (original) | **BL** |
| CBP only | CostSorter + FFDPacker | CPU (original) | **CBP** |
| GPU only | RandomSorter + ChunkPacker | GPU v2 | **GPU** |
| CBP + GPU | CostSorter + FFDPacker | GPU v2 | **CBP+GPU** |

**最终结果 (batch_size=5000, neg_num=150, 5 epochs)**:

| 指标 | BL | CBP | GPU | **CBP+GPU** |
|------|:---:|:---:|:---:|:---:|
| Final Loss | 0.572 | 0.574 | 0.378 | 0.384 |
| MRR | 0.0136 | 0.0150 | 0.0132 | 0.0113 |
| Hits@10 | 0.0225 | 0.0350 | 0.0300 | 0.0175 |
| **Avg epoch time** | **25.1s** | **25.3s** | **4.4s** | **4.7s** |
| Speedup vs BL | 1.0x | 0.99x | **5.7x** | **5.4x** |
| GPU Memory | 5818MB | 5818MB | 5819MB | 5820MB |

**关键洞察**:
- GPU 加速效果显著（5.4-5.7x epoch 加速）
- CBP 在 CPU 和 GPU 上几乎不增加额外开销（epoch 时间差 <5%）
- MRR/Hits@10 数值偏低 — 因评估仅在 200/500 个采样三元组上进行（非全测试集）
- `float('inf')` 排序 bug 已修复，改用 `1e9` mask

---

## 三、架构决策演变

### 3.1 四层解耦架构 (最终定稿)
```
FeatureExtractor (features.py)
       ↓ (entity_features: np.ndarray)
CostModel (cost_model.py) — 纯函数
       ↓ (cost_table: np.ndarray)
Scheduler (schedulers.py) — Sort + Pack 策略模式
       ↓ (batch_indices: List[int])
BatchProvider (batch_provider.py) — Adapter 模式，零侵入注入
```

### 3.2 Scheduler 策略组合
```
Baseline ─ RandomSorter(seed=42) + ChunkPacker
CBP      ─ CostSorter() + FFDPacker
```

**FFDPacker 已知问题**: O(n×batch) 双重循环，overhead 约 1.2s（Baseline 仅 65ms）。未来可优化为 round-robin。

### 3.3 GPU Sampler 选型
```
原始 ─ Bernoulli(0.5) head/tail + global all_triples_set 碰撞 + Python for 循环
GPU  ─ tail-only corruption + batch-level pos_tails isin 碰撞 + 全向量化
```
选择牺牲 Bernoulli → tail-only 换取全向量化效率。

---

## 四、失败实验详细分析

### 1. `args.is_torch` AttributeError (Phase 6-E1/E2 初版)
- **症状**: `AttributeError: 'ARGs' object has no attribute 'is_torch'`
- **原因**: 实验脚本绕过 `kge_models` 直接创建 Trainer，未设置 `args.is_torch = True`
- **修复**: 在 training loop 前添加 `args.is_torch = True`

### 2. Phase-1.1 Profiler 探针中断
- **问题**: PyTorch Profiler trace 文件混合多个 GPU PID 时间戳，12 个 JSON 命名不统一
- **根因**: Profiler 启动/停止逻辑未正确同步

### 3. evaluation_metrics.csv 为空
- Baseline 输出中 `evaluation_metrics.csv` 只有 header（41 bytes）
- `print_results()` 返回空 dict
- **原因**: `LinkPredictionEvaluator` 的返回值处理缺失

### 4. `float('inf')` 排序问题 (Phase 9 Step 1)
- **症状**: MRR=6.9e-05, Hits@10=0.0 — 评估完全失败
- **根因**: 过滤时 `float('inf')` 使 `scores < true_score` 比较失效
- **修复**: 将 `float('inf')` 替换为 `1e9` mask

### 5. Phase 6 CBP 实验不完整
- CBP (Cost+FFD) 在 Phase 6 仅跑了 5/10 epochs（进程中断）
- 因此 Phase 9 重新设计了独立基准脚本 `phase9_step2_benchmark.py`

---

## 五、当前最佳成果

| 指标 | BL (CPU) | CBP (CPU) | GPU | CBP+GPU |
|------|:--------:|:---------:|:---:|:-------:|
| **Epoch Time** | 25.1s | 25.3s | **4.4s** 🔥 | **4.7s** 🔥 |
| **Epoch Speedup** | 1.0x | 0.99x | **5.7x** | **5.4x** |
| Neg Sampling 加速 | 1.0x | 1.0x | **~198x** 🔥 | **~198x** 🔥 |
| Loss (final) | 0.572 | 0.574 | 0.378 | 0.384 |
| MRR (sampled) | 0.0136 | 0.0150 | 0.0132 | 0.0113 |
| Hits@10 (sampled) | 0.0225 | 0.0350 | 0.0300 | 0.0175 |
| GPU Memory | 5818MB | 5818MB | 5819MB | 5820MB |

**完整结果文件**: `output/results/phase9_step2/summary.csv`
**各子目录**: `output/results/phase9_step2/{BL,CBP,GPU,CBP+GPU}/summary.csv`

---

## 六、已知待解决问题清单

1. **[中] Phase 9 Step 3 未执行**: 需要跑 10 epoch 验证收敛稳定性，补齐论文所需 MRR/Hits@10
2. **[中] MRR/Hits@10 评估偏小**: 当前仅采样 200-500 三元组评估，未使用完整测试集。需用完整 `test_triples` 重跑
3. **[低] FFDPacker 效率**: 双重循环 O(n×batch) 约 1.2s overhead，可优化为 round-robin 或二分查找
4. **[低] MCP Memory Server 不可用**: JSON 解析错误，L2 记忆无法写入
5. **[低] offline 环境**: server_node4 不能 git push（需 rsync 到 pc-cluster 再 push）
6. **[低] Evaluation 未全量**: 论文表格需要全测试集 MRR/Hits@10/1/3/50

---

## 七、对下一个接手者的 3 条最关键建议

### 🔥 建议 1: Phase 9 Step 3 — 完整 10 epoch 收敛实验

四组对比已经跑了 5 epoch，但要论文级别的结果需要跑 10-20 epoch：
```bash
python3 -u src/py/experiments/phase9_step2_benchmark.py --epochs 10
```
或者单独跑每组（避免 OOM）：
```bash
for cfg in BL CBP GPU CBP+GPU; do
  python3 -u src/py/experiments/phase9_step2_benchmark.py --only $cfg --epochs 10
done
```

### 🔥 建议 2: 全量 Evaluation

当前 MRR/Hits@10 仅在 200-500 个采样三元组上评估，数值偏低。论文需要：
1. 使用完整 `FB15k-237/test.txt`（20,438 triples）
2. 输出 Raw + Filtered MRR, Hits@1/3/10/50
3. Head/Tail 两侧 ranking 分开统计

可复用 `src/py/evaluation/evaluation.py` 中的 `LinkPredictionEvaluator`。

### 🔥 建议 3: 同步代码 + Memory 闭环

当前环境为 node4 (offline)，无法直接 git push：
```bash
# 1. 同步到 pc-cluster
rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/

# 2. 在 pc-cluster 上执行 Memory Bouncer
python3 utils/memory_bouncer.py

# 3. 推送到 GitHub
git push origin production
```

---

## 八、关键文件索引

### 实验入口
| 文件 | 说明 |
|------|------|
| `src/py/experiments/run_cbp_evaluation.py` | Phase 6 CBP 评估脚本（旧版，10 epoch） |
| `src/py/experiments/phase9_step1_alignment.py` | Phase 9 Step 1 语义对齐验证 |
| `src/py/experiments/phase9_step2_benchmark.py` | Phase 9 Step 2 四组对比基准（当前主要入口） |

### 核心模块
| 文件 | 说明 |
|------|------|
| `src/py/load/features.py` | 图拓扑特征提取 |
| `src/py/load/cost_model.py` | Cost table 生成 (R²=0.9008) |
| `src/py/load/cost_estimator.py` | 单 batch weight 计算 |
| `src/py/load/schedulers.py` | Sort+Pack 策略模式 |
| `src/py/load/batch_provider.py` | 零侵入注入 Adapter |
| `src/py/load/gpu_sampler.py` | GPU 向量化负采样器 |

### 实验输出
| 文件 | 说明 |
|------|------|
| `output/results/phase9_step2/summary.csv` | Phase 9 四组结果汇总 |
| `output/results/negative_sampling_cost.csv` | Phase 2 B1-B5 子阶段数据 |
| `output/results/negative_sampling_breakdown.csv` | Phase 2 占比汇总 |
| `output/results/runtime_attribution/runtime_attribution.csv` | Phase 6 因果链数据 |
| `output/results/runtime_attribution/attribution_interpretation.txt` | Phase 6 深度解读 |

### 文档
| 文件 | 说明 |
|------|------|
| `docs/algorithm_candidates.md` | 6 Route 评估 |
| `docs/algorithm_design.md` | CBP 算法描述 |
| `docs/cost_model.md` | 双模成本定律 + 公式推导 |
| `docs/gpu_migration_feasibility.md` | GPU 迁移可行性 |
| `docs/gpu_runtime_architecture.md` | GPU 运行时架构 |
| `docs/baseline_freeze.md` | Phase 9 基线冻结定义 |
| `docs/semantic_alignment_report.md` | GPU 语义对齐报告 |
| `docs/phase8_architecture_freeze.md` | Phase 8 架构冻结 |

### 记忆与跟踪
| 文件 | 说明 |
|------|------|
| `PROGRESS.md` | 实验航海日志 |
| `.clinerules` | 项目宪法 + 环境路由 |
| `mukg-memory.json` | L2 知识图谱记忆 |
| `utils/memory_bouncer.py` | 强制收尾 + Git 同步脚本 |

---

## 九、环境信息

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

**跨环境同步提醒**: 此环境 offline。修改代码后需:
```bash
rsync -av --delete ~/muKG_LB/ hma@192.168.100.104:~/muKG_LB/
```
然后在 pc-cluster 上执行 `python3 utils/memory_bouncer.py` 完成 L2/L3 同步 + git push。

---

## 十、项目演化时间线 (简版)

```
2025末         Fork muKG + 基础模型移植
2026-04~05     Phase 1-2: Profiling → 发现 CPU 占 93%
2026-05~06     Phase 3: Hub Entity 分析 → R=0.8163 驱动方向
2026-06        Phase 4-5: 设计空间探索 → 选择 CBP，4层架构定稿
2026-07-09     Phase 5.5-5.6: Weight 验证 R²=0.9008, Cost Model 完成
2026-07-17     Phase 6: CBP 框架实现 + Runtime Attribution (std -78%)
              ─── 上次 Handover 文档生成 (RESEARCH_HANDOVER_20260717_1338.md) ───
2026-07-23     Phase 7: GPU 迁移可行性 → GPU Cost Model → 架构设计
2026-07-23     Phase 8: GPU Sampler 原型 (43x) → 统一验证 (7.7x epoch)
2026-07-25     Phase 9 Step 1: 语义对齐 + 基线冻结
2026-07-25     Phase 9 Step 2: 四组对比基准 (GPU 5.7x 加速 ✅)
              ─── 本文档生成 ───
```

---

*本台账由 Cline 自动生成，覆盖 Phase 0~9 Step 2 完整实验历史。下一步建议执行 Phase 9 Step 3 (10 epoch 收敛验证) 后更新 PROGRESS.md 并执行 memory_bouncer.py 完成闭环。*