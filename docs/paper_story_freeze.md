# Paper Story Freeze

**Date**: 2026-07-30  
**Status**: Final (based on all completed experiments through Phase 10)

## 论文标题建议

**"A Cost-aware Runtime Framework for Efficient Knowledge Graph Embedding Training"**

> 备选：*"MuKG: A Unified Runtime for Cost-aware KGE Training with GPU-accelerated Negative Sampling"*

## 整体叙事线（Narrative Arc）

当前知识图谱嵌入（KGE）训练的瓶颈不在模型复杂度，而在**数据加载与负采样**这一系统层面。我们在 FB15k-237 上的深度 Profiling 发现：CPU 负采样占据了训练步骤 65% 的时间（Phase 6），且方差极大（std=28.5ms）。为系统性地理解这一瓶颈，我们首先构建了一个**离线运行时成本模型**（R²=0.9008），将每个实体的拓扑特征（candidate_size）映射为预期的负采样成本。基于此模型，我们设计了 **Cost-aware Batch Packing (CBP)** 调度器，在 CPU 路径上将批次权重方差降低了 78%（batch_size=1000 场景）。然而进一步的消融实验（Phase 9 Step 3/4.5）表明，当回到标准的全训练循环（batch_size=5000）时，CBP 的边际收益被 Python GIL、Tensor 构建等系统噪声淹没。这推动我们做出关键决策：**将负采样迁移到 GPU 上执行**。我们提出了一套**统一运行时框架（Unified Runtime Framework）**，通过 CostModel → Scheduler → BatchProvider → GPUNegativeSampler 四层解耦，实现了 GPU 全向量化负采样。最终实验证明：GPU Runtime 将负采样时间从 596ms 压缩到 3.0ms（198×），epoch 加速 5.7×，且**将负采样标准差从 28.5ms 压到 0.2ms（142×）**——从根本上消除了 KGE 训练中的最大系统瓶颈与方差来源。整个叙事线从"发现瓶颈→建模成本→探索 CPU 调度→迁移 GPU→验证收敛→消融分析"构成一条完整的系统研究链。

---

## Q1: 论文真正解决了什么问题？

知识图谱嵌入训练的核心系统瓶颈在于**大规模负采样过程**——对每个正例三元组生成 N 个伪造负例需要遍历全局三元组集合进行碰撞检测，该过程在 CPU 上占据了训练步骤 65% 的时间（Phase 6），且因 Python for 循环的随机性导致极高方差（std=28.5ms）。现有框架（如 OpenKE、PyKEEN）将负采样视为黑盒数据预处理步骤，并未将其纳入运行时调度与硬件分配的决策中。我们的工作解决了这一问题：**通过构建成本感知的统一运行时框架，将负采样从不可预测的 CPU 瓶颈转变为 GPU 上可调度、可预测的低方差计算模块**，实现了端到端 5.7× 加速且方差压缩 142×。

---

## Q2: 为什么现有方法解决不了？

| 现有方法 | 局限性 | 我们的差异 |
|---------|--------|-----------|
| **纯 CPU 优化**（如多进程 DataLoader、NumPy 向量化） | Python GIL 限制并行度；无法从根本上消除 for 循环随机性 | 将负采样**完全迁移到 GPU**，利用 CUDA 大规模并行消除随机性 |
| **纯 GPU 计算**（如 GPU 训练 + CPU 数据加载） | 负采样仍在 CPU 侧，GPU 等待数据成为新瓶颈 | 提出 **Unified Runtime Framework**，将 CostModel/Scheduler/BatchProvider/GPUNegativeSampler 统一在单一 GPU 运行时中 |
| **静态批处理**（固定 batch_size + 随机 shuffle） | 无法感知每个实体的真实负采样成本（candidate_size 差异可达 100×） | 构建 **Offline Runtime Cost Model**（R²=0.9008），使调度器能预测每个 batch 的预期成本 |
| **通用 DL 框架**（PyTorch DataLoader） | 不支持代价感知调度，不能将"负采样成本"作为排序/打包依据 | 通过 **Cost-aware Batch Packing (CBP)** 提供可插拔的 Sort+Pack 策略组合 |

**关键差异总结**：我们没有单独优化任何一个环节，而是通过 **Cost Model 这一统一抽象**，将"负采样成本"从隐性经验概念转化为显性可计算指标，从而实现了调度层（CBP）与执行层（GPU Sampler）的协同设计。

---

## Q3: 核心贡献（按实验证据强度排序）

### 贡献 1：GPU Runtime — 消除 CPU 负采样瓶颈（端到端加速 5~8×，方差压缩 142×）【最强】

GPU 全向量化负采样器将负采样时间从 596ms 压缩到 3.0ms（198×），epoch 时间从 37s 降到 4.78s（7.7×），并将负采样标准差从 28.5ms 压到 0.2ms（142×）。这是论文的**核心实验贡献**，也是叙事的高潮部分。在论文中位于 Method（GPUNegativeSampler 设计）→ Experiments（Phase 8/9 验证）部分。

### 贡献 2：Unified Runtime Framework — 代价感知调度与执行的统一架构【架构型贡献】

定义了 FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦的运行时架构，并通过 GPUNegativeSampler 将 GPU 执行无缝嵌入。该架构使调度器可感知每个实体的负采样成本，并在 CPU/GPU 两种执行路径上透明切换。在论文中位于 Method（Framework Design）部分，是所有实验的基础。

### 贡献 3：Offline Runtime Cost Model — 从拓扑特征到预期成本的显式映射（R²=0.9008）【理论贡献】

将每个实体的 `candidate_size`（邻居池大小）映射为预期负采样成本 `E[retry] × B3_const`。该模型实现了 R²=0.9008 的预测精度（Phase 5.5/Phase 7 Step 3），使调度器能够在**不执行任何负采样的前提下**估计 batch 成本。在论文中位于 Method（Cost Model）部分，是贡献 2 的前提。

### 贡献 4：Cost-aware Batch Packing (CBP) — CPU 调度探索与框架验证【探索性贡献】

在 CPU 路径上验证了代价感知调度的可行性（Phase 6 在 batch_size=1000 下将 neg_std 降低 78%），并为框架提供了可插拔的 Sort+Pack 策略接口（RandomSorter/CostSorter + ChunkPacker/FFDPacker）。虽然在全训练循环（batch_size=5000）中边际收益被系统噪声稀释（Phase 9 Step 4.5: std 仅降低 8.4%），但 CBP 的实验过程是推动 GPU 迁移的关键动机。在论文中位于 Method（Scheduler）和 Experiments（Ablation）部分，作为从 CPU 到 GPU 叙事转折的论证基础。

---

## Q4: 每个贡献的实验支撑

### 贡献 1 — GPU Runtime
| Phase | 关键指标 | 数据来源 |
|-------|---------|---------|
| Phase 8 Step 1 | GPU Sampler 原型 2.17ms/call，43× 加速 vs CPU | `src/py/load/gpu_sampler.py`; `output/results/gpu_sampler/validation.md` |
| Phase 8 Step 2 | neg_time 596→3.0ms (198×), epoch 37→4.78s (7.7×) | `output/results/unified_runtime/epoch_summary_GPU.md` |
| Phase 9 Step 2 (5 epochs) | GPU epoch 4.4s vs CPU 25.1s (5.7×) | `output/results/phase9_step2/GPU/summary.md`; `output/results/phase9_step2/summary.md` |
| Phase 9 Step 3 (10 epochs) | GPU neg_std 0.2ms vs CPU 28.5ms (142× variance reduction) | `output/results/phase9_step3/GPU/summary.csv`; `paper_assets/figures/fig6_ablation_variance.pdf` |

### 贡献 2 — Unified Runtime Framework
| Phase | 关键指标 | 数据来源 |
|-------|---------|---------|
| Phase 7 Step 4-5 | Route C 架构（CBP + GPU Runtime）推荐 | `docs/gpu_runtime_architecture.md`; `docs/runtime_framework_spec.md` |
| Phase 8 Step 0 | 模块复用表：CostModel/Scheduler/BatchProvider 复用，GPUNegativeSampler 新建 | `docs/phase8_architecture_freeze.md` |
| Phase 8 Step 2 | 统一运行时验证：CBP + GPU Sampler 协同工作 | `output/results/unified_runtime/unified_runtime_validation.md` |
| Phase 9 Step 2 | 四组配置（BL/CBP/GPU/CBP+GPU）全组合验证 | `output/results/phase9_step2/summary.md` |

### 贡献 3 — Offline Runtime Cost Model
| Phase | 关键指标 | 数据来源 |
|-------|---------|---------|
| Phase 5.5 / Phase 6 | Cost Model 公式验证，R²=0.9008 | `scripts/fit_cost_model.py`; `docs/cost_model.md` |
| Phase 7 Step 3 | GPU vs CPU cost model microbench，break-even N*=264k | `output/results/gpu_cost_model/benchmark.md`; `src/py/experiments/gpu_cost_microbench.py` |
| Phase 6 (Runtime Attribution) | CBP 下 Weight vs Neg Sampling r=0.71（成本模型与实测相关性验证） | `output/results/runtime_attribution/attribution_interpretation.md` |

### 贡献 4 — CBP (Cost-aware Batch Packing)
| Phase | 关键指标 | 数据来源 |
|-------|---------|---------|
| Phase 6 (Runtime Attribution) | CBP 在 batch_size=1000 下将 neg_std 降低 78%（15.5→3.4ms） | `output/results/runtime_attribution/runtime_attribution.md` |
| Phase 9 Step 3 (10 epochs) | CPU 路径 CBP vs BL neg_std 几乎相同（28.5ms），提示全流程噪声淹没 CBP 效果 | `output/results/phase9_step3/BL/summary.csv`; `output/results/phase9_step3/CBP/summary.csv` |
| Phase 9 Step 4.5 (隔离实验) | 纯 CPU 负采样阶段 CBP vs BL std 仅降 8.4%（29.5→27.0ms） | `output/results/phase9_step4_5/variance_summary.csv` |
| Phase 9 Step 2 | CBP 在 CPU 上 MRR 略高于 BL（0.0150 vs 0.0136），验证调度对精度的正面影响 | `output/results/phase9_step2/CBP/summary.md` |

---

## 实验证据 vs 贡献对照表

| 贡献 | 支撑实验 Phase | 关键指标 | 文件路径 |
|------|---------------|---------|---------|
| **GPU Runtime** | Phase 8 Step 2 | neg_time 596→3.0ms (198×), epoch 37→4.78s (7.7×) | `output/results/unified_runtime/epoch_summary_GPU.md` |
| | Phase 9 Step 2 | GPU epoch 4.4s vs CPU 25.1s (5.7×) | `output/results/phase9_step2/summary.md` |
| | Phase 9 Step 3 | GPU neg_std 0.2ms vs CPU 28.5ms (142× reduction) | `output/results/phase9_step3/GPU/summary.csv` |
| **Unified Runtime Framework** | Phase 7 Step 4-5 | Route C 架构推荐 | `docs/gpu_runtime_architecture.md` |
| | Phase 8 Step 0-2 | 框架冻结 + 统一验证 | `docs/phase8_architecture_freeze.md` |
| | Phase 9 Step 2 | 四组配置全组合验证 | `output/results/phase9_step2/summary.md` |
| **Offline Cost Model** | Phase 5.5 / 6 | R²=0.9008, Weight vs Neg r=0.71 | `docs/cost_model.md` |
| | Phase 7 Step 3 | GPU cost model, break-even N*=264k | `output/results/gpu_cost_model/benchmark.md` |
| **CBP (Cost-aware Packing)** | Phase 6 | neg_std 降 78% (batch_size=1000, 纯调度场景) | `output/results/runtime_attribution/runtime_attribution.md` |
| | Phase 9 Step 3 | CPU 路径 CBP std 与 BL 相同 (全流程噪声) | `output/results/phase9_step3/CBP/summary.csv` |
| | Phase 9 Step 4.5 | 纯 CPU 负采样隔离：CBP std 仅降 8.4% | `output/results/phase9_step4_5/variance_summary.csv` |

---

## 补充说明

1. **CBP 贡献的定位**：CBP 是统一框架的重要组成部分（实现了可插拔的 Sort+Pack 策略接口），但其独立的方差压缩效果在标准训练条件下（batch_size=5000）被系统噪声显著稀释。在论文叙事中，CBP 应定位为"从 CPU 到 GPU 演进的关键中间步骤"，而非独立的亮点贡献。
2. **精度指标说明**：所有 MRR/Hits@10 数值源自 200-sample 过滤评估（训练子集），为相对对比指标，不直接与文献 SOTA 值比较。如需与文献对比，应使用完整测试集重新评估。
3. **Cost Model 的 R²=0.9008**：该值在 Phase 5.5 中通过 `scripts/fit_cost_model.py` 验证，是基于 `candidate_size → expected_cost` 映射的线性回归结果。该值并非在所有配置下复现（Phase 6 的 Weight vs Neg Sampling r=0.71 是运行时刻测量值，与 Cost Model 的离线预测值在概念上有区别）。