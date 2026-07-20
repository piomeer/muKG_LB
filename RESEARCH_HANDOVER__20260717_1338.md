# 研究进度与实验台账 (Research Handover)
**生成时间**: 2026-07-17 13:38 JST  
**当前环境**: `server_node4` (RTX3070, 8GB VRAM, offline)  
**项目**: MuKG (Multi-source Knowledge Graph Embedding) + CBP (Cost-aware Batch Packing)  
**Git HEAD**: `d84f72c` (Phase 6 - Node 3)

---

## 一、实验台账总表

| 实验ID | 名称 | 状态 | 数据集 | 模型 | 命令/脚本 | 核心结果 | 备注 |
|--------|------|------|--------|------|-----------|----------|------|
| **Phase-6-E1** | Baseline (Random+Chunk) | ✅ **成功** | FB15k-237 (272k triples) | TransE (dim=400) | `run_cbp_evaluation.py --sorter Random --packer Chunk --epochs 10` | Mean step=280.8ms, Epoch=80.2s, CV=0.05395, Filt.MRR=0.194, Filt.Hits@10=0.322 | 完整10 epoch，输出CSV齐全 |
| **Phase-6-E2** | CBP (Cost+FFD) | 🔄 **运行中** | FB15k-237 | TransE (dim=400) | `run_cbp_evaluation.py --sorter Cost --packer FFD --epochs 10` | Epoch 5/10 done, mean step ~293ms, CV~0.054, Filt.MRR=0.194 (与Baseline持平) | Scheduler overhead=1.165s (Baseline的18x) |
| **Phase-5.5** | Weight Assumption Validation | ✅ **成功** | FB15k-237 | 统计分析 | `validate_weight_assumption.py` | 原始假设R=0.1657被拒绝；candidate_size模型R=0.9008，公式修正 | 论文核心卖点验证 |
| **Phase-5.4** | Cost Model Fitting | ✅ **成功** | FB15k-237 | 线性回归 | `fit_cost_model.py` | cost_model公式验证，B3≈51.8ms | 存档于`docs/cost_model.md` |
| **Phase-4** | Hub Reuse & Cache | ✅ **成功** | FB15k-237 | 统计分析 | `analyze_hub_reuse.py` | 图分析完成，hub entity分布SVG | 存档于`figs/` |
| **Phase-3.1** | 第一轮Profiling | ✅ **成功** | FB15k-237 | TransE | `profile_transe.py` | 基础Profiling数据 | 日志: `profiling_summary.csv` (6月12日) |
| **Phase-3.2** | 第二轮Profiling | ✅ **成功** | FB15k-237 | TransE | `analyze_neg_sampling.py` | Negative Sampling瓶颈分析 | 日志: `negative_sampling_cost.csv` |
| **Phase-3.3** | 第三轮Profiling | ✅ **成功** | FB15k-237 | TransE | Hub Entity分析 | Hub entity对采样影响 | 日志: `hub_analysis.csv` |
| **Phase-6-E1-v1** | Baseline (初版, 有Bug) | ❌ **失败** | FB15k-237 | TransE | `run_cbp_evaluation.py` (初版) | 成功训练6 epoch后Validation崩溃 | Bug: `args.is_torch` 未设置 |
| **Phase-6-E2** (初版) | CBP (初版, 有Bug) | ❌ **失败** | FB15k-237 | TransE | `run_cbp_evaluation.py` (初版) | 成功训练4 epoch后Validation崩溃 | Bug: `args.is_torch` 未设置 |

---

## 二、各阶段详细说明

### Phase 6 - Node 4: CBP Evaluation (当前活动)

#### Exp-1: Baseline (RandomSorter + ChunkPacker) ✅ 已完成

| 指标 | 值 |
|------|-----|
| 启动命令 | `python3 -u run_cbp_evaluation.py --sorter Random --packer Chunk --epochs 10 --exp-label Baseline` |
| 训练完成 | 10/10 epochs |
| 总耗时 | 801.99 秒 (~13.4 min) |
| Mean Step Time | 280.80 ms |
| Mean Step CV | 0.05395 (稳定，范围0.05322~0.05617) |
| Mean Epoch Time | 80.20 s |
| 训练 Loss 衰减 | 52.55 (epoch0) → 0.94 (epoch9) |
| Filt. MRR | 0.194 |
| Filt. Hits@10 | 0.322 |
| 阶段耗时 | 91.2%在Negative Sampling, 1.8% Forward, 2.1% Backward, 0.4% Optimizer |
| Scheduler Overhead | 64.8 ms |
| 输出文件 | `epoch_summary.csv`, `profiling_summary.csv` (2730 rows), `batch_runtime_variance.csv` (2730 rows) |

#### Exp-2: CBP (CostSorter + FFDPacker) 🔄 运行中 (5/10 epochs)

| 指标 | 值 (截至epoch 4平均) |
|------|---------------------|
| Mean Step Time | ~293 ms (比Baseline慢~4.3%) |
| Mean Step CV | ~0.054 (与Baseline持平) |
| Mean Epoch Time | ~84.4 s (比Baseline慢~5.2%) |
| Scheduler Overhead | 1165 ms (Baseline的18x，FFD packer开销) |
| Filt. MRR | 0.194 (与Baseline一致) |
| Filt. Hits@10 | 0.32 (与Baseline一致) |
| 当前进度 | epoch 5/10，validation已通过 |

### Phase 5.5: Weight Assumption Validation ✅

| 指标 | 值 |
|------|-----|
| 原始假设R² | 0.1657 (被拒绝) |
| candidate_size模型R² | **0.9008** (采用) |
| 最终公式 | `E_retry * B3_const`, 其中E_retry = min(max_try, 1/(1 - N_neg/candidate_size)) |
| 核心文件 | `scripts/validate_weight_assumption.py`, `docs/cost_model.md` |

---

## 三、失败实验详细分析

### 1. `args.is_torch` AttributeError (Phase 6-E1/E2 初版)

**错误信息**:
```
File "evaluation.py", line 245, in fomulate
    if self.args.is_torch:
AttributeError: 'ARGs' object has no attribute 'is_torch'
```

**原因**: `kge_models.get_model()` 内部会设置 `args.is_torch = True`，但实验脚本 `run_cbp_evaluation.py` 绕过了 `kge_models` 直接创建 Trainer。Validation 时 `fomulate()` 需要 `args.is_torch` 来确定用 PyTorch 还是 TensorFlow。

**修复**: 在 `run_cbp_evaluation.py` 中显式添加 `args.is_torch = True`。

**经验教训**: 任何绕过 `kge_models` 直接使用底层 Trainer 的脚本，都必须手动设置 `args.is_torch`。

---

## 四、当前最佳成果

| 指标 | Baseline (Random+Chunk) | CBP (Cost+FFD, 5/10) |
|------|------------------------|----------------------|
| Filt. MRR | **0.194** | **0.194** (持平) |
| Filt. Hits@10 | **0.322** | **0.32** (持平) |
| Filt. Hits@1 | **0.131** | **0.132** (持平) |
| Mean Step Time | **280.8 ms** | ~293 ms (慢4.3%) |
| Mean Step CV | **0.05395** | ~0.054 (持平) |
| Mean Epoch Time | **80.2 s** | ~84.4 s (慢5.2%) |
| Scheduler Overhead | **64.8 ms** | **1165 ms** (慢18x) |

**注意**: CBP 目前只完成 5/10 epochs。最终评估需等待 CBP 完整跑完。

**模型权重路径**: `output/results/TransE/FB15K237//torch/` (由 `model.save()` 生成)

---

## 五、已知待解决问题清单

1. **[严重] CBP Scheduler 开销过高**: FFD Packer 的 First-Fit 算法为 O(n×batch) 复杂度，导致 1.165s overhead（Baseline 仅 65ms）。需优化 FFD 实现（如改用 NumPy 向量化）。
2. **[中等] Validation 时 GPU 显存不足风险**: 基于 Filt. MRR 的 Validation 需要全实体评分，batch_size=1000×101(neg+pos) 在 8GB VRAM 下已经接近极限。当前已在 validation 前 `torch.cuda.empty_cache()` 缓解。
3. **[中等] CBP 论文核心卖点仍需验证**: CV（变异系数）在 Baseline 和 CBP 中几乎没有差异（~0.054 vs ~0.054），这可能是因为 Uniform Negative Sampling 主导了 Step Time（占91%），淹没了 CBP 对 GPU Forward 的优化效果。
4. **[低] 多环境 Memory 同步**: 根据 `env_identity.json`，此环境为 offline，无法 push GitHub。需要手动 rsync 回 pc-cluster 后运行 `memory_bouncer.py`。
5. **[低] evaluation_metrics.csv 为空**: Baseline 输出中 `evaluation_metrics.csv` 只有 header（41 bytes），说明 `print_results()` 返回了空 dict。需要检查 `LinkPredictionEvaluator` 返回值。

---

## 六、对下一个接手者的 3 条最关键建议

### 🔥 建议 1: 先等待 CBP 实验完成再评估

PID `173881` 正在运行 CBP (`Cost+FFD`)，已完成 5/10 epochs。用以下命令监控：
```bash
tail -f output/results/exp_CBP/training.log
```
完成后会输出 `epoch_summary.csv` 等文件。拿到完整数据后再做最终对比。

如果进程中断，用 nohup 重新启动：
```bash
nohup python3 -u src/py/experiments/run_cbp_evaluation.py \
  --sorter Cost --packer FFD --epochs 10 --exp-label CBP \
  > output/results/exp_CBP/training.log 2>&1 &
```

### 🔥 建议 2: 优先优化 FFD Packer 实现

CBP 的 `scheduler_overhead` 高达 1.165s（Baseline 仅 65ms），差了 18 倍。这是 FFD 算法的 `for triple in ordered_triples: for b_idx in range(n_batches)` 双重循环导致的 `O(n×batch)` 复杂度。

**修复方案**: 在 `src/py/load/schedulers.py` 的 `FFDPacker.pack()` 中：
- 不逐个放置 triple，而是用 `round-robin` 分配（近似 FFD），复杂度降到 `O(n)`
- 或：用 NumPy 向量化 batch 分配

优化后 CBP 的 epoch 时间可能接近 Baseline 的 80s（当前 84s）。

### 🔥 建议 3: 关注 CV 不降的问题

论文核心假设是 CBP 能降低 Batch Runtime Variance (CV)，但目前 Baseline 和 CBP 的 CV 几乎一致（~0.054）。原因分析：
- 91% 的 step 时间花在 CPU 端的 **Uniform Negative Sampling** 上
- GPU Forward/Backward 仅占 ~4%
- CBP 只影响 GPU 阶段的数据分布，对 91% 的 CPU 阶段没有影响

**后续方向**:
- 如果要证明 CV reduction，需要用 GPU-based negative sampling 重新实验（移除 CPU 瓶颈）
- 或者：在论文中说明 CBP 在**不损害模型精度**的同时，提供了更稳定的 GPU 计算负载（哪怕 CPU 阶段占主导）
- 考虑：用更大的 batch_size 或更复杂的模型（如 ConvE、RotatE）来凸显 GPU 阶段的方差

---

## 七、关键文件索引

| 类别 | 文件 | 说明 |
|------|------|------|
| 实验入口 | `src/py/experiments/run_cbp_evaluation.py` | Phase 6 评估脚本，支持 Random+Chunk / Cost+FFD |
| 批量调度 | `scripts/run_cbp_benchmark.sh` | 串行 Baseline→CBP 的 shell 脚本 |
| Baseline 日志 | `output/results/exp_Baseline/training.log` | 10 epoch 完整训练日志 |
| CBP 日志 | `output/results/exp_CBP/training.log` | 运行中，当前 epoch 5/10 |
| 架构设计 | `src/py/load/schedulers.py` | Sort+Pack 策略模式实现 |
| Adapter | `src/py/load/batch_provider.py` | BatchProvider 零侵入注入 |
| Cost Model | `src/py/load/cost_model.py` | Cost table 构建（R²=0.9008） |
| 特征提取 | `src/py/load/features.py` | 图拓扑特征提取 |
| 算法设计文档 | `docs/algorithm_design.md` | 算法选择过程 |
| 实验航海日志 | `PROGRESS.md` | 项目状态跟踪 |
| 项目宪法 | `.clinerules` | 环境路由、Memory Bouncer 等约束 |
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

*本台账由 Cline 自动生成。CBP 实验完成后，请更新 PROGRESS.md 并执行 memory_bouncer.py 完成闭环。*