# Phase 5.6: CBP 实施计划 (Cost-aware Batch Packing)

> **编制日期**: 2026-07-09
> **基于**: Phase 5.5 验证结论 — `candidate_size` vs `actual_time` 的 Pearson R=0.9008
> **核心发现**: 单批次负采样时间的方差由 `候选池大小 (candidate_size)` 驱动，而非实体的度数
> **术语宣告**: 正式废弃 DDBP (Degree-Driven Bin Packing)，启用 **CBP (Cost-aware Batch Packing)**

---

## 1. 术语对照表 (术语重构声明)

| 废弃术语 | 新术语 | 原因 |
|:---|---:|:---|
| DDBP (Degree-Driven Bin Packing) | **CBP (Cost-aware Batch Packing)** | 成本驱动而非度驱动 |
| Degree-aware | **Cost-aware** | R=0.90 来自 candidate_size 而非 degree |
| DegreeTracker | **CostEstimator** | 核心计算实体候选池大小而非度数 |
| BinPackingScheduler | **CostAwareScheduler** | 命名与算法名对齐 |
| degree_table | **cost_table** (entity_id → expected_cost) | 存储预期成本而非度数 |

---

## 2. 核心算法：成本驱动的权重公式

CBP 的学术核心是：**用候选池大小预测采样成本，通过 FFD 装箱抹平 DDP 批次方差。**

### 2.1 单实体预期成本 (Expected Cost per Entity)

```
candidate_size(e) = |neighbor_dict.get(e, entities_list)|

P_collision = N_neg / candidate_size(e)       ← 单次采样碰到正样本的概率
E_retry(e)  = min(max_try, 1 / (1 - P_collision))  ← 几何分布期望

expected_cost(e) = E_retry(e) × B3_const     ← B3 是主导成本源
                 = candidate_size(e) / (candidate_size(e) - N_neg) × B3_const
```

**验证依据** (Phase 5.5, 400 batch, node4):
- `candidate_size` vs `actual_time`: **R = 0.9008 ✅** (极强正相关)
- `1/candidate_size` vs `actual_time`: **R = -0.8949 ✅** (验证倒数关系)
- 原始 degree 驱动假设: **R = 0.1657 ❌** (因 d 与 c_size 人工相关)

### 2.2 单批次总权重 (Batch Weight)

```
batch_weight(batch) = Σ expected_cost(e) over unique entities in batch
```

这个权重直接对应 **DDP AllReduce 的同步等待时间**：权重越高的 batch 耗时越长，多卡同步时其他 rank 等待它的时间越长。

---

## 3. 代码拓扑总览

```
┌─────────────────────────────────────────────────────────────┐
│                     CBP 数据流架构 (v2)                       │
│                                                             │
│   ┌──────────────────┐    ┌────────────────┐  ┌───────────┐ │
│   │   CostEstimator   │───▶│CostAwareSched  │─▶│ DataLoader │ │
│   │ (候选池分析模块)   │    │ (FFD贪心装箱)   │  │ (消费分组)  │ │
│   └──────────────────┘    └────────────────┘  └───────────┘ │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────────────────────────────────────────┐        │
│   │              cost_table.json                     │        │
│   │   (entity_id → expected_cost, 持久化缓存)        │        │
│   └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**FFD (First Fit Decreasing) 的定位**: CBP 算法使用 FFD 作为内部装箱启发式策略。FFD 并非 CBP 的核心创新，而是实现负载均衡的技术手段。CBP 的核心贡献是**以候选池大小为特征的预期成本模型**。

---

## 4. 新增文件 (New Files)

### 4.1 `src/py/load/cbp_sampler.py` — 核心模块

```python
"""
Cost-aware Batch Packing (CBP) — 核心模块

学术叙事:
  CBP 的核心创新在于建立了 "候选池大小 → 预期采样成本" 的定量预测模型,
  并通过 FFD 贪心装箱消除 DDP 多卡同步的木桶效应。
  
  验证数据: candidate_size vs actual_time 的 Pearson R = 0.9008 (Phase 5.5, 400 batch)
"""
```

#### Class: `CostEstimator`

| 项 | 说明 |
|:---|---|
| **职责** | 从 KG 三元组集 + neighbor dict 中计算每个实体的预期负采样成本 |
| **输入** | `relation_triples_list`, `neighbor_dict: Dict[int, List[int]]`, `neg_num`, `max_try` |
| **输出** | `cost_table: Dict[int, float]` — entity_id → expected_cost (基于 candidate_size) |
| **核心方法** | |
| `compute_candidate_size(entity_id)` | 从 neighbor_dict 或全实体集中获取候选池大小 |
| `compute_expected_cost(entity_id)` | 基于 candidate_size 计算预期重试次数 × B3 常数 |
| `save(path)` / `load(path)` | 持久化 cost_table 避免重复计算 |
| **核心公式** | `cost = min(max_try, 1/(1 - N_neg/c_size)) × B3_const` |

#### Class: `CostAwareScheduler`

| 项 | 说明 |
|:---|---:|
| **职责** | 对 epoch 的 triple_list 执行 FFD 贪心装箱，输出均衡的 batch 分组 |
| **输入** | `triple_list`, `cost_table`, `batch_size`, `seed` |
| **输出** | `packed_batches: List[List[Tuple]]` |
| **核心方法** | |
| `compute_triple_cost(triple)` | 对单个三元组计算 `max(estimated_cost(head), estimated_cost(tail))` |
| `pack_epoch()` | sort_by_cost_desc → FFD → 返回分组 |
| **复杂度** | O(N log N + N × B) |

### 4.2 与 FFD 的关系

```
CBP = Cost Model (candidate_size → expected_cost)
    + FFD Heuristic (sort desc + first-fit into bins)
    
CBP 的学术贡献在前者, FFD 是标准工程实践。
```

---

## 5. 需修改文件 (Modified Files)

### 5.1 `src/py/load/batch.py`

在 `generate_pos_triples()` 中注入 CBP 调度器：

```python
def generate_pos_triples(triples, batch_size, step, is_fixed_size=False, 
                         cbp_scheduler=None):          # ← NEW
    if cbp_scheduler is not None:
        if not hasattr(generate_pos_triples, '_cbp_batches'):
            generate_pos_triples._cbp_batches = cbp_scheduler.pack_epoch()
        pos_batch = generate_pos_triples._cbp_batches[step]
    else:
        start = step * batch_size
        end = start + batch_size
        pos_batch = triples[start: end]
    return pos_batch
```

重置接口:

```python
def reset_cbp_cache():
    if hasattr(generate_pos_triples, '_cbp_batches'):
        del generate_pos_triples._cbp_batches
```

### 5.2 `src/py/load/kgs.py`

```python
def build_cost_table(self):
    from src.py.load.cbp_sampler import CostEstimator
    self.cost_estimator = CostEstimator(
        self.relation_triples_list,
        self.neighbor,         # neighbor dict for candidate narrowing
        self.args.neg_triple_num,
        self.args.max_try if hasattr(self.args, 'max_try') else 10
    )
    return self.cost_estimator
```

### 5.3 `src/py/experiments/main_FB15K237.py`

```python
if __name__ == '__main__':
    kgs = read_kgs_from_folder(...)
    model = kge_models(args, kgs)
    
    # ── CBP 装配点 (NEW) ──
    if getattr(args, 'use_cbp_sampler', False):
        estimator = kgs.build_cost_table()
        cbp_scheduler = CostAwareScheduler(
            triple_list=kgs.relation_triples_list,
            cost_table=estimator.cost_table,
            batch_size=args.batch_size,
        )
        model.data_loader.cbp_scheduler = cbp_scheduler
    # ────────────────────
    
    model.get_model('TransE')
    model.run()
```

### 5.4 `src/py/experiments/args_kge/transe_fb15k237_args.json`

```json
{
    "use_cbp_sampler": false,
    "cbp_neg_num": 150,
    "cbp_max_try": 10
}
```

### 5.5 `src/torch/kge_models/pytorch_dataloader.py`

```python
class PyTorchTrainDataLoader(DataLoader):
    def __init__(self, kgs, batch_size, threads, neg_size, cbp_scheduler=None):
        self.cbp_scheduler = cbp_scheduler
        super().__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=(cbp_scheduler is None),   # CBP 启用时禁用 shuffle
            ...
        )
```

---

## 6. Weight 计算详解 (学术叙事核心)

CBP 的 Weight 计算完全基于 **Expected Cost**，而非 Degree：

```
对于 batch 中的每个三元组 (h, r, t):

    c_h = |neighbor_dict[h]|    # 头实体的候选池大小
    c_t = |neighbor_dict[t]|    # 尾实体的候选池大小
    
    # 最严格约束的实体决定成本
    c_effective = min(c_h, c_t)
    
    # 几何分布期望重试次数
    p_collision = N_neg / c_effective
    e_retry = 1 / (1 - p_collision)
    e_retry = min(max_try, e_retry)
    
    # 预期成本 (ms)
    expected_cost = e_retry × B3_const     # B3_const ≈ 51.8ms

batch_weight = Σ expected_cost over batch
```

这个公式被 Phase 5.5 的实验验证为有效（candidate_size R=0.9008）。

---

## 7. 文件修改汇总

| 文件 | 操作 | 修改量 |
|:---|---:|---:|
| `src/py/load/cbp_sampler.py` | **新增** | ~250 行 |
| `src/py/load/batch.py` | 修改 + 新增重置接口 | ~20 行 |
| `src/py/load/kgs.py` | 新增 `build_cost_table()` | ~10 行 |
| `src/py/experiments/main_FB15K237.py` | CBP 装配点 | ~15 行 |
| `src/py/experiments/args_kge/transe_fb15k237_args.json` | 新增 CBP 配置 | ~4 行 |
| `src/torch/kge_models/pytorch_dataloader.py` | `__init__` 接收 cbp_scheduler | ~10 行 |
| **总计** | | **~310 行** |

---

## 8. 验证标准 (Moving Forward)

| 阶段 | 验证内容 | 通过标准 |
|:---|---:|---:|
| CostEstimator UT | 候选池大小计算正确性 | 与 neighbor_dict 一致 |
| CostAwareScheduler UT | FFD 输出 batch 数 = ⌈N/B⌉ | 精确匹配 |
| CBP 权重验证 | batch_weight vs actual_time | **R > 0.85** |
| 精度验证 (node4) | MRR/Hits@K 漂移 | < 0.5% |
| DDP 验证 (node6) | 同步等待时间降低 | > 50% |

---

*本文档标志着 CBP 算法的最终技术落地规划。Phase 6 将逐文件实施。*