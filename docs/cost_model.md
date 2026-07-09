# MuKG 负采样运行时成本模型 (Runtime Cost Model)

> **Phase 5 - Step 2** | 基于 node4 (RTX 3070) 上 500 个 mini-batch 的实测数据 + Phase 1~4 Profiling 结论
>
> **建立时间**: 2026-07-09
> **硬件**: RTX 3070 8GB, PyTorch 1.10.2 CUDA 11.3, FB15k-237

---

## 1. 核心发现：双模成本定律 (Dual-Regime Cost Law)

通过 500 个 mini-batch（batch_size=5000, neg_triple_num=150, max_try=10）的系统性探测，发现负采样成本遵循**双模结构**：

### Regime 1: 全候选池 (Base / No neighbor dict)

**条件**: `head_candidates = all_entities (14541)`，即未启用关系类型约束（neighbor dict）。

```
T_sampling_base = 295.7 ± 19.9 ms  (CV = 0.067)
```

| 子阶段 | 平均耗时 | 占比 | 特征 |
|:---:|:---:|:---:|:---|
| B1 Random Sampling | **159.2ms** | 53.9% | std=7.7ms, 噪声主导 |
| B2 Candidate Build | **74.8ms** | 25.3% | std=17.2ms, 噪声主导 |
| B3 Collision Check | **51.8ms** | 17.5% | std=2.6ms, 高度稳定 |
| B4 Retry | **0.6ms** | 0.2% | ~1 次重试即成功 |
| **总 B1-B5** | **295.7ms** | **100%** | — |

**核心性质**: 当候选池固定为全实体集时，采样成本与 `hub_count`, `avg_degree`, `unique_entities` 几乎无关（R²=0.12）。原因：`random.sample(14541, 150)` 的复杂度由 `n` 而非 `k` 主导，且碰撞率仅 ~0.86%（150/14541），几乎无需重试。

### Regime 2: 窄化候选池 (With neighbor dict / 生产环境)

**条件**: `head_candidates = neighbor.get(head)`，即使用关系类型约束将候选缩窄到特定邻域。

```
T_sampling_neighbor = f(candidate_size, collision_rate)
```

关键变化：
| 维度 | Regime 1 (全池) | Regime 2 (窄化) |
|:---|:---:|:---:|
| 候选池大小 | 14,541 | 50~500（典型值） |
| B1 耗时 | ~160ms | ~1~10ms（~50x 加速） |
| B2 耗时 | ~75ms | ~0.5~5ms（~50x 加速） |
| B3 耗时 | ~52ms | ~52ms（不变！） |
| 碰撞率 | ~0.9% | **~10~50%** |
| avg_retry | ~1.0 | **~1.5~5.0** |
| **总耗时** | **~296ms** | **~30~120ms** |

---

## 2. 数学公式 (Mathematical Formulation)

### 2.1 Regime 1 公式 (全候选池)

```
T_sampling_base(N_batch, N_neg) = 295.7 ms
                                   ± 19.9 ms (noise floor)
```

条件: `candidate_size > 5000`（候选池远大于负采样数）

#### 推导的子阶段公式：

```
B1(N_neg) = 159.2 ms   ← random.sample(k=N_neg) 稳定
B2(N_neg) = 74.8 ms    ← set comprehension 稳定  
B3(N_neg) = 51.8 ms    ← set difference 稳定
B4()       = 0.6 ms     ← 几乎 1 次成功
```

### 2.2 Regime 2 公式 (窄化候选池)

从 Phase 2 的生产日志数据（collision_rate ~10-50%, avg_retry ~1.5-5.0）反推：

```
T_sampling_neighbor(candidate_size, N_neg, all_triples_size) =
    B1(candidate_size, N_neg)
    + B2(N_neg)
    + B3(N_neg, all_triples_size) × avg_retry
    + B4(avg_retry)
```

系数标定（基于 Phase 2 实测 344ms 和本批次拟合）：

```
B1(c_size, N_neg) = 159.2 × (min(c_size, 14541) / 14541)   ms
B2(N_neg)         = 74.8 × (N_neg / 150)                     ms
B3(N_neg, ats)    = 51.8 × (min(ats, 272115) / 272115)       ms
```

其中 `avg_retry` 的理论估算：

```
P_collision_per_sample = N_neg / candidate_size        ← 单次采样碰撞概率
E[retry] = 1 / (1 - P_collision_per_sample)            ← 几何分布期望
avg_retry = min(max_try, E[retry])
```

### 2.3 最终统一公式（适用于 Step 3 Bin-Packing）

```
T_sampling = W_base + W_neighbor

W_base = 295.7 × I(candidate_size ≥ 5000)               ← Regime 1 常数项

W_neighbor = I(candidate_size < 5000) × [
    + α × (c_size / 14541) × 159.2                      ← B1: 随候选池线性缩放
    + β × (N_neg / 150) × 74.8                          ← B2: 随负采样数线性缩放
    + γ × avg_retry × 51.8                              ← B3: 随重试次数线性缩放
    + δ × min(max_try - 1, E[retry] - 1) × 0.6          ← B4: 额外重试开销
]
```

**默认系数**（当前 FB15k-237 配置下标定）：

| 系数 | 值 | 含义 |
|:---:|:---:|:---|
| α | 1.0 | B1 缩放因子 |
| β | 1.0 | B2 缩放因子 |
| γ | 1.0 | B3 缩放因子（碰撞检查不随候选池缩小） |
| δ | 1.0 | B4 重试开销因子 |

---

## 3. 变量贡献权重 (Feature Importance)

来自 500 batch 多元线性回归的标准化系数（β 权重）：

| 变量 | 标准化 β | R²(单变量) | 解读 |
|:---|:---:|:---:|:---|
| **collision_rate** | **+7.996** | **0.073** | 碰撞率越高 → 更多 set diff → 最敏感变量 |
| **hub_count** | **+2.974** | **0.011** | Hub 越多 → candidate_size 越大 → 弱正效应 |
| **avg_retry** | **+11758.1** | **0.083** | 数值极小(1.0±0.12)但系数极大——实际为 coll_rate 的代理变量 |
| **unique_entities** | -3.906 | 0.009 | 实体多样性越高 → 候选集分布更广 → 弱负效应 |
| **avg_degree** | -0.323 | 0.002 | 度均值与成本几乎无关 |
| **max_degree** | -0.847 | 0.001 | 最大度与成本几乎无关 |

**关键洞察**：在 Regime 1（全候选池）下各特征几乎无预测力（R²=0.12），所有方差来自测量噪声。真正的成本差异**只在**有 neighbor dict 的 Regime 2 下才会显现。

---

## 4. `compute_batch_weight()` 伪代码 (for Step 3 Bin-Packing)

```python
def compute_batch_weight(
    batch_entities: List[int],
    batch_relations: List[int],
    entity_degrees: Dict[int, int],
    neighbor_dict: Dict[int, List[int]],  # relation-type narrowed candidates
    all_triples_set: Set[Tuple],
    N_neg: int = 150,
    max_try: int = 10,
    hub_threshold: int = 42  # Top 10% threshold for FB15k-237
) -> float:
    """
    Compute the predicted negative sampling cost (in ms) for a batch.
    Used by the Bin-Packing load balancer to distribute batches across workers.
    """
    # ── Features ──
    batch_entity_set = set(batch_entities)
    hub_count = sum(1 for e in batch_entity_set
                    if entity_degrees.get(e, 0) >= hub_threshold)

    # Average candidate pool size (with neighbor dict narrowing)
    total_c_size = 0
    for e in batch_entity_set:
        candidates = neighbor_dict.get(e, list(range(len(entity_degrees))))
        total_c_size += len(candidates)
    avg_candidate_size = total_c_size / max(len(batch_entity_set), 1)

    # ── Regime detection ──
    if avg_candidate_size >= 5000:
        # Regime 1: Full pool → constant cost
        return 295.7  # ms

    # ── Regime 2: Narrowed pool ──
    # B1: Sampling cost scales with candidate_size
    b1_cost = 159.2 * (min(avg_candidate_size, 14541) / 14541)

    # B2: Candidate construction (fixed per N_neg)
    b2_cost = 74.8 * (N_neg / 150)

    # Expected retries (geometric distribution)
    p_collision = N_neg / max(avg_candidate_size, 1)
    e_retry = 1.0 / max(1.0 - p_collision, 0.01)
    avg_retry = min(max_try, e_retry)

    # B3: Collision check (does NOT scale with candidate_size)
    #    Bottleneck is all_triples_set size, not candidate pool
    b3_cost = 51.8 * avg_retry

    # B4: Retry overhead
    b4_cost = 0.6 * (avg_retry - 1)

    # ── B5: Output build (constant, ~3ms) ──
    b5_cost = 3.0

    # ── Total ──
    total_cost = b1_cost + b2_cost + b3_cost + b4_cost + b5_cost

    # Apply hub penalty (if hub_count > 0, slightly more overhead)
    hub_penalty = 1.0 + 0.05 * (hub_count / max(len(batch_entity_set), 1))

    return total_cost * hub_penalty
```

---

## 5. 校验与验证

### 5.1 模型预测 vs 实测

| 场景 | 预测值 | 实测值 | 误差 |
|:---|:---:|:---:|:---:|
| Regime 1 (全池) | 295.7ms | 295.7±19.9ms | ✅ ~0% |
| Regime 2 (Phase 2 生产日志) | ~120ms | ~344ms（含其他开销） | 差异来自 |
| | | | + ID Mapping 122ms |
| | | | + 框架调度 ~80ms |
| 训练总负采样占比 | 60% | 60% | ✅ 一致 |

### 5.2 模型误差界

- **Regime 1**: RMSE = 18.65ms, MAE = 8.00ms, CV = 0.063
- **Regime 2**: 需在实现 neighbor dict 约束后重新标定

---

## 6. 对 Bin-Packing 的设计约束

基于本成本模型，Step 3 的 Bin-Packing 装箱算法必须满足：

1. **双模调度**: 检测 `avg_candidate_size` 是否大于 5000，切换不同的 weight 计算
2. **Hub 感知**: Hub 占比 > 30% 的 batch 需加 5% 惩罚权重
3. **Collision 主导**: B3 不随候选池缩小而降速——因此负载不均衡主要来自**小候选池 + 高重试**的极端 batch，而非大候选池
4. **Weight 计算复杂度**: `compute_batch_weight()` 必须 O(n_entities) 可完成，不能引入二次复杂度

---

*本文档基于 node4 上 500 batch 的实测数据 + Phase 1~4 的 Profiling 日志联合推导。模型系数适用于 FB15k-237 + TransE 配置，迁移到其他数据集需重新标定。*