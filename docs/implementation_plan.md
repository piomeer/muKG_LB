# Phase 5 - Step 5: DDBP 技术落地规划 (Implementation Planning)

> **编制日期**: 2026-07-09
> **基于**: Phase 5 Step 1~4 全部结论
> **目标**: 明确 DDBP 在 muKG_LB 项目结构中的代码落地位置、修改文件清单、接口设计

---

## 1. 代码拓扑总览

```
┌─────────────────────────────────────────────────────────────┐
│                     DDBP 数据流架构                            │
│                                                             │
│   ┌──────────────┐    ┌──────────────────┐    ┌───────────┐ │
│   │ DegreeTracker │───▶│ BinPackingSched  │───▶│ DataLoader │ │
│   │ (预计算模块)   │    │ (FFD贪心装箱)     │    │ (消费分组)  │ │
│   └──────────────┘    └──────────────────┘    └───────────┘ │
│          │                       │                            │
│          ▼                       ▼                            │
│   ┌─────────────────────────────────────────────────┐        │
│   │              degree_table.json                   │        │
│   │   (持久化缓存，避免每次训练重复计算)               │        │
│   └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 新增文件 (New Files)

### 2.1 `src/py/load/ddbp_sampler.py` — 核心模块

```python
"""
Degree-Driven Bin Packing (DDBP) — 核心模块

包含两个类：DegreeTracker 和 BinPackingScheduler
"""
```

#### Class: `DegreeTracker`

| 项 | 说明 |
|:---|---|
| **职责** | 从 KG 三元组集中统计每个实体的度数，计算理论采样权重 |
| **输入** | `relation_triples_list: List[Tuple[int,int,int]]`, `n_entities: int` |
| **输出** | `degree_table: Dict[int, float]` — entity_id → normalized weight |
| **核心方法** | |
| `compute_degrees()` | 统计每个实体在头尾位置的出现次数 |
| `compute_hub_threshold(percentile=10)` | 计算 Top p% Hub 阈值 |
| `compute_weight(e)` | 根据度数和 neighbor dict 计算预期采样时间 $E[T_{sampling}(e)]$ |
| `save(path)` / `load(path)` | 持久化 degree_table 避免重复计算 |
| **复杂度** | O(\|V\|+\|E\|) — 一次遍历全量三元组 |

#### Class: `BinPackingScheduler`

| 项 | 说明 |
|:---|---:|
| **职责** | 对 epoch 的 triple_list 执行 FFD 贪心装箱，输出均衡的 batch 分组 |
| **输入** | `triple_list`, `degree_table`, `batch_size`, `seed` |
| **输出** | `packed_batches: List[List[Tuple]]` — 权重均衡的 batch 列表 |
| **核心方法** | |
| `compute_sample_weight(triple)` | 对单个三元组计算 $weight = \max(degree(head), degree(tail))$ |
| `_sort_by_weight(triple_list)` | 按 weight 降序排列样本 |
| `_first_fit_decreasing(sorted_samples)` | FFD 贪心装箱: 将高权重样本分散到不同 batch |
| `pack_epoch()` | 主入口: sort → FFD → 返回重组后的 batch 列表 |
| **复杂度** | O(N log N + N × B) — N=样本数, B=batch 数 |

---

## 3. 需修改文件 (Modified Files)

### 3.1 `src/py/load/batch.py` — 拦截现有 batch 生成逻辑

#### 修改点 1: `generate_pos_triples()` — 添加 DDBP 注入点

```python
def generate_pos_triples(triples, batch_size, step, is_fixed_size=False, 
                         ddbp_scheduler=None):          # ← NEW
    if ddbp_scheduler is not None:
        # 首次调用时执行装箱
        if not hasattr(generate_pos_triples, '_ddbp_batches'):
            generate_pos_triples._ddbp_batches = ddbp_scheduler.pack_epoch()
        pos_batch = generate_pos_triples._ddbp_batches[step]
    else:
        # 原有逻辑: 按 start:end 切分
        start = step * batch_size
        end = start + batch_size
        pos_batch = triples[start: end]
    # ... is_fixed_size 处理不变
    return pos_batch
```

#### 修改点 2: 暴露 DDBP 重置接口

```python
def reset_ddbp_cache():
    """每个 epoch 结束后重置 DDBP 缓存，使下一次 epoch 重新装箱"""
    if hasattr(generate_pos_triples, '_ddbp_batches'):
        del generate_pos_triples._ddbp_batches
```

**侵入性评估**: 极小。`ddbp_scheduler=None` 为默认值，现有调用方完全不受影响。

---

### 3.2 `src/py/load/kgs.py` 或 `src/py/load/kg.py` — 构造时提供实体度数

需要评估是在 `read_kgs_from_folder()` 阶段加载 degree_table，还是在训练入口手动初始化。

**推荐方案**: 在 `kgs` 对象上挂载一个 `build_degree_table()` 方法，由训练入口按需调用。

```python
# 在 kgs.py 中添加
def build_degree_table(self):
    from src.py.load.ddbp_sampler import DegreeTracker
    self.degree_tracker = DegreeTracker(self.relation_triples_list, self.entities_num)
    return self.degree_tracker
```

---

### 3.3 `src/py/experiments/main_FB15K237.py` — 装配 DDBP 组件

在 `model.run()` 调用前插入 DDBP 初始化逻辑：

```python
if __name__ == '__main__':
    # ... 原有代码 ...
    kgs = read_kgs_from_folder(...)
    model = kge_models(args, kgs)
    
    # ── DDBP 装配点 (NEW) ──
    if getattr(args, 'use_ddbp_sampler', False):
        tracker = kgs.build_degree_table()
        ddbp_scheduler = BinPackingScheduler(
            triple_list=kgs.relation_triples_list,
            degree_table=tracker.degree_table,
            batch_size=args.batch_size,
            seed=args.seed if hasattr(args, 'seed') else 42
        )
        # 注入到 DataLoader
        model.data_loader.ddbp_scheduler = ddbp_scheduler
    # ──────────────────────
    
    model.get_model('TransE')
    model.run()
    model.test()
```

---

### 3.4 `src/py/args_handler.py` — 增加 DDBP 控制开关

默认 DDBP 关闭，通过 JSON args 或命令行启用：

**在 args JSON 文件中新增字段**（以 `transe_fb15k237_args.json` 为例）：

```json
{
    "use_ddbp_sampler": false,
    "ddbp_hub_percentile": 10,
    "ddbp_batch_weight_mode": "max_degree"
}
```

**`args_handler.py` 无需修改** — `setattr` 自动处理新字段。

---

### 3.5 `src/torch/kge_models/pytorch_dataloader.py` — 传递 DDBP 调度器

在 `PyTorchTrainDataLoader` 中增加 DDBP 属性，并传递给 Dataset：

```python
class PyTorchTrainDataLoader(DataLoader):
    def __init__(self, kgs, batch_size, threads, neg_size, ddbp_scheduler=None):  # ← NEW
        self.ddbp_scheduler = ddbp_scheduler  # ← NEW
        super().__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=(ddbp_scheduler is None),  # DDBP 启用时禁用 shuffle
            ...
        )
```

同时，在 `PyTorchTrainDataset` 的 `collate_fn` 中，当 DDBP 启用时跳过原始的 `steps` 计算。

---

## 4. 依赖与接口 (Interfaces)

### 4.1 核心接口定义

```python
# ── DegreeTracker 输出接口 ──
class DegreeTracker:
    def compute_degrees(self) -> Counter: ...
    def compute_hub_threshold(self, percentile: int = 10) -> int: ...
    def compute_weight(self, entity_id: int, candidate_size: int = None) -> float: ...
    def save(self, path: str): ...
    def load(self, path: str) -> bool: ...

# ── BinPackingScheduler 接口 ──
class BinPackingScheduler:
    def __init__(self, triple_list, degree_table, batch_size, seed, hub_percentile=10): ...
    def compute_sample_weight(self, triple) -> float: ...
    def pack_epoch(self) -> List[List[Tuple]]: ...
    def reset(self): ...

# ── 与 DataLoader 的集成点 ──
# 位置: PyTorchTrainDataLoader.__init__
# 方式: 通过 ddbp_scheduler 参数注入
# 效果: ddbp_scheduler 非 None → 禁用 random shuffle → 使用 DDBP batch 顺序
```

### 4.2 数据流路径

```
epoch_start
    │
    ▼
BinPackingScheduler.pack_epoch()
    │ 对 triple_list 按 weight 降序排列
    │ FFD 装箱 → 输出 batch 分组
    ▼
generate_pos_triples(step=0, ddbp_scheduler=...)
    │ 从预计算 batch 分组中取第 step 组
    ▼
generate_neg_triples_fast()
    │ 原有负采样逻辑不变
    ▼
返回正负样本 → 训练
```

### 4.3 回退机制

```python
# 在训练入口添加降级开关
if args.use_ddbp_sampler:
    try:
        # 尝试 DDBP
        ddbp_scheduler = BinPackingScheduler(...)
    except Exception as e:
        print(f"[WARN] DDBP init failed: {e}. Falling back to random shuffle.")
        args.use_ddbp_sampler = False  # 自动降级
```

---

## 5. 测试计划

| 测试阶段 | 环境 | 验证内容 | 通过标准 |
|:---|---:|---:|:---:|
| 单元测试 | pc-cluster CPU | `DegreeTracker` 度数统计正确性 | 与 `GLOBAL_ENTITY_DEGREE` 数据一致 |
| 单元测试 | pc-cluster CPU | `BinPackingScheduler.pack_epoch()` 输出 batch 数 = ceil(N/B) | 精确匹配 |
| 单元测试 | pc-cluster CPU | batch 间 weight 方差比随机 shuffle 降低 > 50% | 方差比 < 0.5x |
| 集成测试 | node4 GPU | DDBP 启用后 MRR/Hits@K 与原始版本漂移 < 0.5% | MRR 漂移 < 0.005 |
| 性能测试 | node4 GPU | 单卡训练速度不降级 | 训练时间无显著增加 |
| 性能测试 | node6 DDP | DDP 同步等待时间降低 > 50% | 等待时间 < 10ms/step |

---

## 6. 文件修改汇总

| 文件 | 操作 | 修改量 (预估) |
|:---|---:|---:|
| `src/py/load/ddbp_sampler.py` | **新增** | ~250 行 |
| `src/py/load/batch.py` | 修改 `generate_pos_triples()` + 新增 `reset_ddbp_cache()` | ~20 行 |
| `src/py/load/kgs.py` | 新增 `build_degree_table()` 方法 | ~10 行 |
| `src/py/experiments/main_FB15K237.py` | DDBP 装配点 + import | ~15 行 |
| `src/py/experiments/args_kge/transe_fb15k237_args.json` | 新增 DDBP 配置字段 | ~4 行 |
| `src/torch/kge_models/pytorch_dataloader.py` | `__init__` 接收 `ddbp_scheduler` 参数 + 禁用 shuffle | ~10 行 |
| **总计** | | **~310 行** |

---

## 7. Phase 6 实施顺序

```
Phase 6 — DDBP 原型开发 (预估 2~3 天)
├── Day 1: DegreeTracker + BinPackingScheduler 实现 + 单元测试
├── Day 2: batch.py + pytorch_dataloader.py 修改 + 集成测试
├── Day 3: node4 单卡验证 → node6 多卡 Benchmark → 调优
└── 完成后: MRR/Hits@K 对比报告 + 同步等待时间对比
```

---

*本文档标志着 Phase 5 设计阶段的最终收官。Phase 6 将依据本计划逐文件实施代码修改。*