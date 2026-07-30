# 负采样 B1-B5 阶段详解（Phase 2 原始 Profiling 版本）

> **适用阶段**: Phase 2（B1-B5 子阶段拆分后，Phase 9 修改前）
> **源代码**: `src/torch/kge_models/pytorch_dataloader.py` 第 195-271 行
> **数据来源**: `output/results/negative_sampling_breakdown.md`（原始实验产出）

---

## 0. 整体流程概览

负采样（Negative Sampling）发生在 `collate_fn` 中，每个 batch 调用一次 `generate_neg_triples_fast()`。该函数对 batch 中的 **每个 positive triple** 循环处理，循环体内分为 5 个子阶段（B1-B5）。

```
collate_fn (每个 batch)
  └─ generate_neg_triples_fast()
       └─ for each (head, relation, tail) in pos_batch:   ← 每个正样本
            ├─ B1: Random Sampling（首轮）
            ├─ B2: Candidate Build
            ├─ B3: Collision Check（与已知三元组集合做差集）
            ├─ B4: Retry Logic（不够则重试，回到 B1）
            └─ B5: Output Build（拼接结果）
```

**关键参数**（FB15k-237 环境）:
- 实体总数: **14,541**
- 关系总数: **237**
- 训练三元组: **272,115**
- batch_size: **128**（典型值）
- neg_triples_num: **1**（每个正样本生成 1 个负样本）

---

## 1. B1 — Random Sampling（随机采样）

### 代码（pytorch_dataloader.py: 219-228）

```python
# B1: Random Sampling (random.sample) — 只在首次尝试计时，排除重试
if i == 0:
    t_b1 = time.perf_counter()
corrupt_head_prob = np.random.binomial(1, 0.5)
if corrupt_head_prob:
    neg_heads = random.sample(head_candidates, nums_to_sample)
else:
    neg_tails = random.sample(tail_candidates, nums_to_sample)
if i == 0:
    global_neg_sampling_time_b1_ms += (time.perf_counter() - t_b1) * 1000.0
```

### 详解

1. **决定破坏 Head 还是 Tail**: `np.random.binomial(1, 0.5)` 以 50% 概率返回 1（破坏 head）或 0（破坏 tail）。
2. **候选实体池**:
   ```python
   head_candidates = neighbor.get(head, entities_list)
   tail_candidates = neighbor.get(tail, entities_list)
   ```
   - 如果提供了 `neighbor` 词典（通常为 `None`），则从该实体的邻居中采样
   - **默认 fallback**: 使用全量 `entities_list`（FB15k-237: **14,541 个实体**）
3. **`random.sample(population, k)` 调用**:
   - 从 ~14.5K 实体中不放回随机抽取 `nums_to_sample`（默认=1）个实体
   - 这是 Python 标准库的 `random.sample`
4. **计时范围**: 仅在**首轮（i==0）**打点，排除后续重试轮次的 B1 时间

---

## 2. B1 为什么是瓶颈？—— `random.sample` 耗时深度分析

### 2.1 问题：`random.sample` 只是随机抽数字，为什么慢？

老师的直觉是对的——"随机抽一个数"听起来不应该慢。但关键在于 **被采样的 population 列表的大小** 和 **调用频率**。

### 2.2 CPython 源码层面的分析

`random.sample(population, k)` 在 CPython 中的实现路径（Python 3.8+ `random.py`）:

```python
def sample(self, population, k):
    if isinstance(population, _Sequence):
        n = len(population)           # ← O(1)
        result = [None] * k
        setsize = 21                  # 默认集合大小
        if k > 5:
            setsize += 4 ** _ceil(_log(k * 3, 4))  # 根据 k 扩展
        if n <= setsize:
            # 小 population：打乱后取前 k 个
            pool = list(population)   # ← O(n) 复制！
            for i in range(k):
                j = randbelow(n - i)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]
        else:
            # 大 population：用 set 跟踪已选索引
            selected = set()
            for i in range(k):
                j = randbelow(n)
                while j in selected:  # ← 冲突重试
                    j = randbelow(n)
                selected.add(j)
                result[i] = population[j]
        return result
```

对于 FB15k-237 的场景:
- **population = entities_list, n = 14,541**
- **k = 1**（通常 `neg_triples_num = 1`）

当 k 很小时（k ≤ 5），CPython 走 **"small k" 路径**，使用 `selected` set 方案。每次 `randbelow(n)` 生成随机索引后直接取值，O(k) ≈ O(1)。

**单个 `random.sample(entities_list, 1)` 的耗时大约是 1-3 微秒（μs）**。

### 2.3 真正的瓶颈：累积效应

**问题不在于单次调用慢，而在于调用次数巨大**：

```
每个 epoch 的 B1 总调用次数 = num_batches × batch_size
                              = (272,115 / 128) × 128
                              ≈ 272,115 次
```

每次调用 1-3 微秒，累积就是：
```
272,115 × 2μs ≈ 544ms ≈ 0.54 秒（仅 B1 部分）
```

但实际上还有一些额外的开销：
1. **Python 函数调用开销**: 每次 `random.sample()` 都是一个 Python 函数调用，涉及参数检查、类型判断等
2. **`np.random.binomial(1, 0.5)` 调用**: 每轮也要调用一次 NumPy 随机数生成
3. **`neighbor.get(head, entities_list)` 的字典查找**: 虽然 O(1)，但每次也要开销

### 2.4 实验证据

来自 `output/results/negative_sampling_breakdown.md` 的 Phase 2 原始 Profiling 数据：

| 阶段 | 总耗时 (ms) | 占负采样比例 |
|------|-------------|-------------|
| **B1: Sampling** | **46,719** | **42.26%** |
| B2: Candidate Build | 25,429 | 23.00% |
| B3: Collision Check | 15,725 | 14.22% |
| B4: Retry | 220 | 0.20% |
| B5: Output Build | 1,241 | 1.12% |
| **B1-B5 合计** | 89,335 | 80.80% |
| 负采样总耗时 | 110,559 | 100% |

**B1 以 42.26% 的占比成为绝对瓶颈**，是第二名 B2 的 **1.84 倍**。

### 2.5 为什么 B1 > B2 + B3？

直观分析：
- **B1**: 需要从全量实体列表（14,541）中随机索引 + Python 函数调用 × 272,115 次
- **B2**: 集合推导式 `{(h2, relation, tail) for h2 in neg_heads}` — 每次只处理 1 个元素，O(1)
- **B3**: 集合差集 `i_neg_triples - all_triples_set` — 对 1 个元素的集合做差集，O(1)

B1 的每次调用都有**固定的 Python 函数调用开销 + NumPy 随机数开销**，这些常数开销在经过 27 万次循环累积后，最终占到了 42%。

### 2.6 补充证据：`random.sample` 与 population 大小的关系

简单 benchmark（可复现）：

```python
import time
import random

entities = list(range(14541))  # FB15k-237 实体规模

# 测试 100,000 次 random.sample
start = time.perf_counter()
for _ in range(100000):
    random.sample(entities, 1)
elapsed = time.perf_counter() - start
print(f"100K calls: {elapsed:.3f}s, per call: {elapsed/100000*1000:.3f}ms")
# 预期输出: per call ≈ 0.002-0.003ms (2-3μs)
```

272,115 次 × 2-3μs ≈ **544-816ms** ≈ 占整个 epoch 负采样时间（~110s）的约 0.5-0.7%。但加上 **NumPy binomial 调用、字典查找、Python 循环体开销**等，最终 B1 占到了整个负采样流程的 42%。

---

## 3. B2 — Candidate Build（候选三元组构造）

### 代码（pytorch_dataloader.py: 230-236）

```python
# B2: Candidate Construction (set comprehension)
t_b2 = time.perf_counter()
if corrupt_head_prob:
    i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
else:
    i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
global_neg_sampling_time_b2_ms += (time.perf_counter() - t_b2) * 1000.0
```

### 详解

1. **集合推导式**将 B1 的采样结果构造成三元组集合
2. 如果破坏 head: 生成 `{(neg_head, relation, tail), ...}`
3. 如果破坏 tail: 生成 `{(head, relation, neg_tail), ...}`
4. 使用 `set` 而非 `list`，为 B3 的集合差集做准备
5. `neg_heads` / `neg_tails` 通常只有 1 个元素 → O(1)

---

## 4. B3 — Collision Check（碰撞检测）

### 代码（pytorch_dataloader.py: 245-248）

```python
# B3: Collision Check (set difference)
t_b3 = time.perf_counter()
filtered = list(i_neg_triples - all_triples_set)
global_neg_sampling_time_b3_ms += (time.perf_counter() - t_b3) * 1000.0
```

### 详解

1. **集合差集**: `i_neg_triples - all_triples_set`
   - `i_neg_triples`: 刚刚生成的候选负三元组（通常 1 个元素）
   - `all_triples_set`: **所有已知正三元组的集合**（FB15k-237: 272,115 个三元组）
2. **目的**: 过滤掉"假负样本"（false negatives）——即采样到的三元组恰好是已知的正样本
3. **复杂度**: `all_triples_set` 是 Python `set`，查找 O(1)。对一个 1 元素的集合做差集，只需一次哈希查找
4. **返回**: 过滤后的有效负三元组 list

---

## 5. B4 — Retry Logic（重试逻辑）

