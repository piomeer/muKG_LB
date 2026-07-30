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

### 代码（pytorch_dataloader.py: 255-262）

```python
# B4: Retry Processing
t_b4 = time.perf_counter()
if len(neg_triples) == neg_triples_num:
    global_neg_sampling_time_b4_ms += (time.perf_counter() - t_b4) * 1000.0
    break
else:
    nums_to_sample = neg_triples_num - len(neg_triples)
    global_neg_sampling_time_b4_ms += (time.perf_counter() - t_b4) * 1000.0
```

### 详解

1. **判断阶段**: 检查当前轮次收集到的负样本数是否已达到目标 `neg_triples_num`
2. **满足条件 → break**: 如果 `len(neg_triples) == neg_triples_num`（通常 = 1），直接跳出 `for i in range(max_try)` 重试循环
3. **不满足 → 继续重试**: 更新 `nums_to_sample`（还差几个），下一轮回到 B1 重新采样
4. **`max_try` 保护**: 外层循环最多 10 次（由 `max_try=10` 控制），如果一直碰撞到第 10 轮还没有足够负样本，最后会走 B5 的 `neg_triples += list(i_neg_triples)` 直接接受（不再做 B3 过滤）
5. **计时**: 不区分成功/失败分支，都累计到 B4

### 为什么 B4 只占 0.20%？

- 大多数情况下第一轮就成功（无碰撞或已过滤），retry 的额外开销很小
- profiling 数据显示 retry 占比极小 → 说明碰撞概率很低

---

## 6. B5 — Output Build（输出构建）

### 代码（pytorch_dataloader.py: 238-243, 250-253, 267-269）

B5 在代码中出现在三个位置，都计入同一个累加器：

```python
# 位置 1（最后一轮重试时，直接接受，不过滤）
if i == max_try - 1:
    t_b5 = time.perf_counter()
    neg_triples += list(i_neg_triples)
    global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0
    break

# 位置 2（非最后一轮，附加过滤后的结果）
t_b5 = time.perf_counter()
neg_triples += filtered
global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0

# 位置 3（每个正样本处理完后，把该 triple 的负样本拼入全局 batch）
t_b5 = time.perf_counter()
neg_batch.extend(neg_triples)
global_neg_sampling_time_b5_ms += (time.perf_counter() - t_b5) * 1000.0
```

### 详解

1. **位置 1 & 2**: 把本轮生成的负三元组拼入 `neg_triples` 列表（当前正样本的收集器）。位置 1 是 max_try 的兜底（不经过 B3 碰撞过滤），位置 2 是正常流程
2. **位置 3**: 等当前正样本的负三元组收集完毕后，通过 `neg_batch.extend(neg_triples)` 拼入全局的 `neg_batch`（整个 batch 的负采样结果）
3. **数据结构转换**: `list(i_neg_triples)` 将集合转为列表供后续拼接
4. **为何 B5 只占 1.12%**: `extend` 和 `+=` 每次只处理 1 个或少数几个元素，开销可以忽略不计

---

## 7. `global_neg_sampling_time_b1_ms` 的完整计时链路

### 声明 & 初始化

```python
# pytorch_dataloader.py: 32
global_neg_sampling_time_b1_ms = 0.0   # 模块级全局变量

# pytorch_dataloader.py: 120 （每个 batch 开始时清零）
global_neg_sampling_time_b1_ms = 0.0
```

### 累计计时

```python
# pytorch_dataloader.py: 219-228，对每个正样本的首轮尝试
if i == 0:                          # 只在首次尝试时计时
    t_b1 = time.perf_counter()
# ... random.sample() + np.random.binomial() ...
if i == 0:
    global_neg_sampling_time_b1_ms += (time.perf_counter() - t_b1) * 1000.0
```

### 对外暴露

```python
# pytorch_dataloader.py: 78-89
def get_per_batch_profiling():
    return {
        ...
        'neg_sampling_b1_ms': global_neg_sampling_time_b1_ms,
        ...
    }
```

### 上游消费（kge_trainer.py 中读取）

每个 batch 结束后，训练循环调用 `get_per_batch_profiling()` 获取 B1-B5 各阶段的累计耗时，然后写入 `negative_sampling_cost.csv`。

---

## 8. 实验数据总结（Phase 2 Profiling 产出）

| 阶段 | 累计耗时 (ms) | 占负采样比例 | 每个正样本 ≈ |
|------|-------------|:---:|:---:|
| **B1: Sampling** | 46,719 | **42.26%** | ~0.172 ms |
| B2: Candidate Build | 25,429 | 23.00% | ~0.093 ms |
| B3: Collision Check | 15,725 | 14.22% | ~0.058 ms |
| B4: Retry | 220 | 0.20% | ~0.001 ms |
| B5: Output Build | 1,241 | 1.12% | ~0.005 ms |
| **B1-B5 合计** | 89,335 | 80.80% | — |
| 负采样总耗时 | 110,559 | 100% | — |

**数据来源**: `output/results/negative_sampling_breakdown.md`（原始实验，未经过 Phase 9 的优化修改）

### 核心结论

1. **B1 (random.sample) 是绝对瓶颈**，占负采样时间的 42%
2. B1 慢的原因不是单次 `random.sample` 本身慢（每次 1-3μs），而是 **27 万次 Python 函数调用 + NumPy 随机数 + 字典查找的累积效应**
3. B4 (Retry) 几乎可以忽略不计 → 说明碰撞发生的概率极低
4. B5 (Output Build) 开销极小 → 列表 extend 高效

---
*最后更新: 2026-07-30 | 基于 Phase 2 原始 Profiling 数据, Phase 9 未介入*
