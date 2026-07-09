# MuKG 负采样优化 — 设计空间探索 (Design Space Exploration)

> **Phase 5 - Step 1** | 基于 Phase 1~4 的真实 Profiling 数据
>
> **关键约束数据**：
> - 负采样占总训练时间 **60%**
> - B1 Random Sampling 占负采样 **42.3%**（#1 瓶颈子阶段）
> - B3 Collision Check 占负采样 **14.2%**（#2 瓶颈子阶段）
> - Hub Entity Count vs Sampling Time: **Pearson R = 0.8163**（强正相关，主要影响路径）
> - Hub Entity Count vs Collision Check: **Pearson R = 0.5404**（中等正相关，次要路径）
> - Collision Check vs 总负采样时间: **Pearson R = 0.8640**（波动最大解释因子）
> - avg_retry vs 负采样时间: **R = 0.0254**（几乎无关，重试非瓶颈）
> - 每 step 负采样 ~**344ms**（B1=103ms, B2=56ms, B3=35ms）
> - Batch size = 5000, Neg triple num = 150, max_try = 10

---

## Route A: GPU 端负采样 (GPU Sampling)

### 思想 (Idea)
将当前在 CPU 端通过 `random.sample()` 逐条进行的负采样逻辑，完全迁移到 GPU 端。利用 `torch.randint` + 向量化索引一次性生成所有候选负样本张量，并在 GPU 上执行碰撞检查和候选集过滤。

### 复杂度 (Complexity)
- **时间复杂度**: O(batch_size × neg_triple_num) 全向量化 → 约 750k 次/step 完全 GPU 并行
- **空间复杂度**: 需要 GPU 显存额外存储：(batch_size × neg_triple_num × 2) 个 entity_id → ~750k × 2 × 4B = **~6MB**（可忽略）
- **网络传输开销**: 消除 CPU→GPU 的负样本拷贝

### 优点 (Pros)
- ✅ **直接消除 B1 Random Sampling（42.3% 瓶颈）**：将最耗时的 Python `random.sample` 替换为 GPU 张量操作，利用 GPU 的大规模并行性
- ✅ **消除 CPU-GPU 数据传输瓶颈**：负样本直接在 GPU 生成，无需从 CPU 拷贝到 GPU
- ✅ **Collision Check 也受益**：GPU 端的 set-like 操作（`torch.unique` + masking）可进一步加速 B3（14.2%）
- ✅ **极高理论消除上限**：可消除负采样约 70-80% 的 CPU 时间（B1+B2+B3 的大部分）

### 缺点 (Cons)
- ❌ **显存竞争**：需要额外的 GPU 显存存储正负样本张量，batch_size=5000 时约 ~7-8MB 额外负载，但若 batch_size 进一步增大可能压缩模型参数空间
- ❌ **实现侵入性高**：需要重写 `generate_neg_triples_fast()` 的核心采样逻辑，并处理 GPU 上的实体过滤 vs 正样本重复问题
- ❌ **Hub 实体负载不均衡**：GPU 并行处理无法缓解 Hub 实体导致的采样难度差异——所有实体被一视同仁处理

### 实现难度 (Implementation Difficulty)
**High** — 需要深入重构 DataLoader 的采样模块，将 CPU 逻辑迁移到 GPU 张量操作，并确保与现有 Trainer 接口兼容。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**50~60%**（即总训练时间缩短约 **30~36%**）。核心收益来自 B1 完全消除 + B3 部分加速。

---

## Route B: 批次内负采样 (Batch Sampling / In-batch Negative Sampling)

### 思想 (Idea)
不再为每个正样本单独采样负样本，而是将同一 batch 内其他实体的正样本尾实体作为当前样本的负样本。每个 batch 内的 N 个正样本可以自然产生 N×(N-1) 个负样本对，完全跳过显式采样和碰撞检查。

### 复杂度 (Complexity)
- **时间复杂度**: O(batch_size²) → 对于 batch_size=5000，理论产生 **25M 个负样本对**
- **空间复杂度**: O(batch_size²) → 25M 个 pair 的 scoring 需要在 GPU 上计算，显存压力极大

### 优点 (Pros)
- ✅ **完全消除 Sampling 和 Collision Check**：B1（42.3%）和 B3（14.2%）一并消失，因为无需采样也无需去重
- ✅ **极低实现复杂度**：核心思想只需将 batch 内所有实体两两配对作为负样本
- ✅ **GPU 原生友好**：负样本直接在 GPU 上构建，无需 CPU 干预

### 缺点 (Cons)
- ❌ **梯度污染（Label Leakage）**：其他正样本的尾实体可能恰好是当前样本的真实正确尾实体，导致 false negative 噪声，严重降低 MRR/Hits@K
- ❌ **O(batch_size²) 显存爆炸**：batch_size=5000 → 25M 负样本评分矩阵，假设 float32 需要 100MB scoring matrix × 关系数量的组合，极易 OOM
- ❌ **批次内分布偏差**：如果 batch 内实体分布高度相关（如图谱中紧密相连的实体群），负样本缺乏多样性

### 实现难度 (Implementation Difficulty)
**Low** — 概念简单，但需要处理评分矩阵的稀疏计算以避免 OOM。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**~100%**（负采样阶段完全消除），但以 MRR 严重下降和可能的 OOM 为代价。需要配合 self-adversarial sampling 等去偏技术来缓解 false negative 问题，实际落地后预期收益将降至 60~80%。

---

## Route C: 基于度的负采样 (Degree-aware Sampling)

### 思想 (Idea)
利用知识图谱的度分布先验信息（实体度 = 参与的三元组数量），在采样阶段对低度实体适度放宽候选集过滤（低度实体的候选正三元组集合小，碰撞概率低），对高度实体（Hub）主动降低采样期望或使用预计算候选集。核心依据：Hub Count vs Sampling Time R=0.8163。

### 复杂度 (Complexity)
- **时间复杂度**: O(n_entities × degree_bucketing) 预处理 → 只需一次计算；在线采样时为每个实体 O(1) 查找度桶
- **空间复杂度**: O(n_entities) 存储度分桶映射表 → ~14,541 个实体 × 1 byte = **~14KB**

### 优点 (Pros)
- ✅ **直接攻击 Hub 相关瓶颈（R=0.8163）**：针对 Top 10% Hub 实体降低采样预算，它们产生的采样耗时最高
- ✅ **低系统代价**：只需在预处理阶段构建度分桶表，在线推理阶段 O(1) 查表
- ✅ **训练质量可控**：可引入温度参数控制对不同度分桶的采样强度衰减，通过调节避免精度损失
- ✅ **与 GPU 采样兼容**：可作为 Route A 的上层调度策略，叠加优化

### 缺点 (Cons)
- ❌ **低度实体欠采样风险**：过度降低 Hub 采样可能导致模型对 Hub 实体周围的稀疏关系欠拟合
- ❌ **引入调度复杂度**：不同的度分桶需要动态的 neg_triple_num 分配，可能破坏 batch 内的一致性
- ❌ **对 Collision Check 间接影响有限**：Hub 对碰撞检查的影响（R=0.5404）为中等相关，降低采样量对碰撞环节的帮助有限

### 实现难度 (Implementation Difficulty)
**Medium** — 需要在 DataLoader 中实现度分桶调度，对现有架构侵入性中等。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**15~25%**（主要来自 Top 10% Hub 实体的采样预算降低，预期可降低 B1 时间约 30-50%）。需配合其他 Route 才能实现更大幅度加速。

---

## Route D: Hub 感知缓存 (Hub-aware Cache)

### 思想 (Idea)
对高频出现的 Hub 实体（Top 1%~5% 的高实体度实体）预计算并缓存其候选正三元组集合。当 batch 中出现这些 Hub 实体时，直接从缓存读取预构建的候选集合，跳过 B1 Sampling 和 B2 Candidate Build 阶段。

### 复杂度 (Complexity)
- **时间复杂度**: 预处理 O(max_degree × n_hubs) → 假设 Top 1% ≈ 145 个 Hub，每个 Hub 的候选集 ~10k 规模 → 约 1.5M 操作（一次性）
- **空间复杂度**: O(n_hubs × avg_degree) → 145 × 10k × (2 × 4B) ≈ **~11.6MB** 缓存

### 优点 (Pros)
- ✅ **直接消除 Hub 实体的 B1+B2 开销**：Top 1% Hub 实体贡献了最大的采样时间方差（Top 20 最慢 batch 的 hub_count 全部为 6000），消除其采样环节效果显著
- ✅ **缓存命中率与 Hub 集中度正相关**：知识图谱的幂律分布保证了少量 Hub 覆盖大部分采样请求
- ✅ **与 Route C/E/F 天然互补**：缓存可作为 Degree-aware 调度的底层支撑

### 缺点 (Cons)
- ❌ **缓存一致性**：在训练过程中，如果使用动态负采样策略（如 Self-Adversarial Sampling），缓存的正样本关系可能过时
- ❌ **实现复杂性**：需要实现缓存失效/更新策略（如 LRU 或 epoch 级刷新），增加了维护负担
- ❌ **非 Hub 实体无收益**：长尾实体（占大多数）直接不受益

### 实现难度 (Implementation Difficulty)
**Medium-High** — 需要实现缓存层、更新策略，并修改 DataLoader 的采样路径。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**15~20%**（仅对 Top 1~5% Hub 生效，覆盖约 30~40% 的采样请求）。与 Route C 配合可实现 1+1 > 2 的效果。

---

## Route E: 候选集重用 (Candidate Reuse)

### 思想 (Idea)
在同一个 epoch 内的连续 steps 之间，保留上一 step 构建的候选负样本集合（或部分集合），直接重用于当前 step。因为实体在连续 steps 中经常重复出现（batch 从全实体集中抽样，相邻 batch 间有高重叠），可避免重复的 Sampling 和 Candidate Build 操作。

### 复杂度 (Complexity)
- **时间复杂度**: 每个 epoch 首次采样 O(batch_size × neg_triple_num)，后续 step 仅 O(cache_hit_ratio × batch_size) 查表
- **空间复杂度**: O(n_entities × reused_candidates) → 若缓存所有实体的候选集与 Route D 类似，约为 **MB 级别**

### 优点 (Pros)
- ✅ **消除重复采样**：连续 steps 间实体重叠率可达 60~80%（基于当前 batch_size=5000 / 14,541 total entities），可大幅降低采样频率
- ✅ **实现简单**：可以在现有 DataLoader 上增加一个 `prev_candidates` 缓存字典，只需少量修改
- ✅ **低风险**：不改变负采样算法本身，仅增加重用逻辑，不会引入精度下降

### 缺点 (Cons)
- ❌ **Step 间分布偏移**：相邻 step 的实体分布可能变化，重用候选集可能引入采样分布偏差
- ❌ **缓存体积**：若 batch 内实体 5000 × 150 负样本 × 2 = 1.5M 个 ID → 约 6MB/step 缓存，多 step 累积可能膨胀
- ❌ **epoch 边界失效**：epoch 切换时所有候选集必须重建（数据被 shuffle）
- ❌ **收益递减**：重用次数越多，候选集偏差越大，采样质量下降

### 实现难度 (Implementation Difficulty)
**Low-Medium** — 现有 DataLoader 结构支持增量修改，侵入性低。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**10~20%**（假设 50% 的采样请求被重用替代）。收益稳定但有限，适合作为快速见效的低成本优化。

---

## Route F: 近似碰撞检查 (Approximate Collision, Bloom Filter)

### 思想 (Idea)
用 Bloom Filter（布隆过滤器）替代当前基于 Python `set` 的精确碰撞检查。Bloom Filter 使用 k 个哈希函数在位数组中标记元素存在性，可在 **O(k)** 时间内完成碰撞检查，其中 k 通常为 3~7。以可容忍的假阳性率（约 1~5%）换取 10~100x 的查找速度提升。

### 复杂度 (Complexity)
- **时间复杂度**: 碰撞检查从 set 的 O(1) 带哈希计算 → Bloom Filter 的 O(k) 位检查 + 哈希计算，但实际 CPU 指令少 3~5 倍（set 有链表遍历开销）
- **空间复杂度**: Bloom Filter 位数组 → 支持 14,541 实体 × 150 负样本 × 假阳性率 1% 时约 **~1MB**

### 优点 (Pros)
- ✅ **直接攻击 Collision Check 瓶颈（占负采样 14.2%，R=0.8640）**：Bloom Filter 在查表密集型场景下比 Python set 快 3~10 倍
- ✅ **空间效率极高**：1MB 即可覆盖全部实体集，远小于 set 的内存开销
- ✅ **实现可插拔**：可以封装为 `set` 接口的 drop-in replacement，只需修改 collision check 处的几行代码
- ✅ **与 GPU 采样兼容**：Bloom Filter 也可在 GPU 端实现（位数组 + CUDA 核函数）

### 缺点 (Cons)
- ❌ **假阳性 → 模型精度下降**：误将有效负样本判为碰撞而丢弃，相当于减少了有效负样本数，可能导致欠采样问题
- ❌ **无法删除元素**：Bloom Filter 不支持删除操作（除非使用 Counting Bloom Filter，增加 4x 空间），不适合动态更新的实体集
- ❌ **假阳性率与空间博弈**：要降低假阳性率需要更大的位数组，需要在速度和精度之间调优
- ❌ **如果 train/valid/test 共享实体 ID，假阳性会在验证和测试阶段累积误差**

### 实现难度 (Implementation Difficulty)
**Low** — 使用 Python 的标准 `pybloom` 或自实现约 50~100 行代码。可完全封装，现有代码变更量极小。

### 预期收益 (Expected Gain)
对当前负采样 60% 瓶颈的理论消除上限：**8~12%**（B3 Collision Check 占 14.2%，加速 3~10x 后 Collision 降至 ~1~5ms/step）。收益有限但实现代价极低，性价比高。

---

## 跨 Route 对比汇总表

| Route | 核心思想 | 目标子阶段 | 理论消除上限 | 实现难度 | 精度风险 | 与主要瓶颈的对应关系 |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **A** | GPU 端采样 | B1(42.3%) + B3(14.2%) | **50~60%** | High | 低 | ✅ 直接命中 #1 瓶颈 |
| **B** | 批次内负采样 | B1+B2+B3 完全消除 | ~100%（理论） | Low | **极高** | ⚠️ 完全重写范式 |
| **C** | 基于度采样 | B1(42.3%) via Hub(0.8163) | **15~25%** | Medium | 低~中 | ✅ 精准打击 Hub 效应 |
| **D** | Hub 缓存 | B1+B2 for Top Hub | **15~20%** | Medium-High | 低 | ✅ 精准打击 Hub 效应 |
| **E** | 候选集重用 | B1+B2 | **10~20%** | Low-Medium | 低 | 辅助性优化 |
| **F** | Bloom Filter 碰撞 | B3(14.2%) | **8~12%** | **Low** | 中 | ✅ 精准打击 #2 瓶颈 |

---

## 推荐组合策略

基于 Phase 1~4 的 Profiling 数据，推荐的优化优先级为：

### 🥇 Tier 1：核心路径（大胆实施）
1. **Route A (GPU Sampling)** — 直接消除 B1 42.3% 瓶颈，是获得最大收益的必经之路
2. **Route C (Degree-aware Sampling)** — 作为 Route A 的上层调度策略，解决 Hub 不均衡问题（R=0.8163）

### 🥈 Tier 2：辅助加速（低成本高回报）
3. **Route F (Bloom Filter Collision)** — 针对 B3 14.2% 瓶颈，实现难度最低，适合作为快速见效的附加优化

### 🥉 Tier 3：储备方案（等待需求明确）
4. **Route D (Hub Cache)** — 如果在 Route A+C 后仍有 Hub 相关卡点，再引入缓存层
5. **Route E (Candidate Reuse)** — 低风险但收益有限，可随时启用
6. **Route B (Batch Sampling)** — **不推荐**：O(batch_size²) 显存爆炸和精度退化风险超出收益

---

*本文档基于 Phase 1（整体 Profiling）、Phase 2（负采样子阶段分解）、Phase 3（Hub 相关性分析）、Phase 4（稳定性验证）的真实实验数据撰写。所有收益预估基于当前 batch_size=5000, neg_triple_num=150 的配置。*