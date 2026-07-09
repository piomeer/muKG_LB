# Candidate Algorithm Design (Phase 5 - Step 3)

> **基于双模成本定律 (Dual-Regime Cost Law) 与先验 Degree 期望模型设计**

## Algorithm A: Degree-Driven Bin Packing (DDBP) 负载均衡器
**对应路线**: Route C (基于度的调度)
* **核心思想**: 
  在 DDP 模式下，将 Dataset 划分为 mini-batch 之前，利用静态的 `degree_table` 计算每个样本的理论重试期望 $E[retry] = d(e)/(C-d(e))$。使用贪心装箱算法（贪心降序适配，First Fit Decreasing），将高期望（Hub）样本与低期望（长尾）样本混合，确保分发给各个 GPU 的 Batch 拥有极其接近的总 Weight。
* **复杂度**: 时间复杂度 $O(N \log N)$（仅需在每个 epoch 开始前对样本权重排序）。空间复杂度 $O(|V|)$（维护度数表）。
* **优点**: 完全不侵入底层的 C++/CUDA 采样逻辑，纯 Python 层实现，彻底解决 DDP 的 AllReduce 木桶效应。
* **缺点**: 会破坏训练数据原有的随机打乱（Shuffle）分布，可能在极端情况下引发模型收敛轨迹的轻微改变。
* **预期收益**: 消除多卡同步等待时间，预期 DDP 扩展效率（Scaling Efficiency）提升至 90% 以上。

---

## Algorithm B: Bloom Filter Fast-Reject (BFFR) 近似碰撞检查
**对应路线**: Route F (近似碰撞检查)
* **核心思想**: 
  针对 Regime 2 中高昂的 $51.8ms \cdot avg_{retry}$ 乘数，废弃原生的 Python `set.difference()` 或 `in set` 检查。为图中的每个实体预构建一个极小的 Bloom Filter（或使用全局的哈希位图）来存储其真实邻居。
  *注：Bloom Filter 存在假阳性（False Positive），即可能将一个合法的负样本误判为“真实邻居”并触发重试。但在 KGE 中这是绝对安全的！只会略微增加重试次数，但绝不会产生致命的假阴性（False Negative，即把真样本当假样本喂给模型）。*
* **复杂度**: 碰撞检查时间从 $O(|Set|)$ 降至严格的 $O(k)$（k 为哈希函数个数，通常为 3 左右）。空间复杂度增加几十 MB。
* **优点**: 极大地压低 Regime 2 的时间上限，对模型精度 **0 影响**。
* **缺点**: Python 原生实现 Bloom Filter 可能因解释器开销导致加速不明显，可能需要借助 `cython` 或简单的 PyTorch Tensor 位运算来实现。
* **预期收益**: 将 $51.8ms$ 的常数项降低至 $5~10ms$ 级别，总体采样耗时降低约 15%。

---

## Algorithm C: Dual-Regime Adaptive Sampler (DRAS) 自适应双通道采样
**对应路线**: Route A/C 混合体
* **核心思想**: 
  直接基于我们的 Cost Model 结论实施工程分流（If-Else 路由）：
  - **Path 1 (Regime 1)**: 当 candidate pool $\geq 5000$ 时，放弃所有复杂的优化，直接调用原生的均匀随机采样（因为成本恒定在 295ms，优化无意义）。
  - **Path 2 (Regime 2)**: 当 candidate pool $< 5000$（窄化池）且触发 Hub 节点时，切换到预构建的**逆向索引缓存（Inverted Index Cache）**，直接从不在邻居列表中的连续内存块中切片（Slice）负样本，彻底绕过随机数生成与碰撞检查。
* **复杂度**: 初始化需要构建全图的补集缓存，空间复杂度极高 $O(|V|^2)$，但在 $|V|=14541$ 的 FB15k-237 上，位图矩阵仅需约 $26MB$ 显存。
* **优点**: 完美契合数据规律，理论加速比最极端的方案。
* **缺点**: 实现极其复杂，需要重写 `kge_trainer.py` 的数据流转层，且难以扩展到千万级节点的超大图谱（如 Wikidata5M）。
* **预期收益**: Regime 2 的采样耗时趋近于 $0ms$。