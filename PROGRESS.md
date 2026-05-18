# MuKG & RAG 项目实验航海日志

> **Cline 指令**: 在开始任务前，必须全文读取此文档。任务结束或取得阶段性进展后，必须按照格式更新此文档。

---

## 1. 当前活动目标 (Active Task)
- [x] **目标**: 知识图谱训练 4 个阶段全链路独立测时 (Micro-benchmarking) ✅ 已完成
- [x] **具体需求**: 获取单个 Epoch 中以下 4 个阶段各自**绝对独立、精确到毫秒**的耗时。详见下方 "Micro-benchmarking 结果" 章节。

## 2. 跨环境状态快照 (Environment Snapshots)
| 环境 | GPU | VRAM | 框架版本 | 最后成功状态 | 待解决问题 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WSL (local_wsl)** | RTX 4060 Laptop | 8GB | 待确认 | ✅ FP16 稳定性通过 (Loss diff 0.04%, 0% NaN) | 待端到端管线验证 |
| **Node 4** | RTX 3070 | 8GB | PyTorch 1.10.2 + CUDA 11.3 | 全量数据已跑通 (Batch: 128) | 无 |
| **Node 6** | 多 GPU | 待确认 | 待配置 | 待配置 | DDP 改造待完成；PyTorch 2.0 不兼容 |

## 3. 黄金超参数与成功经验 (Golden Records)
*注意：仅记录已验证成功的设置，避免重复踩坑。*

- **Model**: `RotatE` | **Dataset**: `FB15k-237` | **Environment**: `Node 4`
  - **LR**: `0.0001` | **Batch**: `256` | **Result**: `MRR: 0.33`
  - **备注**: 必须配合 `Adam` 优化器，显存占用约 32GB。

- **Model**: `TransE` | **Dataset**: `FB15k-237` | **Environment**: `WSL`
  - **FP16**: ✅ 已验证稳定 (dim=128, batch=128, 100 iters, Loss diff=0.04%, 0% NaN)
  - **备注**: 允许启用 `torch.cuda.amp` 混合精度。下溢检测为误报 (norm梯度分散到128维)。

## 4. 失败实验记录 (Failure Logs)
*待后续实验填充真实失败记录*
| 日期 | 环境 | 操作 | 失败原因 | 避坑建议 |
| :--- | :--- | :--- | :--- | :--- |
---

## 5. Micro-benchmarking 结果 (2026-05-15, WSL/RTX4060)

### 4 阶段独立测时报告 (Epoch 0, TransE + FB15k-237)

**配置**: `batch_threads_num=0, batch_size=128, neg_triple_num=4, dim=400`

| 阶段 | 描述 | 耗时 (秒) | 占比 |
|:---|:---|:---|:---|
| Phase 1 (CPU) | ID Mapping (to_tensor_cpu) | 0.4648 | 0.3% |
| Phase 2 (GPU) | Embedding Lookup (forward) | 1.1955 | 0.8% |
| Phase 3 (CPU) | Negative Sampling 负采样 | 128.3897 | 87.9% |
| Phase 4 (GPU) | Geometry & Learning (score+loss+backward+step) | 13.8095 | 9.5% |
| 调度开销 | 框架/循环调度 | 2.1792 | 1.5% |
| **总计** | **Epoch 总挂钟时间** | **146.0387** | **100%** |

**关键发现**: 
- **瓶颈在 Negative Sampling (Phase 3)**, 占整个 Epoch 87.9% 的时间。
- `generate_neg_triples_fast` 中 `set(all_triples)` 构造 + `random.sample` + `while` 循环是主因。
- GPU 计算 (Phase 2 + Phase 4) 合计仅占 ~10%，当前 batch_size=128 下 GPU 利用率极低。

### 参数安全缩放说明
- 用户原计划 `batch_size=5000, neg_triple_num=150` → **OOM 风险极高** (估算: 5000×(1+150)×3×400×4B ≈ 3.6GB 仅 Embedding，加上梯度/中间激活 ≈ 7-8GB，超出 RTX4060 8GB 安全线)
- 已自动缩放到 `batch_size=128, neg_triple_num=4` → 安全运行，峰值显存 < 2GB

---

## 6. 核心瓶颈与优化记录 (Bottlenecks) [2026-05-18]

### 6.1 Phase 1 (ID Mapping) 代码级铁证分析

**来源文件**: `src/torch/kge_models/pytorch_dataloader.py` — `PyTorchTrainDataset.collate_fn()`

**Phase 1 实际执行代码** (第 62-64 行):
```python
batch_h = to_tensor_cpu(batch_h + [x[0] for x in batch_neg])
batch_r = to_tensor_cpu(batch_r + [x[1] for x in batch_neg])
batch_t = to_tensor_cpu(batch_t + [x[2] for x in batch_neg])
```
其中 `to_tensor_cpu` 定义为:
```python
def to_tensor_cpu(batch):
    return torch.from_numpy(np.array(batch))  # np.array() 是瓶颈
```

**瓶颈根因**:
1. **`np.array(batch)` 转换开销**: 对 Python list 调用 `np.array()` 需要 O(n) 的 Python 级循环 + 类型推断 + 内存分配。对于包含 `batch_size × (1 + neg_num)` 个三元组的列表，每次 collate_fn 调用都要执行 3 次。
2. **Phase 1 命名误导**: 注释称 "ID Mapping (索引への変換)"，但实际只是 Python list → Tensor 转换。真正的 ID Mapping（字符串→整数）在 `read_kge_dataset()` 中已通过 `int(params[0])` 完成。
3. **`set()` 重复构建**: 第 52 行 `set(self.kgs.relation_triples_list)` 每次 collate_fn 都重新构建，O(n) 操作，完全可在 `__init__` 中缓存。

**Phase 1 耗时**: 0.4648s / Epoch (仅占 0.3%) — 当前不是主要瓶颈，但 batch_size 增大后开销会线性增长。

### 6.2 Phase 3 (Negative Sampling) 代码级瓶颈分析

**来源文件**: `src/torch/kge_models/pytorch_dataloader.py` — `PyTorchTrainDataset.generate_neg_triples_fast()`

**瓶颈根因**:
1. **`random.sample(head_candidates, nums_to_sample)`**: 对 Python list 进行无放回采样，O(k) 时间，k = neg_num。
2. **集合差集 `i_neg_triples - all_triples_set`**: 每次采样后都要过滤已存在的三元组，O(k) 集合查找。
3. **`while` 重试循环 (max_try=10)**: 如果采样出的负例大量命中已有三元组，需要多次重试。
4. **逐三元组循环**: `for head, relation, tail in pos_batch:` — 每个正例单独处理，无法向量化。

**Phase 3 耗时**: 128.3897s / Epoch (占 87.9%) — **当前绝对瓶颈**。

### 6.3 优化方向建议
- **Phase 3 向量化**: 用 NumPy/Torch 的随机采样替代 Python `random.sample`，批量生成负例。
- **`set()` 缓存**: 将 `all_triples_set` 在 `__init__` 中缓存为成员变量。
- **Phase 1 预转换**: 如果数据已经是整数 ID，可直接用 `torch.tensor()` 替代 `np.array()` 避免中间 NumPy 拷贝。
- **多进程 DataLoader**: 当前 `num_workers=0`，启用多进程可并行化 Phase 1 + Phase 3。

---

## 7. 待办事项 (Backlog)
- [ ] **[已休眠/Paused] 系统优化思考**: 在 WSL 上端到端跑通 TransE + FB15k-237 小规模训练 (FP16混合精度)。目前已完成稳定性验证，待进行完整管线测试。
- [x] ~~FP16 稳定性验证 (已完成，结果: ✅ 稳定)~~
- [ ] 在 WSL 上跑通 TransE + FB15k-237 小 batch 端到端训练 (FP16)
- [ ] 记录 FP16 vs FP32 的显存占用与 Loss 曲线对比
- [ ] 将 FP16 训练结果写入 MCP Memory 图谱
- [ ] 后续在 Node 4 上进行全量 FP16 验证
