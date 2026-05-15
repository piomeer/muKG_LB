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

## 6. 待办事项 (Backlog)
- [ ] **[已休眠/Paused] 系统优化思考**: 在 WSL 上端到端跑通 TransE + FB15k-237 小规模训练 (FP16混合精度)。目前已完成稳定性验证，待进行完整管线测试。
- [x] ~~FP16 稳定性验证 (已完成，结果: ✅ 稳定)~~
- [ ] 在 WSL 上跑通 TransE + FB15k-237 小 batch 端到端训练 (FP16)
- [ ] 记录 FP16 vs FP32 的显存占用与 Loss 曲线对比
- [ ] 将 FP16 训练结果写入 MCP Memory 图谱
- [ ] 后续在 Node 4 上进行全量 FP16 验证
