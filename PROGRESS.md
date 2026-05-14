# MuKG & RAG 项目实验航海日志

> **Cline 指令**: 在开始任务前，必须全文读取此文档。任务结束或取得阶段性进展后，必须按照格式更新此文档。

---

## 1. 当前活动目标 (Active Task)
- [x] ~~FP16 稳定性排查 (WSL)~~
- [ ] **新目标**: 在 WSL 上端到端跑通 TransE + FB15k-237 小规模训练 (FP16混合精度)
- [ ] **当前瓶颈**: 需要确认 kge_trainer.py 的 FP16 整合是否就绪；首次验证完整训练管线
- [ ] **预期结果**: TransE 在小 batch (batch=128, dim=128) 下 Loss 正常下降，FP16 节省 ~30% 显存

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

## 5. 待办事项 (Backlog)
- [x] ~~FP16 稳定性验证 (已完成，结果: ✅ 稳定)~~
- [ ] 在 WSL 上跑通 TransE + FB15k-237 小 batch 端到端训练 (FP16)
- [ ] 记录 FP16 vs FP32 的显存占用与 Loss 曲线对比
- [ ] 将 FP16 训练结果写入 MCP Memory 图谱
- [ ] 后续在 Node 4 上进行全量 FP16 验证
