# MuKG & RAG 项目实验航海日志

> **Cline 指令**: 在开始任务前，必须全文读取此文档。任务结束或取得阶段性进展后，必须按照格式更新此文档。

---

## 1. 当前活动目标 (Active Task)
- [ ] **目标**: (例如：在 Node 6 上跑通 DDP 分布式训练)
- [ ] **当前瓶颈**: (例如：多卡环境下的 nccl 库通讯超时)
- [ ] **预期结果**: (例如：Hit@10 达到 0.45+)

## 2. 跨环境状态快照 (Environment Snapshots)
| 环境 | 当前分支 | 最后成功状态 | 待解决问题 |
| :--- | :--- | :--- | :--- |
| **WSL** | `main` | 代码逻辑 Debug 完成 | 仅能跑小 batch 验证 |
| **Node 4** | `production` | 全量数据已跑通 (Batch: 128) | 无 |
| **Node 6** | `production` | 待配置 |   |

## 3. 黄金超参数与成功经验 (Golden Records)
*注意：仅记录已验证成功的设置，避免重复踩坑。*

- **Model**: `RotatE` | **Dataset**: `FB15k-237` | **Environment**: `Node 4`
  - **LR**: `0.0001` | **Batch**: `256` | **Result**: `MRR: 0.33`
  - **备注**: 必须配合 `Adam` 优化器，显存占用约 32GB。

## 4. 失败实验记录 (Failure Logs)
- [2026-05-14] [WSL]: 尝试 Batch=512 -> **OOM 崩溃**。
- [2026-05-14] [Node 6]: 尝试升级 PyTorch 2.0 -> **MuKG 算子不兼容**，已回滚。

---
## 5. 待办事项 (Backlog)


