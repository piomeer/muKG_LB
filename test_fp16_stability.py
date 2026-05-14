"""
FP16 vs FP32 稳定性对比测试 (多轮统计版)

测试目标:
  1. 对比 FP16 和 FP32 在 TransE 距离计算中的 Loss 数值差异
  2. 统计 NaN/Inf 出现率
  3. 检测梯度下溢比例

用法:
  python test_fp16_stability.py

环境约束 (WSL / RTX 4060 8GB):
  - 仅用于小 batch Debug 验证
  - 禁止盲目放大 batch_size 或 embedding_dim
"""

import torch
import torch.nn as nn
import numpy as np


def compute_transe_loss(h, r, t_pos, t_neg, margin=1.0):
    """
    模拟 TransE 损失:
      loss = ||h + r - t_pos||^2 + max(0, margin - ||h + r - t_neg||)^2
    确保正例距离小，负例距离被 margin 约束，Loss > 0 有实际意义。
    """
    pos_dist = torch.norm(h + r - t_pos, p=2, dim=-1)
    neg_dist = torch.norm(h + r - t_neg, p=2, dim=-1)
    # 正例: 越小越好；负例: 被 margin 推远
    loss = pos_dist.pow(2) + torch.clamp(margin - neg_dist, min=0).pow(2)
    return loss.mean()


def run_single_test(dtype, device, dim=128, batch_size=128, num_iterations=100):
    """
    在指定精度下运行多轮随机测试。

    Returns:
        dict: {
            'loss_mean': float,
            'loss_std': float,
            'nan_inf_ratio': float,   # NaN 或 Inf 出现的轮次比例
            'underflow_ratio': float,  # 梯度绝对值 < 1e-7 的比例
        }
    """
    print(f"\n{'='*60}")
    print(f"  Running {dtype} test: dim={dim}, batch={batch_size}, iters={num_iterations}")
    print(f"{'='*60}")

    loss_values = []
    nan_inf_count = 0
    underflow_count = 0

    for i in range(num_iterations):
        # 随机生成 h, r
        h = torch.randn(batch_size, dim, device=device, requires_grad=True)
        r = torch.randn(batch_size, dim, device=device, requires_grad=True)
        # 正例: 非常接近 h+r (小噪声)
        t_pos = h + r + torch.randn(batch_size, dim, device=device) * 0.05
        # 负例: 以 margin/2 为半径偏移，使部分负例落入 margin 约束区内
        t_neg = h + r + torch.randn(batch_size, dim, device=device) * (margin * 0.4)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dtype == 'fp16')):
            loss = compute_transe_loss(h, r, t_pos, t_neg, margin=margin)

        # 检查 NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            nan_inf_count += 1
            loss_values.append(float('nan'))
            continue

        loss.backward()

        # 检查梯度下溢
        grad_abs = h.grad.abs()
        underflow_ratio_batch = (grad_abs < 1e-7).float().mean().item()
        if underflow_ratio_batch > 0.5:
            underflow_count += 1

        loss_values.append(loss.item())

        # 清零梯度 (重要!)
        h.grad.zero_()
        r.grad.zero_()

        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1:4d}/{num_iterations}, Loss={loss.item():.6f}")

    # 统计
    valid_losses = [v for v in loss_values if not np.isnan(v)]
    loss_mean = np.mean(valid_losses) if valid_losses else float('nan')
    loss_std = np.std(valid_losses) if valid_losses else float('nan')

    result = {
        'dtype': dtype,
        'loss_mean': loss_mean,
        'loss_std': loss_std,
        'nan_inf_ratio': nan_inf_count / num_iterations * 100,
        'underflow_ratio': underflow_count / num_iterations * 100,
    }

    print(f"\n  --- {dtype.upper()} 汇总 ---")
    print(f"  Valid iterations : {len(valid_losses)}/{num_iterations}")
    print(f"  Loss Mean ± Std  : {loss_mean:.6f} ± {loss_std:.6f}")
    print(f"  NaN/Inf 比例     : {result['nan_inf_ratio']:.2f}%")
    print(f"  梯度下溢比例     : {result['underflow_ratio']:.2f}%")

    return result


def compare_results(fp16_result, fp32_result):
    """打印 FP16 vs FP32 对比报告"""
    print(f"\n{'='*60}")
    print(f"  FP16 vs FP32 对比报告")
    print(f"{'='*60}")

    # Loss 差异
    loss_diff_pct = abs(fp16_result['loss_mean'] - fp32_result['loss_mean']) / fp32_result['loss_mean'] * 100
    print(f"  Loss 差异        : {loss_diff_pct:.2f}% (FP16={fp16_result['loss_mean']:.6f}, FP32={fp32_result['loss_mean']:.6f})")

    # NaN/Inf 风险
    print(f"  NaN/Inf 风险      : FP16={fp16_result['nan_inf_ratio']:.2f}%, FP32={fp32_result['nan_inf_ratio']:.2f}%")

    # 梯度下溢风险
    print(f"  梯度下溢风险     : FP16={fp16_result['underflow_ratio']:.2f}%, FP32={fp32_result['underflow_ratio']:.2f}%")

    # 综合结论
    print(f"\n  --- 综合结论 ---")

    risk_level = "✅ 稳定"
    if fp16_result['nan_inf_ratio'] > 1.0:
        risk_level = "❌ 不稳定 (NaN/Inf 比例 > 1%)"
    elif fp16_result['underflow_ratio'] > 10.0:
        risk_level = "⚠️ 需要关注 (梯度下溢比例 > 10%)"
    elif loss_diff_pct > 5.0:
        risk_level = "⚠️ 需要关注 (Loss 差异 > 5%)"

    print(f"  FP16 稳定性评估  : {risk_level}")
    print(f"{'='*60}\n")
    return risk_level


if __name__ == "__main__":
    print("FP16 vs FP32 稳定性测试 (TransE 距离模拟)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("ERROR: 未检测到 GPU，无法进行 FP16 测试。")
        exit(1)

    print(f"  设备         : {device}")
    print(f"  GPU 名称     : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM 总量    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 超参数
    DIM = 128
    BATCH_SIZE = 128
    NUM_ITERATIONS = 100
    global margin
    margin = 1.0

    # --- FP16 测试 ---
    fp16_result = run_single_test('fp16', device, dim=DIM, batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS)

    # --- FP32 测试 (对照) ---
    fp32_result = run_single_test('fp32', device, dim=DIM, batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS)

    # --- 对比报告 ---
    risk_level = compare_results(fp16_result, fp32_result)

    # --- 显存统计 ---
    max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"  峰值显存占用 : {max_mem:.1f} MB")
    print("=" * 60)
