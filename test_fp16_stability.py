import torch
import torch.nn as nn

def test_stability():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("No GPU found, skipping FP16 test.")
        return

    # 模拟 TransE 距离计算: ||h + r - t||
    dim = 128
    h = torch.randn(1, dim, device=device, requires_grad=True)
    r = torch.randn(1, dim, device=device, requires_grad=True)
    t_pos = h + r + torch.randn(1, dim, device=device) * 1e-6 # 极近
    t_neg = h + r + torch.randn(1, dim, device=device) * 1e4   # 极远

    def compute_loss(h, r, t):
        return torch.norm(h + r - t, p=2, dim=-1)

    print("Testing FP16 stability...")
    try:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss_pos = compute_loss(h, r, t_pos)
            loss_neg = compute_loss(h, r, t_neg)
            loss = loss_pos.mean() + loss_neg.mean()
        
        loss.backward()
        
        print(f"Loss: {loss.item()}")
        print(f"Grad Max: {h.grad.max().item()}, Grad Min: {h.grad.min().item()}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Result: Unstable (NaN/Inf detected)")
        elif (h.grad.abs() < 1e-7).float().mean() > 0.5:
            print("Result: Unstable (Underflow detected)")
        else:
            print("Result: Stable")
            
    except Exception as e:
        print(f"Result: Unstable (Exception: {e})")

if __name__ == "__main__":
    test_stability()
