import torch
import torch.nn as nn

class DeltaLoss(nn.Module):
  def __init__(self, lambda_delta):
    super().__init__()
    self.lambda_delta = lambda_delta

  def forward(self, prev_weight, W):
    if prev_weight is None:
      return torch.tensor(0.0, device=W.device, dtype=W.dtype)
    
    delta = W - prev_weight
    loss = self.lambda_delta * torch.norm(delta, p = 'fro')**2
    return loss