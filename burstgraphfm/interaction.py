"""Differentiable learned non-local Burst relations (not pseudo-labels).

Selection is discrete Top-k, but selected relation scores remain differentiable.
Temporal neighbors are excluded; padding is strictly masked.
"""
import torch
from torch import nn
from torch.nn import functional as F

class LearnedInteraction(nn.Module):
    def __init__(self, hidden=128, heads=4, top_k=3, min_distance=2):
        super().__init__()
        self.top_k = top_k
        self.min_distance = min_distance
        self.norm = nn.LayerNorm(hidden)
        self.pair_net = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden), nn.GELU(),
            nn.Linear(hidden, heads),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, valid):
        b, n, d = h.shape
        idx = torch.arange(n, device=h.device)
        distance = (idx[:, None] - idx[None, :]).abs()
        eligible = (distance >= self.min_distance)[None] & valid[:, :, None] & valid[:, None, :]
        z = self.norm(h)
        zi = z[:, :, None, :].expand(-1, -1, n, -1)
        zj = z[:, None, :, :].expand(-1, n, -1, -1)
        delta = (zi - zj).abs()
        pair = torch.cat((delta, zi * zj, (zi + zj)*0.5,
                          (distance.to(h.dtype)/max(n-1, 1))[None, :, :, None].expand(b,-1,-1,-1)), dim=-1)
        raw = self.pair_net(pair)
        raw = 0.5 * (raw + raw.transpose(1, 2))
        score = raw.mean(dim=-1)
        selected = torch.zeros_like(eligible)
        if n > self.min_distance:
            k = min(self.top_k, n)
            _, index = score.masked_fill(~eligible, -torch.inf).topk(k, dim=-1)
            selected.scatter_(2, index, True)
            selected = (selected | selected.transpose(1,2)) & eligible
        bias = self.scale.tanh() * torch.tanh(raw) * selected.unsqueeze(-1)
        return bias, selected, score.masked_fill(~eligible, 0.), eligible

def relation_stability(score_a, score_b, selected_a, selected_b):
    mask = selected_a | selected_b
    if not mask.any():
        return (score_a.sum() + score_b.sum()) * 0.0
    return F.smooth_l1_loss(score_a[mask], score_b[mask])
