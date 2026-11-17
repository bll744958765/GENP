"""Conventional mean-aggregating CNP controls with the same MDN head family.

No unused geographic/gating modules are included in the parameter count.
"""
import torch
from torch import nn
from torch.nn import functional as F
from gecnp.model import net


class MeanCNP(nn.Module):
    def __init__(
        self, coord_dim, attr_dim, hidden=64, components=2, encoder_width=None
    ):
        super().__init__()
        width = encoder_width or hidden
        self.context_encoder = net(coord_dim + attr_dim + 1, width, hidden)
        self.query_encoder = net(coord_dim + attr_dim, width, hidden)
        self.information_fusion = net(2 * hidden, width, hidden)
        self.prediction_head = net(hidden, hidden, 3 * components)
        self.components = components

    def forward(self, context_s, context_x, context_y, target_s, target_x):
        c = self.context_encoder(torch.cat((context_s, context_x, context_y), -1)).mean(
            1, keepdim=True
        )
        c = c.expand(-1, target_s.shape[1], -1)
        q = self.query_encoder(torch.cat((target_s, target_x), -1))
        representation = self.information_fusion(torch.cat((c, q), -1))
        mu, raw_sigma, logits = self.prediction_head(representation).chunk(3, -1)
        sigma = F.softplus(raw_sigma) + 0.001
        lp = logits.log_softmax(-1)
        pi = lp.exp()
        mean = (pi * mu).sum(-1)
        variance = (
            (pi * (sigma.square() + (mu - mean[..., None]).square()))
            .sum(-1)
            .clamp_min(1e-8)
        )
        return dict(
            mu=mu,
            sigma=sigma,
            log_pi=lp,
            mean=mean,
            variance=variance,
            std=variance.sqrt(),
        )


def matched_cnp_width(coord_dim, attr_dim, target_parameters, components=2):
    # Count formulas, rather than constructing candidate random networks.
    def mlp_count(n, w, o):
        return (n + 1) * w + (w + 1) * w + (w + 1) * o

    def count(w):
        return (
            mlp_count(coord_dim + attr_dim + 1, w, 64)
            + mlp_count(coord_dim + attr_dim, w, 64)
            + mlp_count(128, w, 64)
            + mlp_count(64, 64, 3 * components)
        )

    return min(range(16, 513), key=lambda w: abs(count(w) - target_parameters))
