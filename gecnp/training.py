"""Shared, seed-matched episode sampling and predictive scoring."""
import math
import random
from typing import Dict
import numpy as np
import torch


def mixture_nll(
    output: Dict[str, torch.Tensor], target_y: torch.Tensor
) -> torch.Tensor:
    y = target_y.unsqueeze(-1)
    z = (y - output["mu"]) / output["sigma"]
    log_component = (
        output["log_pi"]
        - output["sigma"].log()
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * z.square()
    )
    return -torch.logsumexp(log_component, dim=-1).mean()


def sample_episode(context_s, context_x, context_y):
    """Sample disjoint context/target sets; half use a buffered spatial split."""
    n = context_s.shape[1]
    if np.random.rand() < 0.5:
        perm = torch.randperm(n, device=context_s.device)
        nc = max(4, int(np.random.uniform(0.25, 0.50) * n))
        ci, rest = perm[:nc], perm[nc:]
        nt = max(4, int(np.random.uniform(0.70, 1.0) * len(rest)))
        ti = rest[:nt]
    else:
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        direction = context_s.new_tensor([np.cos(angle), np.sin(angle)])
        projection = (context_s[0] * direction).sum(dim=-1)
        qlo, qhi = torch.quantile(projection, projection.new_tensor([0.35, 0.65]))
        left = torch.where(projection <= qlo)[0]
        right = torch.where(projection >= qhi)[0]
        if np.random.rand() < 0.5:
            left, right = right, left
        # Coincident quantiles make the tails overlap; fall back to a disjoint
        # random episode rather than violating the context/target boundary.
        if bool((qlo < qhi).item()) and min(len(left), len(right)) >= 4:
            nc = max(4, int(np.random.uniform(0.50, 1.0) * len(left)))
            nt = max(4, int(np.random.uniform(0.70, 1.0) * len(right)))
            ci = left[torch.randperm(len(left), device=context_s.device)[:nc]]
            ti = right[torch.randperm(len(right), device=context_s.device)[:nt]]
        else:
            perm = torch.randperm(n, device=context_s.device)
            nc = max(4, int(0.35 * n))
            ci, ti = perm[:nc], perm[nc:]
    return (
        context_s[:, ci],
        context_x[:, ci],
        context_y[:, ci, None],
        context_s[:, ti],
        context_x[:, ti],
        context_y[:, ti],
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score(output, y, y_std):
    err = output["mean"].squeeze(0) - y.squeeze(0)
    mse = err.square().mean().item()
    denom = (y.squeeze(0) - y.squeeze(0).mean()).square().sum().item()
    return {
        "mae": err.abs().mean().item() * y_std,
        "mse": mse * y_std**2,
        "r2": 1.0 - err.square().sum().item() / max(denom, 1e-12),
        "nll": mixture_nll(output, y).item() + math.log(y_std),
        "mean_predictive_variance": output["variance"].mean().item() * y_std**2,
        "coverage_90": (
            (y >= output["mean"] - 1.645 * output["std"])
            & (y <= output["mean"] + 1.645 * output["std"])
        )
        .float()
        .mean()
        .item(),
        "mean_interval_width_90": (2 * 1.645 * output["std"]).mean().item() * y_std,
    }
