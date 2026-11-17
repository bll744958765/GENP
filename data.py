from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


DATA_DIR = Path(
    os.environ.get("GENP_DATA_DIR", Path(__file__).resolve().parent / "data")
).resolve()
DATASETS = {
    "chengdu": "Chengdu_housing.csv",
    "election": "election result.csv",
    "california": "fetch_california_housing.csv",
    "synthetic": "generated_data_with_target_T=3.csv",
}


@dataclass
class DataSplit:
    context_s: torch.Tensor
    context_x: torch.Tensor
    context_y: torch.Tensor
    validation_s: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    test_s: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    context_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    coord_dim: int
    attr_dim: int
    target_mean: float
    target_std: float


def load_split(
    name: str, seed: int = 1, mode: str = "random", device: str = "cpu"
) -> DataSplit:
    frame = pd.read_csv(DATA_DIR / DATASETS[name])
    arr = frame.to_numpy(dtype=np.float64)
    ids = arr[:, 0]
    s = arr[:, 1:3]
    x = arr[:, 3:-1].copy()
    # California Housing's per-household ratios have extreme right tails from
    # small household denominators; log1p these ratios and counts before scaling.
    if name == "california":
        for col in (2, 3, 4, 5):
            x[:, col] = np.log1p(np.maximum(x[:, col], 0.0))
    y = arr[:, -1]

    if mode == "random":
        ids_dev, ids_test = train_test_split(
            np.arange(len(y)), test_size=0.85, random_state=seed
        )
        ids_context, ids_val = train_test_split(
            ids_dev, train_size=2 / 3, random_state=seed + 10000
        )
    elif mode == "spatial_block":
        # Hold out a contiguous upper-right quadrant; select a sparse context pool
        # only from the remaining region, then reserve validation points there.
        med_s = np.median(s, axis=0)
        test_mask = (s[:, 0] >= med_s[0]) & (s[:, 1] >= med_s[1])
        ids_test = np.flatnonzero(test_mask)
        ids_dev = np.flatnonzero(~test_mask)
        desired = max(8, int(round(0.10 * len(y))))
        desired = min(desired, max(1, len(ids_dev) - 1))
        ids_context, ids_val = train_test_split(
            ids_dev, train_size=desired, random_state=seed + 10000
        )
    else:
        raise ValueError("mode must be random or spatial_block")

    # The context pool is 10% of the full dataset in the random protocol.
    # For block protocol it is also about 10% of the full dataset, drawn only
    # from the non-test domain; validation uses the remaining non-test rows (capped).
    if mode == "random":
        desired = max(8, int(round(0.10 * len(y))))
        if len(ids_context) > desired:
            ids_context, ids_val = train_test_split(
                ids_dev, train_size=desired, random_state=seed + 10000
            )

    # Cap model-selection evaluation size for the large spatial-block protocol.
    if len(ids_val) > 2048:
        _, ids_val = train_test_split(
            ids_val, test_size=2048, random_state=seed + 20000
        )

    if name == "california":
        # Train-only winsorization limits influence from census-derived ratios
        # with very small household denominators; no observations are removed.
        for col in (2, 3, 5):  # AveRooms, AveBedrms, AveOccup
            upper = np.quantile(x[ids_context, col], 0.99)
            x[:, col] = np.minimum(x[:, col], upper)
    s_mu, s_sd = s[ids_context].mean(0), s[ids_context].std(0)
    x_mu, x_sd = x[ids_context].mean(0), x[ids_context].std(0)
    y_mu, y_sd = y[ids_context].mean(), y[ids_context].std()
    s_sd[s_sd < 1e-8] = 1.0
    x_sd[x_sd < 1e-8] = 1.0
    y_sd = y_sd if y_sd > 1e-8 else 1.0
    s = (s - s_mu) / s_sd
    x = (x - x_mu) / x_sd
    y = (y - y_mu) / y_sd

    def tensor(a, ix):
        return torch.as_tensor(a[ix], dtype=torch.float32, device=device).unsqueeze(0)

    return DataSplit(
        tensor(s, ids_context),
        tensor(x, ids_context),
        tensor(y, ids_context),
        tensor(s, ids_val),
        tensor(x, ids_val),
        tensor(y, ids_val),
        tensor(s, ids_test),
        tensor(x, ids_test),
        tensor(y, ids_test),
        ids[ids_context],
        ids[ids_val],
        ids[ids_test],
        2,
        x.shape[1],
        float(y_mu),
        float(y_sd),
    )
