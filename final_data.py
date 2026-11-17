"""Final-study preprocessing; historical development splits remain reproducible.

Nominal Chengdu labels use context-fitted one-hot columns, including an unknown
category. Continuous attributes retain the context-only transformations in data.py.
Indicator columns stay in {0,1}; standardizing them would distort rare categories.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from data import DATA_DIR, DATASETS, load_split

PREPROCESSING_VERSION = "final-v2-context-onehot"


def load_final_split(name, seed=1, mode="random", device="cpu"):
    split = load_split(name, seed, mode, device)
    if name != "chengdu":
        return split
    frame = pd.read_csv(DATA_DIR / DATASETS[name]).set_index("Id")
    attributes = list(frame.columns[2:-1])
    nominal = [c for c in attributes if c.endswith("_Label")]
    categories = {
        c: sorted(frame.loc[split.context_ids, c].unique().tolist()) for c in nominal
    }

    def encode(ids, previous):
        raw = frame.loc[ids]
        columns = []
        for i, column in enumerate(attributes):
            if column in categories:
                values = raw[column].to_numpy()
                known = categories[column]
                columns.extend([(values == k).astype(np.float32) for k in known])
                columns.append((~np.isin(values, known)).astype(np.float32))
            else:
                columns.append(previous[0, :, i].detach().cpu().numpy())
        return torch.as_tensor(np.stack(columns, -1), device=device).unsqueeze(0)

    cx = encode(split.context_ids, split.context_x)
    vx = encode(split.validation_ids, split.validation_x)
    tx = encode(split.test_ids, split.test_x)
    return replace(
        split, context_x=cx, validation_x=vx, test_x=tx, attr_dim=cx.shape[-1]
    )
