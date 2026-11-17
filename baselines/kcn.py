"""Evaluate Gaussian KCN on the same episodic splits as the neural comparators."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from gecnp.data import load_final_split
from baselines.neural_processes import marginal_nll
from gecnp.training import sample_episode, seed_all
from experiments.train import EPOCHS, EPISODES, OUT as FROZEN, training_identity
from experiments.reference_np import write_json
from experiments.sensitivity import PROTOCOLS
from experiments.summarize_supplemental import archive_metrics, summarize

OUT = Path(__file__).resolve().parents[1] / "results" / "kcn_300e30ep"
VARIANT = "kcn"
NEIGHBORS = 5
WIDTHS = (8, 8, 8)
DROPOUT = 0.1


class GaussianKCN(nn.Module):
    """Dense form of GCNConv on weighted query-local graphs.

    Based on the KCN formulation of Appleby et al. (AAAI 2020) and
    the Tufts ML PyTorch reference. The query label is masked.
    """

    def __init__(self, input_dim, length_scale):
        super().__init__()
        widths = (input_dim + 2, *WIDTHS)
        self.layers = nn.ModuleList(
            nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:])
        )
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        self.head = nn.Linear(widths[-1], 2, bias=False)
        self.length_scale = length_scale

    def graph(self, cs, cx, cy, ts, tx):
        assert cs.shape[0] == 1
        count = min(NEIGHBORS, cs.shape[1])
        distances = torch.cdist(ts, cs, compute_mode="donot_use_mm_for_euclid_dist")
        idx = distances[0].argsort(dim=-1, stable=True)[:, :count]
        coordinates = torch.cat((ts[0, :, None], cs[0][idx]), dim=1)
        covariates = torch.cat((tx[0, :, None], cx[0][idx]), dim=1)
        inputs = torch.cat((coordinates, covariates), dim=-1)
        label = torch.cat((cy.new_zeros((len(idx), 1, 1)), cy[0][idx]), dim=1)
        center = torch.zeros_like(label)
        center[:, 0] = 1.0
        edges = torch.cdist(
            coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        weights = torch.exp(-0.5 * (edges / self.length_scale).square())
        inverse_degree = weights.sum(-1).rsqrt()
        adjacency = inverse_degree[..., None] * weights * inverse_degree[:, None, :]
        return torch.cat((inputs, label, center), dim=-1), adjacency

    def forward(self, cs, cx, cy, ts, tx):
        h, adjacency = self.graph(cs, cx, cy, ts, tx)
        for layer in self.layers:
            h = F.dropout(
                F.relu(adjacency @ layer(h)), p=DROPOUT, training=self.training
            )
        mu, raw_scale = self.head(h[:, 0]).chunk(2, dim=-1)
        return mu[None], (F.softplus(raw_scale) + 0.001)[None]

    def predict(self, context, coords, attributes):
        parts = [
            self(*context, s, x)
            for s, x in zip(coords.split(256, 1), attributes.split(256, 1))
        ]
        return tuple(torch.cat(values, dim=1) for values in zip(*parts))


def context_length_scale(coords):
    distances, _ = cKDTree(coords).query(coords, k=min(NEIGHBORS + 1, len(coords)))
    positive = distances[:, 1:][distances[:, 1:] > 0]
    return float(np.median(positive)) if positive.size else 1.0


def fit(dataset, mode, seed, out=OUT, epochs=EPOCHS, episodes=EPISODES):
    folder = out / dataset / mode
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"seed_{seed}_result.json"
    config = dict(
        neighbors=NEIGHBORS, widths=WIDTHS, dropout=DROPOUT, learning_rate=0.001
    )
    frozen = training_identity(dataset, seed, mode, VARIANT, config, epochs, episodes)
    source = Path(__file__).resolve()
    identity = hashlib.sha256(frozen.encode() + source.read_bytes()).hexdigest()
    if result.is_file():
        if json.loads(result.read_text(encoding="utf8"))["identity"] != identity:
            raise RuntimeError(f"Incompatible completed run: {result}")
        return
    data = load_final_split(dataset, seed, mode, "cpu")
    context = data.context_s, data.context_x, data.context_y[..., None]
    length_scale = context_length_scale(data.context_s[0].numpy())
    seed_all(seed)
    model = GaussianKCN(data.coord_dim + data.attr_dim, length_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    dropout_rng = torch.Generator().manual_seed(seed + 55000).get_state()
    seed_all(seed + 900000)
    start_epoch, best_epoch, best, best_state, digest = 0, 0, float("inf"), None, ""
    resume = folder / f"seed_{seed}_resume.pt"
    if resume.exists():
        state = torch.load(resume, weights_only=False, map_location="cpu")
        if state["identity"] != identity:
            raise RuntimeError(f"Incompatible checkpoint: {resume}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_epoch, best = (
            state["epoch"],
            state["best_epoch"],
            state["best"],
        )
        best_state, digest, dropout_rng = (
            state["best_state"],
            state["digest"],
            state["dropout_rng"],
        )
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["sampler_rng"])
    for epoch in range(start_epoch, epochs):
        model.train()
        for _ in range(episodes):
            *inputs, target = sample_episode(
                data.context_s, data.context_x, data.context_y
            )
            digest = hashlib.sha256(
                digest.encode()
                + b"".join(t.numpy().tobytes() for t in (*inputs, target))
            ).hexdigest()
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(dropout_rng)
                loss = marginal_nll(*model(*inputs), target)
                dropout_rng = torch.get_rng_state()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite KCN loss: {dataset}/{mode}/{seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = marginal_nll(
                *model.predict(context, data.validation_s, data.validation_x),
                data.validation_y,
            ).item()
        if not np.isfinite(val):
            raise RuntimeError(
                f"Non-finite KCN validation NLL: {dataset}/{mode}/{seed}"
            )
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            tmp = resume.with_suffix(".tmp")
            torch.save(
                dict(
                    identity=identity,
                    epoch=epoch + 1,
                    best_epoch=best_epoch,
                    best=best,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    best_state=best_state,
                    digest=digest,
                    dropout_rng=dropout_rng,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    sampler_rng=torch.get_rng_state(),
                ),
                tmp,
            )
            tmp.replace(resume)
            print(
                f"{dataset}/{mode}/{seed} epoch {epoch+1}/{epochs}, validation NLL {val:.5f}",
                flush=True,
            )
    if (epochs, episodes) == (EPOCHS, EPISODES):
        with (FROZEN / f"{dataset}_runs.jsonl").open(encoding="utf8") as file:
            matched = next(
                row
                for row in map(json.loads, file)
                if row["mode"] == mode
                and row["seed"] == seed
                and row["variant"] == "final"
            )
        if digest != matched["episode_sha256"]:
            raise RuntimeError(
                f"KCN sampled different training episodes: {dataset}/{mode}/{seed}"
            )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mu, sd = model.predict(context, data.test_s, data.test_x)
    y_std, y_mean = data.target_std, data.target_mean
    archive = folder / f"seed_{seed}_mixture.npz"
    np.savez_compressed(
        archive,
        ids=data.test_ids,
        y=data.test_y[0].numpy() * y_std + y_mean,
        mu=mu[0].numpy() * y_std + y_mean,
        sigma=sd[0].numpy() * y_std,
        pi=np.ones_like(mu[0].numpy()),
    )
    reference = FROZEN / dataset / mode / "final" / f"seed_{seed}_mixture.npz"
    metrics = archive_metrics(archive, reference, 1)
    torch.save(best_state, folder / f"seed_{seed}_best.pt")
    write_json(
        result,
        dict(
            dataset=dataset,
            mode=mode,
            variant=VARIANT,
            seed=seed,
            identity=identity,
            epochs=epochs,
            episodes_per_epoch=episodes,
            episode_sha256=digest,
            best_epoch=best_epoch,
            best_validation_nll_scaled=best,
            config=config,
            length_scale=length_scale,
            **metrics,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.smoke:
        fit("synthetic", "random", 1, OUT / "smoke", epochs=1, episodes=2)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (OUT / "suite.log").open("a", encoding="utf8", buffering=1)
        sys.stderr = sys.stdout
    status = OUT / "suite_status.json"
    try:
        for first, last in ((1, 5), (6, 10)):
            write_json(
                status,
                dict(state="running", first_seed=first, last_seed=last, expected=70),
            )
            for seed in range(first, last + 1):
                for dataset, mode in PROTOCOLS:
                    fit(dataset, mode, seed)
            rows = [
                json.loads(p.read_text(encoding="utf8"))
                for p in OUT.glob("*/*/*_result.json")
            ]
            frame = pd.DataFrame(rows)
            frame = frame[frame.seed <= last].sort_values(["dataset", "mode", "seed"])
            assert (
                len(frame) == 7 * last
                and not frame.duplicated(["dataset", "mode", "seed"]).any()
            )
            frame.to_csv(OUT / f"metrics_{last}seeds.csv", index=False)
            summarize(frame).to_csv(OUT / f"summary_{last}seeds.csv", index=False)
        write_json(status, dict(state="complete", completed=70))
    except Exception as exc:
        write_json(status, dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
