"""Recompute and plot California spatial-holdout context disagreement."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
DATA = ROOT / "results" / "finalization" / "california_context_disagreement.csv"
NEIGHBOR_COUNTS = (8, 16, 32)


def collect():
    import torch
    from scipy.spatial.distance import cdist
    from scipy.special import softmax
    from gecnp.data import load_final_split

    torch.set_num_threads(1)
    folder = ROOT / "results/final_300e30ep/california/spatial_block/final"
    chunks = []
    for seed in range(1, 11):
        split = load_final_split("california", seed, "spatial_block")
        cs, cx, cy, ts, tx = [
            getattr(split, name)[0].numpy().astype(np.float64)
            for name in ("context_s", "context_x", "context_y", "test_s", "test_x")
        ]
        checkpoint = torch.load(
            folder / f"seed_{seed}_best.pt", map_location="cpu", weights_only=True
        )
        weights = checkpoint["attribute_logits"].softmax(0).numpy().astype(np.float64)
        tau_d = checkpoint["log_distance_temperature"].exp().clamp_min(1e-4).item()
        tau_a = checkpoint["log_similarity_temperature"].exp().clamp_min(1e-4).item()
        for start in range(0, len(ts), 256):
            sl = slice(start, start + 256)
            spatial_distance = cdist(ts[sl], cs, metric="sqeuclidean")
            attribute_distance = cdist(
                tx[sl] * weights, cx * weights, metric="cityblock"
            )
            spatial_order = np.argsort(spatial_distance, axis=1, kind="stable")[
                :, : max(NEIGHBOR_COUNTS)
            ]
            attribute_order = np.argsort(attribute_distance, axis=1, kind="stable")[
                :, : max(NEIGHBOR_COUNTS)
            ]
            rows = dict(seed=seed, query_id=split.test_ids[sl])
            for m in NEIGHBOR_COUNTS:
                same = spatial_order[:, :m, None] == attribute_order[:, None, :m]
                rows[f"shared_{m}"] = same.any(axis=2).sum(axis=1)
            wd = softmax(-spatial_distance / tau_d, axis=1)
            wa = softmax(-attribute_distance / tau_a, axis=1)
            gap = np.abs(((wd - wa) * cy).sum(axis=1))
            rows["gap_context_sd"] = gap
            rows["gap_usd"] = gap * split.target_std * 100_000
            chunks.append(pd.DataFrame(rows))
    records = pd.concat(chunks, ignore_index=True)
    assert records.groupby("seed").size().eq(492).all()
    assert records.groupby("query_id").size().eq(10).all()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(DATA, index=False)


def plot():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    records = pd.read_csv(DATA)
    assert (
        records.shape[0] == 4_920 and not records.duplicated(["seed", "query_id"]).any()
    )
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.15, 2.55), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.subplots_adjust(left=0.12, right=0.976, bottom=0.27, top=0.81, wspace=0.44)
    blue = "#356383"
    means = []
    for i, m in enumerate(NEIGHBOR_COUNTS):
        shared = records[f"shared_{m}"]
        disjoint = shared.eq(0).groupby(records.seed).mean()
        mean = disjoint.mean()
        means.append(mean)
        a.scatter(
            i + np.linspace(-0.10, 0.10, len(disjoint)),
            disjoint,
            s=15,
            color="#7992a4",
            alpha=0.7,
            zorder=3,
            linewidths=0,
        )
        a.annotate(
            f"{mean:.3%}",
            (i, mean),
            xytext=(11, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=blue,
            fontsize=8,
        )
        print(
            f"m={m}: disjoint={mean:.9f}, Jaccard={(shared / (2*m-shared)).mean():.9f}"
        )
    a.plot(
        range(3), means, color=blue, marker="D", markersize=4, linewidth=1.2, zorder=4
    )
    a.set(
        xlim=(-0.32, 2.76),
        xticks=range(3),
        xticklabels=NEIGHBOR_COUNTS,
        ylim=(0, 1.02),
        xlabel="Neighbors per criterion",
        ylabel="Queries with disjoint sets",
    )
    a.set_title("A   Neighborhood overlap", loc="left", fontweight="bold", pad=10)

    grid = np.unique(np.r_[0, 0.5, records.gap_context_sd.to_numpy()])
    curves = []
    for _, seed_records in records.groupby("seed"):
        gaps = np.sort(seed_records.gap_context_sd.to_numpy())
        survival = 1 - np.searchsorted(gaps, grid, side="left") / len(gaps)
        curves.append(survival)
        b.step(grid, survival, where="pre", color="#a2b2bd", alpha=0.50, linewidth=0.65)
    b.step(grid, np.mean(curves, axis=0), where="pre", color=blue, linewidth=1.55)
    share = records.gap_context_sd.ge(0.5).mean()
    b.vlines(0.5, 0, share, color="#92714f", linewidth=0.8, linestyle="--")
    b.scatter(0.5, share, color=blue, s=18, zorder=5)
    b.annotate(
        f"{share:.3%} at 0.5 SD",
        (0.5, share),
        xytext=(8, 8),
        textcoords="offset points",
        color=blue,
        fontsize=8,
    )
    b.set(
        xlim=(0, float(grid[-1]) * 1.04),
        ylim=(0, 1.02),
        xlabel="Absolute response gap (context SD)",
        ylabel="Queries with at least this gap",
    )
    b.set_title(
        "B   Observed response disagreement", loc="left", fontweight="bold", pad=10
    )
    for ax in (a, b):
        ax.set_yticks(np.linspace(0, 1, 5), labels=None)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", linewidth=0.45, color="#dce1e4")
        ax.set_axisbelow(True)
    print(
        f"gap_sd={records.gap_context_sd.mean():.9f}, gap_usd={records.gap_usd.mean():.4f}, share_half_sd={share:.9f}"
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"california_context_disagreement.{suffix}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Recompute source data from frozen checkpoints.",
    )
    if parser.parse_args().collect:
        collect()
    else:
        plot()
