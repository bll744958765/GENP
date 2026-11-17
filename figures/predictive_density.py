"""Compare archived California random-holdout densities on matched records."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_300e30ep"
FIGURES = ROOT / "results" / "figures"
SOURCE = ROOT / "results" / "supplemental_300e30ep" / "visualization_source_data"
REFERENCE = ROOT / "results" / "reference_np_300e30ep"
METHODS = {
    "smacnp": ("SMACNP", "#A36236", "--", REFERENCE),
    "final": ("GECNP", "#456F92", "-", RESULTS),
}
NAME = "california_predictive_density"


def build():
    response_edges = np.linspace(-0.55, 5.75, 64)
    response_grid = np.linspace(response_edges[0], response_edges[-1], 631)
    pit_edges = np.linspace(0, 1, 11)
    observed = np.zeros(len(response_edges) - 1)
    density = {key: np.zeros_like(response_grid) for key in METHODS}
    pit = {key: np.zeros(len(pit_edges) - 1) for key in METHODS}
    nll = {key: [] for key in METHODS}
    upper_fraction = []
    folder = RESULTS / "california" / "random"
    for seed in range(1, 11):
        with np.load(folder / "final" / f"seed_{seed}_mixture.npz") as archive:
            ids, y = archive["ids"], archive["y"].astype(float)
        assert len(ids) == len(np.unique(ids)) == len(y) == 17544
        assert np.all((response_edges[0] <= y) & (y <= response_edges[-1]))
        observed += np.histogram(y, bins=response_edges)[0] / (
            len(y) * np.diff(response_edges)
        )
        upper_fraction.append(np.isclose(y, 5.00001, rtol=0, atol=2e-6).mean())
        for key, (_, _, _, directory) in METHODS.items():
            with np.load(
                directory / "california" / "random" / key / f"seed_{seed}_mixture.npz"
            ) as archive:
                np.testing.assert_array_equal(ids, archive["ids"])
                np.testing.assert_allclose(y, archive["y"], rtol=0, atol=1e-6)
                mu, sigma, pi = (
                    archive[name].astype(float) for name in ("mu", "sigma", "pi")
                )
            assert (
                mu.shape
                == sigma.shape
                == pi.shape
                == (len(y), 2 if key == "final" else 1)
            )
            assert (
                np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()
            )
            assert np.isfinite(pi).all() and (pi >= 0).all()
            np.testing.assert_allclose(pi.sum(axis=1), 1, rtol=0, atol=1e-6)
            pi /= pi.sum(axis=1, keepdims=True)
            z = (y[:, None] - mu) / sigma
            values = np.clip((ndtr(z) * pi).sum(axis=1), 0, 1)
            pit[key] += np.histogram(values, bins=pit_edges)[0] / len(y)
            with np.errstate(divide="ignore"):
                scores = logsumexp(
                    np.log(pi) - np.log(sigma) - 0.5 * (z**2 + np.log(2 * np.pi)),
                    axis=1,
                )
            nll[key].append(-scores.mean())
            weighted_scale = pi / (sigma * np.sqrt(2 * np.pi))
            for start in range(0, len(response_grid), 32):
                part = slice(start, start + 32)
                standardized = (response_grid[part, None, None] - mu) / sigma
                density[key][part] += (
                    (np.exp(-0.5 * standardized**2) * weighted_scale)
                    .sum(axis=2)
                    .mean(axis=1)
                )

    summary = pd.read_csv(REFERENCE / "comparison_summary.csv")
    summary = summary.loc[
        (summary.dataset == "california") & (summary["mode"] == "random")
    ].set_index("variant")
    for key in METHODS:
        np.testing.assert_allclose(
            np.mean(nll[key]), summary.loc[key, "nll_mean"], atol=1e-5, rtol=0
        )
        density[key] /= 10
        pit[key] /= 10
        np.testing.assert_allclose(pit[key].sum(), 1)
        print(
            f"{key}: NLL {np.mean(nll[key]):.3f}, PIT L1 {np.abs(pit[key] - .1).sum():.3f}"
        )
    observed /= 10
    print(f"Observed upper-end concentration: {100 * np.mean(upper_fraction):.3f}%")

    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"response": response_grid, **density}).to_csv(
        SOURCE / f"{NAME}_curves.csv", index=False
    )
    pd.DataFrame(
        {"left": response_edges[:-1], "right": response_edges[1:], "density": observed}
    ).to_csv(SOURCE / f"{NAME}_observed.csv", index=False)
    pd.DataFrame({"left": pit_edges[:-1], "right": pit_edges[1:], **pit}).to_csv(
        SOURCE / f"{NAME}_pit.csv", index=False
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )
    fig, (ax, calibration) = plt.subplots(
        1, 2, figsize=(7.15, 2.65), width_ratios=(1.6, 1)
    )
    fig.subplots_adjust(left=0.11, right=0.975, bottom=0.23, top=0.75, wspace=0.43)
    ax.stairs(
        observed,
        response_edges,
        fill=True,
        color="#D5DADE",
        edgecolor="#78838B",
        linewidth=0.65,
        label="Observed",
    )
    for key, (name, color, style, _) in METHODS.items():
        ax.plot(
            response_grid,
            density[key],
            color=color,
            linestyle=style,
            lw=1.5,
            label=name,
        )
        calibration.stairs(
            pit[key], pit_edges, baseline=None, color=color, linestyle=style, lw=1.5
        )
    calibration.axhline(
        0.1, color="#7B838A", lw=0.9, linestyle=":", label="Uniform PIT"
    )
    ax.set(
        xlim=(response_edges[0], response_edges[-1]),
        ylim=(0, None),
        xlabel=r"Median house value (\$100,000)",
        ylabel="Probability density",
    )
    calibration.set(
        xlim=(0, 1),
        ylim=(0, None),
        xlabel="PIT value",
        ylabel="Fraction per bin",
        xticks=np.linspace(0, 1, 6),
    )
    ax.set_title("A   Pooled response density", loc="left", fontweight="bold", pad=7)
    calibration.set_title("B   PIT by model", loc="left", fontweight="bold", pad=7)
    for axis in (ax, calibration):
        axis.grid(axis="y", color="#E8EBEE", lw=0.5)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_formatter("{x:.3f}")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{NAME}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
