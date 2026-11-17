"""Build publication panels from archived predictions and seed-level scores."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from calibration import mixture_quantile
from data import DATA_DIR, DATASETS
from final_data import load_final_split
from run_supplemental_sensitivity import FROZEN, OUT


FIGURES = Path(__file__).resolve().parent / "results" / "figures"
PAIRS = [
    (ds, mode)
    for ds in ("california", "chengdu", "election", "synthetic")
    for mode in ("random", "spatial_block")
    if (ds, mode) != ("synthetic", "spatial_block")
]
BLUE, ORANGE, INK = "#456F92", "#A36236", "#344755"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    }
)


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", pad=8)


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def label(ds, mode):
    return f"{ds.title()}  /  {'random' if mode == 'random' else 'block'}"


def results_figure():
    folder = FROZEN / "phase2_10seeds"
    data = pd.read_csv(folder / "metrics_per_seed.csv")
    comparisons = pd.read_csv(folder / "paired_comparisons.csv").set_index(
        ["dataset", "mode", "comparator", "metric"]
    )
    rows = []
    for ds, mode in PAIRS:
        for control in ("single_gaussian", "equal_gate"):
            row = comparisons.loc[ds, mode, control, "nll"]
            assert row.n == 10
            rows.append(
                dict(
                    dataset=ds,
                    mode=mode,
                    control=control,
                    delta=row.mean_delta,
                    low=row.bootstrap_ci_low,
                    high=row.bootstrap_ci_high,
                )
            )
    effect = pd.DataFrame(rows)
    coverage = data[data.variant == "final"].copy()
    directory = OUT / "visualization_source_data"
    directory.mkdir(parents=True, exist_ok=True)
    effect.to_csv(directory / "head_and_gate_nll.csv", index=False)
    coverage[
        ["dataset", "mode", "seed", "coverage50", "coverage80", "coverage90"]
    ].to_csv(directory / "coverage.csv", index=False)

    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.7), sharey=True, width_ratios=[1, 1, 1.1]
    )
    fig.subplots_adjust(left=0.245, right=0.976, top=0.85, bottom=0.25, wspace=0.35)
    positions = np.arange(len(PAIRS))[::-1]
    for ax, control, letter, title, color in zip(
        axes[:2],
        ("single_gaussian", "equal_gate"),
        "AB",
        ("Mixture head", "Learned gate"),
        (BLUE, ORANGE),
    ):
        df = effect[effect.control == control]
        ax.axvline(0, color="#748095", lw=0.8, ls="--", zorder=1)
        ax.hlines(positions, df.low, df.high, color=color, lw=1.1)
        ax.scatter(df.delta, positions, color=color, s=18, zorder=2)
        panel(ax, letter, title)
        ax.set_xlabel(r"$\Delta$ NLL" + "\nGECNP − control")
    axes[0].set_yticks(positions, [label(*pair) for pair in PAIRS])
    axes[0].set_ylim(-0.55, len(PAIRS) - 0.45)
    ax = axes[2]
    ax.axvline(0.9, color="#748095", lw=0.8, ls="--")
    for pos, (ds, mode) in zip(positions, PAIRS):
        points = coverage[
            (coverage.dataset == ds) & (coverage["mode"] == mode)
        ].coverage90
        ax.scatter(
            points, np.full(len(points), pos), color=BLUE, alpha=0.33, s=13, linewidth=0
        )
        ax.scatter(points.mean(), pos, marker="D", color=INK, s=20, zorder=3)
    ax.set_xlabel("Empirical 90% coverage")
    ax.set_xticks([0.6, 0.75, 0.9, 1.0])
    ax.set_xlim(min(0.59, coverage.coverage90.min() - 0.015), 1.015)
    panel(ax, "C", "Interval coverage")
    save(fig, FIGURES, "paired_nll_and_coverage")


def spatial_figure():
    dataset, mode, seed = "california", "spatial_block", 1
    split = load_final_split(dataset, seed, mode, "cpu")
    raw = pd.read_csv(DATA_DIR / DATASETS[dataset]).set_index("id")
    path = FROZEN / dataset / mode / "final" / f"seed_{seed}_mixture.npz"
    with np.load(path) as archive:
        ids, y, mu, sd, pi = [archive[k] for k in ("ids", "y", "mu", "sigma", "pi")]
    np.testing.assert_array_equal(ids, split.test_ids)
    pi = pi.astype(float)
    pi /= pi.sum(axis=1, keepdims=True)
    mean = (pi * mu).sum(axis=1)
    lower, upper = [mixture_quantile(q, mu, sd, pi) for q in (0.05, 0.95)]
    context, validation, test = [
        raw.loc[i, ["Latitude", "Longitude"]]
        for i in (split.context_ids, split.validation_ids, split.test_ids)
    ]
    source = test.copy()
    source["observed"], source["predicted"] = y, mean
    source["interval_lower"], source["interval_upper"] = lower, upper
    source.to_csv(OUT / "visualization_source_data" / "california_spatial_seed1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.981, top=0.85, bottom=0.29, wspace=0.22)
    for ax in axes:
        ax.set_aspect(1 / np.cos(np.deg2rad(raw.Latitude.mean())), adjustable="box")
        ax.scatter(raw.Longitude, raw.Latitude, s=0.7, color="#E3E7EB", linewidth=0)
        ax.axvline(raw.Longitude.median(), color="#979CA4", lw=0.55, ls="--", zorder=0)
        ax.axhline(raw.Latitude.median(), color="#979CA4", lw=0.55, ls="--", zorder=0)
        ax.set_xticks([-124, -120, -116])
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{abs(value):.0f}")
        )
        ax.set_yticks([33, 36, 39, 42])
    axes[0].set_ylabel("Latitude (°N)")
    for points, c, name in (
        (validation, "#98A2AF", "Validation"),
        (context, ORANGE, "Context"),
        (test, BLUE, "Test"),
    ):
        axes[0].scatter(
            points.Longitude, points.Latitude, s=2.5, color=c, linewidth=0, label=name
        )
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=6.3,
        handletextpad=0.15,
        columnspacing=0.6,
        markerscale=2,
    )
    panel(axes[0], "A", "Spatial holdout")
    error = mean - y
    cap = np.quantile(np.abs(error), 0.98)
    errmap = axes[1].scatter(
        test.Longitude,
        test.Latitude,
        c=error,
        s=3,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap),
        linewidth=0,
    )
    panel(axes[1], "B", "Signed prediction error")
    width = upper - lower
    wm = axes[2].scatter(
        test.Longitude,
        test.Latitude,
        c=width,
        s=3,
        cmap="viridis",
        vmin=0,
        vmax=np.quantile(width, 0.98),
        linewidth=0,
    )
    panel(axes[2], "C", "90% interval width")
    for ax, artist, title, extend in (
        (axes[1], errmap, "Prediction − observation", "both"),
        (axes[2], wm, "Upper − lower quantile", "max"),
    ):
        box = ax.get_position()
        color_ax = fig.add_axes([box.x0 + 0.008, 0.165, box.width - 0.016, 0.031])
        cbar = fig.colorbar(
            artist, cax=color_ax, orientation="horizontal", extend=extend
        )
        cbar.set_label(title, fontsize=8.5)
        cbar.ax.tick_params(labelsize=8.5)
    fig.text(0.2, 0.02, "Longitude (°W)", ha="center", fontsize=9.5)
    save(fig, FIGURES, "california_holdout_maps")


def sensitivity_figure(folder, n_seeds):
    data = pd.read_csv(folder / "paired_nll.csv")
    variants = [
        ("single_gaussian", r"$K=1$"),
        ("gecnp_k3", r"$K=3$"),
        ("gecnp_k4", r"$K=4$"),
        ("gecnp_lr_low", r"$\eta=3\times10^{-4}$"),
        ("gecnp_lr_high", r"$\eta=3\times10^{-3}$"),
        ("gecnp_width32", r"$d_h=32$"),
        ("gecnp_width128", r"$d_h=128$"),
    ]
    matrix = data.pivot(
        index=["dataset", "mode"], columns="variant", values="nll_difference"
    )
    matrix = matrix.loc[PAIRS, [v for v, _ in variants]].to_numpy()
    cap = max(np.quantile(np.abs(matrix), 0.95), 1e-12)
    fig, ax = plt.subplots(figsize=(7.15, 3))
    fig.subplots_adjust(left=0.24, right=0.85, top=0.87, bottom=0.27)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-cap, vmax=cap, aspect="auto")
    for i, j in np.ndindex(matrix.shape):
        x = matrix[i, j]
        ax.text(
            j,
            i,
            f"{x:+.3f}",
            ha="center",
            va="center",
            color="white" if abs(x) > cap * 0.62 else INK,
            fontsize=8.2,
        )
    ax.set_xticks(
        range(len(variants)),
        [name for _, name in variants],
        rotation=26,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(PAIRS)), [label(*pair) for pair in PAIRS])
    ax.set_title(
        f"NLL change relative to GECNP defaults ({n_seeds} paired seeds)",
        loc="left",
        fontweight="bold",
    )
    color_ax = fig.add_axes([0.88, 0.27, 0.02, 0.60])
    fig.colorbar(im, cax=color_ax, extend="both", label="Variant − default")
    save(fig, folder, "sensitivity_nll")


def cnp_extension_figure():
    """Show paired effects of the full model against the four CNP configurations."""
    data = pd.read_csv(FROZEN / "phase2_10seeds" / "paired_comparisons.csv")
    variants = ("cnp_single", "cnp_two", "cnp_single_matched", "cnp_two_matched")
    selected = data[data.comparator.isin(variants) & data.metric.isin(("nll", "crps"))]
    lookup = selected.set_index(
        ["metric", "comparator", "dataset", "mode"]
    ).sort_index()
    assert len(lookup) == 56 and lookup.index.is_unique and (lookup.n == 10).all()
    directory = OUT / "visualization_source_data"
    directory.mkdir(parents=True, exist_ok=True)
    selected.to_csv(directory / "cnp_extension_paired_effects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.15), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.83, bottom=0.15, wspace=0.23)
    positions = np.arange(len(PAIRS))[::-1]
    labels = ("CNP, $K=1$", "CNP, $K=2$", "CNP-m, $K=1$", "CNP-m, $K=2$")
    colors = (BLUE, "#6F5A87", "#52796F", ORANGE)
    for ax, metric, letter in zip(axes, ("nll", "crps"), "AB"):
        ax.axvline(0, color="#748095", lw=0.8, ls="--")
        for pos in positions[:-1]:
            ax.axhline(pos - 0.5, color="#E6E9EC", lw=0.45)
        for offset, variant, name, color, marker in zip(
            (-0.28, -0.095, 0.095, 0.28), variants, labels, colors, ("o", "s", "^", "D")
        ):
            effect = lookup.loc[(metric, variant)].loc[PAIRS]
            y = positions + offset
            ax.hlines(
                y,
                effect.bootstrap_ci_low,
                effect.bootstrap_ci_high,
                color=color,
                lw=0.85,
            )
            ax.scatter(
                effect.mean_delta,
                y,
                marker=marker,
                color=color,
                s=12,
                zorder=3,
                label=name,
            )
        panel(ax, letter, metric.upper())
        ax.set_xlabel("GECNP − CNP variant")
    axes[0].set_yticks(positions, [label(*pair) for pair in PAIRS])
    axes[0].set_ylim(-0.55, len(PAIRS) - 0.45)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.99),
        ncol=4,
        handletextpad=0.3,
        columnspacing=1.15,
    )
    save(fig, FIGURES, "cnp_extension_effects")


def election_prediction_figure():
    """Plot aligned original-scale observations and predictive means on one split."""
    dataset, mode, seed = "election", "spatial_block", 1
    raw = pd.read_csv(DATA_DIR / DATASETS[dataset]).set_index("id")
    arrays = {}
    for variant in ("cnp_single", "final"):
        with np.load(
            FROZEN / dataset / mode / variant / f"seed_{seed}_mixture.npz"
        ) as archive:
            pi = archive["pi"].astype(float)
            pi /= pi.sum(axis=1, keepdims=True)
            arrays[variant] = (
                archive["ids"],
                archive["y"],
                (pi * archive["mu"]).sum(axis=1),
            )
    ids, observed, cnp_mean = arrays["cnp_single"]
    np.testing.assert_array_equal(ids, arrays["final"][0])
    np.testing.assert_array_equal(observed, arrays["final"][1])
    gecnp_mean = arrays["final"][2]
    source = raw.loc[ids, ["INTPTLONG", "INTPTLAT"]].copy()
    source["observed"], source["cnp"], source["gecnp"] = observed, cnp_mean, gecnp_mean
    directory = OUT / "visualization_source_data"
    directory.mkdir(parents=True, exist_ok=True)
    source.to_csv(directory / "election_spatial_seed1_prediction.csv")
    x_bounds, y_bounds = (-91, -86), (42, 46)
    inside = source.INTPTLONG.between(*x_bounds) & source.INTPTLAT.between(*y_bounds)
    local = source.loc[inside].copy()
    local["cnp_absolute_error"] = (local.cnp - local.observed).abs()
    local["gecnp_absolute_error"] = (local.gecnp - local.observed).abs()
    local["absolute_error_reduction"] = (
        local.cnp_absolute_error - local.gecnp_absolute_error
    )
    assert len(local) == 61 and (local.absolute_error_reduction > 0).sum() == 38
    marked = local.nlargest(3, "absolute_error_reduction").index
    local["highlighted"] = local.index.isin(marked)
    local.to_csv(directory / "election_spatial_seed1_inset.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.34, wspace=0.19)
    values = (observed, cnp_mean, gecnp_mean)
    lower = min(float(np.min(v)) for v in values)
    upper = np.quantile(np.concatenate(values), 0.98)
    titles = (
        "Observed",
        f"CNP (inset MAE {local.cnp_absolute_error.mean():.3f})",
        f"GECNP (inset MAE {local.gecnp_absolute_error.mean():.3f})",
    )
    for ax, letter, title, z in zip(axes, "ABC", titles, values):
        ax.set_aspect(1 / np.cos(np.deg2rad(raw.INTPTLAT.mean())), adjustable="box")
        ax.scatter(raw.INTPTLONG, raw.INTPTLAT, s=1.1, color="#E3E7EB", linewidth=0)
        artist = ax.scatter(
            source.INTPTLONG,
            source.INTPTLAT,
            c=z,
            s=6,
            linewidth=0,
            cmap="viridis",
            vmin=lower,
            vmax=upper,
        )
        ax.set(xlim=(-126, -65), ylim=(24, 51))
        inset = ax.inset_axes([0.055, 0.07, 0.47, 0.52])
        inset.set_facecolor("white")
        inset.scatter(
            local.INTPTLONG,
            local.INTPTLAT,
            c=z[inside.to_numpy()],
            s=9,
            linewidth=0,
            cmap="viridis",
            norm=artist.norm,
        )
        highlighted = local.loc[marked]
        inset.scatter(
            highlighted.INTPTLONG,
            highlighted.INTPTLAT,
            s=23,
            facecolors="none",
            edgecolors="#2B3038",
            linewidths=0.8,
        )
        inset.set(xlim=x_bounds, ylim=y_bounds, xticks=[], yticks=[])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#575D68")
        ax.indicate_inset_zoom(inset, edgecolor="#575D68", linewidth=0.7, alpha=0.9)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{abs(value):.0f}")
        )
        panel(ax, letter, title)
    axes[0].set_ylabel("Latitude (°N)")
    fig.text(0.54, 0.25, "Longitude (°W)", ha="center", fontsize=9.5)
    cax = fig.add_axes([0.27, 0.155, 0.51, 0.032])
    fig.colorbar(
        artist,
        cax=cax,
        orientation="horizontal",
        extend="max",
        label="Republican votes (10,000s)",
    )
    save(fig, FIGURES, "election_spatial_predictions")


def component_ablation_figures():
    """One six-metric component-ablation figure per dataset."""
    summary = pd.read_csv(FROZEN / "phase2_10seeds" / "summary.csv")
    methods = {
        "equal_gate": "Equal gate",
        "single_gaussian": "Single\nGaussian",
        "cosine": "Cosine",
        "spherical": "Spherical",
        "final": "GECNP",
    }
    metrics = {
        "mae": "MAE ↓",
        "mse": "MSE ↓",
        "r2": r"$R^2$ ↑",
        "nll": "NLL ↓",
        "crps": "CRPS ↓",
        "coverage90": "90% coverage",
    }
    source = summary[summary.variant.isin(methods)].copy()
    columns = ["dataset", "mode", "variant"] + [
        f"{key}_{stat}" for key in metrics for stat in ("mean", "std", "count")
    ]
    directory = OUT / "visualization_source_data"
    directory.mkdir(parents=True, exist_ok=True)
    source[columns].to_csv(directory / "component_ablation_by_dataset.csv", index=False)

    for dataset, subset in source.groupby("dataset"):
        component_ablation_lines(dataset, subset, methods, metrics)


def component_ablation_lines(dataset, summary, methods, metrics):
    variants = [variant for variant in methods if variant in set(summary.variant)]
    labels = [methods[variant] for variant in variants]
    positions = np.arange(len(variants))
    protocols = []
    for mode, marker, color, name in (
        ("random", "o", BLUE, "Random holdout"),
        ("spatial_block", "s", ORANGE, "Spatial block"),
    ):
        block = summary.loc[summary["mode"] == mode].set_index("variant")
        if not block.empty:
            protocols.append((block.loc[variants], marker, color, name))
    assert len(summary) == len(variants) * len(protocols)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.2), sharex=True)
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.84, bottom=0.18, wspace=0.44, hspace=0.71
    )
    for ax, (metric, title), letter in zip(axes.flat, metrics.items(), "ABCDEF"):
        for block, marker, color, name in protocols:
            assert (block[f"{metric}_count"] == 10).all()
            ax.errorbar(
                positions,
                block[f"{metric}_mean"],
                yerr=block[f"{metric}_std"],
                color=color,
                marker=marker,
                ms=3,
                lw=0.9,
                elinewidth=0.8,
                capsize=2,
                label=name,
            )
        if metric == "coverage90":
            ax.axhline(0.9, lw=0.75, ls="--", color="#748095")
        tick_scale = 1e3 if dataset == "synthetic" and metric == "mse" else 1
        if tick_scale != 1:
            title += r" ($\times 10^{-3}$)"
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _, scale=tick_scale: f"{value * scale:.3f}")
        )
        ax.grid(axis="y", color="#E6E9EC", lw=0.5)
        ax.set_axisbelow(True)
        panel(ax, letter, title)
    for ax in axes.flat:
        ax.set_xticks(positions, labels, rotation=32, ha="right")
        ax.tick_params(axis="x", labelbottom=True, labelsize=8.5)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=len(protocols),
    )
    save(fig, FIGURES, f"ablation_{dataset}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cnp-extensions", action="store_true")
    parser.add_argument("--component-ablations", action="store_true")
    args = parser.parse_args()
    if args.cnp_extensions:
        cnp_extension_figure()
        election_prediction_figure()
    if args.component_ablations:
        component_ablation_figures()
    if not (args.cnp_extensions or args.component_ablations):
        results_figure()
        spatial_figure()
