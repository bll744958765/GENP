"""Plot the archived synthetic field and ten-seed random-holdout comparisons."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate_independent_synthetic import OUTPUT as INDEPENDENT_TEST
from data import DATA_DIR, DATASETS

ROOT = Path(__file__).resolve().parent
BASELINES = {
    "ordinary_kriging": "Ordinary kriging",
    "regression_kriging": "Regression kriging",
    "cnp_single": "CNP",
    "anp_standard": "ANP",
    "smacnp": "SMACNP",
    "sfconvcnp": "SFConvCNP+",
    "sfvconvcnp": "SFVConvCNP+",
    "te_istnp": "TE-ISTNP+",
    "kcn": "KCN",
}


FIGURES = ROOT / "results" / "figures"
SOURCE = ROOT / "results" / "supplemental_300e30ep" / "visualization_source_data"
BLUE = "#456F92"
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


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def synthetic_fields():
    data_path = DATA_DIR / DATASETS["synthetic"]
    frame = pd.read_csv(data_path)
    assert len(frame) == 2500 and not frame.duplicated(["x", "y"]).any()
    np.testing.assert_allclose(frame["T"], 0.5 * (frame.x + frame.y))
    np.testing.assert_allclose(
        frame.y_target,
        frame["T"]
        + 0.3 * frame.f1
        + 0.5 * np.sqrt(np.abs(frame.f2))
        - 0.2 * np.sin(frame.f3),
        atol=1e-12,
    )
    development, test = train_test_split(
        np.arange(len(frame)), test_size=0.85, random_state=1
    )
    context, validation = train_test_split(
        development, train_size=2 / 3, random_state=10001
    )
    assert (len(context), len(validation), len(test)) == (250, 125, 2125)
    archive = (
        ROOT
        / "results"
        / "final_300e30ep"
        / "synthetic"
        / "random"
        / "final"
        / "seed_1_mixture.npz"
    )
    with np.load(archive) as stored:
        np.testing.assert_array_equal(frame.id.iloc[test], stored["ids"])
        np.testing.assert_allclose(
            frame.y_target.iloc[test], stored["y"], atol=1e-6, rtol=0
        )
    frame["split"] = "test"
    frame.loc[context, "split"] = "context"
    frame.loc[validation, "split"] = "validation"
    frame.to_csv(SOURCE / "synthetic_fields_seed1.csv", index=False)

    independent = pd.read_csv(INDEPENDENT_TEST / "test_field.csv")
    assert len(independent) == 2500 and not independent.duplicated(["x", "y"]).any()
    response_min = min(frame.y_target.min(), independent.y_target.min())
    response_max = max(frame.y_target.max(), independent.y_target.max())
    norm = matplotlib.colors.Normalize(response_min, response_max)
    fig = plt.figure(figsize=(7.15, 4.4))
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=(1.33, 1, 0.055),
        left=0.045,
        right=0.985,
        top=0.94,
        bottom=0.11,
        hspace=0.27,
        wspace=0.20,
    )
    for column, letter in enumerate("ABC"):
        cell = grid[0, 4 * column : 4 * (column + 1)].subgridspec(
            1, 2, width_ratios=(1, 0.052), wspace=0.18
        )
        ax = fig.add_subplot(cell[0], projection="3d")
        axis = np.sort(frame.x.unique())
        xx, yy = np.meshgrid(axis, axis)
        zz = frame.pivot(index="y", columns="x", values=f"f{column + 1}").to_numpy()
        surface = ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="viridis",
            edgecolor="#354248",
            linewidth=0.14,
            rstride=1,
            cstride=1,
        )
        ax.set(xlabel="$s_1$", ylabel="$s_2$", xticks=(0, 0.5, 1), yticks=(0, 0.5, 1))
        ax.xaxis.labelpad = -3
        ax.yaxis.labelpad = -3
        ax.zaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
        ax.tick_params(labelsize=9, pad=-1)
        ax.view_init(elev=26, azim=-57)
        ax.set_box_aspect((1, 1, 0.72), zoom=0.91)
        ax.set_title(
            f"{letter}   Field $f_{column + 1}$", loc="left", fontweight="bold", pad=0
        )
        bar = fig.colorbar(surface, cax=fig.add_subplot(cell[1]))
        bar.locator = plt.MaxNLocator(3)
        bar.update_ticks()
        bar.ax.tick_params(labelsize=8.5, pad=1, length=2)

    half_step = 0.5 / 49
    extent = (-half_step, 1 + half_step, -half_step, 1 + half_step)
    panels = (
        ("D   Context (10%)", frame.loc[frame.split.eq("context")]),
        ("E   Validation (5%)", frame.loc[frame.split.eq("validation")]),
        ("F   Independent test", independent),
    )
    for column, (title, points) in enumerate(panels):
        ax = fig.add_subplot(grid[1, 4 * column : 4 * (column + 1)])
        if len(points) == 2500:
            values = points.pivot(index="y", columns="x", values="y_target").to_numpy()
            assert values.shape == (50, 50) and np.isfinite(values).all()
            artist = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="viridis",
                norm=norm,
                interpolation="nearest",
            )
        else:
            ax.scatter(
                points.x,
                points.y,
                c=points.y_target,
                cmap="viridis",
                norm=norm,
                marker="s",
                s=3.0,
                linewidths=0,
            )
        ax.set(
            xlim=extent[:2],
            ylim=extent[2:],
            aspect="equal",
            xticks=(0, 1),
            yticks=(0, 1),
        )
        ax.tick_params(labelsize=9, length=2, pad=2, labelleft=column == 0)
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        if column == 0:
            ax.set_ylabel("$s_2$")
        ax.set_xlabel("$s_1$", labelpad=1)
    colorbar = fig.colorbar(
        artist, cax=fig.add_axes([0.30, 0.075, 0.42, 0.018]), orientation="horizontal"
    )
    colorbar.set_label("Response scale for D–F", labelpad=2, fontsize=10)
    save(fig, "synthetic_benchmark_fields")


def random_holdout_comparison():
    summary = pd.read_csv(ROOT / "results" / "comparison_summary.csv")
    datasets = ("chengdu", "election", "california")
    scores = []
    for dataset in datasets:
        rows = summary.loc[
            summary.dataset.eq(dataset) & summary["mode"].eq("random")
        ].set_index("variant")
        assert rows.loc[list(BASELINES), "nll_count"].eq(10).all()
        for metric in ("mae", "mse", "r2", "nll"):
            column = metric + "_mean"
            candidates = rows.loc[list(BASELINES), column]
            baseline = candidates.idxmax() if metric == "r2" else candidates.idxmin()
            b, g = candidates.loc[baseline], rows.loc["final", column]
            scores.append((dataset, metric, BASELINES[baseline], b, g))
    data = pd.DataFrame(
        scores, columns=["dataset", "metric", "baseline", "baseline_mean", "gecnp_mean"]
    )
    data.to_csv(SOURCE / "random_holdout_baseline_contrasts.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35))
    fig.subplots_adjust(
        left=0.085, right=0.98, bottom=0.15, top=0.85, wspace=0.29, hspace=0.68
    )
    x = np.arange(len(datasets))
    for ax, title, metric in zip(
        axes.flat,
        ("A   MAE ↓", "B   MSE ↓", r"C   $R^2$ ↑", "D   NLL ↓"),
        ("mae", "mse", "r2", "nll"),
    ):
        selected = data.loc[data.metric.eq(metric)]
        for column, offset, color, label in (
            ("baseline_mean", -0.19, "#A5AFB8", "Strongest external baseline"),
            ("gecnp_mean", 0.19, BLUE, "GECNP"),
        ):
            values = selected[column].to_numpy()
            bars = ax.bar(x + offset, values, width=0.33, color=color, label=label)
            ax.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values],
                padding=2.5,
                fontsize=8,
            )
        ax.axhline(0, color="#66717B", lw=0.65)
        ax.grid(axis="y", color="#E8EBEE", lw=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
        ax.set_xticks(
            x,
            [
                f"{ds.capitalize()}\n({name})"
                for ds, name in zip(datasets, selected.baseline)
            ],
            fontsize=9,
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.98), ncol=2)
    save(fig, "random_holdout_baseline_contrasts")
    print(data.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    SOURCE.mkdir(parents=True, exist_ok=True)
    synthetic_fields()
    random_holdout_comparison()
