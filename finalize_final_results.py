"""Wait locally for the recorded workers, then validate and summarize saved runs.

No model fitting, test-guided tuning, API calls, or manuscript conclusion writing.
"""
import argparse
import itertools
import json
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from calibration import mixture_cdf, mixture_quantile, mixture_crps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/final_300e30ep"


def expected_keys(seeds=range(1, 11)):
    from run_final_experiments import CONFIGS

    return {
        (ds, mode, v, seed)
        for ds in ("california", "chengdu", "election", "synthetic")
        for mode in ("random", "spatial_block")
        if not (ds == "synthetic" and mode == "spatial_block")
        for v in CONFIGS
        if not (v == "spherical" and ds != "california")
        for seed in seeds
    }


def paired_signflip(difference):
    d = np.asarray(difference, dtype=float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(d))))
    null = np.abs((signs * d).mean(1))
    observed = abs(d.mean())
    return float(np.mean(null >= observed - 1e-12))


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    result = np.zeros(len(p))
    last = 0.0
    for i, index in enumerate(order):
        last = max(last, min(1.0, (len(p) - i) * p[index]))
        result[index] = last
    return result


def collect(seeds=None, report_dir=None):
    destination = Path(report_dir) if report_dir else OUT
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    keys = set()
    validation = []
    errors = []
    for ledger in sorted(OUT.glob("*runs.jsonl")):
        for line in ledger.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = tuple(r[k] for k in ("dataset", "mode", "variant", "seed"))
            if seeds is not None and r["seed"] not in seeds:
                continue
            if key in keys:
                raise RuntimeError(f"Duplicate run key: {key}")
            keys.add(key)
            if (r["epochs"], r["episodes_per_epoch"], r["optimizer_steps"]) != (
                300,
                30,
                9000,
            ):
                errors.append(f"Incorrect budget: {key}")
            path = (
                OUT
                / r["dataset"]
                / r["mode"]
                / r["variant"]
                / f"seed_{r['seed']}_mixture.npz"
            )
            if not path.exists():
                errors.append(f"Missing density archive: {path}")
                continue
            with np.load(path) as a:
                y, mu, sd, pi = [
                    a[k].astype(np.float64) for k in ("y", "mu", "sigma", "pi")
                ]
            if (
                not all(np.isfinite(v).all() for v in (y, mu, sd, pi))
                or np.any(sd <= 0)
                or np.any(pi < 0)
            ):
                errors.append(f"Invalid mixture parameters: {key}")
                continue
            if not np.allclose(pi.sum(-1), 1.0, atol=1e-5):
                errors.append(f"Unnormalized mixture: {key}")
            # Exported float32 exp(log_pi) sums can differ from one by ~1e-7.
            # Normalize in float64 before treating the archive as a probability
            # distribution; otherwise saturated PIT values can exceed one and
            # be silently dropped by np.histogram(range=(0,1), density=True).
            raw_pi_sum_error = float(np.max(np.abs(pi.sum(-1) - 1)))
            raw_pit = mixture_cdf(y, mu, sd, pi)
            raw_pit_outside = int(np.sum((raw_pit < 0) | (raw_pit > 1)))
            pi = pi / pi.sum(-1, keepdims=True)
            mean = (pi * mu).sum(-1)
            variance = (pi * (sd**2 + (mu - mean[:, None]) ** 2)).sum(-1)
            mse = float(np.mean((y - mean) ** 2))
            r2 = float(1 - np.sum((y - mean) ** 2) / np.sum((y - y.mean()) ** 2))
            with np.errstate(divide="ignore"):
                logp = logsumexp(
                    np.log(pi)
                    - np.log(sd)
                    - 0.5 * math.log(2 * math.pi)
                    - 0.5 * ((y[:, None] - mu) / sd) ** 2,
                    axis=-1,
                )
            nll = float(-logp.mean())
            checks = dict(
                dataset=key[0],
                mode=key[1],
                variant=key[2],
                seed=key[3],
                mse_absolute_error=abs(mse - r["test"]["mse"]),
                r2_absolute_error=abs(r2 - r["test"]["r2"]),
                nll_absolute_error=abs(nll - r["test"]["nll"]),
                minimum_sigma=float(sd.min()),
                maximum_pi_sum_error=raw_pi_sum_error,
                raw_archive_pit_outside_unit_interval=raw_pit_outside,
            )
            validation.append(checks)
            for field, value in (("mse", mse), ("r2", r2), ("nll", nll)):
                if abs(value - r["test"][field]) > 1e-3 * (1 + abs(value)):
                    errors.append(f"Saved metric mismatch: {key}/{field}")
            row = {
                k: r[k]
                for k in (
                    "dataset",
                    "mode",
                    "variant",
                    "seed",
                    "parameters",
                    "elapsed_seconds",
                    "best_epoch",
                    "episode_sha256",
                    "training_identity",
                )
            }
            row.update(
                mae=float(np.mean(np.abs(y - mean))),
                mse=mse,
                r2=r2,
                nll=nll,
                mean_predictive_variance=float(variance.mean()),
                crps=float(mixture_crps(y, mu, sd, pi).mean()),
            )
            pit = np.clip(mixture_cdf(y, mu, sd, pi), 0.0, 1.0)
            row.update(
                pit_mean=float(pit.mean()),
                pit_std=float(pit.std()),
                pit_histogram_l1=float(
                    np.abs(
                        np.histogram(pit, bins=10, range=(0, 1), density=True)[0] - 1
                    ).mean()
                ),
            )
            for coverage in (0.5, 0.8, 0.9):
                alpha = (1 - coverage) / 2
                lower = mixture_quantile(alpha, mu, sd, pi)
                upper = mixture_quantile(1 - alpha, mu, sd, pi)
                row[f"coverage{int(100*coverage)}"] = float(
                    np.mean((y >= lower) & (y <= upper))
                )
                row[f"width{int(100*coverage)}"] = float(np.mean(upper - lower))
            row["coverage90_error"] = abs(row["coverage90"] - 0.9)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if rows:
        frame.to_csv(destination / "metrics_per_seed.csv", index=False)
        pd.DataFrame(validation).to_csv(
            destination / "metric_identity_checks.csv", index=False
        )
        # Episode checks compare all saved variants within the same dataset/split/seed.
        for group, g in frame.groupby(["dataset", "mode", "seed"]):
            if g.episode_sha256.nunique() != 1:
                errors.append(f"Mismatched training episodes: {group}")
    return frame, keys, errors


def summarize(frame, report_dir=None, seed_count=10):
    destination = Path(report_dir) if report_dir else OUT
    destination.mkdir(parents=True, exist_ok=True)
    metrics = [
        "mae",
        "mse",
        "r2",
        "nll",
        "crps",
        "coverage50",
        "coverage80",
        "coverage90",
        "width50",
        "width80",
        "width90",
        "coverage90_error",
        "pit_mean",
        "pit_std",
        "pit_histogram_l1",
        "elapsed_seconds",
        "parameters",
        "best_epoch",
    ]
    summary = frame.groupby(["dataset", "mode", "variant"])[metrics].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(destination / "summary.csv", index=False)
    pairs = []
    rng = np.random.default_rng(20261103)
    for (ds, mode), g in frame.groupby(["dataset", "mode"]):
        main = g[g.variant == "final"].set_index("seed")
        for variant in sorted(set(g.variant) - {"final"}):
            other = g[g.variant == variant].set_index("seed")
            seeds = sorted(set(main.index) & set(other.index))
            if len(seeds) != seed_count:
                continue
            for metric in ("mae", "mse", "nll", "crps", "coverage90_error"):
                delta = (
                    main.loc[seeds, metric].to_numpy()
                    - other.loc[seeds, metric].to_numpy()
                )
                bootstrap = delta[
                    rng.integers(0, len(delta), size=(20000, len(delta)))
                ].mean(1)
                ci = np.quantile(bootstrap, [0.025, 0.975])
                pairs.append(
                    dict(
                        dataset=ds,
                        mode=mode,
                        main="final",
                        comparator=variant,
                        metric=metric,
                        n=seed_count,
                        mean_delta=float(delta.mean()),
                        bootstrap_ci_low=ci[0],
                        bootstrap_ci_high=ci[1],
                        p_exact_signflip=paired_signflip(delta),
                        direction="negative favors final",
                    )
                )
    paired = pd.DataFrame(pairs)
    if len(paired):
        paired["p_holm_global"] = holm(paired.p_exact_signflip)
        paired.to_csv(destination / "paired_comparisons.csv", index=False)
    lines = [
        "% Automatically generated from complete validated full-budget runs.",
        f"% Mean +/- sample SD across {seed_count} seeds. Bold denotes descriptive best mean only.",
    ]
    for mode in ("random", "spatial_block"):
        table = summary[summary["mode"] == mode]
        if table.empty:
            continue
        lines.extend(
            [
                r"\begin{table*}[p]",
                r"\centering\scriptsize",
                r"\caption{Full-budget "
                + mode.replace("_", " ")
                + f" results (300 epochs, 30 episodes per epoch, {seed_count} seeds). "
                + r"NLL, CRPS and interval widths are in the original target units; bold NLL denotes the smallest mean, not statistical significance. Covariate-adapted recent baselines use the documented scaled configurations.}",
                r"\begin{tabular}{llrrrrr}\hline",
                r"Dataset & Method & MAE & MSE & NLL & CRPS & Cov.90 \\ \hline",
            ]
        )
        for ds, g in table.groupby("dataset"):
            best = g.nll_mean.min()
            for _, r in g.iterrows():
                nll = f"{r.nll_mean:.4g} $\\pm$ {r.nll_std:.2g}"
                if r.nll_mean == best:
                    nll = r"\textbf{" + nll + "}"
                name = r["variant"].replace("_", r"\_")
                lines.append(
                    f"{ds} & {name} & {r.mae_mean:.4g} & {r.mse_mean:.4g} & {nll} & {r.crps_mean:.4g} & {r.coverage90_mean:.3f}"
                    + " "
                    + chr(92) * 2
                )
            lines.append(r"\hline")
        lines.extend([r"\end{tabular}", r"\end{table*}"])
    (destination / "final_tables.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )
    report = [
        "# Full-budget experiment collection",
        "",
        "All expected runs have completed and saved-density checks passed. This is an experiment collection, not a finalized manuscript.",
        "",
        f"- Budget: 300 epochs x 30 episodes, {seed_count} seeds. No short pilot is substituted.",
        "- summary.csv: means and sample SD across seeds.",
        "- paired_comparisons.csv: paired differences, bootstrap intervals, exact sign-flip tests and global Holm correction.",
        "- metrics_per_seed.csv: exact mixture intervals at 50/80/90%, PIT and CRPS, plus point metrics.",
        "- metric_identity_checks.csv: saved distribution versus stored score consistency.",
        "- final_tables.tex: generated result tables for later review and inclusion in the existing manuscript.",
        "",
        "The manuscript still requires result-grounded narrative revision, reviewer-tracker audit, and compilation in the built-in LaTeX editor. No universal superiority or gate benefit is inferred merely from the best mean.",
    ]
    (destination / "RESULTS_README.md").write_text(
        "\n".join(report) + "\n", encoding="utf8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert paired_signflip([1] * 5) == 0.0625
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        from scipy.stats import norm

        q = mixture_quantile(0.95, np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
        np.testing.assert_allclose(q, norm.ppf(0.95), atol=1e-6)
        rounded_pi = np.array([[0.5, 0.50000006]])
        saturated = mixture_cdf(
            np.array([20.0]), np.zeros((1, 2)), np.ones((1, 2)), rounded_pi
        )
        assert saturated[0] > 1
        fixed = np.clip(
            mixture_cdf(
                np.array([20.0]),
                np.zeros((1, 2)),
                np.ones((1, 2)),
                rounded_pi / rounded_pi.sum(-1, keepdims=True),
            ),
            0,
            1,
        )
        assert np.histogram(fixed, bins=10, range=(0, 1))[0].sum() == 1
        print("Postprocessing reference checks passed.")
        return
    frame, keys, errors = collect()
    missing = expected_keys() - keys
    status = dict(
        completed_runs=len(keys),
        expected_runs=len(expected_keys()),
        missing_runs=len(missing),
        errors=errors,
        status="experiments_complete"
        if not missing and not errors
        else "incomplete_or_failed",
    )
    if not missing and not errors:
        summarize(frame, OUT / "phase2_10seeds")
    else:
        status["missing_keys"] = sorted(missing)
        (OUT / "FAILURE_OR_INCOMPLETE.md").write_text(
            "Experiments incomplete; no final manuscript claim is permitted.\n\n"
            + json.dumps(status, indent=2),
            encoding="utf8",
        )
    (OUT / "completion_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf8"
    )
    print(
        json.dumps({k: v for k, v in status.items() if k != "missing_keys"}), flush=True
    )

    if missing or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
