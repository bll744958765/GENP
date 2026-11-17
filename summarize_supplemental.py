"""Validate paired prediction archives and summarize the added experiments."""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from calibration import mixture_crps, mixture_quantile
from run_supplemental_sensitivity import FROZEN, OUT, PROTOCOLS, SETTINGS


KRIGING = ("ordinary_kriging", "regression_kriging")
METRICS = ("mae", "mse", "r2", "nll", "crps", "coverage90")
KEYS = ["dataset", "mode", "variant", "seed"]


def archive_metrics(path, reference, count):
    with np.load(path) as archive:
        ids, y, mu, sd, pi = (archive[key] for key in ("ids", "y", "mu", "sigma", "pi"))
    with np.load(reference) as archive:
        np.testing.assert_array_equal(ids, archive["ids"])
        np.testing.assert_array_equal(y, archive["y"])
    if (
        mu.shape != (len(y), count)
        or sd.shape != mu.shape
        or pi.shape != mu.shape
        or not all(np.isfinite(arr).all() for arr in (y, mu, sd, pi))
        or (sd <= 0).any()
        or (pi < 0).any()
    ):
        raise ValueError(f"Invalid predictive parameters: {path}")
    np.testing.assert_allclose(pi.sum(axis=1), 1, atol=1e-6)
    y, mu, sd, pi = (arr.astype(np.float64) for arr in (y, mu, sd, pi))
    pi /= pi.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        logp = logsumexp(
            np.log(pi)
            - np.log(sd)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((y[:, None] - mu) / sd) ** 2,
            axis=1,
        )
    error = y - (mu * pi).sum(axis=1)
    lo, hi = (mixture_quantile(q, mu, sd, pi) for q in (0.05, 0.95))
    return dict(
        mae=np.abs(error).mean(),
        mse=np.square(error).mean(),
        r2=1 - np.square(error).sum() / np.square(y - y.mean()).sum(),
        nll=-logp.mean(),
        crps=mixture_crps(y, mu, sd, pi).mean(),
        coverage90=((y >= lo) & (y <= hi)).mean(),
        width90=(hi - lo).mean(),
    )


def collect(variants, n_seeds, frozen_metrics):
    rows = []
    index = frozen_metrics.set_index(KEYS)
    for (ds, mode), variant, seed in itertools.product(
        PROTOCOLS, variants, range(1, n_seeds + 1)
    ):
        folder = OUT / ds / mode / variant
        row = json.loads(
            (folder / f"seed_{seed}_result.json").read_text(encoding="utf8")
        )
        assert tuple(row[k] for k in KEYS) == (ds, mode, variant, seed)
        if variant in SETTINGS:
            count, rate, width = SETTINGS[variant]
            assert row["config"] == dict(
                components=count, learning_rate=rate, hidden=width
            )
            assert (
                row["epochs"],
                row["episodes_per_epoch"],
                row["optimizer_steps"],
            ) == (300, 30, 9000)
            assert (
                row["episode_sha256"]
                == index.loc[(ds, mode, "final", seed), "episode_sha256"]
            )
        else:
            count = 1
        reference = FROZEN / ds / mode / "final" / f"seed_{seed}_mixture.npz"
        metrics = archive_metrics(folder / f"seed_{seed}_mixture.npz", reference, count)
        if variant in SETTINGS:
            for key in ("mse", "r2", "nll"):
                np.testing.assert_allclose(
                    metrics[key], row["test"][key], rtol=1e-3, atol=1e-3
                )
        rows.append({**{key: row[key] for key in KEYS}, **metrics})
    return pd.DataFrame(rows)


def paired_deltas(frame):
    rows = []
    rng = np.random.default_rng(41007)
    for (ds, mode), group in frame.groupby(["dataset", "mode"]):
        nll = group.pivot(index="seed", columns="variant", values="nll")
        if nll.isna().any().any():
            raise ValueError(f"Unpaired runs for {ds}/{mode}")
        for variant in nll.drop(columns="final"):
            delta = (nll[variant] - nll["final"]).to_numpy()
            lo, hi = np.quantile(
                delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1),
                [0.025, 0.975],
            )
            rows.append(
                dict(
                    dataset=ds,
                    mode=mode,
                    variant=variant,
                    n=len(delta),
                    nll_difference=delta.mean(),
                    bootstrap_low=lo,
                    bootstrap_high=hi,
                )
            )
    return pd.DataFrame(rows)


def summarize(frame):
    result = frame.groupby(KEYS[:-1])[list(METRICS)].agg(["mean", "std", "count"])
    result.columns = ["_".join(column) for column in result.columns]
    return result.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kriging-only", action="store_true")
    parser.add_argument("--seeds", type=int, choices=(5, 10), default=10)
    args = parser.parse_args()
    frozen = pd.read_csv(FROZEN / "phase2_10seeds" / "metrics_per_seed.csv")
    if args.kriging_only:
        added = collect(KRIGING, 10, frozen)
        folder = OUT / "kriging_10seeds"
        combined = pd.concat([frozen, added], ignore_index=True)
        scope = combined[combined.variant.isin(["final", *KRIGING])]
    else:
        added = collect(SETTINGS, args.seeds, frozen)
        folder = OUT / f"sensitivity_{args.seeds}seeds"
        reference = frozen[
            (frozen.seed <= args.seeds)
            & frozen.variant.isin(["final", "single_gaussian"])
        ]
        combined = pd.concat([reference, added], ignore_index=True)
        scope = combined
    folder.mkdir(parents=True, exist_ok=True)
    added.to_csv(folder / "added_metrics_per_seed.csv", index=False)
    summarize(combined).to_csv(folder / "summary.csv", index=False)
    paired_deltas(scope).to_csv(folder / "paired_nll.csv", index=False)
    if not args.kriging_only:
        from plot_supplemental_results import sensitivity_figure

        sensitivity_figure(folder, args.seeds)
    print(f"Verified and summarized {len(added)} new predictive archives: {folder}")


if __name__ == "__main__":
    main()
