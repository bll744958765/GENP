"""Evaluate frozen checkpoints on one independently generated full synthetic grid."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "independent_synthetic_grid"
FIELD_SEED = 20261106


def generate_field():
    axis = np.linspace(0, 1, 50)
    x, y = (a.ravel() for a in np.meshgrid(axis, axis))
    rng = np.random.default_rng(FIELD_SEED)
    separation = axis[:, None] - axis
    fields = []
    for length in (0.1, 0.2, 0.3):
        factor = np.linalg.cholesky(
            np.exp(-0.5 * (separation / length) ** 2) + 1e-12 * np.eye(len(axis))
        )
        correlated = factor @ rng.standard_normal((50, 50)) @ factor.T
        fields.append(
            (correlated + np.sqrt(0.05) * rng.standard_normal((50, 50))).ravel()
        )
    f1, f2, f3 = fields
    trend = 0.5 * (x + y)
    response = trend + 0.3 * f1 + 0.5 * np.sqrt(np.abs(f2)) - 0.2 * np.sin(f3)
    return pd.DataFrame(dict(x=x, y=y, f1=f1, f2=f2, f3=f3, T=trend, y_target=response))


def main():
    import torch
    from final_data import load_final_split
    from run_final_experiments import (
        CONFIGS,
        OUT,
        build_model,
        geometry_metadata,
        predict,
    )
    from calibration import mixture_crps, mixture_quantile
    from data import DATA_DIR, DATASETS
    from scipy.special import logsumexp

    torch.set_num_threads(1)
    field = pd.read_csv(OUTPUT / "test_field.csv")
    assert len(field) == 2500 and not field.duplicated(["x", "y"]).any()
    coords = field[["x", "y"]].to_numpy()
    attributes = field[["f1", "f2", "f3", "T"]].to_numpy()
    observed = field.y_target.to_numpy()
    original = pd.read_csv(DATA_DIR / DATASETS["synthetic"]).set_index("id")
    records = []
    for seed in range(1, 11):
        split = load_final_split("synthetic", seed, "random")
        context = original.loc[split.context_ids, ["f1", "f2", "f3", "T"]].to_numpy()
        mean, std, scale = geometry_metadata(split, "synthetic")
        attr_mean, attr_std = context.mean(axis=0), context.std(axis=0)
        target_s = torch.tensor((coords - mean) / std, dtype=torch.float32).unsqueeze(0)
        target_x = torch.tensor(
            (attributes - attr_mean) / attr_std, dtype=torch.float32
        ).unsqueeze(0)
        for variant in ("cnp_single", "final"):
            folder = OUT / "synthetic" / "random" / variant
            model = build_model(split, mean, std, scale, CONFIGS[variant])
            model.load_state_dict(
                torch.load(
                    folder / f"seed_{seed}_best.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            model.eval()
            check = predict(model, split, split.test_s, split.test_x)
            with np.load(folder / f"seed_{seed}_mixture.npz") as archive:
                np.testing.assert_array_equal(archive["ids"], split.test_ids)
                np.testing.assert_allclose(
                    check["mu"][0].numpy() * split.target_std + split.target_mean,
                    archive["mu"],
                    atol=2e-6,
                    rtol=1e-5,
                )
                np.testing.assert_allclose(
                    check["sigma"][0].numpy() * split.target_std,
                    archive["sigma"],
                    atol=2e-6,
                    rtol=1e-5,
                )
                np.testing.assert_allclose(
                    check["log_pi"][0].exp().numpy(),
                    archive["pi"],
                    atol=2e-6,
                    rtol=1e-5,
                )
            pred = predict(model, split, target_s, target_x)
            mu = (
                pred["mu"][0].numpy().astype(float) * split.target_std
                + split.target_mean
            )
            sigma = pred["sigma"][0].numpy().astype(float) * split.target_std
            pi = pred["log_pi"][0].numpy().astype(float)
            pi = np.exp(pi - logsumexp(pi, axis=1, keepdims=True))
            assert (
                np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()
            )
            assert np.isfinite(pi).all()
            error = (pi * mu).sum(axis=1) - observed
            z = (observed[:, None] - mu) / sigma
            nll = -logsumexp(
                np.log(pi) - np.log(sigma) - 0.5 * (z**2 + np.log(2 * np.pi)), axis=1
            )
            lo = mixture_quantile(0.05, mu, sigma, pi)
            hi = mixture_quantile(0.95, mu, sigma, pi)
            records.append(
                dict(
                    variant=variant,
                    seed=seed,
                    n=len(observed),
                    mae=np.abs(error).mean(),
                    mse=(error**2).mean(),
                    r2=1 - (error**2).mean() / observed.var(),
                    nll=nll.mean(),
                    crps=mixture_crps(observed, mu, sigma, pi).mean(),
                    coverage90=((observed >= lo) & (observed <= hi)).mean(),
                )
            )
            np.savez_compressed(
                OUTPUT / f"{variant}_seed_{seed}_mixture.npz",
                y=observed,
                mu=mu,
                sigma=sigma,
                pi=pi,
            )
    scores = pd.DataFrame(records)
    scores.to_csv(OUTPUT / "metrics_per_seed.csv", index=False)
    print(
        scores.groupby("variant")[["mae", "mse", "r2", "nll", "crps", "coverage90"]]
        .agg(["mean", "std"])
        .to_string()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate the test grid without importing PyTorch.",
    )
    if parser.parse_args().generate:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        generate_field().to_csv(OUTPUT / "test_field.csv", index=False)
    else:
        main()
