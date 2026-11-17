"""Coordinate-only ordinary kriging and covariate-adjusted regression kriging."""
import argparse
import itertools
import json
import os
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
import torch

from gecnp.data import load_final_split
from experiments.sensitivity import FROZEN, OUT, PROTOCOLS


RANGES = (0.25, 0.75, 2.0)
NUGGET_FRACTIONS = (0.01, 0.10, 0.35)
RIDGE_ALPHA = 1.0


def spatial_covariance(distance, spatial_range, nugget_fraction):
    return (1.0 - nugget_fraction) * np.exp(-distance / spatial_range)


def fit_kriging(distance, response, spatial_range, nugget_fraction, ordinary):
    covariance = spatial_covariance(distance, spatial_range, nugget_fraction)
    covariance.flat[:: len(response) + 1] += nugget_fraction
    factor, _ = cho_factor(covariance, lower=True, check_finite=False)
    inverse_ones = cho_solve((factor, True), np.ones(len(response)))
    precision = inverse_ones.sum()
    center = (inverse_ones @ response) / precision if ordinary else 0.0
    centered = response - center
    coefficients = cho_solve((factor, True), centered)
    scale_squared = max(float(centered @ coefficients) / len(response), 1e-6)
    return factor, inverse_ones, precision, center, coefficients, scale_squared


def predict_kriging(fit, distance, trend, spatial_range, nugget_fraction, ordinary):
    factor, inverse_ones, precision, center, coefficients, scale_squared = fit
    means, scales = [], []
    for lo in range(0, len(distance), 512):
        sl = slice(lo, lo + 512)
        cross_covariance = spatial_covariance(
            distance[sl], spatial_range, nugget_fraction
        )
        projected = solve_triangular(
            factor, cross_covariance.T, lower=True, check_finite=False
        )
        variance = 1.0 - np.sum(projected**2, axis=0)
        if ordinary:
            variance += (1.0 - cross_covariance @ inverse_ones) ** 2 / precision
        means.append(trend[sl] + center + cross_covariance @ coefficients)
        scales.append(np.sqrt(np.maximum(scale_squared * variance, 1e-6)))
    return np.concatenate(means), np.concatenate(scales)


def vector(tensor):
    return tensor[0].numpy().astype(np.float64)


def evaluate(dataset, mode, seed, name, data):
    started = time.perf_counter()
    ordinary = name == "ordinary_kriging"
    context, validation, test = map(
        vector, (data.context_s, data.validation_s, data.test_s)
    )
    response, validation_y = map(vector, (data.context_y, data.validation_y))
    d_context = cdist(context, context)
    d_validation = cdist(validation, context)
    if ordinary:
        trend_context, trend_validation, trend_test = map(
            np.zeros, (len(context), len(validation), len(test))
        )
    else:
        x, xv, xt = map(vector, (data.context_x, data.validation_x, data.test_x))
        x_mean, y_mean = x.mean(0), response.mean()
        centered = x - x_mean
        penalty = RIDGE_ALPHA * np.eye(x.shape[1])
        coefficients = np.linalg.solve(
            centered.T @ centered + penalty, centered.T @ (response - y_mean)
        )
        trend_context, trend_validation, trend_test = (
            y_mean + (matrix - x_mean) @ coefficients for matrix in (x, xv, xt)
        )
    residual = response - trend_context
    best = None
    for spatial_range, nugget in itertools.product(RANGES, NUGGET_FRACTIONS):
        fit = fit_kriging(d_context, residual, spatial_range, nugget, ordinary)
        mu, sd = predict_kriging(
            fit, d_validation, trend_validation, spatial_range, nugget, ordinary
        )
        nll = np.mean(
            0.5 * np.log(2.0 * np.pi)
            + np.log(sd)
            + 0.5 * ((validation_y - mu) / sd) ** 2
        )
        if best is None or nll < best[0]:
            best = nll, spatial_range, nugget, fit
    nll, spatial_range, nugget, fit = best
    mu, sd = predict_kriging(
        fit, cdist(test, context), trend_test, spatial_range, nugget, ordinary
    )
    mean, scale = data.target_mean, data.target_std
    y = data.test_y[0].numpy() * scale + mean
    path = OUT / dataset / mode / name
    path.mkdir(parents=True, exist_ok=True)
    original_mu, original_sd = (mu[:, None] * scale + mean), (sd[:, None] * scale)
    with np.load(
        FROZEN / dataset / mode / "final" / f"seed_{seed}_mixture.npz"
    ) as reference:
        np.testing.assert_array_equal(data.test_ids, reference["ids"])
        np.testing.assert_array_equal(y, reference["y"])
    if (
        not np.isfinite(original_mu).all()
        or not np.isfinite(original_sd).all()
        or (original_sd <= 0).any()
    ):
        raise RuntimeError(
            f"Invalid predictive density for {dataset}/{mode}/{seed}/{name}"
        )
    np.savez_compressed(
        path / f"seed_{seed}_mixture.npz",
        ids=data.test_ids,
        y=y,
        mu=original_mu,
        sigma=original_sd,
        pi=np.ones_like(original_mu),
    )
    row = dict(
        dataset=dataset,
        mode=mode,
        variant=name,
        seed=seed,
        range=spatial_range,
        nugget_fraction=nugget,
        variance_scale=fit[-1],
        validation_nll=float(nll),
        elapsed_seconds=time.perf_counter() - started,
    )
    temporary = path / f"seed_{seed}_result.tmp"
    temporary.write_text(json.dumps(row, allow_nan=False), encoding="utf8")
    temporary.replace(path / f"seed_{seed}_result.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted({ds for ds, _ in PROTOCOLS})
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=range(1, 11), default=range(1, 11)
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    for dataset, mode in PROTOCOLS:
        if args.datasets and dataset not in args.datasets:
            continue
        for seed in args.seeds:
            data = load_final_split(dataset, seed, mode, "cpu")
            for name in ("ordinary_kriging", "regression_kriging"):
                path = OUT / dataset / mode / name / f"seed_{seed}_result.json"
                if not path.exists():
                    evaluate(dataset, mode, seed, name, data)
            print(f"Kriging completed {dataset}/{mode}/{seed}", flush=True)


if __name__ == "__main__":
    main()
