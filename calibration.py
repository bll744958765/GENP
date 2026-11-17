from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LEVELS = (0.50, 0.80, 0.90)


def mixture_cdf(q, mu, sigma, pi):
    return np.sum(pi * ndtr((q[..., None] - mu) / sigma), axis=-1)


def mixture_quantile(level, mu, sigma, pi):
    lo = np.min(mu - 8.0 * sigma, axis=-1)
    hi = np.max(mu + 8.0 * sigma, axis=-1)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        below = mixture_cdf(mid, mu, sigma, pi) < level
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def mixture_crps(y, mu, sigma, pi):
    z = (y[:, None] - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    first = np.sum(
        pi * (2.0 * sigma * phi + (y[:, None] - mu) * (2.0 * ndtr(z) - 1.0)), axis=-1
    )
    diff = mu[:, :, None] - mu[:, None, :]
    pair_sd = np.sqrt(sigma[:, :, None] ** 2 + sigma[:, None, :] ** 2)
    z_pair = diff / pair_sd
    phi_pair = np.exp(-0.5 * z_pair * z_pair) / np.sqrt(2.0 * np.pi)
    a_pair = 2.0 * pair_sd * phi_pair + diff * (2.0 * ndtr(z_pair) - 1.0)
    second = 0.5 * np.sum(pi[:, :, None] * pi[:, None, :] * a_pair, axis=(1, 2))
    return first - second


def analyze(path: Path):
    d = np.load(path)
    y, mu, sigma, pi = d["y"], d["mu"], d["sigma"], d["pi"]
    pit = mixture_cdf(y, mu, sigma, pi)
    row = {
        "mode": path.parents[2].name,
        "dataset": path.parents[1].name,
        "variant": path.parents[0].name,
        "seed": int(path.stem.split("_")[1]),
        "crps": float(np.mean(mixture_crps(y, mu, sigma, pi))),
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "pit_histogram_l1": float(
            np.abs(
                np.histogram(pit, bins=10, range=(0, 1), density=True)[0] - 1.0
            ).mean()
        ),
    }
    for level in LEVELS:
        alpha = (1.0 - level) / 2.0
        lower = mixture_quantile(alpha, mu, sigma, pi)
        upper = mixture_quantile(1.0 - alpha, mu, sigma, pi)
        key = int(round(100 * level))
        row[f"coverage_{key}"] = float(np.mean((y >= lower) & (y <= upper)))
        row[f"width_{key}"] = float(np.mean(upper - lower))
    return row


def main():
    files = sorted(RESULTS.glob("*/*/*/*_mixture.npz"))
    rows = [analyze(p) for p in files]
    if not rows:
        raise SystemExit("No exported mixture parameter files found.")
    out = RESULTS / "calibration_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Analyzed {len(rows)} saved predictive mixtures: {out}")


if __name__ == "__main__":
    main()
