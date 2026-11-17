"""Additional full-budget, paired GECNP sensitivity runs.

Use the frozen suite's split, sampler, validation and prediction routines.
Run outputs and checkpoints are isolated from the frozen experiment.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from calibration import mixture_crps, mixture_quantile
from data import DATASETS
from final_data import PREPROCESSING_VERSION, load_final_split
from final_model import FinalGECNP
from training_utils import mixture_nll
from recent_baselines import real_parameter_count
from training_utils import sample_episode, score, seed_all
from run_final_experiments import (
    EPOCHS,
    EPISODES,
    OUT as FROZEN,
    geometry_metadata,
    predict,
    training_identity,
)

OUT = Path(__file__).resolve().parent / "results" / "supplemental_300e30ep"
SETTINGS = {
    "gecnp_k3": (3, 0.001, 64),
    "gecnp_k4": (4, 0.001, 64),
    "gecnp_lr_low": (2, 0.0003, 64),
    "gecnp_lr_high": (2, 0.003, 64),
    "gecnp_width32": (2, 0.001, 32),
    "gecnp_width128": (2, 0.001, 128),
}
PROTOCOLS = [
    (ds, mode)
    for ds in DATASETS
    for mode in ("random", "spatial_block")
    if not (ds == "synthetic" and mode == "spatial_block")
]


def config_for(name):
    components, learning_rate, hidden = SETTINGS[name]
    return dict(components=components, learning_rate=learning_rate, hidden=hidden)


def build_sensitivity_model(data, mean, sd, config):
    model = FinalGECNP(
        data.coord_dim,
        data.attr_dim,
        hidden=config["hidden"],
        coordinate_mean=mean,
        coordinate_std=sd,
    )
    count = config["components"]
    if count != 2:
        old = model.prediction_head[-1]
        new = nn.Linear(old.in_features, 3 * count)
        # Keep the corresponding locations, scales and logits of the original
        # two components. Additional components have independent initial weights.
        with torch.no_grad():
            for family in range(3):
                start = family * count
                new.weight[start : start + 2].copy_(
                    old.weight[family * 2 : family * 2 + 2]
                )
                new.bias[start : start + 2].copy_(old.bias[family * 2 : family * 2 + 2])
        model.prediction_head[-1] = new
        model.components = count
    return model


def train(data, dataset, mode, seed, name, config, epochs=EPOCHS, episodes=EPISODES):
    started = time.perf_counter()
    mean, sd, scale = geometry_metadata(data, dataset)
    identity = hashlib.sha256(
        (
            training_identity(dataset, seed, mode, name, config, epochs, episodes)
            + Path(__file__).read_bytes().hex()
        ).encode()
    ).hexdigest()
    seed_all(seed)
    model = build_sensitivity_model(data, mean, sd, config)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    seed_all(seed + 900000)
    episode_digest = ""
    best = math.inf
    best_state = None
    best_epoch = 0
    start_epoch = 0
    curve = []
    previous_seconds = 0.0
    folder = OUT / dataset / mode / name
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / f"seed_{seed}_resume.pt"
    if resume.exists():
        import random

        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("training_identity") != identity:
            raise RuntimeError(
                "Source, data, or configuration changed since checkpoint; refusing mixed-provenance resume"
            )
        if (saved["epochs"], saved["episodes"], saved["preprocessing"]) != (
            epochs,
            episodes,
            PREPROCESSING_VERSION,
        ):
            raise RuntimeError("Incompatible resume budget or preprocessing")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        best = saved["best"]
        best_epoch = saved["best_epoch"]
        best_state = saved["best_state"]
        start_epoch = saved["epoch"] + 1
        episode_digest = saved["episode_digest"]
        curve = saved["learning_curve"]
        previous_seconds = saved["elapsed_seconds"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for step in range(episodes):
            cs, cx, cy, ts, tx, ty = sample_episode(
                data.context_s, data.context_x, data.context_y
            )
            # Include attributes/responses too: coincident coordinates need not identify
            # the same observation (particularly in Chengdu).
            episode_digest = hashlib.sha256(
                episode_digest.encode()
                + b"".join(t.numpy().tobytes() for t in (cs, cx, cy, ts, tx, ty))
            ).hexdigest()
            out = model(cs, cx, cy, ts, tx)
            loss = mixture_nll(out, ty)
            if not torch.isfinite(loss):
                raise RuntimeError((name, mode, seed, "nonfinite loss"))
            train_loss += loss.detach().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mixture_nll(
                predict(model, data, data.validation_s, data.validation_x),
                data.validation_y,
            ).item()
        curve.append(
            dict(
                epoch=epoch + 1,
                train_nll_scaled=train_loss / episodes,
                validation_nll_scaled=v,
            )
        )
        if v < best:
            best = v
            best_epoch = epoch + 1
            best_state = {
                k: t.detach().cpu().clone() for k, t in model.state_dict().items()
            }
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            import random

            snapshot = dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                best=best,
                best_epoch=best_epoch,
                best_state=best_state,
                epoch=epoch,
                epochs=epochs,
                episodes=episodes,
                preprocessing=PREPROCESSING_VERSION,
                training_identity=identity,
                episode_digest=episode_digest,
                learning_curve=curve,
                elapsed_seconds=previous_seconds + time.perf_counter() - started,
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
            )
            temp = resume.with_suffix(".tmp")
            torch.save(snapshot, temp)
            temp.replace(resume)
            print(
                f"  epoch {epoch+1}/{epochs}; validation NLL {v:.5f}; best {best:.5f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = predict(model, data, data.test_s, data.test_x)
        metrics = score(out, data.test_y, data.target_std)
        metrics["moment_matched_coverage90"] = metrics.pop("coverage_90")
        metrics["moment_matched_width90"] = metrics.pop("mean_interval_width_90")
        ys, ym = data.target_std, data.target_mean
        y = data.test_y[0].numpy() * ys + ym
        mu = out["mu"][0].numpy() * ys + ym
        sigma = out["sigma"][0].numpy() * ys
        pi = out["log_pi"][0].exp().numpy()
        lo = mixture_quantile(0.05, mu, sigma, pi)
        hi = mixture_quantile(0.95, mu, sigma, pi)
        metrics.update(
            exact_coverage90=float(np.mean((y >= lo) & (y <= hi))),
            exact_width90=float(np.mean(hi - lo)),
            crps=float(np.mean(mixture_crps(y, mu, sigma, pi))),
        )
        diagnostics = {}
        if "gate_weights" in out:
            g = out["gate_weights"][0, :512].numpy()
            gen = torch.Generator().manual_seed(seed + 40000)
            ix = torch.randperm(data.context_s.shape[1], generator=gen)[
                : data.context_s.shape[1] // 2
            ]
            half = model(
                data.context_s[:, ix],
                data.context_x[:, ix],
                data.context_y[:, ix, None],
                data.test_s[:, :512],
                data.test_x[:, :512],
            )["gate_weights"][0].numpy()
            diagnostics.update(
                spatial_gate_mean=float(g[:, 0].mean()),
                spatial_gate_query_sd=float(g[:, 0].std()),
                half_context_gate_l1=float(np.abs(g - half).sum(-1).mean()),
            )
        diagnostics.update(
            spatial_bandwidth=float(
                model.log_distance_temperature.exp().clamp_min(1e-4).sqrt()
            )
            if hasattr(model, "log_distance_temperature")
            else None,
            attribute_temperature=float(
                model.log_similarity_temperature.exp().clamp_min(1e-4)
            )
            if hasattr(model, "log_similarity_temperature")
            else None,
        )
        if config.get("geometry") == "haversine":
            diagnostics["spatial_bandwidth_km"] = (
                diagnostics["spatial_bandwidth"] * scale
            )
        if config.get("attribute_kernel") == "weighted_l1":
            diagnostics["attribute_weights"] = torch.softmax(
                model.attribute_logits, 0
            ).tolist()
    torch.save(best_state, folder / f"seed_{seed}_best.pt")
    np.savez_compressed(
        folder / f"seed_{seed}_mixture.npz",
        ids=data.test_ids,
        y=y,
        mu=mu,
        sigma=sigma,
        pi=pi,
    )
    return dict(
        dataset=dataset,
        mode=mode,
        variant=name,
        seed=seed,
        config=config,
        epochs=epochs,
        episodes_per_epoch=episodes,
        optimizer_steps=epochs * episodes,
        preprocessing=PREPROCESSING_VERSION,
        training_identity=identity,
        hidden=config["hidden"],
        learning_rate=config["learning_rate"],
        best_validation_nll_scaled=best,
        best_validation_nll=best + math.log(ys),
        best_epoch=best_epoch,
        test=metrics,
        diagnostics=diagnostics,
        parameters=real_parameter_count(model),
        episode_sha256=episode_digest,
        coordinate_mean=mean.tolist(),
        coordinate_std=sd.tolist(),
        distance_scale_km=scale,
        learning_curve=curve,
        elapsed_seconds=previous_seconds + time.perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("mode", choices=["random", "spatial_block"])
    parser.add_argument("variant", choices=list(SETTINGS))
    parser.add_argument("seed", type=int, choices=range(1, 11))
    args = parser.parse_args()
    if (args.dataset, args.mode) not in PROTOCOLS:
        parser.error("unsupported dataset/protocol")
    torch.set_num_threads(1)
    config = config_for(args.variant)
    path = (
        OUT / args.dataset / args.mode / args.variant / f"seed_{args.seed}_result.json"
    )
    if path.is_file():
        row = json.loads(path.read_text(encoding="utf8"))
        expected = hashlib.sha256(
            (
                training_identity(
                    args.dataset,
                    args.seed,
                    args.mode,
                    args.variant,
                    config,
                    EPOCHS,
                    EPISODES,
                )
                + Path(__file__).read_bytes().hex()
            ).encode()
        ).hexdigest()
        if row["training_identity"] != expected:
            raise RuntimeError(f"Cannot reuse an incompatible completed result: {path}")
        return
    with (FROZEN / f"{args.dataset}_runs.jsonl").open(encoding="utf8") as source:
        reference = next(
            row
            for row in map(json.loads, source)
            if row["mode"] == args.mode
            and row["variant"] == "final"
            and row["seed"] == args.seed
        )
    data = load_final_split(args.dataset, args.seed, args.mode, "cpu")
    row = train(data, args.dataset, args.mode, args.seed, args.variant, config)
    if row["episode_sha256"] != reference["episode_sha256"]:
        raise RuntimeError(
            f"Training episodes differ from the paired reference: {path}"
        )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, allow_nan=False), encoding="utf8")
    tmp.replace(path)
    print(
        f"Completed {args.dataset}/{args.mode}/{args.variant}/{args.seed}", flush=True
    )


if __name__ == "__main__":
    main()
