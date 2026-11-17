"""Final frozen-model runner. All artifacts are separate from development studies."""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import DATA_DIR, DATASETS
from final_data import load_final_split, PREPROCESSING_VERSION
from final_controls import MeanCNP, matched_cnp_width
from recent_baselines import RecentCovariateNP, BASELINE_SPEC, real_parameter_count
from training_utils import mixture_nll
from final_model import FinalGECNP, SingleGaussianControl
from geographic_kernel_model import GeographicKernelMDN
from training_utils import seed_all, sample_episode, score
from calibration import mixture_quantile, mixture_crps
from coordinate_geometry import california_geometry_metadata as spherical_metadata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/final_300e30ep"
EPOCHS = 300
EPISODES = 30
NUMERICAL_SOURCES = (
    "final_model.py",
    "geographic_kernel_model.py",
    "specialized_model.py",
    "coordinate_geometry.py",
    "final_data.py",
    "final_controls.py",
    "recent_baselines.py",
    "data.py",
    "training_utils.py",
    "run_final_experiments.py",
    "calibration.py",
)


def training_identity(dataset, seed, mode, name, config, epochs, episodes):
    source = {
        s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
        for s in NUMERICAL_SOURCES
    }
    source.update(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "external/fourier_volterra_py39/nps").rglob("*.py"))
        }
    )
    dataset_hash = hashlib.sha256(
        (DATA_DIR / DATASETS[dataset]).read_bytes()
    ).hexdigest()
    content = dict(
        source=source,
        dataset_sha256=dataset_hash,
        seed=seed,
        mode=mode,
        name=name,
        config=config,
        epochs=epochs,
        episodes=episodes,
        preprocessing=PREPROCESSING_VERSION,
    )
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


CONFIGS = {
    "final": dict(
        kind="final",
        geometry="standardized",
        spatial_kernel="gaussian",
        attribute_kernel="weighted_l1",
        gate_mode="learned",
    ),
    "equal_gate": dict(
        kind="final",
        geometry="standardized",
        spatial_kernel="gaussian",
        attribute_kernel="weighted_l1",
        gate_mode="uniform",
    ),
    "single_gaussian": dict(
        kind="single",
        geometry="standardized",
        spatial_kernel="gaussian",
        attribute_kernel="weighted_l1",
        gate_mode="learned",
    ),
    "cosine": dict(
        kind="attention",
        geometry="standardized",
        spatial_kernel="gaussian",
        attribute_kernel="cosine",
        gate_mode="learned",
    ),
    "spherical": dict(
        kind="attention",
        geometry="haversine",
        spatial_kernel="gaussian",
        attribute_kernel="weighted_l1",
        gate_mode="learned",
    ),
    "cnp_two": dict(
        kind="cnp", components=2, geometry="standardized", attribute_kernel="none"
    ),
    "cnp_single": dict(
        kind="cnp", components=1, geometry="standardized", attribute_kernel="none"
    ),
    "cnp_two_matched": dict(
        kind="cnp_matched",
        components=2,
        geometry="standardized",
        attribute_kernel="none",
    ),
    "cnp_single_matched": dict(
        kind="cnp_matched",
        components=1,
        geometry="standardized",
        attribute_kernel="none",
    ),
}
CONFIGS.update(
    {
        v: dict(
            kind="recent",
            recent_variant=v,
            geometry="standardized",
            attribute_kernel="covariate_token_features",
            paper_config=spec,
        )
        for v, spec in BASELINE_SPEC.items()
    }
)


def geometry_metadata(data, dataset):
    frame = pd.read_csv(DATA_DIR / DATASETS[dataset])
    raw = (
        frame.set_index(frame.columns[0])
        .loc[data.context_ids]
        .iloc[:, :2]
        .to_numpy(dtype=np.float64)
    )
    mean = raw.mean(0)
    sd = raw.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    scale = spherical_metadata(data, dataset)[2] if dataset == "california" else 1.0
    return mean, sd, scale


def build_model(data, mean, sd, scale, config):
    common = dict(coord_dim=data.coord_dim, attr_dim=data.attr_dim, hidden=64)
    if config["kind"] == "recent":
        return RecentCovariateNP(
            data.coord_dim, data.attr_dim, config["recent_variant"]
        )
    if config["kind"] == "final":
        return FinalGECNP(
            **common,
            gate_mode=config["gate_mode"],
            coordinate_mean=mean,
            coordinate_std=sd,
        )
    if config["kind"] == "single":
        return SingleGaussianControl(**common, coordinate_mean=mean, coordinate_std=sd)
    if config["kind"] in ("cnp", "cnp_matched"):
        width = None
        if config["kind"] == "cnp_matched":
            # Construct the reference without consuming this control's initialization RNG.
            with torch.random.fork_rng(devices=[]):
                count = sum(p.numel() for p in FinalGECNP(**common).parameters())
            width = matched_cnp_width(
                data.coord_dim, data.attr_dim, count, config["components"]
            )
        return MeanCNP(**common, components=config["components"], encoder_width=width)
    return GeographicKernelMDN(
        **common,
        coordinate_mean=mean,
        coordinate_std=sd,
        geometry_scale_km=scale,
        **{
            k: config[k]
            for k in ["geometry", "spatial_kernel", "attribute_kernel", "gate_mode"]
        },
    )


def predict(model, data, s, x, chunk_size=256):
    # Query chunking is exact for the frozen pointwise-query architecture.
    parts = []
    with torch.no_grad():
        for start in range(0, s.shape[1], chunk_size):
            out = model(
                data.context_s,
                data.context_x,
                data.context_y[..., None],
                s[:, start : start + chunk_size],
                x[:, start : start + chunk_size],
            )
            parts.append(
                {
                    k: v.detach().cpu()
                    for k, v in out.items()
                    if k
                    in (
                        "mu",
                        "sigma",
                        "log_pi",
                        "mean",
                        "variance",
                        "std",
                        "gate_weights",
                    )
                }
            )
    return {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}


def run(data, dataset, mode, seed, name, config, epochs=EPOCHS, episodes=EPISODES):
    started = time.perf_counter()
    mean, sd, scale = geometry_metadata(data, dataset)
    identity = training_identity(dataset, seed, mode, name, config, epochs, episodes)
    seed_all(seed)
    model = build_model(data, mean, sd, scale, config)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
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
        hidden=config.get("paper_config", {}).get("embed", 64),
        learning_rate=0.001,
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    p.add_argument(
        "--variants", nargs="+", choices=list(CONFIGS), default=list(CONFIGS)
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["random", "spatial_block"],
        default=["random", "spatial_block"],
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Start the selected full-budget jobs",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    # Independent per-dataset ledgers allow separate CPU workers without appending
    # to the same file. All ledgers are read on resume.
    done = set()
    for ledger in OUT.glob("*runs.jsonl"):
        for line in ledger.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["mode"], r["variant"], r["seed"]))
    jobs = []
    for ds in args.datasets:
        for mode in args.modes:
            if ds == "synthetic" and mode == "spatial_block":
                continue
            for seed in args.seeds:
                for variant in args.variants:
                    if variant == "spherical" and ds != "california":
                        continue
                    if (ds, mode, variant, seed) not in done:
                        jobs.append((ds, mode, variant, seed))
    if not args.execute:
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    epochs=EPOCHS,
                    episodes_per_epoch=EPISODES,
                    seeds=args.seeds,
                    optimizer_steps_per_job=EPOCHS * EPISODES,
                    execution="dry_run_only",
                )
            )
        )
        return
    missing_data = sorted(
        {
            DATA_DIR / DATASETS[ds]
            for ds, _, _, _ in jobs
            if not (DATA_DIR / DATASETS[ds]).is_file()
        }
    )
    if missing_data:
        p.error("Missing input CSV files: " + ", ".join(map(str, missing_data)))
    for i, (ds, mode, name, seed) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {ds}/{mode}/{name}/{seed}", flush=True)
        data = load_final_split(ds, seed, mode, "cpu")
        row = run(data, ds, mode, seed, name, CONFIGS[name])
        # Include dataset in final checkpoint folders to avoid overwriting other datasets.
        path = OUT / f"{ds}_runs.jsonl"
        with path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row["test"]), flush=True)


if __name__ == "__main__":
    main()
