"""Run standard ANP and released-core SMACNP on the frozen evaluation splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from final_data import load_final_split
from np_reference_models import AttentiveNeuralProcess, SMACNP, marginal_nll
from training_utils import sample_episode, seed_all
from run_final_experiments import EPOCHS, EPISODES, OUT as FROZEN, training_identity
from run_supplemental_sensitivity import OUT as SUPPLEMENTAL, PROTOCOLS
from summarize_supplemental import archive_metrics, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reference_np_300e30ep"
VARIANTS = ("anp_standard", "smacnp")
DRAWS = 32
WIDTH = {"anp_standard": 64, "smacnp": 128}


def write_json(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, allow_nan=False, indent=2), encoding="utf8")
    tmp.replace(path)


def build(data, variant):
    if variant == "anp_standard":
        return AttentiveNeuralProcess(data.coord_dim + data.attr_dim, WIDTH[variant])
    return SMACNP(data.attr_dim, WIDTH[variant])


def fit(ds, mode, variant, seed, out=OUT, epochs=EPOCHS, episodes=EPISODES):
    torch.set_num_threads(1)
    folder = out / ds / mode / variant
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"seed_{seed}_result.json"
    count = DRAWS if variant == "anp_standard" else 1
    config = dict(width=WIDTH[variant], draws=count, learning_rate=0.001)
    frozen_identity = training_identity(
        ds, seed, mode, variant, config, epochs, episodes
    )
    own_sources = [Path(__file__), ROOT / "np_reference_models.py"]
    if variant == "smacnp":
        own_sources.extend(
            ROOT / "external/smacnp_upstream" / name
            for name in ("model_smanp.py", "attention.py")
        )
    identity = hashlib.sha256(
        frozen_identity.encode() + b"".join(p.read_bytes() for p in own_sources)
    ).hexdigest()
    if result.exists():
        assert json.loads(result.read_text(encoding="utf8"))["identity"] == identity
        return
    data = load_final_split(ds, seed, mode, "cpu")
    seed_all(seed)
    model = build(data, variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    noise = torch.randn(
        count, config["width"], generator=torch.Generator().manual_seed(seed + 67000)
    )
    model_rng = torch.Generator().manual_seed(seed + 44000).get_state()
    seed_all(seed + 900000)
    digest, start, best, best_epoch, best_state = "", 0, float("inf"), 0, None
    resume = folder / f"seed_{seed}_resume.pt"
    if resume.exists():
        state = torch.load(resume, weights_only=False, map_location="cpu")
        if state["identity"] != identity:
            raise RuntimeError(f"Run fingerprint changed: {resume}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        digest, start, best = state["digest"], state["epoch"], state["best"]
        best_epoch, best_state, model_rng = (
            state["best_epoch"],
            state["best_state"],
            state["model_rng"],
        )
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["sampler_rng"])
    started = time.perf_counter()
    context = data.context_s, data.context_x, data.context_y[..., None]
    for epoch in range(start, epochs):
        model.train()
        for _ in range(episodes):
            episode = sample_episode(data.context_s, data.context_x, data.context_y)
            digest = hashlib.sha256(
                digest.encode() + b"".join(t.numpy().tobytes() for t in episode)
            ).hexdigest()
            # Isolate model randomness so dropout and latent draws do not
            # change the frozen sequence of paired training episodes.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(model_rng)
                loss = model.training_loss(*episode)
                model_rng = torch.get_rng_state()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training objective: {ds}/{mode}/{variant}/{seed}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model.predictions(
                *context, data.validation_s, data.validation_x, noise
            )
            value = marginal_nll(*predictions, data.validation_y).item()
        if not np.isfinite(value):
            raise RuntimeError(
                f"Non-finite validation NLL: {ds}/{mode}/{variant}/{seed}"
            )
        if value < best:
            best, best_epoch = value, epoch + 1
            best_state = {
                key: val.detach().clone() for key, val in model.state_dict().items()
            }
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            tmp = resume.with_suffix(".tmp")
            torch.save(
                dict(
                    identity=identity,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    digest=digest,
                    epoch=epoch + 1,
                    best=best,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    model_rng=model_rng,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    sampler_rng=torch.get_rng_state(),
                ),
                tmp,
            )
            tmp.replace(resume)
            print(f"epoch {epoch + 1}/{epochs} validation NLL {value:.5f}", flush=True)
    reference = FROZEN / ds / mode / "final" / f"seed_{seed}_mixture.npz"
    if epochs == EPOCHS and episodes == EPISODES:
        with (FROZEN / f"{ds}_runs.jsonl").open(encoding="utf8") as source:
            match = next(
                row
                for row in map(json.loads, source)
                if row["mode"] == mode
                and row["seed"] == seed
                and row["variant"] == "final"
            )
        if digest != match["episode_sha256"]:
            raise RuntimeError("Training episodes differ from the frozen reference")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mu, sigma = model.predictions(*context, data.test_s, data.test_x, noise)
    mean, scale = data.target_mean, data.target_std
    y = data.test_y[0].numpy() * scale + mean
    archive = folder / f"seed_{seed}_mixture.npz"
    np.savez_compressed(
        archive,
        ids=data.test_ids,
        y=y,
        mu=mu[0].numpy() * scale + mean,
        sigma=sigma[0].numpy() * scale,
        pi=np.full(mu[0].shape, 1 / count, dtype=np.float32),
    )
    metrics = archive_metrics(archive, reference, count)
    torch.save(best_state, folder / f"seed_{seed}_best.pt")
    write_json(
        result,
        dict(
            dataset=ds,
            mode=mode,
            variant=variant,
            seed=seed,
            identity=identity,
            epochs=epochs,
            episodes_per_epoch=episodes,
            config=config,
            best_epoch=best_epoch,
            validation_nll_scaled=best,
            episode_sha256=digest,
            elapsed_seconds=time.perf_counter() - started,
            **metrics,
        ),
    )


def suite():
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = OUT / "suite_status.json"
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    env["PATH"] = (
        str(Path(sys.executable).parent / "Library" / "bin") + os.pathsep + env["PATH"]
    )

    def launch(job):
        with (logs / ("_".join(map(str, job)) + ".log")).open(
            "a", encoding="utf8"
        ) as stream:
            subprocess.run(
                [sys.executable, "-u", __file__, *map(str, job)],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        for seeds in (range(1, 6), range(6, 11)):
            write_json(
                status,
                dict(state="running", seeds=list(seeds), pid=os.getpid(), expected=140),
            )
            jobs = [
                (ds, mode, variant, seed)
                for seed in seeds
                for variant in VARIANTS
                for ds, mode in PROTOCOLS
            ]
            with ThreadPoolExecutor(max_workers=2) as workers:
                list(workers.map(launch, jobs))
            rows = [
                json.loads(path.read_text(encoding="utf8"))
                for path in OUT.glob("*/*/*/*_result.json")
            ]
            added = pd.DataFrame(rows)
            added = added[added.seed <= seeds.stop - 1].sort_values(
                ["dataset", "mode", "variant", "seed"]
            )
            expected = 2 * len(PROTOCOLS) * (seeds.stop - 1)
            assert len(added) == expected
            added.to_csv(OUT / f"metrics_{seeds.stop - 1}seeds.csv", index=False)
        prior = pd.read_csv(SUPPLEMENTAL / "kriging_10seeds" / "summary.csv")
        combined = pd.concat([prior, summarize(added)], ignore_index=True)
        assert len(combined) == 107 and (combined.nll_count == 10).all()
        combined.to_csv(OUT / "comparison_summary.csv", index=False)
        write_json(status, dict(state="complete", completed=140))
    except Exception as exc:
        write_json(status, dict(state="failed", error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("mode", nargs="?")
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    parser.add_argument("seed", nargs="?", type=int, choices=range(1, 11))
    parser.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    if args.suite:
        suite()
    else:
        if (
            (args.dataset, args.mode) not in PROTOCOLS
            or args.variant is None
            or args.seed is None
        ):
            parser.error(
                "Provide a valid dataset, holdout protocol, baseline and seed."
            )
        fit(args.dataset, args.mode, args.variant, args.seed)


if __name__ == "__main__":
    main()
