"""Run all six one-factor sensitivity variants on the seven paper settings."""
import argparse
from pathlib import Path
import subprocess
import sys

from run_supplemental_sensitivity import PROTOCOLS, SETTINGS

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11))
    )
    args = parser.parse_args()
    for seed in args.seeds:
        for dataset, protocol in PROTOCOLS:
            for variant in SETTINGS:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "run_supplemental_sensitivity.py"),
                        dataset,
                        protocol,
                        variant,
                        str(seed),
                    ],
                    cwd=ROOT,
                    check=True,
                )


if __name__ == "__main__":
    main()
