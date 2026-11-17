"""Combine completed ten-seed study summaries without writing to the manuscript."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "results"


def main():
    reference = pd.read_csv(ROOT / "reference_np_300e30ep" / "comparison_summary.csv")
    kcn = pd.read_csv(ROOT / "kcn_300e30ep" / "summary_10seeds.csv")
    combined = pd.concat([reference, kcn], ignore_index=True)
    key = ["dataset", "mode", "variant"]
    if len(combined) != 114 or combined.duplicated(key).any():
        raise ValueError("Expected 114 unique method/dataset/protocol combinations")
    if not combined["nll_count"].eq(10).all():
        raise ValueError("Every configuration must have ten completed seed runs")
    destination = ROOT / "comparison_summary.csv"
    combined.sort_values(key).to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
