# GENP | Gated Expert Conditional Neural Process (GECNP)

Source code for probabilistic spatial prediction from sparse observations. GECNP
aggregates observed responses with two query-dependent attention kernels, one
for coordinates and one for covariates. The common expert mean-pools a separate
coordinate-and-covariate encoding. A two-way gate mixes the local expert
representations. A single mixture density head outputs Gaussian means, positive
standard deviations, and normalized weights.

## Repository layout

| Directory | Contents |
| --- | --- |
| gecnp/ | Model architecture and matched GECNP controls, data preparation, shared training utilities and predictive scoring. |
| baselines/ | CNP, ANP, SMACNP, Fourier and transformer adaptations, KCN, and kriging. The _vendor/ directory contains the imported upstream cores used by these methods. |
| experiments/ | Training, sensitivity runs, independent synthetic evaluation and result summaries. |
| figures/ | Plotting scripts that read locally generated data and experiment archives. |

Run the module commands below from the repository root. Source fingerprints
include the new file paths. Use a fresh results directory when running this
layout to keep its checkpoints and ledgers separate from earlier releases.

## Installation

Use Python 3.9 or later in a fresh environment. The manuscript runs used Python
3.9.23 and PyTorch 2.7.1 on CPU with one PyTorch thread per worker.

~~~bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m experiments.train
python -m experiments.summarize_main --self-check
~~~

The first command in the last two lines is a dry run listing the 790 reference
jobs. The second checks sign-flip enumeration, Holm adjustment, and mixture
quantile scoring without input data.

## Benchmark inputs

Place the prepared CSV files below in the local data directory, or set the
GENP_DATA_DIR environment variable to the directory containing them. Files are
read in column order: unique observation ID, two coordinate columns, covariate
columns, and response in the last column. Preserve row order, units, and column
names to reproduce the published splits and preprocessing.

| Key | Prepared file | Input notes |
| --- | --- | --- |
| chengdu | Chengdu_housing.csv | 14,838 Tianchi housing records. Nominal columns end in _Label and use a context-fitted one-hot encoding with an unknown level. [Source](https://tianchi.aliyun.com/dataset/154888). |
| california | fetch_california_housing.csv | 20,640 California Housing records. Coordinate columns must be named Latitude, Longitude in that order. Keep the eight predictors in their original order. Response is in USD 100,000. |
| election | election result.csv | 3,107 county records from the [Election benchmark](https://doi.org/10.1145/3394486.3403101). Response is 2016 Republican votes in units of 10,000. Predictors include same-year Democratic votes. |
| synthetic | generated_data_with_target_T=3.csv | Archived 50 by 50 training field. Columns are id, x, y, f1, f2, f3, T, y_target. The independent test generator uses a separately drawn field and cannot replace the archived training input for numeric reproduction. |

This repository ships code, model configuration, and setup documentation.
Obtain the exact prepared benchmark files from the authors for numeric
reproduction. Raw data sources alone may differ in record selection, formatting,
or coordinate matching. Generated logs, scores, predictive archives, plots, and
checkpoints are written locally under results/ and ignored by Git.

### Evaluation protocol

- Random holdout reserves about 85% of observations for test, 10% for the
  context-training pool, and 5% for validation.
- Spatial block holdout uses the quadrant at or above both coordinate medians
  for test. Roughly 10% of the full dataset is sampled as the context-training
  pool from the non-test region. Validation uses remaining non-test records,
  capped at 2,048. Only the three observational datasets use this protocol.
- Fit standardization, California tail handling, and Chengdu category mapping
  using the context-training pool. Validation and test use this pool as context.
- Neural methods share seed-specific splits and 300 epochs of 30 disjoint
  context/target episodes. Episodes use a mixture of random and buffered spatial
  sampling; validation selects a checkpoint. Kriging uses the context-training
  set directly and selects covariance settings on validation.
- All scores are on the input response scale. Report point predictions from
  the mixture mean, NLL from the actual predictive density, CRPS from that
  density, and interval coverage and width from numerically inverted mixture
  quantiles. For a single Gaussian, the mixture has one component. ANP averages
  32 conditional densities from context-prior draws for predictive scoring.

## Run the comparison and ablation experiments

Run from the repository root. First check an experiment selection, then add
--execute to start. For example, a full-budget single-seed California run:

~~~bash
python -m experiments.train --datasets california --modes random --variants final equal_gate single_gaussian cosine spherical --seeds 1
python -m experiments.train --datasets california --modes random --variants final equal_gate single_gaussian cosine spherical --seeds 1 --execute
~~~

The following commands run the full comparison suite in dependency order.
The first training command is computationally expensive even on CPU workers.

~~~bash
python -m experiments.train --execute
python -m experiments.summarize_main
python -m baselines.kriging
python -m experiments.summarize_supplemental --kriging-only
python -m experiments.reference_np --suite
python -m baselines.kcn
python -m experiments.summarize_comparisons
~~~

| Runner | Variants and scope |
| --- | --- |
| experiments/train.py | Reference GECNP; equal gate; single Gaussian; cosine attribute attention; California-only spherical distance; standard and parameter-matched CNP with one and two components; covariate-aware SFConvCNP, SFVConvCNP and TE-ISTNP. |
| baselines/kriging.py | Ordinary and regression kriging on the same split and evaluation records. |
| experiments/reference_np.py | Standard ANP and SMACNP under the shared episode budget. To run just one job, supply dataset, mode, variant (anp_standard or smacnp), and seed, in that order. |
| baselines/kcn.py | Query-centered weighted graph convolution, five spatial neighbors, masked query label, Gaussian predictive head. |
| experiments/sensitivity.py | K=3 and K=4, Adam learning rate 0.0003 and 0.003, and hidden widths 32 and 128. K=1 is the single_gaussian control above. |

The supplemental baseline runners use the matching reference GECNP artifacts
to verify identical test IDs and, for neural baselines, episode hashes. Run
the reference suite first. The main runner writes one ledger per dataset and
skips completed jobs on restart. Use a separate checkout when revising inputs
or hyperparameters, and assign each dataset to at most one worker.

### Hyperparameter sensitivity and mechanism diagnostics

~~~bash
python -m experiments.sensitivity --suite --seeds 1 2 3 4 5 6 7 8 9 10
python -m experiments.summarize_supplemental --seeds 10
python -m figures.ablations_and_maps --component-ablations
python -m figures.ablations_and_maps --cnp-extensions
python -m figures.ablations_and_maps
python -m figures.context_disagreement --collect
python -m experiments.independent_synthetic --generate
python -m experiments.independent_synthetic
python -m figures.benchmark
python -m figures.predictive_density
~~~

Sensitivity runs change one factor at a time from K=2, learning rate 0.001,
and width 64. The context-disagreement diagnostic uses the trained California
spatial checkpoints to compare the spatial and attribute rankings and their
attention-weighted observed responses. The independent synthetic test
reuses the ten synthetic random-holdout reference and CNP checkpoints on
one newly generated 2,500-point field.

### Reading local outputs

- results/final_300e30ep/phase2_10seeds/ contains main means, sample SDs,
  per-seed scores and paired differences. Comparisons use a two-sided exact
  paired sign-flip test and a Holm correction across the main test family.
- results/comparison_summary.csv combines main and supplemental methods into
  one row per dataset, protocol and method. Rank comparisons within each
  dataset and protocol. For nominal 90% coverage, rank the absolute
  difference from 0.90, separately from interval width.
- results/supplemental_300e30ep/sensitivity_10seeds/ contains one-factor
  sensitivity summaries and its paired NLL figure.
- results/figures/ contains regenerated mechanism, ablation and diagnostic
  figures. results/independent_synthetic_grid/ contains the locally generated
  test field and its scores.

## Baseline provenance

- CNP: [Garnelo et al., ICML 2018](https://proceedings.mlr.press/v80/garnelo18a.html).
- ANP: [Kim et al., ICLR 2019](https://arxiv.org/abs/1901.05761).
- SMACNP: [Bao et al., Neural Networks 2024](https://doi.org/10.1016/j.neunet.2024.106201).
  The author-released model and attention source files come from
  [SMACNP](https://github.com/bll744958765/SMACNP/tree/a48531306e55f732623db52bdf9b6b9c835a4244);
  the adapter evaluates its variance as a Gaussian standard deviation after
  taking the square root.
- KCN: [Appleby et al., AAAI 2020](https://doi.org/10.1609/aaai.v34i04.5716),
  adapted from the [Tufts PyTorch reference](https://github.com/tufts-ml/kcn-torch/tree/d0cdb5e99d0f62d207faaba7377468d10e3ca293).
- Fourier baselines: [Mohseni et al., ICML 2026](https://proceedings.mlr.press/v306/mohseni26a.html).
  TE-ISTNP: [Ashman et al., ICML 2024](https://proceedings.mlr.press/v235/ashman24a.html).
  Their adapters use the pinned [Fourier/Volterra author code](https://github.com/peiman-m/fourier-volterra-nps/tree/25c3b0bd4b27a3eda0cea099effc2ee450a87c49)
  with the Python 3.9 compatibility changes from the study. Its MIT licence is included.
- Kriging: [Matheron, 1963](https://doi.org/10.2113/gsecongeo.58.8.1246) and
  [Hengl et al., 2007](https://doi.org/10.1016/j.cageo.2007.05.001).

These comparisons evaluate study-specific implementations and configurations
under a common protocol. Their scores describe that evaluation.
