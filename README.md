# haramo

**H**olistic **A**utoML-driven **R**obust pipeline for **A**pplied **M**ulti-**O**mics

Authors: Nikolay Simankov & Helene Soyeurt

---

## Overview

haramo is an AutoML pipeline for binary classification on high-dimensional tabular data, particularly multi-omics datasets. It automates feature selection, scaling, algorithm selection, and hyperparameter tuning inside a nested cross-validation framework, and reports a comprehensive set of classification metrics on the outer folds.

---

## Installation

```bash
pip install haramo
pip install greedyboruta
```

---

## How it works

The pipeline has four sequential steps optimized jointly by Optuna:

```
VarianceThreshold → Feature Selector → Scaler → Classifier
```

**Outer loop** (4-fold `StratifiedKFold` or `StratifiedGroupKFold`): splits data into train/test sets for unbiased evaluation.

**Inner loop** (4-fold `StratifiedKFold`): runs Optuna search on the training split of each outer fold.

haramo is designed to be run in two stages:

### Stage 1 — Pipeline structure search (`hyperparameters="default"`)

Uses `GridSampler` to exhaustively evaluate all combinations of algorithm, scaler, and feature selector with default hyperparameters. With all options open this covers up to 240 configurations (15 algorithms × 4 scalers × 4 feature selector combinations) per outer fold. The goal is to identify which pipeline structures work best on the data.

### Stage 2 — Hyperparameter optimization (`hyperparameters="optimize"`)

Takes the top pipeline structures identified in stage 1 and runs Bayesian (`TPESampler`, multivariate) hyperparameter search on each one, tuning every step of the pipeline jointly.

**Model selection**: the best pipeline across outer folds is selected by maximum MCC.

---

## Quick start

```python
from pathlib import Path
import pandas as pd
from haramo.classification import magic_now

X = pd.read_csv("features.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()

# --- Stage 1: find the best pipeline structures ---
_, _, studies = magic_now(
    X=X,
    y=y,
    hyperparameters="default",
    output_dir=Path("results/stage1"),
    tag="_stage1",
)

# Extract top 3 structures from any fold (they are consistent across folds)
top3 = sorted(studies["fold_1"].trials, key=lambda t: t.value, reverse=True)[:3]
for i, trial in enumerate(top3):
    print(f"#{i+1}: {trial.params}  score={trial.value:.4f}")

# --- Stage 2: optimize hyperparameters of the best structure ---
# Example: stage 1 identified LGBM + robust scaler + boruta as best
validation, pipeline, studies = magic_now(
    X=X,
    y=y,
    algorithm="LGBM",
    scaler="robust",
    feature_selector="boruta",
    hyperparameters="optimize",
    n_trials=200,
    output_dir=Path("results/stage2"),
    tag="_lgbm_robust_boruta",
)

print(validation)
```

---

## API reference

### `magic_now`

```python
from haramo.classification import magic_now

validation, pipeline, studies = magic_now(
    X,
    y,
    scoring="balanced_accuracy",
    task="classification",
    feature_selector="optimize",
    scaler="optimize",
    algorithm="optimize",
    hyperparameters="optimize",
    random_state=42,
    n_trials=100,
    output_dir=None,
    tag="",
    groups=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Feature matrix. Must be a DataFrame with named columns. |
| `y` | `pd.Series` | — | Binary target vector. |
| `scoring` | `str` or callable | `"balanced_accuracy"` | Scoring metric for the inner Optuna objective. Any sklearn scorer name or callable. |
| `task` | `str` | `"classification"` | Task type. Currently only `"classification"` is supported. |
| `feature_selector` | `str` or `list` | `"optimize"` | Feature selection strategy. See [Feature selection](#feature-selection). |
| `scaler` | `str` or `list` | `"optimize"` | Scaling strategy. See [Scalers](#scalers). |
| `algorithm` | `str` or `list` | `"optimize"` | Classifier strategy. See [Algorithms](#algorithms). |
| `hyperparameters` | `str` | `"optimize"` | `"optimize"` tunes all hyperparameters via Optuna. `"default"` fixes classifier hyperparameters and grid-searches only the pipeline structure. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |
| `n_trials` | `int` | `100` | Number of Optuna trials per outer fold. Ignored when `hyperparameters="default"`. |
| `output_dir` | `Path` | `None` | **Required.** Root directory for all outputs. Created automatically with subdirectories. |
| `tag` | `str` | `""` | Optional suffix appended to all output filenames. |
| `groups` | array-like | `None` | Group labels for samples. When provided, uses `StratifiedGroupKFold` instead of `StratifiedKFold` for the outer loop, preventing data leakage across groups (e.g. repeated measures, patients). |

**Returns**

| Name | Type | Description |
|---|---|---|
| `validation` | `pd.DataFrame` | Metrics per outer fold (rows = folds). Columns: MCC, F1-score, Kappa, Bal. Acc., Precision, Sensitivity, Selectivity. |
| `pipeline` | `sklearn.pipeline.Pipeline` | Best pipeline selected by maximum MCC across outer folds. |
| `studies` | `dict` | Optuna `Study` objects keyed by fold name (`"fold_1"`, ..., `"fold_4"`). |

**Output files**

```
output_dir/
    results/  validation{tag}.tsv    # validation metrics per fold
    models/   pipelines{tag}.pkl     # best fitted pipeline
    trials/   studies{tag}.pkl       # Optuna study objects
```

---

### `train`

Lower-level function that runs the Optuna search on each outer fold and returns unfitted best pipelines.

```python
from haramo.classification import train

pipelines, studies = train(
    X, y,
    scoring="balanced_accuracy",
    task="classification",
    feature_selector="optimize",
    scaler="optimize",
    algorithm="optimize",
    hyperparameters="optimize",
    random_state=42,
    n_trials=100,
    groups=None,
)
```

Returns `pipelines` (dict of `Pipeline` keyed by fold) and `studies` (dict of Optuna `Study`).

---

### `nested_crossval`

Retrains a set of pipelines on the outer folds and collects predictions for evaluation.

```python
from haramo.classification import nested_crossval

validation, best_pipeline = nested_crossval(
    X, y,
    pipelines=pipelines,
    random_state=42,
    groups=None,
)
```

Returns the validation `DataFrame` and the best pipeline by MCC.

---

## Feature selection

Controlled by the `feature_selector` parameter.

| Value | Behaviour |
|---|---|
| `"optimize"` | Optuna decides whether to apply a p-value filter and/or a GreedyBoruta filter (all four combinations are searchable). |
| `"pvalue"` | Only p-value filter (Pearson correlation, threshold optimized by Optuna). |
| `"boruta"` | Only GreedyBoruta filter. |
| `None` | No feature selection (identity). |
| `list` | Optuna picks from the provided list, e.g. `["pvalue", "boruta", None]`. |

A `VarianceThreshold` step always precedes the feature selector in the pipeline (threshold is also optimized).

**GreedyBoruta hyperparameters** (when `hyperparameters="optimize"`):

- `perc` — percentile threshold (80–100, step 10)
- `boruta_max_leaf_nodes` — max leaf nodes of the internal RF (10–50, step 10)

---

## Scalers

Controlled by the `scaler` parameter.

| Value | Behaviour |
|---|---|
| `"optimize"` | Optuna selects among `None`, `"standard"`, `"minmax"`, `"robust"`. |
| `"standard"` | `StandardScaler` (`with_mean` and `with_std` optimized). |
| `"minmax"` | `MinMaxScaler` (feature range optimized: `(0,1)` or `(-1,1)`). |
| `"robust"` | `RobustScaler` (centering, scaling, quantile range optimized). |
| `None` | No scaling (identity). |
| `list` | Optuna picks from the provided list, e.g. `["standard", "robust"]`. |

---

## Algorithms

Controlled by the `algorithm` parameter.

| Key | Model |
|---|---|
| `"LSVM"` | `SVC` with linear or polynomial kernel |
| `"RBFSVM"` | `SVC` with RBF kernel |
| `"NuLSVM"` | `NuSVC` with linear or polynomial kernel |
| `"NuRBFSVM"` | `NuSVC` with RBF kernel |
| `"SGD"` | `SGDClassifier` |
| `"MLP"` | `MLPClassifier` |
| `"RF"` | `RandomForestClassifier` |
| `"ET"` | `ExtraTreesClassifier` |
| `"LGBM"` | `LGBMClassifier` |
| `"XGB"` | `XGBClassifier` |
| `"CatB"` | `CatBoostClassifier` |
| `"KNN"` | `KNeighborsClassifier` |
| `"ENet"` | `LogisticRegression` (Elastic Net, saga solver) |
| `"PLR"` | `LogisticRegression` (L2 primal) |
| `"DLR"` | `LogisticRegression` (L2 dual, liblinear) |
| `"Ridge"` | `RidgeClassifier` |
| `"LDA"` | `LinearDiscriminantAnalysis` |

Pass `algorithm="optimize"` to let Optuna select among all of the above, or pass a list to restrict the search space:

```python
magic_now(..., algorithm=["LGBM", "RF", "RBFSVM"])
```

All classifiers are instantiated with `class_weight="balanced"` to handle class imbalance.

---

## Usage examples

### Two-stage workflow (recommended)

```python
from pathlib import Path
import pandas as pd
from haramo.classification import magic_now

X = pd.read_csv("features.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()

# Stage 1: exhaustive grid over all pipeline structures
_, _, studies = magic_now(
    X=X,
    y=y,
    algorithm="optimize",       # all 15 algorithms
    scaler="optimize",          # all 4 scalers
    feature_selector="optimize", # all 4 feature selector combinations
    hyperparameters="default",
    output_dir=Path("results/stage1"),
)

# Inspect top 3 structures from fold 1
# Params keys: "algorithm", "scaling_method", "add_pvalue_filter", "add_boruta_filter"
top3 = sorted(studies["fold_1"].trials, key=lambda t: t.value, reverse=True)[:3]
for i, t in enumerate(top3):
    print(f"#{i+1} score={t.value:.4f}  params={t.params}")

# Stage 2: run the top 3 structures with full hyperparameter optimization
# Repeat for each of the top 3 identified structures
for i, trial in enumerate(top3):
    p = trial.params
    # Map add_pvalue_filter / add_boruta_filter back to feature_selector
    if p.get("add_pvalue_filter") and p.get("add_boruta_filter"):
        fs = "optimize"
    elif p.get("add_pvalue_filter"):
        fs = "pvalue"
    elif p.get("add_boruta_filter"):
        fs = "boruta"
    else:
        fs = None

    validation, pipeline, studies2 = magic_now(
        X=X,
        y=y,
        algorithm=p["algorithm"],
        scaler=p.get("scaling_method"),
        feature_selector=fs,
        hyperparameters="optimize",
        n_trials=200,
        output_dir=Path("results/stage2"),
        tag=f"_top{i+1}",
    )
    print(f"\nTop {i+1} — {p['algorithm']} / {p.get('scaling_method')} / fs={fs}")
    print(validation)
```

### Restrict search space in stage 1

```python
# Only consider a subset of algorithms
_, _, studies = magic_now(
    X=X,
    y=y,
    algorithm=["LGBM", "RF", "RBFSVM", "ENet"],
    scaler="optimize",
    feature_selector="optimize",
    hyperparameters="default",
    output_dir=Path("results/stage1_subset"),
)
```

### Group-aware cross-validation (e.g. repeated measures, patients)

```python
# Prevents samples from the same group appearing in both train and test splits
_, _, studies = magic_now(
    X=X,
    y=y,
    groups=subject_ids,        # array-like of group labels, same length as y
    hyperparameters="default",
    output_dir=Path("results/stage1_grouped"),
)
```

### Custom scoring metric

```python
from sklearn.metrics import make_scorer, matthews_corrcoef

mcc_scorer = make_scorer(matthews_corrcoef)

validation, pipeline, studies = magic_now(
    X=X,
    y=y,
    algorithm="LGBM",
    scaler="robust",
    feature_selector="boruta",
    scoring=mcc_scorer,
    hyperparameters="optimize",
    n_trials=200,
    output_dir=Path("results/stage2_mcc"),
)
```

---

## Output metrics

The `validation` DataFrame contains one row per outer fold:

| Metric | Description |
|---|---|
| MCC | Matthews Correlation Coefficient |
| F1-score | Binary F1 score |
| Kappa | Cohen's Kappa |
| Bal. Acc. | Balanced accuracy |
| Precision | Positive predictive value |
| Sensitivity | True positive rate (recall) |
| Selectivity | True negative rate |

---

## License

MIT License. See `LICENSE` for details.
