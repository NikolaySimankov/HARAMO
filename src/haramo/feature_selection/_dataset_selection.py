###########
# Imports #
###########

from __future__ import annotations

from itertools import combinations as _iter_combinations
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC

from ..utils import reduce_dataset

#############
# Functions #
#############


def _build_default_pipeline(task: str, random_state: int):
    """
    Build a minimal pipeline with default hyperparameters for fast dataset evaluation.

    Uses LGBM + StandardScaler with no feature selection beyond the variance
    filter.  Passing concrete strings to ``instantiate_pipeline`` bypasses every
    ``trial.suggest_*`` branch, so the dummy Optuna trial is never queried.
    """
    # Deferred import to avoid a circular dependency at module load time
    # (feature_selection ← classification ← feature_selection).
    from optuna import create_study
    from ..classification import instantiate_pipeline

    study = create_study()
    trial = study.ask()

    return instantiate_pipeline(
        trial,
        task=task,
        feature_selector="pvalue",  # identity – no suggest call
        scaler="standard",  # concrete string – no suggest call
        algorithm="RBFSVM",  # concrete string – no suggest call
        hyperparameters="default",
        random_state=random_state,
        n_jobs=1,
    )


def _score_dataset_combo(
    X_combo: pd.DataFrame,
    y: pd.Series,
    scoring: Union[str, Callable],
    task: str,
    random_state: int,
    groups,
) -> float:
    """
    Score a single dataset combination via a lightweight 3-fold CV.

    This helper intentionally skips the ``reduce_dataset`` step used in the
    main training loop — the goal is a cheap informative signal, not a full
    simulation of training.

    Returns
    -------
    float
        Mean CV score, or ``nan`` if every fold raised an exception.
    """
    pipeline = _build_default_pipeline(task, random_state)

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=3)
        splits = list(cv.split(X_combo, y.astype(str), groups=groups))
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        splits = list(cv.split(X_combo, y.astype(str)))

    def _eval_fold(train_idx, test_idx):
        pipe = clone(pipeline)
        X_tr, X_te = X_combo.iloc[train_idx], X_combo.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        try:
            reduced_index = reduce_dataset(
                X=X_tr,
                y=y_tr,
                target_size=1000,
                difficulty_model=SVC(
                    kernel="rbf", random_state=random_state, class_weight="balanced"
                ),
                stage2_shrink=0.9,
                class_weight="balanced",
                random_state=random_state,
                verbose=False,
            )
            pipe.fit(X_tr.loc[reduced_index], y_tr.loc[reduced_index])
            return scorer(pipe, X_te, y_te)
        except Exception:
            return np.nan

    scores = [_eval_fold(tr, te) for tr, te in splits]
    return float(np.nanmean(scores))


def select_best_dataset_combo(
    datasets: Dict[str, pd.DataFrame],
    y: pd.Series,
    scoring: Union[str, Callable] = "balanced_accuracy",
    task: str = "classification",
    random_state: int = 42,
    groups=None,
    n_jobs: int = 1,
    max_combos: int = 300,
) -> Tuple[str, pd.DataFrame, pd.Series]:
    """
    Enumerate non-empty subsets of *datasets*, score each with a default
    pipeline, and return the winning combination.

    For up to 5 datasets every subset is tested (2^5 − 1 = 31 combinations,
    matching the default ``max_combos``).  Beyond that the search is restricted
    to singletons and pairs to keep computation manageable.

    Parameters
    ----------
    datasets : dict
        Mapping ``name → DataFrame``.  All DataFrames must share the same
        index as ``y``.
    y : pd.Series
        Target vector.
    scoring : str or callable, default ``"balanced_accuracy"``
        Scorer passed to ``sklearn.metrics.get_scorer`` or used directly.
    task : str, default ``"classification"``
        Forwarded to the pipeline builder.
    random_state : int, default 42
    groups : array-like, optional
        Group labels forwarded to ``StratifiedGroupKFold``.
    n_jobs : int, default 1
        Total parallel workers.  Inner CV parallelism is scaled proportionally.
    max_combos : int, default 31
        Maximum number of subsets to evaluate before falling back to
        singletons + pairs.

    Returns
    -------
    best_combo_name : str
        Dataset names joined by ``" + "``.
    best_X : pd.DataFrame
        Concatenated feature matrix of the winning combination.
    scores : pd.Series
        Mean CV score for every evaluated combination, sorted descending.
    """
    names = list(datasets.keys())
    n = len(names)

    # Build candidate subsets
    all_combos: List[tuple] = []
    for r in range(1, n + 1):
        for combo in _iter_combinations(names, r):
            all_combos.append(combo)

    if len(all_combos) > max_combos:
        print(
            f"[Dataset Selection] {len(all_combos)} possible combinations exceed "
            f"max_combos={max_combos}; restricting to singletons and pairs."
        )
        all_combos = [(name,) for name in names]
        all_combos += list(_iter_combinations(names, 2))

    print(
        f"[Dataset Selection] Scoring {len(all_combos)} combination(s) "
        f"across {n} dataset(s) using default pipeline …"
    )

    def _score_one(combo: tuple):
        X_combo = pd.concat([datasets[name] for name in combo], axis=1)
        score = _score_dataset_combo(X_combo, y, scoring, task, random_state, groups)
        return combo, score

    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(combo) for combo in all_combos
    )

    scores_dict = {" + ".join(combo): score for combo, score in results}
    scores = pd.Series(scores_dict, name="score").sort_values(ascending=False)

    best_combo, _ = max(
        results,
        key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
    )
    best_combo_name = " + ".join(best_combo)
    best_X = pd.concat([datasets[name] for name in best_combo], axis=1)

    return best_combo_name, best_X, scores
