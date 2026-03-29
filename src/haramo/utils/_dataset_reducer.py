from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import Birch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict


def _extract_medoids_from_clusters(
    X_cls: np.ndarray,
    global_idx: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Discover natural clusters in one class using Birch(n_clusters=None).
    Extract one medoid per cluster (nearest real point to cluster center).

    Returns
    -------
    medoid_indices : global indices of extracted medoids
    cluster_info : dict mapping local cluster label to list of local indices in that cluster
    """
    n = len(X_cls)
    if n < 3:
        return global_idx.copy(), {0: np.arange(n)}

    model = Birch(n_clusters=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(X_cls)

    unique_labels = np.unique(labels)
    centers = np.array([X_cls[labels == k].mean(0) for k in unique_labels])

    medoid_locals: list[int] = []
    cluster_info = {}

    for k in unique_labels:
        mask = labels == k
        if not mask.any():
            continue

        local_indices = np.where(mask)[0]
        cluster_info[k] = local_indices.copy()

        dists = np.linalg.norm(
            X_cls[local_indices] - centers[list(unique_labels).index(k)], axis=1
        )
        medoid_local = local_indices[np.argmin(dists)]
        medoid_locals.append(medoid_local)

    medoid_indices = global_idx[np.array(medoid_locals)]
    return medoid_indices, cluster_info


def _balance_classes_via_clusters(
    X: np.ndarray,
    y: np.ndarray,
    X_sc: np.ndarray,
    target_size: int,
    random_state: int = 42,
    class_weight: str = "proportional",
) -> np.ndarray:
    """
    Discover clusters in each class, extract medoids, then allocate the
    remaining budget across classes.

    Parameters
    ----------
    class_weight : {"proportional", "balanced"}
        ``"proportional"`` preserves the natural class distribution.
        ``"balanced"`` gives every class an equal share of the budget.

    Returns global indices of selected samples.
    """
    if class_weight not in ("proportional", "balanced"):
        raise ValueError(
            f"class_weight must be 'proportional' or 'balanced', got {class_weight!r}"
        )
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    all_medoids = []
    class_clusters = {}

    for cls in classes:
        mask = y == cls
        global_indices = np.where(mask)[0]
        X_cls_sc = X_sc[mask]

        medoids, cluster_info = _extract_medoids_from_clusters(
            X_cls_sc, global_indices, random_state
        )
        class_clusters[cls] = (medoids, cluster_info)
        all_medoids.append(medoids)

    all_medoids_arr = np.concatenate(all_medoids)

    n_total = len(y)
    n_classes = len(classes)

    if class_weight == "balanced":
        class_target = {cls: target_size / n_classes for cls in classes}
    else:
        class_target = {
            cls: target_size * float((y == cls).sum()) / n_total for cls in classes
        }

    raw_targets = {cls: max(1, int(class_target[cls])) for cls in classes}
    while sum(raw_targets.values()) > target_size and any(
        v > 1 for v in raw_targets.values()
    ):
        over = max(
            (cls for cls in classes if raw_targets[cls] > 1),
            key=lambda c: raw_targets[c] - class_target[c],
        )
        raw_targets[over] -= 1

    total_medoids = len(all_medoids_arr)

    if total_medoids >= target_size:
        selected = set()
        for cls in classes:
            medoids = class_clusters[cls][0]
            cls_share = min(raw_targets[cls], len(medoids))
            if cls_share > 0:
                selected_idx = rng.choice(len(medoids), cls_share, replace=False)
                selected.update(medoids[selected_idx].tolist())

        if len(selected) < target_size:
            shortfall = target_size - len(selected)
            remaining_medoids = np.array(
                [idx for idx in all_medoids_arr if idx not in selected]
            )
            if len(remaining_medoids) > 0:
                to_add = rng.choice(
                    remaining_medoids,
                    min(shortfall, len(remaining_medoids)),
                    replace=False,
                )
                selected.update(to_add.tolist())

        return np.sort(list(selected))[:target_size]

    selected = set(all_medoids_arr.tolist())

    class_allocation = {
        cls: max(0, raw_targets[cls] - len(class_clusters[cls][0])) for cls in classes
    }

    total_allocated = sum(class_allocation.values())
    remaining_budget = target_size - total_medoids
    if total_allocated < remaining_budget:
        leftover = remaining_budget - total_allocated
        sorted_classes = sorted(
            classes, key=lambda c: -(class_target[c] - len(class_clusters[c][0]))
        )
        i = 0
        while leftover > 0:
            class_allocation[sorted_classes[i % n_classes]] += 1
            leftover -= 1
            i += 1

    for cls in classes:
        medoids, cluster_info = class_clusters[cls]
        share = class_allocation[cls]

        if share <= 0:
            continue

        cls_mask = y == cls
        global_cls_idx = np.where(cls_mask)[0]
        X_cls_sc = X_sc[cls_mask]

        available = np.array(
            [i for i in range(len(X_cls_sc)) if global_cls_idx[i] not in selected]
        )

        if len(available) > 0:
            if len(medoids) > 0:
                medoid_coords = X_sc[medoids]
                dists_to_medoids = np.min(
                    np.linalg.norm(
                        X_cls_sc[available][:, None, :] - medoid_coords[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                w = dists_to_medoids
            else:
                w = np.ones(len(available))

            w_sum = w.sum()
            w = w / w_sum if w_sum > 0 else np.ones(len(w)) / len(w)
            take = min(share, len(available))
            sampled_local = rng.choice(available, size=take, replace=False, p=w)
            sampled_global = global_cls_idx[sampled_local]
            selected.update(sampled_global.tolist())

    return np.sort(list(selected))[:target_size]


def stage1_reduce(
    X: np.ndarray,
    y: np.ndarray,
    target_size: int,
    *,
    random_state: int = 42,
    class_weight: str = "proportional",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1: cluster-based structure-preserving reduction.

    Discovers natural clusters per class (Birch, n_clusters=None), extracts
    medoids, and allocates the budget across classes according to class_weight.

    Returns
    -------
    X_reduced, y_reduced, global_indices
    """
    target_size = min(target_size, len(X))
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    idx = _balance_classes_via_clusters(
        X, y, X_sc, target_size, random_state, class_weight=class_weight
    )
    return X[idx], y[idx], idx


def _build_difficulty_model(difficulty_model=None, random_state: int = 42):
    """Clone the difficulty model and inject random_state when supported."""
    if difficulty_model is None:
        raise ValueError(
            "difficulty_model must be provided when running Stage 2 refinement"
        )

    model = clone(difficulty_model)
    try:
        model.set_params(random_state=random_state)
    except (TypeError, ValueError):
        pass
    return model


def _difficulty_from_decision(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Convert decision_function outputs into [0, 1] difficulty values."""
    if scores.ndim == 1:
        margin = np.abs(scores)
        return 1.0 / (1.0 + np.exp(margin))

    true_scores = scores[np.arange(len(y_true)), y_true]
    masked = scores.copy()
    masked[np.arange(len(y_true)), y_true] = -np.inf
    best_other = masked.max(axis=1)
    margin = true_scores - best_other
    return 1.0 / (1.0 + np.exp(margin))


def stage2_refine(
    X: np.ndarray,
    y: np.ndarray,
    target_size: int,
    *,
    difficulty_model=None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 2: difficulty-weighted refinement on Stage-1 output.

    1. Fit the provided sklearn classifier on the Stage-1 data.
    2. Score every instance: difficulty = 1 − P(correct class).
    3. Re-cluster per class (Birch); compute mean difficulty per cluster.
    4. Collect all instances from *hard* clusters (mean difficulty > 0.5).
    5. Fill remaining budget from easy clusters (uniform random).
    6. Trim to target_size.
    """
    n = len(X)
    if target_size >= n:
        return X.copy(), y.copy(), np.arange(n)

    rng = np.random.RandomState(random_state)

    model = _build_difficulty_model(difficulty_model, random_state)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_cv = max(2, min(5, np.bincount(y_enc).min()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if hasattr(model, "predict_proba"):
                proba = cross_val_predict(
                    model, X, y_enc, cv=n_cv, method="predict_proba"
                )
                difficulty = 1.0 - proba[np.arange(n), y_enc]
            elif hasattr(model, "decision_function"):
                scores = cross_val_predict(
                    model, X, y_enc, cv=n_cv, method="decision_function"
                )
                difficulty = _difficulty_from_decision(np.asarray(scores), y_enc)
            else:
                pred = cross_val_predict(model, X, y_enc, cv=n_cv, method="predict")
                difficulty = (pred != y_enc).astype(float)
        except Exception:
            model.fit(X, y_enc)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                difficulty = 1.0 - proba[np.arange(n), y_enc]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                difficulty = _difficulty_from_decision(np.asarray(scores), y_enc)
            else:
                pred = model.predict(X)
                difficulty = (pred != y_enc).astype(float)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    classes = np.unique(y)

    hard_pool: list[int] = []
    easy_pool: list[int] = []

    for cls in classes:
        mask = y == cls
        global_idx = np.where(mask)[0]
        X_cls_sc = X_sc[mask]

        if len(global_idx) < 3:
            for gi in global_idx:
                (hard_pool if difficulty[gi] > 0.5 else easy_pool).append(int(gi))
            continue

        birch = Birch(n_clusters=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = birch.fit_predict(X_cls_sc)

        for cluster_label in np.unique(labels):
            cluster_global = global_idx[labels == cluster_label]
            mean_diff = difficulty[cluster_global].mean()
            target_pool = hard_pool if mean_diff > 0.5 else easy_pool
            target_pool.extend(cluster_global.tolist())

    hard_arr = np.array(hard_pool)
    easy_arr = np.array(easy_pool)

    if len(hard_arr) >= target_size:
        chosen = rng.choice(hard_arr, size=target_size, replace=False)
    else:
        chosen = hard_arr.tolist()
        remaining = target_size - len(chosen)
        if len(easy_arr) > 0:
            take = min(remaining, len(easy_arr))
            chosen = (
                list(chosen) + rng.choice(easy_arr, size=take, replace=False).tolist()
            )

    idx = np.sort(chosen)
    return X[idx], y[idx], idx


def reduce_dataset(
    X,
    y,
    *,
    reduction_factor: float = 0.2,
    target_size: Optional[int] = None,
    stage2_shrink: float = 0.8,
    difficulty_model=None,
    class_weight: str = "proportional",
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Two-stage dataset reduction.

    Accepts both numpy arrays and pandas DataFrames/Series as inputs.
    Always returns positional integer indices into the original X/y,
    suitable for use with both ``X[indices]`` (numpy) and
    ``X.iloc[indices]`` (pandas).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. DataFrame index labels are not used; only
        integer positions are tracked.
    y : array-like of shape (n_samples,)
        Class labels.
    reduction_factor : float, default 0.2
        Fraction of rows to keep (0.20 = 20 %). Ignored when
        ``target_size`` is set.
    target_size : int, optional
        Explicit final sample count. Overrides ``reduction_factor``.
    stage2_shrink : float, default 0.8
        Stage 1 produces ``target_size / stage2_shrink`` rows; Stage 2
        then refines them down to ``target_size``. Set to 1.0 to skip
        Stage 2.
    difficulty_model : sklearn estimator, optional
        Classifier used by Stage 2 to score sample difficulty. Required
        when ``stage2_shrink < 1.0``.
    class_weight : {"proportional", "balanced"}, default "proportional"
        ``"proportional"`` preserves the original class frequencies.
        ``"balanced"`` gives every class an equal share of the budget.
    random_state : int, default 42
    verbose : bool, default True

    Returns
    -------
    reduced_index : pandas Index or np.ndarray
        Selected row identifiers in the same format as the input index.
        - If X is a DataFrame, returns a pandas Index of the original
          index labels (e.g. "POTY00123") for use with ``X.loc[reduced_index]``.
        - If X is a plain numpy array, returns sorted positional integers
          for use with ``X[reduced_index]``.
    """
    # Capture the original index labels before converting to numpy so we
    # can map positional results back to string labels at the end.
    original_index = X.index if isinstance(X, pd.DataFrame) else None

    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    y_arr = np.asarray(y) if not isinstance(y, np.ndarray) else y

    n = len(X_arr)
    n_classes = len(np.unique(y_arr))

    if target_size is None:
        target_size = max(int(n * reduction_factor), 30 * n_classes)
    target_size = min(target_size, n)

    s1_target = int(target_size / stage2_shrink) if stage2_shrink < 1.0 else target_size
    s1_target = min(s1_target, n)

    if verbose:
        orig_dist = {c: int((y_arr == c).sum()) for c in np.unique(y_arr)}
        print(f"[Reducer] {n:,} rows, {n_classes} classes → target {target_size:,}")
        print(f"          Original distribution: {orig_dist}")
        print(
            f"[Stage 1] cluster discovery to ~{s1_target:,} (class_weight={class_weight!r}) …"
        )

    X1, y1, idx1 = stage1_reduce(
        X_arr,
        y_arr,
        s1_target,
        random_state=random_state,
        class_weight=class_weight,
    )

    if verbose:
        d1 = {c: int((y1 == c).sum()) for c in np.unique(y1)}
        print(f"[Stage 1] → {len(X1):,} rows  |  class distribution: {d1}")

    if stage2_shrink < 1.0 and len(X1) > target_size:
        if verbose:
            print(f"[Stage 2] difficulty refinement to ~{target_size:,} …")
        _, _, local = stage2_refine(
            X1,
            y1,
            target_size,
            difficulty_model=difficulty_model,
            random_state=random_state,
        )
        final_idx = idx1[local]
    else:
        final_idx = idx1

    if verbose:
        y_final = y_arr[final_idx]
        red_dist = {c: int((y_final == c).sum()) for c in np.unique(y_final)}
        print(f"[Done]    {n:,} → {len(final_idx):,} ({len(final_idx)/n:.2%})  |  {red_dist}")

    final_idx = np.sort(final_idx)

    if original_index is not None:
        return original_index[final_idx]

    return final_idx
