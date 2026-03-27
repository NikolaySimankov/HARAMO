# -*- coding: utf-8 -*-

###########
# Imports #
###########

import pandas as pd
import numpy as np

import pickle

from joblib import Parallel, delayed
from typing import Union, Callable
from os import PathLike

from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import get_scorer

from optuna import (
    Trial,
    create_study,
)

from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import GridSampler

from ._instantiation import instantiate_pipeline
from ..utils import classification_report

#############
# Functions #
#############


def _score_fold(pipeline, X, y, scoring, train_index, test_index, sample_weight=None):
    """Fit a cloned pipeline on one CV fold and return the score."""
    pipe = clone(pipeline)
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if sample_weight is not None:
        sample_weight_train = sample_weight.iloc[train_index]
        try:
            pipe.fit(X_train, y_train, model__sample_weight=sample_weight_train)
        except:
            pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train)

    return scorer(pipe, X_test, y_test)


def pipeline_cross_val(pipeline, X, y, scoring, cv, sample_weight=None, n_jobs=1):
    """
    Custom cross-validation function that maintains DataFrame format.

    Parameters:
    -----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : pd.DataFrame
        The data to fit.
    y : pd.Series
        The target variable to try to predict in the case of supervised learning.
    scoring : Union[str, callable]
        A scorer callable object / function with signature scorer(estimator, X, y) or a string.
    cv : iterable
        Cross-validation splitting strategy.
    sample_weight : pd.Series, optional
        Sample weights to be used in training.
    n_jobs : int, default=1
        Number of parallel jobs for cross-validation folds.

    Returns:
    --------
    scores : list of float
        Array of scores of the estimator for each run of the cross-validation.
    """
    splits = list(cv)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score_fold)(
            pipeline, X, y, scoring, train_idx, test_idx, sample_weight
        )
        for train_idx, test_idx in splits
    )

    return scores


def objective(
    trial: Trial,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series, list],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series, list],
    sample_weight: Union[np.ndarray, pd.DataFrame] = None,
    scoring: Union[str, Callable] = "balanced_accuracy",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_cv_jobs: int = 1,
    groups: Union[np.ndarray, pd.Series, list] = None,
):
    """
    Objective function for hyperparameter optimization using cross-validation.
    Parameters:
    -----------
    trial : Trial
        An Optuna trial object for suggesting hyperparameters.
    X : Union[np.ndarray, pd.DataFrame],
        Feature matrix for training the model.
    y : Union[np.ndarray, pd.Series, list],
        Target vector for training the model.
    scoring : Union[str, Callable], default="balanced_accuracy"
        Scoring method to evaluate the predictions on the test set.
    algorithm : str, default="LSVM"
        The machine learning algorithm to be used.
    random_state : int, default=42
        Random seed for reproducibility.
    sample_weight : Union[np.ndarray, pd.DataFrame], optional
        Sample weights to be used in training.
    n_cv_jobs : int, default=1
        Number of threads for the model's native parallelism.
    groups : Union[np.ndarray, pd.Series, list], optional
        Group labels for StratifiedGroupKFold. When provided, inner CV
        ensures no group appears in both train and test, so hyperparameters
        are optimized for generalization to unseen groups.
    Returns:
    --------
    float
        Mean cross-validation score.
    """

    pipeline = instantiate_pipeline(
        trial,
        task=task,
        feature_selector=feature_selector,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    # Give CPU budget to the model's native threading (bypasses GIL)
    # rather than parallelizing CV folds via joblib (GIL-limited)
    try:
        pipeline.set_params(model__n_jobs=n_cv_jobs)
    except ValueError:
        pass

    if groups is not None:
        strat_kfold_inner = StratifiedGroupKFold(n_splits=4)
    else:
        strat_kfold_inner = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )

    scores = pipeline_cross_val(
        pipeline,
        X=X_train,
        y=y_train,
        sample_weight=sample_weight,
        scoring=scoring,
        cv=strat_kfold_inner.split(
            X_train.values, y_train.astype("str"), groups=groups
        ),
    )

    score = np.mean(scores)

    return score


def get_search_space(feature_selector, scaler, algorithm):
    search_space = {}
    n_trials = 1

    if feature_selector == "optimize":
        search_space["add_pvalue_filter"] = [True, False]
        search_space["add_boruta_filter"] = [True, False]
        n_trials *= len(search_space["add_pvalue_filter"]) * len(
            search_space["add_boruta_filter"]
        )
    elif isinstance(feature_selector, list):
        search_space["feature_selection_method"] = feature_selector
        n_trials *= len(feature_selector)

    if scaler == "optimize":
        search_space["scaling_method"] = [None, "standard", "minmax", "robust"]
        n_trials *= len(search_space["scaling_method"])
    elif isinstance(scaler, list):
        search_space["scaling_method"] = scaler
        n_trials *= len(scaler)

    if algorithm == "optimize":
        search_space["algorithm"] = [
            "LSVM",
            "RBFSVM",
            "NuLSVM",
            "NuRBFSVM",
            "SGD",
            "MLP",
            "RF",
            "ET",
            "LGBM",
            "KNN",
            "ENet",
            "PLR",
            "DLR",
            "Ridge",
            "LDA",
        ]
        n_trials *= len(search_space["algorithm"])
    elif isinstance(algorithm, list):
        search_space["algorithm"] = algorithm
        n_trials *= len(algorithm)

    return search_space, n_trials


def _train_fold(
    fold_idx,
    X,
    y,
    sample_weight,
    train_index,
    test_index,
    scoring,
    task,
    feature_selector,
    scaler,
    algorithm,
    hyperparameters,
    random_state,
    n_trials,
    n_cv_jobs,
    groups=None,
):
    """Run Optuna optimization for a single outer fold."""
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    w_train = sample_weight.iloc[train_index]
    groups_train = groups.iloc[train_index] if groups is not None else None

    if hyperparameters == "default":
        search_space, n_trials = get_search_space(feature_selector, scaler, algorithm)
        sampler = GridSampler(search_space)
    else:
        sampler = TPESampler(
            seed=random_state,
            multivariate=True,
        )

    study = create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(reduction_factor=2),
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_weight=w_train,
            scoring=scoring,
            task=task,
            feature_selector=feature_selector,
            scaler=scaler,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state,
            n_cv_jobs=n_cv_jobs,
            groups=groups_train,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    model = instantiate_pipeline(
        trial=study.best_trial,
        feature_selector=feature_selector,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    fold_name = f"fold_{fold_idx}"
    return fold_name, model, study


def train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    scoring: Union[str, Callable] = "balanced_accuracy",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_trials: int = 100,
    groups: Union[np.ndarray, pd.Series, list] = None,
    n_jobs: int = 16,
):
    """
    Train a model using stratified k-fold cross-validation and hyperparameter optimization.

    Parameters:
    -----------
    X : Union[np.ndarray, pd.DataFrame],
        Feature matrix. Can be a NumPy array or a pandas DataFrame.
    y : Union[np.ndarray, pd.Series, list],
        Target vector. Can be a NumPy array, pandas Series, or a list.
    scoring : Union[str, Callable], default="balanced_accuracy"
        Scoring method to evaluate the predictions on the test set.
    task : str, default="classification"
        The type of task to perform. Currently supports "classification".
    feature_selector : Union[str, list], default="optimize"
        Feature selection method(s) to be used.
    scaler : Union[str, list], default="optimize"
        Scaling method(s) to be used.
    algorithm : Union[str, list], default="optimize"
        Algorithm(s) to be used.
    hyperparameters : str, default="optimize"
        Hyperparameters to be optimized.
    random_state : int, default=42
        Random seed for reproducibility.
    n_trials : int, default=100
        Number of trials for hyperparameter optimization. igored if hyperparameters = "default"
    n_jobs : int, default=16
        Total number of parallel CPUs. Split as 4 outer folds × (n_jobs // 4) inner CV jobs.

    Returns:
    --------
    models : dict
        A dictionary containing trained models for each fold.
    """

    sample_weight = pd.Series(
        compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
    )

    if groups is not None:
        strat_kfold_outer = StratifiedGroupKFold(n_splits=4)
    else:
        strat_kfold_outer = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )

    splits = list(strat_kfold_outer.split(X, y.astype("str"), groups=groups))
    n_outer_folds = len(splits)
    n_cv_jobs = max(1, n_jobs // n_outer_folds)

    results = Parallel(n_jobs=n_outer_folds)(
        delayed(_train_fold)(
            fold_idx=fold_idx,
            X=X,
            y=y,
            sample_weight=sample_weight,
            train_index=train_index,
            test_index=test_index,
            scoring=scoring,
            task=task,
            feature_selector=feature_selector,
            scaler=scaler,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state,
            n_trials=n_trials,
            n_cv_jobs=n_cv_jobs,
            groups=groups,
        )
        for fold_idx, (train_index, test_index) in enumerate(splits, start=1)
    )

    models = {name: model for name, model, _ in results}
    studies = {name: study for name, _, study in results}

    return models, studies


def _fit_and_predict(pipeline, X, y, sample_weight, train_index, test_index):
    """Fit a cloned pipeline on a single CV split and return predictions."""
    pipe = clone(pipeline)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    w_train = sample_weight.iloc[train_index]

    try:
        pipe.fit(X_train, y_train, model__sample_weight=w_train)
    except:
        pipe.fit(X_train, y_train)

    return pd.DataFrame(
        np.column_stack((y_test, pipe.predict(X_test))),
        columns=["true", "predicted"],
        index=y_test.index,
    )


def nested_crossval(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    pipelines: dict,
    random_state: int = 42,
    groups: Union[np.ndarray, pd.Series, list] = None,
    n_jobs: int = 16,
):
    """
    Perform nested cross-validation on a set of models.

    Parameters:
    -----------
    X : Union[np.ndarray, pd.DataFrame]
        Feature matrix.
    y : Union[np.ndarray, pd.Series, list]
        Target vector.
    models : dict
        Dictionary of models to be evaluated, where keys are fold identifiers and values are classifier instances.
    groups : Union[np.ndarray, pd.Series, list], optional
        Group labels for the samples, by default None
    n_jobs : int, default=16
        Number of parallel jobs for cross-validation.

    Returns:
    --------
    validation : pd.DataFrame
        DataFrame containing classification reports for each fold.
    best_model : classifier instance
        The model corresponding to the best fold based on the Kappa score.
    """

    sample_weight = pd.Series(
        compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
    )

    if groups is not None:
        strat_kfold_outer = StratifiedGroupKFold(n_splits=4)
    else:
        strat_kfold_outer = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )

    splits = list(strat_kfold_outer.split(X, y.astype("str"), groups=groups))

    # Build all (fold, split) tasks and run in parallel
    tasks = [
        (fold, pipelines[fold], train_idx, test_idx)
        for fold in pipelines
        for train_idx, test_idx in splits
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_predict)(pipe, X, y, sample_weight, train_idx, test_idx)
        for _, pipe, train_idx, test_idx in tasks
    )

    # Aggregate results by fold
    validation = pd.DataFrame()
    n_splits = len(splits)
    for i, fold in enumerate(pipelines):
        fold_results = results[i * n_splits : (i + 1) * n_splits]
        all_predicted_values = pd.concat(fold_results, axis=0)

        validation[fold] = classification_report(
            all_predicted_values["true"],
            all_predicted_values["predicted"],
        )

    # Find the best fold based on MCC and refit on all data
    best_fold = validation.T["MCC"].idxmax()
    best_model = clone(pipelines[best_fold])

    # Let the final refit use all cores on estimators that support n_jobs
    try:
        best_model.set_params(model__n_jobs=n_jobs)
    except ValueError:
        pass

    try:
        best_model.fit(
            X,
            y,
            model__sample_weight=sample_weight,
        )

    except:
        best_model.fit(
            X,
            y,
        )

    validation = validation.T

    return validation, best_model


def magic_now(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    scoring: Union[str, Callable] = "balanced_accuracy",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_trials: int = 100,
    output_dir: Union[str, "PathLike[str]"] = None,
    groups: Union[np.ndarray, pd.Series, list] = None,
    tag: str = "",
    n_jobs: int = 16,
):

    if not output_dir:
        raise ValueError("Output directory must be specified.")

    results = output_dir / "results"
    results.mkdir(exist_ok=True)

    models = output_dir / "models"
    models.mkdir(exist_ok=True)

    trials = output_dir / "trials"
    trials.mkdir(exist_ok=True)

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)

    pipelines, studies = train(
        X=X,
        y=y,
        scoring=scoring,
        task=task,
        feature_selector=feature_selector,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_trials=n_trials,
        groups=groups,
        n_jobs=n_jobs,
    )

    validation, pipeline = nested_crossval(
        X=X,
        y=y,
        pipelines=pipelines,
        groups=groups,
        n_jobs=n_jobs,
    )

    with open(
        models / f"pipelines{tag}.pkl",
        "wb",
    ) as handle:
        pickle.dump(pipeline, handle)

    with open(
        trials / f"studies{tag}.pkl",
        "wb",
    ) as handle:
        pickle.dump(studies, handle)

    validation.to_csv(results / f"validation{tag}.tsv", sep="\t", index=True)

    return validation, pipeline, studies
