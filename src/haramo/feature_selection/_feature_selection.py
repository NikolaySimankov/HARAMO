###########
# Imports #
###########

import pandas as pd
import numpy as np
from typing import Union

from boruta import BorutaPy
from ._tools import BorutaPyWrapper

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from ..utils import (
    PValueFeatureSelector,
    filter_args,
    instantiate_identity_function,
)

from optuna import Trial

#############
# Functions #
#############


def instantiate_variance_filter(trial: Trial, hyperparameters: str = "optimize"):
    """
    Instantiate a VarianceThreshold filter based on the trial suggestion.

    Parameters:
    -----------
    trial : Trial
        Optuna trial object for hyperparameter optimization.

    Returns:
    --------
    VarianceThreshold
        The VarianceThreshold object with the specified variance threshold.
    """
    if hyperparameters == "optimize":
        params = {
            "threshold": trial.suggest_float(
                "variance_threshold", 0.001, 0.05, step=0.001
            )
        }
    elif hyperparameters == "default":
        params = {"threshold": 0.001}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    filter = VarianceThreshold(**params)

    return filter


def instantiate_boruta_filter(
    trial: Trial,
    task: str = "classification",
    random_state: int = 42,
    hyperparameters: str = "optimize",
):

    if hyperparameters == "optimize":
        params = {
            "perc": trial.suggest_int("perc", 80, 100),
            "max_leaf_nodes": trial.suggest_int(
                "boruta_max_leaf_nodes", 10, 51, step=5
            ),
            "n_estimators": trial.suggest_int("boruta_n_estimators", 200, 501, step=50),
        }

    elif hyperparameters == "default":
        params = {"max_iter": 20}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    params.update(
        {
            "random_state": random_state,
            "n_jobs": -1,
        },
    )

    if task == "regression":
        estimator = RandomForestRegressor(
            **filter_args(
                RandomForestRegressor,
                **params,
            )
        )
    elif task == "classification":
        estimator = RandomForestClassifier(
            **filter_args(
                RandomForestClassifier,
                **params,
            )
        )
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    # np.int = np.int32
    # np.float = np.float64
    # np.bool = np.bool_

    # define Boruta feature selection method
    filter = BorutaPyWrapper(
        estimator=estimator,
        verbose=2,
        max_iter=100,
        **(filter_args(BorutaPyWrapper, **params)),
    )

    return filter


def instantiate_pvalue_filter(trial: Trial, hyperparameters: str = "optimize"):
    """
    Filters features in X based on point biserial correlation with targets in y.

    Parameters:
    X (pd.DataFrame): DataFrame containing features.
    y (pd.DataFrame): DataFrame containing binary target variables.
    p_value_threshold (float): Threshold for p-value to consider a feature significant.

    Returns:
    pd.DataFrame: DataFrame containing only significant features.
    """

    if hyperparameters == "optimize":
        params = {
            "threshold": trial.suggest_categorical(
                "significance_threshold",
                [0.001, 0.01, 0.05],
            )
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    filter = PValueFeatureSelector(**params)

    return filter


def combined_feature_selector(
    trial: Trial,
    task: str = "classification",
    hyperparameters: str = "optimize",
    random_state: int = 42,
):
    """
    Create a combined feature selector pipeline with variance, p-value, and Boruta filters.

    Parameters:
    -----------
    trial : Trial
        Optuna trial object for hyperparameter optimization.

    Returns:
    --------
    Pipeline
        A pipeline that applies variance, p-value, and Boruta feature selection in sequence.
    """
    steps = []

    if hyperparameters == "default":
        add_variance_filter = True
        add_pvalue_filter = True
        add_boruta_filter = True
    else:
        # Decide whether to add variance filter
        add_variance_filter = trial.suggest_categorical(
            "add_variance_filter",
            [True, False],
        )
        # Decide whether to add p-value filter
        add_pvalue_filter = trial.suggest_categorical(
            "add_pvalue_filter",
            [True, False],
        )
        # Decide whether to add Boruta filter
        add_boruta_filter = trial.suggest_categorical(
            "add_boruta_filter",
            [True, False],
        )

    if add_variance_filter:
        variance_filter = instantiate_variance_filter(trial, hyperparameters)
        steps.append(("variance", variance_filter))

    if add_pvalue_filter:
        pvalue_filter = instantiate_pvalue_filter(trial, hyperparameters)
        steps.append(("pvalue", pvalue_filter))

    if add_boruta_filter:
        boruta_filter = instantiate_boruta_filter(
            trial,
            task=task,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )
        steps.append(("boruta", boruta_filter))

    # If no filters are added, use identity function
    if not steps:
        no_filter = instantiate_identity_function(trial)
        steps.append(("no_filter", no_filter))

    combined_selector = Pipeline(steps)

    return combined_selector


def instantiate_feature_selector(
    trial: Trial,
    task: str = "classification",
    method: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
):
    """
    Instantiate a feature selector based on the trial suggestion.

    Parameters:
    -----------
    trial : Trial
        An Optuna trial object used for suggesting hyperparameters.
    method : Union[str, list], optional
        The feature selection method to use. If 'Optimize', the method is suggested by the trial.
        Default is 'Optimize'.

    Returns:
    --------
    selector : TransformerMixin
        The instantiated feature selector object.
    """

    if method == "optimize":
        method = trial.suggest_categorical(
            "feature_selection_method",
            [
                None,
                "variance",
                "pvalue",
                "boruta",
                "combine",
            ],
        )

    elif isinstance(method, list):
        method = trial.suggest_categorical("feature_selection_method", method)
    else:
        pass

    if method == None:
        selector = instantiate_identity_function(trial)

    elif method == "variance":
        selector = instantiate_variance_filter(trial, hyperparameters)

    elif method == "pvalue":
        selector = instantiate_pvalue_filter(trial, hyperparameters)

    elif method == "boruta":
        selector = instantiate_boruta_filter(
            trial,
            task=task,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )

    elif method == "combine":
        selector = combined_feature_selector(
            trial,
            task=task,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Invalid feature selection method: {method}. Valid methods are: None, 'variance', 'pvalue', 'boruta', 'combine'."
        )

    return selector
