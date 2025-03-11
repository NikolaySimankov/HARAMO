###########efault
# Imports #
###########

from typing import Union

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    FunctionTransformer,
)

from optuna import Trial

import pandas as pd
import numpy as np

from ..utils import TransformerWrapper

#############
# Functions #
#############


def instantiate_standard_scaler(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        params = {
            "with_mean": trial.suggest_categorical("with_mean", [True, False]),
            "with_std": trial.suggest_categorical("with_std", [True, False]),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = StandardScaler(**params)
    return TransformerWrapper(scaler)


def instantiate_minmax_scaler(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        params = {
            "feature_range": trial.suggest_categorical(
                "feature_range", [(0, 1), (-1, 1)]
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = MinMaxScaler(**params)
    return TransformerWrapper(scaler)


def instantiate_robust_scaler(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        params = {
            "with_centering": trial.suggest_categorical(
                "with_centering", [True, False]
            ),
            "with_scaling": trial.suggest_categorical("with_scaling", [True, False]),
            "quantile_range": trial.suggest_categorical(
                "quantile_range", [(25, 75), (2.35, 97.65), (13.5, 86.5)]
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = RobustScaler(**params)
    return TransformerWrapper(scaler)


def instantiate_identity_function(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        pass
    elif hyperparameters == "default":
        pass
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    function = FunctionTransformer()
    return TransformerWrapper(function)


def instantiate_scaler(
    trial: Trial,
    method: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
):
    """
    Instantiate a scaler based on the trial suggestion.

    Parameters:
    -----------
    trial : Trial
        An Optuna trial object used for suggesting hyperparameters.
    method : Union[str, list], optional
        The scaling method to use. If 'optimize', the method is suggested by the trial.
        default is 'optimize'.
    hyperparameters : str, optional
        The hyperparameters setting to use. Must be 'optimize' or 'default'.
        default is 'optimize'.

    Returns:
    --------
    scaler : TransformerMixin
        The instantiated scaler object.
    """

    if method == "optimize":
        method = trial.suggest_categorical(
            "scaling_method", [None, "standard", "minmax", "robust"]
        )
    elif isinstance(method, list):
        method = trial.suggest_categorical("scaling_method", method)
    else:
        pass

    if method is None:
        scaler = instantiate_identity_function(trial)
    elif method == "standard":
        scaler = instantiate_standard_scaler(trial, hyperparameters)
    elif method == "minmax":
        scaler = instantiate_minmax_scaler(trial, hyperparameters)
    elif method == "robust":
        scaler = instantiate_robust_scaler(trial, hyperparameters)
    else:
        raise ValueError(
            f"Invalid scaling method: {method}. Valid methods are: None, 'standard', 'minmax', 'robust'."
        )
    return scaler
