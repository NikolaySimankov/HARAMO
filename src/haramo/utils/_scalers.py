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
        pass
    elif hyperparameters == "default":
        pass
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = StandardScaler(with_mean=True, with_std=True)
    return TransformerWrapper(scaler)


def instantiate_minmax_scaler(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        pass
    elif hyperparameters == "default":
        pass
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = MinMaxScaler(feature_range=(0, 1))
    return TransformerWrapper(scaler)


def instantiate_robust_scaler(trial: Trial, hyperparameters: str = "optimize"):
    if hyperparameters == "optimize":
        params = {
            "quantile_range": trial.suggest_categorical(
                "quantile_range", [(25, 75), (10, 90), (5, 95)]
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    scaler = RobustScaler(**params, with_centering=True, with_scaling=True)
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
