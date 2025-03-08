###########
# Imports #
###########

import numpy as np
import pandas as pd

from typing import Union

from category_encoders import WOEEncoder

from optuna import Trial

#############
# Functions #
#############


def instantiate_woe_encoder(trial: Trial):
    params = {
        "sigma": trial.suggest_float("sigma", 0.001, 5),
        "regularization": trial.suggest_float("regularization", 0, 5),
        "randomized": trial.suggest_categorical("randomized", [True, False]),
    }

    encoder = WOEEncoder(**params)

    return encoder


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

Encoder = OrdinalEncoder | OneHotEncoder | WOEENcoder


def instantiate_encoder(trial: Trial) -> Encoder:
    method = trial.suggest_categorical("encoding_method", ["ordinal", "onehot", "woe"])

    if method == "ordinal":
        encoder = instantiate_ordinal_encoder(trial)
    elif method == "onehot":
        encoder = instantiate_onehot_encoder(trial)
    elif method == "woe":
        encoder = instantiate_woe_encoder(trial)

    return encoder
