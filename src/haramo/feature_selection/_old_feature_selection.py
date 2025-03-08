###########
# Imports #
###########

import numpy as np
import pandas as pd

from typing import Union
from functools import reduce

from joblib import (
    Parallel,
    delayed,
)

from sklearn.feature_selection import (
    f_classif,
    mutual_info_regression,
)

from boruta import BorutaPy

from sklearn.feature_selection import (
    SelectPercentile,
    VarianceThreshold,
    SequentialFeatureSelector,
    RFECV,
)

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
)

from sklearn.preprocessing import FunctionTransformer

from ..utils import (
    spearman_scorer,
    kendall_scorer,
    biserial_scorer,
)

#############
# Functions #
#############


def rank_best(X, target, score_func, percentile, n_jobs=-1):
    """
    Helper function to fit SelectKBest with a given scoring function.
    Allows specifying the number of jobs for parallel computation.
    """
    X_tmp = X.copy()

    if score_func == mutual_info_regression:
        # mutual_info_classif supports n_jobs parameter
        selector = SelectPercentile(
            score_func=lambda X_tmp, target: score_func(
                X_tmp, target, random_state=42, n_jobs=n_jobs
            ),
            percentile=percentile,
        ).fit(X_tmp, target)
    else:
        selector = SelectPercentile(score_func=score_func, percentile=percentile).fit(
            X_tmp, target
        )
    return selector.get_feature_names_out()


def detect_dtype(x: Union[np.ndarray, pd.Series, list]):
    """
    Detect the type of a x: continuous, discrete, interval, ratio, binary, or nominal.
    Ensure binary variables are in the format 0 or 1.
    """
    if isinstance(x, list):
        x = pd.Series(x)
    elif isinstance(x, np.ndarray):
        x = pd.Series(x)

    if is_bool_dtype(x):
        x = x.astype(int)
        return "binary"

    elif isinstance(x.dtype, pd.CategoricalDtype()):
        unique_values = x.dropna().unique()
        if len(unique_values) == 2:
            return "binary"
        else:
            return "nominal"

    elif is_numeric_dtype(x):
        unique_values = x.dropna().unique()
        if len(unique_values) == 2:
            return "binary"
        else:  # Arbitrary threshold for discrete vs continuous
            differences = np.diff(np.sort(unique_values))
            if np.all(differences == differences[0]):
                return "interval/ordinal"
            elif (x >= 0).all() and (x / x.min()).dropna().apply(
                float.is_integer
            ).all():
                return "ratio"
            else:
                return "continuous"


def statistical_selection(X, y, percentile=25, operation="intersection", verbose=True):
    """
    Perform statistical feature selection with customized CPU usage.
    """

    target = y.copy().astype(np.float32)

    # Define scoring functions and their parallelization strategy
    scoring_functions = [
        (kendall_scorer, 1),  # Use 1 CPU
        (spearman_scorer, 1),  # Use 1 CPU
        (biserial_scorer, 1),  # Use 1 CPU
        (f_classif, 1),  # Use 1 CPU
        (mutual_info_regression, -5),  # Use all other available CPUs
    ]

    # Parallel execution of fitting SelectKBest for each scoring function
    results = Parallel(n_jobs=-1)(
        delayed(rank_best)(
            X,
            target,
            score_func,
            percentile,
            n_jobs,
        )
        for score_func, n_jobs in scoring_functions
    )

    # Extract feature names from results
    kendall, spearman, biserial, anova, mutual = results

    # Operation to perform on the selected features
    if operation == "intersection":
        selected_features = reduce(
            np.intersect1d, (kendall, spearman, biserial, anova, mutual)
        )
        if len(selected_features) < 100:
            selected_features = reduce(
                np.union1d, (kendall, spearman, biserial, anova, mutual)
            )

    if operation == "union":
        selected_features = reduce(
            np.union1d, (kendall, spearman, biserial, anova, mutual)
        )

    if verbose:
        print(f"{operation} = {len(selected_features)}")

    return selected_features


def boruta_selection(
    trial: Trial,
    objective: str,
    random_state: int = 42,
):

    params = {
        "percentage": trial.suggest_int("percentage", 80, 0.100),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 40),
        "n_estimators": trial.suggest_int("n_estimators", 200, 501, step=50),
        "random_state": random_state,
        "n_jobs": -1,
    }

    if "regression" in objective:

        model = RandomForestRegressor(filter_args(RandomForestRegressor, **params))

    elif "classification" in objective:
        model = RandomForestClassifier(filter_args(RandomForestClassifier, **params))

    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_

    # define Boruta feature selection method
    boruta = BorutaPy(
        model,
        perc=percentage,
        n_estimators=250,
        verbose=2,
        random_state=random_state,
        max_iter=200,
    )
    # find all relevant features
    boruta.fit(X, y)

    # Filter to keep only features with a ranking of 1
    X_new = boruta.transform(X, return_df=True)

    return X_new
