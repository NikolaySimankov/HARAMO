###########
# Imports #
###########
import numpy as np
import pandas as pd

from typing import Union

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from optuna import Trial

#############
# Functions #
#############


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def instantiate_numerical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_numerical_simple_imputer(trial)),
            ("scaler", instantiate_scaler(trial)),
        ]
    )
    return pipeline


def instantiate_categorical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_categorical_simple_imputer(trial)),
            ("encoder", instantiate_encoder(trial)),
        ]
    )
    return pipeline


def instantiate_processor(
    trial: Trial, numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:

    numerical_pipeline = instantiate_numerical_pipeline(trial)
    categorical_pipeline = instantiate_categorical_pipeline(trial)

    selected_numerical_columns = choose_columns(numerical_columns)
    selected_categorical_columns = choose_columns(categorical_columns)

    processor = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                numerical_pipeline,
                selected_numerical_columns,
            ),
            (
                "categorical_pipeline",
                categorical_pipeline,
                selected_categorical_columns,
            ),
        ]
    )

    return processor


def objective(
    trial: Trial, X: DataFrame, y: DataFrame, seed: int = 42, base: int = 2, n_rungs=5
) -> float:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=seed
    )
    model = instantiate_extra_trees(trial, warm_start=False)

    for n_samples in pruner_sampling(y_train, base, n_rungs):
        X_train_sample = X_train.sample(n_samples, random_state=seed)
        y_train_sample = y_train.sample(n_samples, random_state=seed)

        model.fit(X_train_sample, y_train_sample.values.ravel())

        score = (y_test, model.predict_proba(X_test)[:, 1])
        trial.report(score, n_samples)

        if trial.should_prune():
            raise TrialPruned()

        kfold = KFold(shuffle=True, random_state=seed)
    roc_auc = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=roc_auc)

    return np.min([np.mean(scores), np.median(scores)])


study = create_study(
    direction="maximize",
    pruner=SuccessiveHalvingPruner(reduction_factor=factor),
    sampler=RandomSampler(seed=42),  # not necessary, helps with reproducibility
)

study.optimize(
    lambda trial: objective(trial, X, y, base=factor, n_rungs=4), n_trials=60
)
