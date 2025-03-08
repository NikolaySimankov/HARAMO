# -*- coding: utf-8 -*-
# %%
###########
# Imports #
###########

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from tqdm import tqdm

from dawgz import job, after, schedule

pd.set_option("display.max_rows", None)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from optuna import (
    Trial,
    create_study,
)
from optuna.samplers import TPESampler

from utils._tools import (
    regression_report,
    group_small,
    plot_confusion_matrix,
    set_verbosity,
    mean_report,
    Shifted_Model,
)

#############
# Functions #
#############


def get_parser():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--file",
        dest="file",
        help="name of file containing the data",
        required=True,
    )

    parser.add_argument(
        "--n_trials",
        default=250,
        type=int,
        dest="n_trials",
        help="number of trials for the hyperparameter optimization process",
        required=False,
    )

    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        dest="verbose",
        help="whether to print the progress of the training and validation",
        required=False,
    )

    parser.add_argument(
        "--backend",
        default="slurm",
        type=str,
        dest="backend",
        help="backend to use for the parallelization of the jobs",
        required=True,
    )

    return parser


def instantiate_LSVM(
    trial: Trial,
):
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e0),
        "epsilon": trial.suggest_float("epsilon", 1e-2, 25e-2),
    }

    estimator = SVR(
        tol=1e-4,
        kernel="linear",
        cache_size=1000,
        **params,
    )

    return estimator


def instantiate_RBFSVM(
    trial: Trial,
):
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e0),
        "epsilon": trial.suggest_float("epsilon", 1e-2, 25e-2),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1),
    }

    estimator = SVR(
        tol=1e-4,
        kernel="rbf",
        cache_size=1000,
        **params,
    )

    return estimator


def instantiate_ElasticNet(
    trial: Trial,
):
    params = {
        "l1_ratio": trial.suggest_float("l1_ratio", 1e-1, 9e-1),
        "alpha": trial.suggest_float("alpha", 1e-4, 1e-1),
        "max_iter": trial.suggest_int("max_iter", 2**9, 2**12),
    }

    estimator = ElasticNet(
        random_state=42,
        **params,
    )

    return estimator


def instantiate_RF(
    trial: Trial,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 10, 50),
        "max_features": trial.suggest_int("max_features", 0.1, 1.0),
    }

    estimator = RandomForestRegressor(
        random_state=42,
        **params,
    )

    return estimator


def instantiate_XGBT(
    trial: Trial,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_leaves": trial.suggest_int("num_leaves", 5, 75),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e1),
        "tree_method": trial.suggest_categorical("tree_method", ["exact", "approx"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    estimator = XGBRegressor(
        booster="gbtree",
        random_state=42,
        **params,
    )

    return estimator


def instantiate_LGBM(
    trial: Trial,
):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 5, 75),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-0),
        "n_estimators": trial.suggest_int("n_estimators", 2**7, 2**9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        verbose=-1,
        n_jobs=1,
        **params,
    )

    return model


def instantiate_MLP(
    trial: Trial,
):
    params = {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            list(itertools.product([50, 100, 200], repeat=2))
            + list(itertools.product([50, 100, 200], repeat=3)),
        ),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1),
        "max_iter": trial.suggest_int("max_iter", 2**10, 2**13),
    }

    estimator = MLPRegressor(
        random_state=32,
        **params,
    )

    return estimator


def instantiate_learner(
    trial: Trial,
    algorithm: str = "LSVM",
):
    if algorithm == "LSVM":
        model = instantiate_LSVM(trial)

    elif algorithm == "RBFSVM":
        model = instantiate_RBFSVM(trial)

    elif algorithm == "ENet":
        model = instantiate_ElasticNet(trial)

    elif algorithm == "XGBL":
        model = instantiate_XGBL(trial)

    elif algorithm == "RF":
        model = instantiate_RF(trial)

    elif algorithm == "LGBM":
        model = instantiate_LGBM(trial)

    elif algorithm == "XGBT":
        model = instantiate_XGBT(trial)

    return model


def objective(
    trial: Trial,
    X,
    y,
    algorithm: str,
    sample_weight=None,
):
    model = instantiate_learner(
        trial,
        algorithm=algorithm,
    )

    strat_kfold_inner = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    try:
        scores = cross_val_score(
            model,
            X,
            y,
            scoring="neg_root_mean_squared_log_error",
            n_jobs=-1,
            cv=strat_kfold_inner.split(X, y.astype("str")),
            params={"sample_weight": sample_weight},  # Sample weights
        )

    except:
        scores = cross_val_score(
            model,
            X,
            y,
            scoring="neg_root_mean_squared_log_error",
            n_jobs=-1,
            cv=strat_kfold_inner.split(X, y.astype("str")),
        )

    return scores.mean()


def train(
    X,
    y,
    algorithm: str,
    antibiotic: str,
    n_trials: int,
):
    path = Path(".")

    trials = path / "trials"
    trials.mkdir(exist_ok=True)

    models = {}

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    strat_kfold_outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    outer_fold = 0

    for train_index, test_index in tqdm(strat_kfold_outer.split(X, y.astype("str"))):
        X_train, _ = (
            X.iloc[train_index],
            X.iloc[test_index],
        )
        y_train, _ = (
            y.iloc[train_index],
            y.iloc[test_index],
        )
        w_train, _ = (
            weights[train_index],
            weights[test_index],
        )

        outer_fold += 1

        study = create_study(
            direction="maximize",
            sampler=TPESampler(
                seed=42,
                multivariate=True,
            ),
        )

        joblib.dump(study, trials / f"study_{antibiotic}_{algorithm}.pkl")

        study.optimize(
            lambda trial: objective(
                trial,
                X=X_train,
                y=y_train,
                algorithm=algorithm,
                sample_weight=w_train,
            ),
            n_trials=n_trials,
            n_jobs=-1,
        )

        model = instantiate_learner(
            trial=study.best_trial,
            algorithm=algorithm,
        )
        try:
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
            )
        except:
            model.fit(
                X_train,
                y_train,
            )
        models[f"fold_{outer_fold}"] = model

    return models


def nested_crossval(
    X,
    y,
    models,
    algorithm: str,
    antibiotic: str,
    shift,
    threshold: str,
):
    validation = pd.DataFrame()
    fold_predictions = {}

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    strat_kfold_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold in tqdm(models.keys()):
        regressor = models[fold]

        outer_fold = 0

        all_predicted_values = pd.DataFrame()

        for train_index, test_index in strat_kfold_outer.split(X, y.astype("str")):
            X_train, X_test = (
                X.iloc[train_index],
                X.iloc[test_index],
            )
            y_train, y_test = (
                y.iloc[train_index],
                y.iloc[test_index],
            )
            w_train, _ = (
                weights[train_index],
                weights[test_index],
            )

            outer_fold += 1

            try:
                regressor.fit(
                    X_train,
                    y_train,
                    sample_weight=w_train,
                )
            except:
                regressor.fit(
                    X_train,
                    y_train,
                )

            predicted_values = pd.DataFrame(
                np.column_stack((y_test, regressor.predict(X_test))),
                columns=["true", "predicted"],
                index=y_test.index,
            )

            all_predicted_values = pd.concat(
                [
                    all_predicted_values,
                    predicted_values,
                ],
                axis=0,
            )

        # Store predictions for each fold
        fold_predictions[fold] = all_predicted_values

        # Collect validation metrics for each fold
        validation[fold] = regression_report(
            all_predicted_values["true"],
            all_predicted_values["predicted"],
            threshold - shift,
        )

        models[fold] = regressor

    # Find the best fold based on RM
    best_fold = validation.T["neg_root_mean_squared_log_error"].idxmin()

    plot_confusion_matrix(
        fold_predictions[best_fold]["true"].round() + shift,
        fold_predictions[best_fold]["predicted"].round() + shift,
        f"{algorithm}_{antibiotic}",
    )

    # Select the model corresponding to the best fold
    best_model = Shifted_Model(models[best_fold], shift)

    return validation, best_model


# %%

if __name__ == "__main__":
    path = Path(".")

    data = path / "data"
    data.mkdir(exist_ok=True)

    models = path / "models"
    models.mkdir(exist_ok=True)

    tmp = path / "tmp"
    tmp.mkdir(exist_ok=True)

    args = get_parser().parse_args()

    set_verbosity(args.verbose)

    # open Streptococcus 1312 Data
    with open(data / f"{args.file}", "rb") as handle:  # file
        Bacteria = pickle.load(handle)

    # Methods
    methods = ["ENet", "XGBL", "RF", "LGBM", "XGBT", "LSVM", "RBFSVM"]
    kwargs = {"cpus": 16, "ram": "8GB", "time": "01:00:00"}

    modelisation_jobs = []

    for antibiotic in Bacteria["Target"]:
        X = Bacteria["Selection"][antibiotic].drop(index=Bacteria["Anomalies"][antibiotic])

        mic = group_small(
            Bacteria["Target"][antibiotic].drop(index=Bacteria["Anomalies"][antibiotic])
        )

        shift = mic.min() - 1
        y = mic - shift

        threshold = Bacteria["Threshold"][antibiotic]

        for algorithm in methods:

            @job(
                name=f"Modelisation_{antibiotic}_{algorithm}",
                **kwargs,
            )
            def modelisation():
                model = train(
                    X=X,
                    y=y,
                    algorithm=algorithm,
                    antibiotic=antibiotic,
                    n_trials=args.n_trials,
                )

                validation, model = nested_crossval(
                    X=X,
                    y=y,
                    models=model,
                    algorithm=algorithm,
                    antibiotic=antibiotic,
                    shift=shift,
                    threshold=threshold,
                )

                with open(
                    models / f"model_{antibiotic}_{algorithm}.pkl",
                    "wb",
                ) as handle:
                    pickle.dump(model, handle)

                with open(
                    tmp / f"validation_{antibiotic}_{algorithm}.pkl",
                    "wb",
                ) as handle:
                    pickle.dump(validation, handle)

            modelisation_jobs.append(modelisation)

    @after(*modelisation_jobs)  # Waits for all selection jobs to complete
    @job(
        name="collect_results",
        **kwargs,
    )
    def collect():
        # Get all .pkl files in the directory
        files = list(tmp.glob("validation*"))
        is_initialized = False

        results = pd.DataFrame()
        results.columns = pd.MultiIndex.from_product([[""], results.columns])

        for file in files:
            # Open the .pkl file
            with open(file, "rb") as pkl_file:
                content = pickle.load(pkl_file)
            antibio = file.name.split("_")[1]
            model = file.name.split("_")[-1].replace(".pkl", "")
            content.columns = pd.MultiIndex.from_tuples(
                [(antibio, model, col) for col in content.columns]
            )

            if not is_initialized:
                # If results hasn't been initialized, set it to the first DataFrame
                results = content
                is_initialized = True
            else:
                # If results has been initialized, merge the new DataFrame with it
                results = results.merge(content, left_index=True, right_index=True)

        results.columns.names = ["Antibiotic", "Model", "Fold"]
        results = results.T
        results.to_csv(path / f"{args.file.split('.')[0]}_validation_results.csv")

        results.reset_index(inplace=True)
        mean_results = mean_report(results)
        mean_results.to_csv(
            path / f"{args.file.split('.')[0]}_validation_mean_results.csv", index=False
        )

    schedule(
        collect,
        name="MicRoProGen",
        backend=args.backend,
        env=[
            "source ~/.bashrc",
            "conda activate ml_magitics",
        ],
    )
