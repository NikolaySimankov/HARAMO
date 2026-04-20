#!/usr/bin/env python

###########
# Imports #
###########

import numpy as np
import pandas as pd

from pathlib import Path
import pickle

import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

from joblib import Parallel, delayed

from dawgz import (
    job,
    schedule,
)

#############
# Functions #
#############


def get_parser():

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--d",
        dest="folder",
        help="destination folder",
        required=True,
    )

    parser.add_argument(
        "--b",
        default="slurm",
        type=str,
        dest="backend",
        help="backend to use for the parallelization of the jobs",
        required=True,
    )

    return parser


def _predict_one(protein, target, X_all, prot_ids, taxo, models, results_dir, val_meta):
    algorithms = ["LGBM", "XGB", "CatB"]

    ds_path = results_dir / f"dataset_selection_{protein}_{target}.tsv"
    best_combo = ""
    if ds_path.exists():
        try:
            ds_scores = pd.read_csv(ds_path, sep="\t", index_col=0)
            best_combo = ds_scores.index[0]
        except Exception:
            pass

    probas = []
    for alg in algorithms:
        pipeline_path = models / f"pipelines_{alg}_{protein}_{target}.pkl"
        if pipeline_path.exists():
            with open(pipeline_path, "rb") as handle:
                pipeline = pickle.load(handle)
            try:
                X_pred = X_all[pipeline[-1].feature_names_in_]
                probas.append(pipeline[-1].predict_proba(X_pred)[:, 1])
            except Exception:
                pass

    if not probas:
        return None

    proba = np.mean(probas, axis=0)
    meta = val_meta.get((protein, target), {})

    new_predictions = taxo.loc[prot_ids].copy()
    new_predictions.insert(0, "Target", target)
    new_predictions.insert(0, "Protein", protein)
    new_predictions.insert(0, "Best_combo", best_combo)
    new_predictions.insert(0, "n_groups", meta.get("n_groups", np.nan))
    new_predictions.insert(0, "class_imbalance", meta.get("class_imbalance", np.nan))
    new_predictions.insert(0, "negatives", meta.get("negatives", np.nan))
    new_predictions.insert(0, "positives", meta.get("positives", np.nan))
    new_predictions.insert(0, "Expected_Selectivity", meta.get("Selectivity", np.nan))
    new_predictions.insert(0, "Expected_Sensitivity", meta.get("Sensitivity", np.nan))
    new_predictions.insert(0, "Probability", proba.round(3))

    return new_predictions


########
# Main #
########

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    path = Path(".")

    output_dir = path / args.folder
    output_dir.mkdir(exist_ok=True)

    data = path / "data"
    data.mkdir(exist_ok=True)

    outputs = output_dir / "outputs"
    outputs.mkdir(exist_ok=True)

    models = output_dir / "models"
    models.mkdir(exist_ok=True)

    kwargs_heavy = {"cpus": 12, "ram": "128GB", "time": "03:00:00"}

    @job(name=f"Predict {args.folder}", **kwargs_heavy)
    def predict():

        # Build (Protein, Target) → validation metadata lookup
        val_results_path = output_dir / "results" / f"{args.folder}_validation_results.tsv"
        val_meta = {}
        if val_results_path.exists():
            vr = pd.read_csv(val_results_path, sep="\t")
            for _, row in vr.iterrows():
                key = (row.get("Protein", ""), row.get("Target", ""))
                val_meta[key] = row.to_dict()

        # Load the target file into a DataFrame
        all_targets = pd.read_csv(
            data / "DATABASE_SEED.tsv", sep="\t", index_col="Virus_Species"
        )

        target_counts = all_targets.apply(sum)
        consistent_targets = target_counts[target_counts >= 12].index
        all_targets = all_targets[consistent_targets]
        all_targets.reset_index(inplace=True)

        # Load the feature DataFrames
        feature_files = {
            # "ctd": "X_ctd.tsv",
            "ctdc": "X_ctdc.tsv",
            "ctdt": "X_ctdt.tsv",
            "ctdd": "X_ctdd.tsv",
            # "aac": "X_aac.tsv",
            "b2b": "X_b2btools.tsv",
            "nsp": "X_netsurfp.tsv",
            # "residue": "X_residue.tsv",
            "biophys": "X_biophys.tsv",
            "class": "X_class.tsv",
        }

        all_X = {
            name: pd.read_csv(data / fname, sep="\t", index_col="Prot_ID")
            for name, fname in feature_files.items()
        }

        # Load the Protein metadata file into a DataFrame
        taxo = pd.read_csv(
            data / "clustered_proteins_V5.2.tsv", sep="\t", index_col="Prot_ID"
        )

        # Restrict metadata to proteins present in every feature dataset and with a known name
        common_index = taxo.index
        for X in all_X.values():
            common_index = common_index.intersection(X.index)
        taxo = taxo.loc[common_index]
        taxo = taxo[taxo["Definitive_name"].notna()]

        proteins = [
            "DNA replication protein",
            "RNA-dependent RNA polymerase",
            "DNA-RNA polymerase superfamily",
            "Reverse transcriptase",
            "Coat protein",
            "Movement protein",
            "Transactivator-Viroplasmin protein",
            "RNA silencing suppressor",
            "Vector transmission protein",
            "RNA-dependent RNA polymerase complex",
            "Reverse transcriptase complex",
            "Glycoprotein",
        ]

        # Build list of tasks: one per (protein, target) pair
        tasks = []
        for protein in proteins:
            pattern = rf"\b{re.escape(protein)}\b(?!\s+\w)"
            biophys = taxo.loc[
                taxo["Definitive_name"].str.contains(
                    pattern, case=False, regex=True, na=False
                )
            ]
            biophys = biophys.reset_index()[["Prot_ID", "Virus_Species"]]

            if len(biophys) < 500:
                continue

            intersect = pd.merge(biophys, all_targets, how="inner", on="Virus_Species")
            prot_ids = intersect["Prot_ID"]

            datasets = {name: X.loc[prot_ids].fillna(0) for name, X in all_X.items()}
            X_all = pd.concat(list(datasets.values()), axis=1)

            targets_df = intersect[["Prot_ID"] + list(all_targets.columns)]
            targets_df.drop(columns=["Virus_Species"], inplace=True)
            targets_df.set_index("Prot_ID", inplace=True)

            prot_counts = targets_df.apply(lambda x: x.sum(), axis=0).sort_values(
                ascending=False
            )
            consistant_targets = prot_counts[prot_counts >= 100].index

            for target in consistant_targets:
                tasks.append((protein, target, X_all, prot_ids, taxo))

        # Run all (protein, target) predictions in parallel
        results_dir = output_dir / "results"
        frames = Parallel(n_jobs=kwargs_heavy["cpus"])(
            delayed(_predict_one)(protein, target, X_all, prot_ids, taxo, models, results_dir, val_meta)
            for protein, target, X_all, prot_ids, taxo in tasks
        )

        frames = [f for f in frames if f is not None]
        if not frames:
            return

        result = pd.concat(frames, sort=False)
        result = result[result["Probability"] >= 0.1]

        extra_cols = ["Probability", "Best_combo", "Protein", "Target", "Expected_Sensitivity", "Expected_Selectivity", "positives", "negatives", "class_imbalance", "n_groups"]
        sorted_columns = extra_cols + [c for c in taxo.columns if c not in extra_cols]
        result = result[[c for c in sorted_columns if c in result.columns]]
        result.sort_values(by=["Virus_Species"], inplace=True)

        result.to_csv(
            outputs / f"{args.folder}_autoprediction.tsv", sep="\t", index=True
        )

    schedule(
        predict,
        name="haramo_predict",
        backend=args.backend,
        env=[
            "source ~/.bashrc",
            "conda activate tabml",
        ],
    )
