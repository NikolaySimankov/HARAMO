#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd

from pathlib import Path

from dawgz import (
    job,
    ensure,
    schedule,
)

from itertools import product

import os
import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from haramo.classification import magic_now

from sklearn.metrics import make_scorer, matthews_corrcoef

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
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
        "--t",
        default=250,
        type=int,
        dest="n_trials",
        help="number of trials for the hyperparameter optimization process",
        required=False,
    )

    parser.add_argument(
        "--v",
        default=1,
        type=int,
        dest="verbose",
        help="whether to print the progress of the training and validation",
        required=False,
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


def vectors_to_matrix(*vectors):
    return [list(item) for item in product(*vectors)]


########
# Main #
########

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    mcc_scorer = make_scorer(matthews_corrcoef)

    path = Path(".")

    output_dir = path / args.folder
    output_dir.mkdir(exist_ok=True)

    data = path / "data"
    data.mkdir(exist_ok=True)

    logs = path / "logs"
    logs.mkdir(exist_ok=True)

    jobs = []

    # Load the target file into a DataFrame
    all_targets = pd.read_csv(
        data / "Plant_Viruses_host_species.tsv", sep="\t", index_col="Virus_Species"
    )
    counts = all_targets.apply(sum)
    consistent_targets = counts[counts >= 8].index
    all_targets = all_targets[consistent_targets]
    all_targets.reset_index(inplace=True)

    # Load the Biophysical properties file into a DataFrame
    X = pd.read_csv(data / "predomics_biophys_V2.tsv", sep="\t", index_col="Prot_ID")

    # Load the Protein metadata file into a DataFrame
    taxo = pd.read_csv(
        data / "clustered_proteins_V5.2.tsv", sep="\t", index_col="Prot_ID"
    )

    # Merge the Biophysical properties and Protein metadata DataFrames
    joined = pd.merge(taxo, X, how="inner", left_index=True, right_index=True)
    joined = joined[joined["Definitive_name"].notna()]

    # Split comma-separated values and remove duplicates
    proteins = [
        "DNA replication protein",
        "RNA-dependent RNA polymerase",
        "Reverse transcriptase",
        "Coat protein",
        "Movement protein",
        "Transactivator/viroplasmin protein",
        "RNA silencing suppressor",
        "Vector transmission protein",
        "RNA-dependent RNA polymerase complex",
        "Reverse transcriptase complex",
    ]

    for protein in proteins:

        # forbid matches where the phrase is followed by another word (e.g. "... complex")
        pattern = rf"\b{re.escape(protein)}\b(?!\s+\w)"
        biophys = joined.loc[
            joined["Definitive_name"].str.contains(
                pattern, case=False, regex=True, na=False
            )
        ]
        biophys.reset_index(inplace=True)

        if len(biophys) >= 500:

            intersect = pd.merge(biophys, all_targets, how="inner", on="Virus_Species")
            X = intersect[["Prot_ID"] + list(X.columns)]
            X.set_index("Prot_ID", inplace=True)

            targets = intersect[["Prot_ID"] + list(all_targets.columns)]
            targets.drop(columns=["Virus_Species"], inplace=True)
            targets.set_index("Prot_ID", inplace=True)

            groups = intersect["Virus_Species"]

            sum = targets.apply(lambda x: x.sum(), axis=0).sort_values(ascending=False)
            consistant_targets = sum[sum >= 200].index

            kwargs_heavy = {"cpus": 12, "ram": "32GB", "time": "03:00:00"}

            @job(name=f"Optimisation {args.folder}: {protein}_sp200", **kwargs_heavy)
            def optimisation():

                for target in consistant_targets:
                    try:
                        log_path = logs / f"{args.folder}_{protein}_{target}.log"
                        if not os.path.exists(log_path):
                            y = targets[target]

                            X.dropna(axis=1, inplace=True)

                            magic_now(
                                X=X,
                                y=y,
                                groups=groups,
                                scoring=mcc_scorer,
                                algorithm=["RBFSVM","LGBM"],
                                scaler="standard",
                                feature_selector="boruta",
                                hyperparameters="default",
                                n_trials=args.n_trials,
                                output_dir=output_dir,
                                tag=f"_{protein}_{target}",
                                n_jobs=12,
                            )

                            with open(log_path, "w") as file:
                                file.write("done")

                    except Exception as e:
                        with open(log_path, "w") as file:
                            import traceback
    
                            file.write(traceback.format_exc())
                        raise
                    break
            jobs.append(optimisation)
        
    schedule(
        *jobs,
        name="Haramo4PredOmics",
        backend=args.backend,
        # prune=True,
        env=[
            "source ~/.bashrc",
            "conda activate tabml",
        ],
    )
