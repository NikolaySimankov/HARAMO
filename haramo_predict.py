#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd

from pathlib import Path
import pickle

from itertools import product

import os
import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

from dawgz import (
    job,
    ensure,
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


def vectors_to_matrix(*vectors):
    return [list(item) for item in product(*vectors)]


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

    kwargs_heavy = {"cpus": 4, "ram": "128GB", "time": "03:00:00"}

    @job(name=f"Predict {args.folder}", **kwargs_heavy)
    def predict():

        result = pd.DataFrame()

        # Load the target file into a DataFrame
        all_targets = pd.read_csv(
            data / "DATABASE_SEED.tsv", sep="\t", index_col="Virus_Species"
        )
    
        counts = all_targets.apply(sum)
        consistent_targets = counts[counts >= 12].index
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
    
        # Split comma-separated values and remove duplicates
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
    
        for protein in proteins:
    
            # forbid matches where the phrase is followed by another word (e.g. "... complex")
            pattern = rf"\b{re.escape(protein)}\b(?!\s+\w)"
            biophys = taxo.loc[
                taxo["Definitive_name"].str.contains(
                    pattern, case=False, regex=True, na=False
                )
            ]
            biophys = biophys.reset_index()[["Prot_ID", "Virus_Species"]]
    
            if len(biophys) >= 500:
    
                intersect = pd.merge(biophys, all_targets, how="inner", on="Virus_Species")
    
                prot_ids = intersect["Prot_ID"]
    
                # Align each feature dataset to the intersected protein IDs; fill all-NaN values by 0
                datasets = {name: X.loc[prot_ids].fillna(0) for name, X in all_X.items()}
    
                targets = intersect[["Prot_ID"] + list(all_targets.columns)]
                targets.drop(columns=["Virus_Species"], inplace=True)
                targets.set_index("Prot_ID", inplace=True)
    
                groups = intersect.set_index("Prot_ID")["Virus_Species"]
    
                counts = targets.apply(lambda x: x.sum(), axis=0).sort_values(ascending=False)
                consistant_targets = counts[counts >= 100].index

                for target in targets.column:
                
                    y = targets[target]

                    X.dropna(axis=1, inplace=True)

                    # Load the pipeline and get predict_proba
                    pipeline_path = models / f"pipeline_{protein}_{target}.pkl"
                    if pipeline_path.exists():
                        with open(pipeline_path, "rb") as handle:
                            pipeline = pickle.load(handle)
                            
#                        X = X[(pipeline.model.feature_names_in_)]
#                        new_samples = pd.DataFrame(
#                            np.column_stack((y, model.predict(X))),
#                            columns=["true", "predicted"],
#                            index=y.index,
#                        )

                        proba = pipeline.predict_proba(X)[:, 1]
                        
                        new_predictions = tax.copy()
                        new_predictions.insert(0, "Seed_Trasmission", target)
                        new_predictions.insert(0, "Probability", proba.round(3))

                        result = pd.concat([result, new_predictions], sort=False)
                        result = result[result["Probability"] >= 0.1]


        # Get the remaining columns and sort them
        predictions = sorted([col for col in result.columns if col not in tax.columns])
        # Combine the first columns with the sorted remaining columns
        sorted_columns = list(tax.columns) + predictions
        # Reorder the DataFrame columns
        result = result[sorted_columns]
        result.sort_values(by=["Virus_Species"], inplace=True)

        result.to_csv(outputs / f"{args.folder}_autoprediction.tsv", sep="\t", index=True)

    schedule(
        predict,
        name="haramo_predict",
        backend=args.backend,
        env=[
            "source ~/.bashrc",
            "conda activate tabml",
        ],
    )