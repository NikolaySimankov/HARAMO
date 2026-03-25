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


def filter_by_annotations(group):
    """
    Filters a DataFrame group of proteins by species based on reference length or annotation count.
    Parameters:
    group (pd.DataFrame): A DataFrame containing protein information with columns "RefSeq", "Length",
                          "Gene_Ontology", "Gene3D", "Pfam", and "SUPFAM".
    Returns:
    pd.DataFrame: A filtered DataFrame where proteins have lengths within 5% of the reference length.
    The function performs the following steps:
    1. If any RefSeq values are present, use the length of the first RefSeq entry as the reference length.
    2. If no RefSeq values are present, compute the total annotation count from "Gene_Ontology", "Gene3D",
       "Pfam", and "SUPFAM" columns.
    3. If the annotation sum varies within the group, use the protein with the highest annotation count as the reference.
    4. If all annotation counts are the same, use the 75th percentile length as the reference length.
    5. Define an acceptable length range (5% deviation from the reference length).
    6. Filter out proteins that are outside the acceptable length range.
    """

    if group["RefSeq"].notna().any():
        # Use RefSeq length if available
        reference_length = group.loc[group["RefSeq"].notna(), "Length"].values[0]
    else:
        # Compute the total annotation count from multiple columns
        annotation_cols = ["Gene_Ontology", "Gene3D", "Pfam", "SUPFAM"]
        group["Annotation_Sum"] = (
            group[annotation_cols]
            .fillna("")
            .map(lambda col: col.count(";"))
            .sum(axis=1)
        )

        # Check if the annotation sum varies in the group
        if group["Annotation_Sum"].nunique() > 1:
            # Use the protein with the highest annotation count as the reference
            most_complete_idx = group["Annotation_Sum"].idxmax()
            reference_length = group.loc[most_complete_idx, "Length"]
        else:
            # If all annotation counts are the same, use the 75th percentile length
            reference_length = group["Length"][
                group["Length"] >= group["Length"].quantile(0.75)
            ].mode()[0]

    # Define acceptable range (5% deviation)
    lower_bound = reference_length * 0.95
    upper_bound = reference_length * 1.05

    # Filter out proteins that are outside the acceptable range
    group = group[(group["Length"] >= lower_bound) & (group["Length"] <= upper_bound)]

    return group


def filter_by_size(group):
    """
    Filters a DataFrame group by the 'Length' column using the Interquartile Range (IQR) method.
    Rows with 'Length' values outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] are removed,
    except for rows where the 'RefSeq' column is not null.
    Parameters:
    group (pd.DataFrame): The DataFrame group to be filtered. It must contain 'Length' and 'RefSeq' columns.
    Returns:
    pd.DataFrame: The filtered DataFrame group.
    """

    # Calculate the interquartile range (IQR)
    Q1 = group["Length"].quantile(0.25)
    Q3 = group["Length"].quantile(0.75)
    IQR = Q3 - Q1

    # Define acceptable range
    lower_bound = Q1 - IQR
    upper_bound = Q3 + IQR

    # Filter out proteins that are outside the acceptable range unless RefSeq is not null
    group = group[
        (group["Length"] >= lower_bound) & (group["Length"] <= upper_bound)
        | (group["RefSeq"].notna())
    ]

    return group


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
    # all_targets = pd.read_csv(data / "all_genopredict_targets.tsv", sep="\t")
    # all_targets.drop(columns=["SAR"], inplace=True)
    all_targets = pd.read_csv(data / "Plant_Viruses_host_species.tsv", sep="\t")

    # Load the Biophysical properties file into a DataFrame
    X = pd.read_csv(data / f"{args.folder}_Fasta_EMBL.fasta.out.plearn.csv", sep=",")

    # Load the Protein metadata file into a DataFrame
    # taxo = pd.read_csv(data / f"{args.folder}_Taxo_Modules_EMBL_NS.tsv", sep="\t")
    taxo = pd.read_csv(data / f"{args.folder}_Modules_2025_NS.tsv", sep="\t")

    # Merge the Biophysical properties and Protein metadata DataFrames
    joined = pd.merge(taxo, X, how="inner", left_index=True, right_index=True)
    joined = joined[joined["Abbreviation"].notna()]
    joined.drop(columns=["sequence_id"], inplace=True)
    abbreviations = joined["Abbreviation"].unique()

    # Split comma-separated values and remove duplicates
    proteins = set()
    for item in abbreviations:
        proteins.update(item.split(","))
    proteins = list(proteins)

    for protein in proteins:

        biophys = joined.loc[
            joined["Abbreviation"].str.contains(
                rf"(?<![a-zA-Z]){protein}(?![a-zA-Z])", case=False
            )
        ]
        biophys = biophys.groupby("Virus_TaxID", group_keys=False).apply(
            filter_by_annotations
        )
        biophys = biophys.groupby("Taxonomic_Lineage_Family", group_keys=False).apply(
            filter_by_size
        )
        biophys = biophys.drop_duplicates(subset="Entry")

        if len(biophys) >= 500:

            # subset.to_csv(f"C:\\Users\\nikol\\Documents\\GitHub\\virus-db\\data\\{name}_{protein}.tsv", index=False, sep='\t')

            intersect = pd.merge(biophys, all_targets, how="inner", on="Virus_Species")
            biophys = intersect[biophys.columns]
            biophys.set_index("Entry", inplace=True)

            targets = intersect[["Entry"] + list(all_targets.columns)]
            targets.drop(columns=["Virus_Species"], inplace=True)
            targets.set_index("Entry", inplace=True)

            sum = targets.apply(lambda x: x.sum(), axis=0).sort_values(ascending=False)
            consistant_targets = sum[sum >= 200].index

            kwargs_heavy = {"cpus": 16, "ram": "64GB", "time": "16:00:00"}

            @job(name=f"Optimisation {args.folder}: {protein}_sp200", **kwargs_heavy)
            def optimisation():

                for target in consistant_targets:
                    try:
                        log_path = logs / f"{args.folder}_{protein}_{target}.log"
                        if not os.path.exists(log_path):
                            X = biophys.loc[:, "A":]
                            y = targets[target]

                            X.dropna(axis=1, inplace=True)

                            magic_now(
                                X=X,
                                y=y,
                                scoring=mcc_scorer,
                                algorithm=["RF", "LGBM", "DLR"],
                                scaler=[None, "robust"],
                                feature_selector="pvalue",
                                hyperparameters="optimize",
                                n_trials=args.n_trials,
                                output_dir=output_dir,
                                tag=f"_{protein}_{target}",
                            )

                            with open(log_path, "w") as file:
                                file.write("done")

                    except:
                        pass

            jobs.append(optimisation)

    schedule(
        *jobs,
        name="haramo4genopredict",
        backend=args.backend,
        # prune=True,
        env=[
            "source ~/.bashrc",
            "conda activate tabml",
        ],
    )
