#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd

import pandas as pd
import glob
import os

from pathlib import Path

from itertools import product
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
    
    outputs = output_dir / "outputs"
    outputs.mkdir(exist_ok=True)

    results = output_dir / "results"
    results.mkdir(exist_ok=True)

    # Get all validation_*.tsv files
    files = list(results.glob(f"validation_*.tsv"))

    # List to store processed dataframes
    df_list = []

    for file in files:
        # Read TSV file
        df = pd.read_csv(file, sep="\t")

        # Get the row with the highest MCC
        max_mcc_row = df.loc[df["MCC"].idxmax()]

        # Extract protein and target from filename  (strip leading "validation_" and ".tsv")
        stem = os.path.basename(file)[len("validation_"):].replace(".tsv", "")
        parts = stem.split("_", 1)
        protein = parts[0] if len(parts) > 0 else ""
        target = parts[1] if len(parts) > 1 else ""

        # Convert the Series to DataFrame and add new columns
        max_mcc_row = max_mcc_row.to_frame().T
        max_mcc_row["Protein"] = protein
        max_mcc_row["Target"] = target

        # Add best dataset combo if dataset_selection file exists
        ds_path = results / f"dataset_selection_{protein}_{target}.tsv"
        if ds_path.exists():
            try:
                ds_scores = pd.read_csv(ds_path, sep="\t", index_col=0)
                max_mcc_row["Best_combo"] = ds_scores.index[0]
            except Exception:
                max_mcc_row["Best_combo"] = ""
        else:
            max_mcc_row["Best_combo"] = ""

        # Append to the list
        df_list.append(max_mcc_row)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)

    # Save to a new TSV file
    final_df.to_csv(
        outputs / f"{args.folder}_validation_results.tsv", sep="\t", index=False
    )

    print("Processing complete. Results saved in validation_results.tsv")
