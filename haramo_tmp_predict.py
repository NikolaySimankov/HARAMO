#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd
import os
import re
import warnings
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from joblib import Parallel, delayed

from dawgz import job, schedule

warnings.filterwarnings("ignore")

#############
# Functions #
#############


def get_parser():
    parser = ArgumentParser(
        description="Backfill y_1 / y_0 / ratio_1_0 / n_groups into validation TSVs "
        "produced by older haramo versions that did not record these columns.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--d", dest="folder", help="output folder", required=True)
    parser.add_argument(
        "--b",
        default="slurm",
        type=str,
        dest="backend",
        help="backend to use for the parallelization of the jobs",
        required=True,
    )
    return parser


def _patch_file(val_path, lookup):
    stem = val_path.stem[len("validation_") :]
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return f"[skip] cannot parse protein/target from {val_path.name}"

    protein, target = parts[0], parts[1]

    if (protein, target) not in lookup:
        return f"[skip] no data for ({protein!r}, {target!r})"

    df = pd.read_csv(val_path, sep="\t")

    if "positives" in df.columns and df["positives"].notna().any():
        return f"[already up to date] {val_path.name}"

    y, groups = lookup[(protein, target)]
    n_1 = int((y == 1).sum())
    n_0 = int((y == 0).sum())

    df["positives"] = n_1
    df["negatives"] = n_0
    df["class_imbalance"] = round(n_1 / n_0, 4) if n_0 > 0 else float("inf")
    df["n_groups"] = int(groups.nunique())

    df.to_csv(val_path, sep="\t", index=False)
    return f"[patched] {val_path.name}  (y_1={n_1}, y_0={n_0}, n_groups={groups.nunique()})"


########
# Main #
########

if __name__ == "__main__":

    args = get_parser().parse_args()

    path = Path(".")
    output_dir = path / args.folder
    output_dir.mkdir(exist_ok=True)

    kwargs_heavy = {"cpus": 12, "ram": "64GB", "time": "03:00:00"}

    @job(name=f"Predict {args.folder}", **kwargs_heavy)
    def predict():

        results = output_dir / "results"
        data = path / "data"

        # ------------------------------------------------------------------ #
        # Load data — mirrors haramo_cluster_exsp.py exactly                  #
        # ------------------------------------------------------------------ #
        all_targets = pd.read_csv(
            data / "DATABASE_SEED.tsv", sep="\t", index_col="Virus_Species"
        )
        counts = all_targets.apply(sum)
        consistent_targets = counts[counts >= 12].index
        all_targets = all_targets[consistent_targets]
        all_targets.reset_index(inplace=True)

        all_X = {
            name: pd.read_csv(data / fname, sep="\t", index_col="Prot_ID")
            for name, fname in {
                "ctdc": "X_ctdc.tsv",
                "ctdt": "X_ctdt.tsv",
                "ctdd": "X_ctdd.tsv",
                "b2b": "X_b2btools.tsv",
                "nsp": "X_netsurfp.tsv",
                "biophys": "X_biophys.tsv",
                "class": "X_class.tsv",
            }.items()
        }

        taxo = pd.read_csv(
            data / "clustered_proteins_V5.2.tsv", sep="\t", index_col="Prot_ID"
        )
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

        # ------------------------------------------------------------------ #
        # Build lookup: (protein, target) -> (y, groups)                      #
        # ------------------------------------------------------------------ #
        lookup: dict = {}

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
            targets = intersect[["Prot_ID"] + list(all_targets.columns)]
            targets.drop(columns=["Virus_Species"], inplace=True)
            targets.set_index("Prot_ID", inplace=True)
            groups = intersect.set_index("Prot_ID")["Virus_Species"]

            prot_counts = targets.apply(lambda x: x.sum(), axis=0).sort_values(
                ascending=False
            )
            consistant_targets = prot_counts[prot_counts >= 100].index

            for target in consistant_targets:
                lookup[(protein, target)] = (targets[target], groups)

        # ------------------------------------------------------------------ #
        # Patch every validation TSV in parallel                              #
        # ------------------------------------------------------------------ #
        val_files = sorted(results.glob("validation_*.tsv"))

        outcomes = Parallel(n_jobs=kwargs_heavy["cpus"])(
            delayed(_patch_file)(val_path, lookup) for val_path in val_files
        )

        patched = sum(1 for msg in outcomes if msg.startswith("[patched]"))
        skipped = sum(1 for msg in outcomes if msg.startswith("[already"))

        for msg in outcomes:
            print(msg)

        print(f"\nDone — {patched} file(s) patched, {skipped} already up to date.")

    schedule(
        predict,
        name="haramo_tmp_predict",
        backend=args.backend,
        env=[
            "source ~/.bashrc",
            "conda activate tabml",
        ],
    )
