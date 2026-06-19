"""Regenerate a subject-indexed magnitude-judgement behaviour table.

The shared CSV (data/magjudge_behavior_DNumRisk.csv) has no `subject` column,
so it can only support a *pooled* fit. A hierarchical DDM needs per-subject
data.

This reproduces numrisk.behavior_magjudge.utils.get_data (same columns, same
`choice = chose_n2`, same group join), but is **robust**: numrisk's loader
crashes (`No objects to concatenate`) on any sub-* dir that has no magjudge
events.tsv, and it ignores its own subject_list arg. Here we skip such
subjects (and any whose events don't parse) and report them.

Run on the cluster, where the BIDS folder lives:

    srun -c2 --mem 8G --time 15 --account=zne.uzh \
        python prepare_data.py --bids-folder /shares/zne.uzh/mrenke/ds-dnumrisk \
        --out ~/git/dyscalculic_ddm/data/magjudge_behavior_subjectwise.csv

then rsync the CSV back to the laptop.
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd


def cleanup_behavior(df_):
    """Identical to numrisk.behavior_magjudge.utils.cleanup_behavior."""
    df = df_[[]].copy()
    df["rt"] = df_.loc[:, ("onset", "choice")] - df_.loc[:, ("onset", "stimulus 2")]
    df["n1"], df["n2"] = df_["n1"]["stimulus 1"], df_["n2"]["stimulus 1"]
    df["choice"] = df_[("choice", "choice")]
    df["chose_n2"] = (df["choice"] == 2.0)
    df["frac"] = df["n2"] / df["n1"]
    df["log(n2/n1)"] = np.log(df["frac"])
    df["log(n1)"] = np.log(df["n1"])
    df = df.droplevel(-1, 1)
    return df


def get_behavior(bids_folder, session=1, runs=range(1, 7)):
    subject_list = sorted(
        f[4:] for f in os.listdir(bids_folder)
        if f[:3] == "sub" and len(f) == 6
    )
    print(f"Found {len(subject_list)} sub-* dirs")

    df_all, skipped = [], []
    for subject in subject_list:
        df_sub = []
        for run in runs:
            fn = op.join(
                bids_folder,
                f"sub-{subject}/ses-{session}/func/"
                f"sub-{subject}_ses-{session}_task-magjudge_run-{run}_events.tsv",
            )
            if op.exists(fn):
                d = pd.read_csv(fn, sep="\t", index_col=["trial_nr", "trial_type"])
                d["subject"], d["run"] = int(subject), run
                df_sub.append(d)

        if not df_sub:                     # numrisk crashes here; we skip instead
            skipped.append((subject, "no magjudge run files"))
            continue
        try:
            df_sub = pd.concat(df_sub)
            df_sub = (df_sub.reset_index()
                      .set_index(["subject", "run", "trial_type", "trial_nr"])
                      .unstack("trial_type"))
            df_all.append(cleanup_behavior(df_sub))
        except (KeyError, ValueError) as e:
            skipped.append((subject, f"parse error: {e}"))

    if skipped:
        print(f"Skipped {len(skipped)} subjects:")
        for s, why in skipped:
            print(f"  sub-{s}: {why}")

    return pd.concat(df_all)


def main(bids_folder, out):
    df = get_behavior(bids_folder)

    # Join group from the participants table (same as numrisk.get_data).
    parts = pd.read_csv(
        op.join(bids_folder, "add_tables", "subjects_recruit_scan_scanned-final.csv"),
        header=0,
    )
    parts = (parts.loc[:, ["subject ID", "age", "group", "gender"]]
             .rename(columns={"subject ID": "subject"})
             .dropna()
             .astype({"subject": int, "group": int})
             .set_index("subject"))
    df = df.join(parts["group"], on="subject", how="left").dropna()
    df["choice"] = df["chose_n2"]

    os.makedirs(op.dirname(op.abspath(out)), exist_ok=True)
    df.reset_index().to_csv(out, index=False)

    n_sub = df.index.get_level_values("subject").nunique()
    print(f"\nWrote {out}: {len(df)} trials, {n_sub} subjects with group, "
          f"groups={df['group'].value_counts().to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bids-folder", default="/shares/zne.uzh/mrenke/ds-dnumrisk")
    p.add_argument("--out", default="data/magjudge_behavior_subjectwise.csv")
    args = p.parse_args()
    main(args.bids_folder, args.out)
