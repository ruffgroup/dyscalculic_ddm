# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # DNumRisk magnitude judgement — quick behavioural look
#
# Two-alternative magnitude comparison: on each trial the participant sees
# `n1` then `n2` and chooses which is larger (`choice` / `chose_n2` = chose the
# second number). Groups: **control** vs **dyscalc**(ulic).
#
# The signed difficulty axis is `log(n2/n1)`: positive → n2 larger (correct
# answer is "n2"), magnitude → easy. `|log(n2/n1)|` is unsigned difficulty
# (the classic ratio / distance effect: harder pairs are slower and less
# accurate).
#
# We hunt for the usual qualitative signatures and, above all, for a
# **group difference** in any of them.

# %%
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PAL = {"control": "#4C72B0", "dyscalc": "#C44E52"}

# Maike's DNumRisk table is the canonical full sample (66 subjects, has a
# `subject` column + `correct`); prefer it. The *_subjectwise.csv is the
# partial 42-subject cluster regeneration kept only as a fallback.
CANDIDATES = ["../data/magjudge_behavior_DNumRisk.csv",
              "data/magjudge_behavior_DNumRisk.csv",
              "../data/magjudge_behavior_subjectwise.csv",
              "data/magjudge_behavior_subjectwise.csv"]
csv = next(p for p in CANDIDATES if op.exists(p))
df = pd.read_csv(csv)
HAS_SUBJECT = "subject" in df.columns
print(f"Loaded {csv}  ({len(df)} trials, subject column: {HAS_SUBJECT})")

# Harmonise the group label to strings regardless of source encoding.
if df["group"].dtype != object:
    df["group"] = df["group"].map({0: "control", 1: "dyscalc"}).fillna(df["group"])

for col in ("log(n2/n1)", "log(n1)"):
    if col not in df.columns:
        df["log(n2/n1)"] = np.log(df["n2"] / df["n1"])
        df["log(n1)"] = np.log(df["n1"])
df["abs_log_ratio"] = df["log(n2/n1)"].abs()
df["chose_n2"] = df["chose_n2"].astype(bool)
# Derive correctness if the source CSV doesn't carry it (the subjectwise
# fallback doesn't), so the notebook works with either table.
if "correct_answer_n2" not in df.columns:
    df["correct_answer_n2"] = df["n2"] > df["n1"]
if "correct" not in df.columns:
    df["correct"] = df["correct_answer_n2"] == df["chose_n2"]
df["correct"] = df["correct"].astype(bool)

# Bin the (signed and unsigned) difficulty for clean lines.
df["ratio_bin"] = pd.qcut(df["log(n2/n1)"], 9, duplicates="drop")
df["ratio_bin_c"] = df["ratio_bin"].apply(lambda iv: iv.mid).astype(float)
df["absdiff_bin"] = pd.qcut(df["abs_log_ratio"], 6, duplicates="drop")
df["absdiff_bin_c"] = df["absdiff_bin"].apply(lambda iv: iv.mid).astype(float)

print(df.groupby("group").agg(
    n_trials=("rt", "size"),
    accuracy=("correct", "mean"),
    median_rt=("rt", "median")))


# %%
def by_subject(frame, xcol, ycol):
    """Per-(group, x-bin) means; per subject first when we have subjects,
    so seaborn's CI is across subjects rather than across trials."""
    keys = ["group", xcol]
    if HAS_SUBJECT:
        keys = ["subject", "group", xcol]
    out = frame.groupby(keys, observed=True)[ycol].mean().reset_index()
    return out


# %% [markdown]
# ## 1. Psychometric & ratio (distance) effect
# Left: choice curve — P(chose n2) should rise sigmoidally with `log(n2/n1)`;
# a **shallower slope = noisier number sense**. Right: accuracy vs unsigned
# difficulty — the ratio effect.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

psych = by_subject(df, "ratio_bin_c", "chose_n2")
sns.lineplot(data=psych, x="ratio_bin_c", y="chose_n2", hue="group",
             palette=PAL, marker="o", ax=axes[0], errorbar=("ci", 95))
axes[0].axhline(0.5, ls=":", c="grey"); axes[0].axvline(0, ls=":", c="grey")
axes[0].set(xlabel="Log(n2/n1)", ylabel="P(chose n2)", title="Psychometric curve")

acc = by_subject(df, "absdiff_bin_c", "correct")
sns.lineplot(data=acc, x="absdiff_bin_c", y="correct", hue="group",
             palette=PAL, marker="o", ax=axes[1], errorbar=("ci", 95))
axes[1].axhline(0.5, ls=":", c="grey")
axes[1].set(xlabel="|Log(n2/n1)|  (difficulty → easy)", ylabel="P(correct)",
            title="Accuracy vs ratio")
fig.tight_layout()


# %% [markdown]
# ## 2. Chronometric effect
# RT should peak near the hardest pairs (`log(n2/n1) ≈ 0`) and fall for easy
# ones. A flatter or higher curve in one group is a candidate effect.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

rt_sign = by_subject(df, "ratio_bin_c", "rt")
sns.lineplot(data=rt_sign, x="ratio_bin_c", y="rt", hue="group",
             palette=PAL, marker="o", ax=axes[0], errorbar=("ci", 95))
axes[0].axvline(0, ls=":", c="grey")
axes[0].set(xlabel="Log(n2/n1)", ylabel="RT (s)", title="Chronometric curve")

# RT by difficulty, split correct vs error (error RTs are diagnostic for DDMs).
rt_diff = df.groupby(
    (["subject"] if HAS_SUBJECT else []) + ["group", "correct", "absdiff_bin_c"],
    observed=True)["rt"].mean().reset_index()
sns.lineplot(data=rt_diff, x="absdiff_bin_c", y="rt", hue="group",
             style="correct", palette=PAL, marker="o", ax=axes[1],
             errorbar=("ci", 95))
axes[1].set(xlabel="|Log(n2/n1)|  (difficulty → easy)", ylabel="RT (s)",
            title="RT by difficulty & correctness")
fig.tight_layout()


# %% [markdown]
# ## 3. Size effect & RT distributions
# Left: does accuracy depend on the absolute magnitude `log(n1)` (Weber)?
# Right: RT distributions per group — shape matters for the DDM (skew, floor).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

df["n1_bin"] = pd.qcut(df["log(n1)"], 5, duplicates="drop")
df["n1_bin_c"] = df["n1_bin"].apply(lambda iv: iv.mid).astype(float)
size = by_subject(df, "n1_bin_c", "correct")
sns.lineplot(data=size, x="n1_bin_c", y="correct", hue="group",
             palette=PAL, marker="o", ax=axes[0], errorbar=("ci", 95))
axes[0].set(xlabel="Log(n1)", ylabel="P(correct)", title="Size effect")

sns.kdeplot(data=df[df.rt < 5], x="rt", hue="group", palette=PAL,
            fill=True, alpha=.25, common_norm=False, ax=axes[1])
axes[1].axvline(0.2, ls=":", c="k")  # the rt<0.20 s cutoff used for the DDM
axes[1].set(xlabel="RT (s)", title="RT distribution (line = 0.2 s DDM cutoff)")
fig.tight_layout()


# %% [markdown]
# ## 4. Effect hunt — group contrast summary
# Per-subject summaries (when available) so the comparison is across people,
# not trials. Look for separation in accuracy / RT / psychometric slope.

# %%
if HAS_SUBJECT:
    persub = df.groupby(["subject", "group"]).agg(
        accuracy=("correct", "mean"),
        median_rt=("rt", "median"),
        mean_rt=("rt", "mean")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, y, ttl in zip(
            axes, ["accuracy", "median_rt", "mean_rt"],
            ["Accuracy", "Median RT (s)", "Mean RT (s)"]):
        sns.boxplot(data=persub, x="group", y=y, hue="group", palette=PAL,
                    legend=False, ax=ax, showfliers=False,
                    order=["control", "dyscalc"])
        sns.stripplot(data=persub, x="group", y=y, color="k", alpha=.5,
                      size=3, ax=ax, order=["control", "dyscalc"])
        ax.set(title=ttl, xlabel="")
    fig.tight_layout()

    # Quick Welch t-tests across subjects (descriptive, not the real model).
    from scipy import stats
    c = persub[persub.group == "control"]
    d = persub[persub.group == "dyscalc"]
    print(f"N: control={len(c)}, dyscalc={len(d)}")
    for y in ["accuracy", "median_rt"]:
        t, p = stats.ttest_ind(c[y], d[y], equal_var=False)
        print(f"{y:>10}: control={c[y].mean():.3f}  dyscalc={d[y].mean():.3f}"
              f"  Welch t={t:+.2f}, p={p:.3f}")
else:
    print("No subject column in this CSV — regenerate the subject-wise table "
          "(prepare_data.py) for an across-subject contrast.")
