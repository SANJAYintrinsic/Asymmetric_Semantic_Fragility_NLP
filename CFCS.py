# ==========================================
# 9. NOVEL ADDITION: CROSS-MODEL FRAGILITY CONSENSUS SCORING (CFCS)
# ==========================================
"""
Novel Contribution: Cross-Model Fragility Consensus Scoring (CFCS)
------------------------------------------------------------------
Motivation:
    Existing ASF research treats each model's DFS in isolation.
    This module introduces CFCS — a meta-metric that measures how many
    models *agree* that a sentence pair is semantically fragile.

    Taxonomy:
        - UNIVERSAL FRAGILITY  : All models flip or score high DFS  → Deepest linguistic blind spots
        - PARTIAL FRAGILITY    : Majority of models show fragility   → Structural bias, not model-specific
        - ISOLATED FRAGILITY   : Only 1 model breaks                 → Model-specific quirk

    Why this is novel:
        CFCS transforms per-model DFS vectors into a cross-model consensus landscape,
        enabling identification of linguistically universal failure modes vs.
        architecture-dependent artefacts — a distinction the literature has not formalized.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr

# ---- Thresholds (tunable) ----
DFS_FRAGILE_THRESHOLD = 0.1   # A pair is "fragile" for a model if DFS >= this
UNIVERSAL_CONSENSUS   = 1.0   # 100% of models agree
PARTIAL_CONSENSUS_MIN = 0.5   # >= 50% but < 100%


def compute_cfcs(df_results: pd.DataFrame,
                 dfs_col: str = "DFS",
                 model_col: str = "model",
                 pair_id_cols: list = None,
                 dfs_threshold: float = DFS_FRAGILE_THRESHOLD) -> pd.DataFrame:
    """
    Computes the Cross-Model Fragility Consensus Score for each sentence pair.

    Args:
        df_results     : Output DataFrame from ExperimentRunner (one row per model×pair).
        dfs_col        : Column name for the Directional Fragility Score.
        model_col      : Column name for the model identifier.
        pair_id_cols   : Columns that uniquely identify a sentence pair.
                         Defaults to ['sentence_a', 'sentence_b'].
        dfs_threshold  : DFS value above which a pair is considered fragile for a model.

    Returns:
        DataFrame with one row per sentence pair, with CFCS metrics appended.
    """
    if pair_id_cols is None:
        pair_id_cols = ["sentence_a", "sentence_b"]

    all_models = df_results[model_col].unique()
    n_models   = len(all_models)

    # --- Step 1: Tag each (model, pair) as fragile or robust ---
    df = df_results.copy()
    df["is_fragile"] = (df[dfs_col] >= dfs_threshold).astype(int)#main Goes through every row (every model × sentence pair combination)

    # --- Step 2: Pivot → rows = pairs, columns = models ---       reshapes the data
    pivot = df.pivot_table(
        index=pair_id_cols,
        columns=model_col,
        values=["is_fragile", dfs_col],
        aggfunc="mean"    # handles any duplicate rows gracefully
    )
    pivot.columns = ["_".join(c).strip() for c in pivot.columns.values]
    pivot = pivot.reset_index()

    # --- Step 3: Compute consensus metrics ---
    fragile_cols = [c for c in pivot.columns if c.startswith("is_fragile_")]
    dfs_cols     = [c for c in pivot.columns if c.startswith(f"{dfs_col}_")]

    pivot["fragile_model_count"] = pivot[fragile_cols].sum(axis=1)
    pivot["consensus_ratio"]     = pivot["fragile_model_count"] / n_models
    pivot["mean_dfs"]            = pivot[dfs_cols].mean(axis=1)
    pivot["std_dfs"]             = pivot[dfs_cols].std(axis=1)    # cross-model disagreement
    pivot["max_dfs"]             = pivot[dfs_cols].max(axis=1)

    # --- Step 4: Fragility Taxonomy ---
    def classify(row):
        r = row["consensus_ratio"]
        if r >= UNIVERSAL_CONSENSUS:
            return "UNIVERSAL"
        elif r >= PARTIAL_CONSENSUS_MIN:
            return "PARTIAL"
        elif r > 0:
            return "ISOLATED"
        else:
            return "ROBUST"

    pivot["fragility_class"] = pivot.apply(classify, axis=1)

    return pivot


def visualize_cfcs(cfcs_df: pd.DataFrame):
    """
    Renders three panels:
        1. Fragility class distribution (pie/bar)
        2. Mean DFS vs. Consensus Ratio scatter (the "consensus landscape")
        3. Cross-model DFS std — measures how much models *disagree*
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Cross-Model Fragility Consensus Scoring (CFCS)",
                 fontsize=18, fontweight="bold", y=1.02)

    # --- Panel 1: Fragility Class Distribution ---
    class_counts = cfcs_df["fragility_class"].value_counts()
    color_map = {
        "UNIVERSAL": "#d62728",
        "PARTIAL":   "#ff7f0e",
        "ISOLATED":  "#1f77b4",
        "ROBUST":    "#2ca02c"
    }
    colors = [color_map.get(c, "grey") for c in class_counts.index]
    axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Fragility Taxonomy Distribution")
    axes[0].set_xlabel("Fragility Class")
    axes[0].set_ylabel("Number of Sentence Pairs")
    for i, (cls, val) in enumerate(class_counts.items()):
        axes[0].text(i, val + 0.3, str(val), ha="center", fontsize=12, fontweight="bold")

    # --- Panel 2: Consensus Landscape (Mean DFS vs Consensus Ratio) ---
    scatter_colors = cfcs_df["fragility_class"].map(color_map).fillna("grey")
    axes[1].scatter(
        cfcs_df["consensus_ratio"],
        cfcs_df["mean_dfs"],
        c=scatter_colors,
        alpha=0.7,
        s=80,
        edgecolors="white",
        linewidths=0.5
    )
    axes[1].axvline(x=PARTIAL_CONSENSUS_MIN, color="orange", linestyle="--", linewidth=1.5, label=f"Partial threshold ({PARTIAL_CONSENSUS_MIN})")
    axes[1].axvline(x=UNIVERSAL_CONSENSUS,   color="red",    linestyle="--", linewidth=1.5, label=f"Universal threshold ({UNIVERSAL_CONSENSUS})")
    axes[1].axhline(y=DFS_FRAGILE_THRESHOLD, color="grey",   linestyle=":",  linewidth=1.2, label=f"DFS threshold ({DFS_FRAGILE_THRESHOLD})")
    axes[1].set_title("Consensus Landscape")
    axes[1].set_xlabel("Consensus Ratio (fraction of models that break)")
    axes[1].set_ylabel("Mean DFS across models")
    axes[1].legend(fontsize=9)

    # Add Pearson correlation annotation
    valid = cfcs_df[["consensus_ratio", "mean_dfs"]].dropna()
    if len(valid) > 2:
        r, p = pearsonr(valid["consensus_ratio"], valid["mean_dfs"])
        axes[1].text(0.05, 0.92, f"r = {r:.3f}, p = {p:.3e}",
                     transform=axes[1].transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Panel 3: Cross-Model DFS Standard Deviation ---
    axes[2].hist(
        cfcs_df["std_dfs"].dropna(),
        bins=20,
        color="#9467bd",
        edgecolor="black",
        linewidth=0.7
    )
    axes[2].set_title("Cross-Model DFS Disagreement (σ)")
    axes[2].set_xlabel("Std Dev of DFS across models")
    axes[2].set_ylabel("Count")
    axes[2].axvline(cfcs_df["std_dfs"].median(), color="red", linestyle="--",
                    linewidth=2, label=f"Median σ = {cfcs_df['std_dfs'].median():.4f}")
    axes[2].legend()

    # Legend patch for fragility classes
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    fig.legend(handles=patches, title="Fragility Class", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=True)

    plt.tight_layout()
    plt.show()


def print_cfcs_summary(cfcs_df: pd.DataFrame):
    """
    Prints a structured report of the CFCS results, highlighting
    universally fragile pairs for presentation.
    """
    print("\n" + "=" * 70)
    print("         CROSS-MODEL FRAGILITY CONSENSUS REPORT")
    print("=" * 70)

    total = len(cfcs_df)
    for cls in ["UNIVERSAL", "PARTIAL", "ISOLATED", "ROBUST"]:
        subset = cfcs_df[cfcs_df["fragility_class"] == cls]
        pct = 100 * len(subset) / total if total > 0 else 0
        print(f"  {cls:<12}: {len(subset):>4} pairs  ({pct:.1f}%)")

    print("-" * 70)
    print(f"  Mean cross-model DFS  : {cfcs_df['mean_dfs'].mean():.4f}")
    print(f"  Mean consensus ratio  : {cfcs_df['consensus_ratio'].mean():.4f}")
    print(f"  Mean DFS std (σ)      : {cfcs_df['std_dfs'].mean():.4f}  ← lower = models agree more")
    print("=" * 70)

    # Top 3 universally fragile pairs
    universal = cfcs_df[cfcs_df["fragility_class"] == "UNIVERSAL"].nlargest(3, "mean_dfs")
    if len(universal) > 0:
        print("\n🔴 TOP UNIVERSALLY FRAGILE PAIRS (all models fail):")
        for _, row in universal.iterrows():
            print(f"\n  Sentence A : {row['sentence_a']}")
            print(f"  Sentence B : {row['sentence_b']}")
            print(f"  Mean DFS   : {row['mean_dfs']:.4f}  |  Max DFS: {row['max_dfs']:.4f}")
            print(f"  σ (disagreement): {row['std_dfs']:.4f}")
            print("  " + "-" * 60)

    # Top 3 model-isolated quirks
    isolated = cfcs_df[cfcs_df["fragility_class"] == "ISOLATED"].nlargest(3, "max_dfs")
    if len(isolated) > 0:
        print("\n🔵 TOP ISOLATED FRAGILITY PAIRS (only 1 model fails — architectural quirk):")
        for _, row in isolated.iterrows():
            print(f"\n  Sentence A : {row['sentence_a']}")
            print(f"  Sentence B : {row['sentence_b']}")
            print(f"  Max DFS    : {row['max_dfs']:.4f}  |  σ: {row['std_dfs']:.4f}")
            print("  " + "-" * 60)


# ==========================================
# USAGE — Run after ExperimentRunner produces df_results
# ==========================================
# Assumes df_results has columns: sentence_a, sentence_b, model, DFS

# Step 1: Compute CFCS
cfcs_df = compute_cfcs(df_results)

# Step 2: Print structured report
print_cfcs_summary(cfcs_df)

# Step 3: Visualize consensus landscape
visualize_cfcs(cfcs_df)

# Step 4: Optional — export for paper/report
cfcs_df.to_csv(env.cfg.output_dir / "cfcs_results.csv", index=False)
env.logger.info(f"✅ CFCS results saved to {env.cfg.output_dir / 'cfcs_results.csv'}")
