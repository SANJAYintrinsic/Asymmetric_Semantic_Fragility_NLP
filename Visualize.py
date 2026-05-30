import matplotlib.pyplot as plt
import seaborn as sns

def visualize_fragility_distribution(df):
    """
    Plots the Kernel Density Estimation (KDE) of the Fragility Score.
    This shows the 'Shape' of the failure.
    """
    plt.figure(figsize=(10, 6))

    # Set professional style (use set_theme for seaborn >= 0.13 compatibility)
    sns.set_theme(style="whitegrid", context="talk")  # Replaces deprecated set_style/set_context

    # Plot histogram + density
    sns.histplot(
        data=df,
        x="DFS",
        hue="model",
        element="step",
        stat="density",
        common_norm=False,
        palette="viridis",
        alpha=0.5
    )

    sns.kdeplot(data=df, x="DFS", hue="model", common_norm=False, palette="viridis", linewidth=3)

    plt.title("Distribution of Semantic Fragility (DFS)", fontsize=16, fontweight='bold')
    plt.xlabel("Fragility Score (0 = Robust, 1 = Broken)", fontsize=12)
    plt.ylabel("Density of Samples", fontsize=12)
    plt.xlim(0, 0.5) # Focus on the interesting part

    plt.show()

# Run it
visualize_fragility_distribution(df_results)
