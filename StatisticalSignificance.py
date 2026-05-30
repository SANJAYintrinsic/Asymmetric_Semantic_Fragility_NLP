from scipy import stats

def calculate_statistical_significance(df_results):
    """
    Performs a Paired T-Test to prove the asymmetry is real.
    Null Hypothesis (H0): The model treats A and B exactly the same.
    """
    print("\n--- 📉 STATISTICAL SIGNIFICANCE TEST ---")

    # We compare the confidence scores of A vs. B
    conf_a = df_results["conf_a"]
    conf_b = df_results["conf_b"]

    # 1. Paired T-Test
    # We use this because the two samples (A and B) are dependent (same meaning).
    t_stat, p_val = stats.ttest_rel(conf_a, conf_b)

    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_val:.4e}") # Scientific notation usually impresses

    # Interpretation for the Panel
    alpha = 0.05
    if p_val < alpha:
        print("✅ RESULT: Statistically Significant (Reject H0).")
        print("   The model definitively distinguishes between the two sentence forms.")
    else:
        print("⚠️ RESULT: Not Significant (Fail to Reject H0).")
        print("   The differences might be due to chance.")

# Run it on your results
calculate_statistical_significance(df_results)
