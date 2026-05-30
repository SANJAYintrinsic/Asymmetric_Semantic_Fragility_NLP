def print_top_failures(df, n=3):
    """
    Formats the top failures for a slide deck copy-paste.
    """
    sorted_df = df.sort_values(by="DFS", ascending=False).head(n)

    print("\n--- 🚨 CRITICAL FAILURES FOR PRESENTATION ---")
    for idx, row in sorted_df.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"🔴 Sentence A (Conf: {row['conf_a']:.4f}): \"{row['sentence_a']}\"")
        print(f"🔵 Sentence B (Conf: {row['conf_b']:.4f}): \"{row['sentence_b']}\"")
        print(f"🔥 Fragility Score: {row['DFS']:.4f}")
        print("-" * 60)

print_top_failures(df_results)
