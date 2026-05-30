from IPython.display import display
# ==========================================
# ⚡ AUTOMATED STRESS TEST SUITE
# ==========================================

def run_stress_test_suite(model_wrapper):
    print(f"💣 RUNNING STRESS TEST ON: {model_wrapper.model_name}")
    print("=" * 80)

    # The "Trap" Dataset
    traps = [
        # 1. Passive Voice
        ("The sudden crash shattered the silence.", "The silence was shattered by the sudden crash.", "Passive Voice"),
        # 2. Fronting
        ("I absolutely refuse to accept this proposal.", "This proposal, I absolutely refuse to accept.", "Syntactic Fronting"),
        # 3. Distractor Clause
        ("The algorithm produces accurate results.", "The algorithm, despite being old, produces accurate results.", "Distractor Clause"),
        # 4. Double Negation
        ("The solution is effective.", "The solution is not ineffective.", "Double Negation"),
        # 5. Lexical Rarity
        ("The movie was scary.", "The film was petrifying.", "Lexical Rarity"),
        # 6. Modifier Placement
        ("She only eats an apple.", "She eats only an apple.", "Modifier Placement")
    ]

    results = []

    for sent_a, sent_b, category in traps:
        # Run Inference
        pred_a, conf_a = model_wrapper.predict_batch([sent_a])
        pred_b, conf_b = model_wrapper.predict_batch([sent_b])

        # Calculate DFS
        dfs = abs(conf_a[0] - conf_b[0])
        flipped = pred_a[0] != pred_b[0]

        results.append({
            "Category": category,
            "DFS": dfs,
            "Flipped": "YES" if flipped else "No",
            "Conf_A": conf_a[0],
            "Conf_B": conf_b[0],
            "Sentence Pair": f"A: {sent_a} \nB: {sent_b}"
        })

    # Convert to DataFrame for ranking
    df_stress = pd.DataFrame(results)

    # Sort by DFS (Highest Fragility First)
    df_sorted = df_stress.sort_values(by="DFS", ascending=False)

    # Display Professional Table
    display(df_sorted[["Category", "DFS", "Flipped", "Conf_A", "Conf_B"]])

    # Return the winner for the presentation
    winner = df_sorted.iloc[0]
    print("\n🏆 THE WINNER (Highest Fragility):")
    print(f"Category: {winner['Category']}")
    print(f"DFS Score: {winner['DFS']:.4f}")
    print(f"Sentences:\n{winner['Sentence Pair']}")

# --- EXECUTE ---
# Assuming 'inference_engine' is already loaded from previous steps
run_stress_test_suite(inference_engine)
