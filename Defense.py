def run_custom_diagnostic(model_wrapper: ModelInferenceWrapper, pair_a: str, pair_b: str):
    """
    Runs a 'PhD-Level' deep dive on a single custom pair.
    Generates a micro-report on stability, confidence shifts, and fragility.
    Includes built-in validation of custom prompts.
    """
    print(f"\n🔬 DIAGNOSTIC REPORT FOR MODEL: {model_wrapper.model_name}")
    print("=" * 60)

    # ⚡ DEFENSE MECHANISM: Validate custom inputs first
    print("🛡️ Validating input prompts...")
    is_vague_a, reason_a = PromptValidator.is_vague(pair_a)
    is_vague_b, reason_b = PromptValidator.is_vague(pair_b)

    validation_failed = is_vague_a or is_vague_b

    if is_vague_a:
        print(f"⚠️ Input A is vague: {reason_a}")
        print(f"   Input: '{pair_a}'")

    if is_vague_b:
        print(f"⚠️ Input B is vague: {reason_b}")
        print(f"   Input: '{pair_b}'")

    if validation_failed:
        if model_wrapper.env.cfg.raise_on_vague_input:
            raise ValueError(
                f"Vague prompts detected. Input A: {reason_a}, Input B: {reason_b}"
            )
        else:
            print("⚠️ Proceeding with vague inputs (validation warnings above).\n")
    else:
        print("✅ Both inputs passed validation. Proceeding.\n")

    # 1. Run Inference (Using the same engine as the main experiment)
    pred_a, conf_a = model_wrapper.predict_batch([pair_a])
    pred_b, conf_b = model_wrapper.predict_batch([pair_b])

    # Unpack lists (since batch size is 1)
    label_a, prob_a = pred_a[0], conf_a[0]
    label_b, prob_b = pred_b[0], conf_b[0]

    # 2. Calculate Metrics
    dfs = abs(prob_a - prob_b)
    is_flipped = (label_a != label_b)

    # 3. Generate The "Journal" Output
    print(f"📝 INPUT A: \"{pair_a}\"")
    print(f"   └── Prediction: Class {label_a} | Confidence: {prob_a:.4f}")

    print(f"📝 INPUT B: \"{pair_b}\"")
    print(f"   └── Prediction: Class {label_b} | Confidence: {prob_b:.4f}")

    print("-" * 60)

    # 4. The "PhD Analysis"
    print(f"📊 ANALYTICS:")
    print(f"   ➤ Directional Fragility Score (DFS): {dfs:.4f}")

    if is_flipped:
        print("   ➤ 🚨 CRITICAL FAILURE: LABEL FLIP DETECTED")
        print("      The model completely changed its decision based on phrasing.")
    elif dfs > 0.10:
        print("   ➤ ⚠️ HIGH INSTABILITY")
        print("      The prediction held, but confidence dropped significantly.")
    else:
        print("   ➤ ✅ ROBUST")
        print("      The model treated both sentences identically.")
    print("=" * 60)

# ==========================================
# 🎯 HOW TO USE IT (Example)
# ==========================================

# 1. Load the model you want to probe (if not already loaded)
# (Assuming 'env' is already defined from Module 1)
inference_engine = ModelInferenceWrapper(env, config.target_models[0])

# 2. Define your "Trap" sentences
custom_A = "The generated image was surprisingly realistic and vivid."
custom_B = "Surprisingly realistic and vivid was the generated image."

# 3. Run the Probe
run_custom_diagnostic(inference_engine, custom_A, custom_B)
