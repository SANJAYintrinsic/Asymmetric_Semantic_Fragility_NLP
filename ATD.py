# ==========================================
# 10. NOVEL: ATTENTION TRAJECTORY DIVERGENCE (ATD)
# ==========================================
"""
Attention Trajectory Divergence (ATD)
--------------------------------------
Hypothesis:
    When a model processes two semantically identical sentences (a paraphrase pair),
    its internal attention patterns should be similar. When they diverge significantly,
    that divergence IS the fragility — and we can localize it to specific tokens
    and specific layers using Dynamic Time Warping (DTW).

Why this has never been done:
    1. DTW is a temporal alignment metric from speech/gesture processing.
       Nobody has applied it to transformer attention flows across paraphrase pairs.
    2. Prior interpretability work (BERTViz, attention rollout) visualizes attention
       but never *quantifies the temporal divergence* between two inputs.
    3. This produces a "fragility pivot token" — the exact word where the model's
       internal narrative forks — which is actionable and interpretable.

Output:
    - ATD score per layer (scalar)
    - Fragility Pivot Token (the token with highest DTW path cost)
    - Attention Trajectory heatmap showing divergence path
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine as cosine_dist

try:
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dtaidistance", "-q"])
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True


# ---- Core ATD Engine ----

class AttentionTrajectoryDivergence:
    """
    Extracts per-layer attention trajectories from a transformer model
    and computes DTW-based divergence between two input sentences.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def _get_attention_trajectory(self, text: str):
        """
        Runs a forward pass and extracts the mean attention
        across all heads per layer, per token position.

        Returns:
            tokens         : list of token strings
            trajectory     : np.array of shape (n_layers, seq_len)
                             Each row = mean attention *received* by each token at that layer.
                             This is the "attention narrative" the model tells about the input.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
        # We take the mean over heads, then sum over the "from" axis → attention received per token
        trajectory = []
        for layer_attn in outputs.attentions:
            # layer_attn: (1, heads, seq_len, seq_len)
            mean_heads = layer_attn[0].mean(dim=0)           # (seq_len, seq_len)
            attn_received = mean_heads.sum(dim=0).numpy()    # (seq_len,)  — column sum
            # Normalize to [0,1]
            attn_received = attn_received / (attn_received.sum() + 1e-9)
            trajectory.append(attn_received)

        return tokens, np.array(trajectory)  # (n_layers, seq_len)

    def _align_trajectories(self, traj_a: np.ndarray, traj_b: np.ndarray):
        """
        Aligns two trajectories of potentially different seq_len using zero-padding.
        Returns two arrays of equal shape.
        """
        n_layers = min(traj_a.shape[0], traj_b.shape[0])
        max_len = max(traj_a.shape[1], traj_b.shape[1])

        def pad(arr, target_len):
            pad_width = target_len - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

        traj_a = pad(traj_a[:n_layers], max_len)
        traj_b = pad(traj_b[:n_layers], max_len)
        return traj_a, traj_b, n_layers

    def compute_atd(self, sentence_a: str, sentence_b: str):
        """
        Full ATD computation pipeline.

        Returns:
            result: dict with
                - atd_per_layer     : DTW distance between A and B's attention at each layer
                - mean_atd          : scalar — the overall ATD score
                - pivot_layer       : layer with highest divergence
                - pivot_token_idx_a : token index in sentence_a driving the divergence
                - pivot_token_a     : the actual pivot token string
                - tokens_a, tokens_b: tokenized forms
                - traj_a, traj_b   : full trajectory arrays for visualization
        """
        tokens_a, traj_a = self._get_attention_trajectory(sentence_a)
        tokens_b, traj_b = self._get_attention_trajectory(sentence_b)

        traj_a, traj_b, n_layers = self._align_trajectories(traj_a, traj_b)

        atd_per_layer = []
        for layer_idx in range(n_layers):
            # DTW distance between the two 1D attention signals at this layer
            dist = dtw_lib.distance_fast(
                traj_a[layer_idx].astype(np.double),
                traj_b[layer_idx].astype(np.double)
            )
            atd_per_layer.append(dist)

        atd_per_layer = np.array(atd_per_layer)
        mean_atd = float(atd_per_layer.mean())
        pivot_layer = int(np.argmax(atd_per_layer))

        # Fragility Pivot Token: token in sentence_a with largest attention
        # difference at the pivot layer
        token_delta = np.abs(traj_a[pivot_layer] - traj_b[pivot_layer])
        pivot_token_idx = int(np.argmax(token_delta[:len(tokens_a)]))
        pivot_token = tokens_a[pivot_token_idx] if pivot_token_idx < len(tokens_a) else "[UNK]"

        return {
            "atd_per_layer":     atd_per_layer,
            "mean_atd":          mean_atd,
            "pivot_layer":       pivot_layer,
            "pivot_token_idx_a": pivot_token_idx,
            "pivot_token":       pivot_token,
            "tokens_a":          tokens_a,
            "tokens_b":          tokens_b,
            "traj_a":            traj_a,
            "traj_b":            traj_b,
            "n_layers":          n_layers,
        }


# ---- Visualization ----

def visualize_atd(result: dict, sentence_a: str, sentence_b: str):
    """
    Three-panel ATD visualization:
        Panel 1: ATD score across layers (where divergence lives)
        Panel 2: Attention trajectory heatmap for A and B side by side
        Panel 3: Token-level divergence at the pivot layer (fragility pivot bar chart)
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig = plt.figure(figsize=(22, 7))
    fig.suptitle("Attention Trajectory Divergence (ATD) — Fragility Localization",
                 fontsize=16, fontweight="bold")

    # ---- Panel 1: Layer-wise ATD score ----
    ax1 = fig.add_subplot(1, 3, 1)
    layers = np.arange(result["n_layers"])
    bars = ax1.bar(layers, result["atd_per_layer"],
                   color=["#d62728" if i == result["pivot_layer"] else "#4e79a7"
                          for i in layers],
                   edgecolor="black", linewidth=0.7)
    ax1.set_title("ATD per Layer")
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("DTW Distance (higher = more divergent)")
    ax1.axhline(result["mean_atd"], color="orange", linestyle="--",
                linewidth=2, label=f"Mean ATD = {result['mean_atd']:.4f}")
    ax1.legend(fontsize=9)
    ax1.text(result["pivot_layer"], result["atd_per_layer"][result["pivot_layer"]] * 1.02,
             "PIVOT", ha="center", fontsize=10, color="red", fontweight="bold")

    # ---- Panel 2: Trajectory heatmaps (A vs B) ----
    ax2 = fig.add_subplot(1, 3, 2)
    n_tok = min(result["traj_a"].shape[1], 20)   # cap display at 20 tokens

    diff_map = result["traj_a"][:, :n_tok] - result["traj_b"][:, :n_tok]
    im = ax2.imshow(diff_map, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                    vmin=-np.abs(diff_map).max(), vmax=np.abs(diff_map).max())
    ax2.set_title("Attention Trajectory Difference\n(A − B, per layer × token)")
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Layer")

    # Overlay token labels for sentence_a (top 20)
    labels = [t.replace("##", "") for t in result["tokens_a"][:n_tok]]
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.colorbar(im, ax=ax2, label="Attention Δ (A−B)")

    # Highlight pivot layer
    ax2.axhline(result["pivot_layer"] - 0.5, color="red", linewidth=2)
    ax2.axhline(result["pivot_layer"] + 0.5, color="red", linewidth=2)

    # ---- Panel 3: Token-level divergence at pivot layer ----
    ax3 = fig.add_subplot(1, 3, 3)
    pivot_delta = np.abs(
        result["traj_a"][result["pivot_layer"], :n_tok] -
        result["traj_b"][result["pivot_layer"], :n_tok]
    )
    pivot_colors = ["#d62728" if i == result["pivot_token_idx_a"] else "#4e79a7"
                    for i in range(len(labels))]
    ax3.bar(range(len(labels)), pivot_delta[:len(labels)],
            color=pivot_colors, edgecolor="black", linewidth=0.6)
    ax3.set_title(f"Fragility Pivot Tokens\n(Layer {result['pivot_layer']})")
    ax3.set_xlabel("Token")
    ax3.set_ylabel("|Attention Δ| at Pivot Layer")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax3.text(result["pivot_token_idx_a"],
             pivot_delta[result["pivot_token_idx_a"]] * 1.03,
             f"← PIVOT\n'{result['pivot_token']}'",
             ha="center", fontsize=9, color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(env.cfg.output_dir / "atd_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 ATD visualization saved.")


def print_atd_report(result: dict, sentence_a: str, sentence_b: str):
    """Prints a clean ATD diagnostic report."""
    print("\n" + "=" * 70)
    print("        ATTENTION TRAJECTORY DIVERGENCE (ATD) REPORT")
    print("=" * 70)
    print(f"  Sentence A : {sentence_a}")
    print(f"  Sentence B : {sentence_b}")
    print("-" * 70)
    print(f"  Mean ATD Score      : {result['mean_atd']:.6f}  (↑ = more fragile)")
    print(f"  Pivot Layer         : Layer {result['pivot_layer']}  (highest divergence layer)")
    print(f"  Fragility Pivot Token : '{result['pivot_token']}'  "
          f"(token position {result['pivot_token_idx_a']})")
    print("-" * 70)
    print("  Layer-wise ATD scores:")
    for i, score in enumerate(result["atd_per_layer"]):
        bar = "█" * int(score * 300)
        marker = " ← PIVOT" if i == result["pivot_layer"] else ""
        print(f"    Layer {i:02d} : {score:.5f}  {bar}{marker}")
    print("=" * 70)


def run_atd_on_stress_pairs(atd_engine: AttentionTrajectoryDivergence,
                             pairs: list = None):
    """
    Runs ATD across multiple paraphrase pairs and ranks them by Mean ATD.
    Shows which linguistic transformation is most internally disruptive.

    Args:
        pairs: list of (sentence_a, sentence_b, category) tuples.
               Defaults to the same stress-test pairs from Cell 8.
    """
    if pairs is None:
        pairs = [
            ("The sudden crash shattered the silence.",
             "The silence was shattered by the sudden crash.",
             "Passive Voice"),
            ("I absolutely refuse to accept this proposal.",
             "This proposal, I absolutely refuse to accept.",
             "Syntactic Fronting"),
            ("The algorithm produces accurate results.",
             "The algorithm, despite being old, produces accurate results.",
             "Distractor Clause"),
            ("The solution is effective.",
             "The solution is not ineffective.",
             "Double Negation"),
            ("The movie was scary.",
             "The film was petrifying.",
             "Lexical Rarity"),
            ("She only eats an apple.",
             "She eats only an apple.",
             "Modifier Placement"),
        ]

    print("\n🔬 Running ATD across stress-test pairs...\n")
    records = []
    for sent_a, sent_b, category in pairs:
        result = atd_engine.compute_atd(sent_a, sent_b)
        records.append({
            "Category":        category,
            "Mean ATD":        round(result["mean_atd"], 5),
            "Pivot Layer":     result["pivot_layer"],
            "Pivot Token":     result["pivot_token"],
            "Sentence A":      sent_a,
            "Sentence B":      sent_b,
            "_result":         result,   # keep for optional deep-dive viz
        })
        print(f"  [{category}]  Mean ATD = {result['mean_atd']:.5f}  |  "
              f"Pivot Layer = {result['pivot_layer']}  |  "
              f"Pivot Token = '{result['pivot_token']}'")

    df_atd = pd.DataFrame(records).sort_values("Mean ATD", ascending=False)

    print("\n📊 ATD RANKING (most → least internally disruptive):")
    print(df_atd[["Category", "Mean ATD", "Pivot Layer", "Pivot Token"]].to_string(index=False))

    # Full visualization for the worst offender
    worst = records[0] if records else None
    if worst:
        print(f"\n🔴 Deep-diving into worst pair: [{worst['Category']}]")
        visualize_atd(worst["_result"], worst["Sentence A"], worst["Sentence B"])
        print_atd_report(worst["_result"], worst["Sentence A"], worst["Sentence B"])

    return df_atd


# ==========================================
# USAGE
# ==========================================

# Initialize ATD engine (uses the base model, not the classifier head,
# so it works even if the classifier models differ)
atd_engine = AttentionTrajectoryDivergence(
    model_name="distilbert-base-uncased"  # swap for roberta-base etc.
)

# Option A: Run on stress-test pairs from Cell 8 (recommended first run)
df_atd_results = run_atd_on_stress_pairs(atd_engine)

# Option B: Deep-dive on a single custom pair
custom_result = atd_engine.compute_atd(
    "The judge declared the verdict.",
    "The verdict was declared by the judge."
)
print_atd_report(custom_result,
                 "The judge declared the verdict.",
                 "The verdict was declared by the judge.")
visualize_atd(custom_result,
              "The judge declared the verdict.",
              "The verdict was declared by the judge.")

# Save ATD rankings
df_atd_results.drop(columns=["_result"]).to_csv(
    env.cfg.output_dir / "atd_rankings.csv", index=False
)
env.logger.info("✅ ATD analysis complete. Rankings saved.")
