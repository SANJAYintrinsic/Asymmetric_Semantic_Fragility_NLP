from __future__ import annotations
import os
import sys
import random
import logging
import torch
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re

# suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# ==========================================
# 0. PROMPT VALIDATION & DEFENSE MECHANISM
# ==========================================
class PromptValidator:
    """
    Defense mechanism to filter out vague, empty, or low-quality prompts
    before they reach the model. Ensures robust input quality.
    """

    # Vague indicators - common meaningless phrases
    VAGUE_INDICATORS = {
        "stuff", "things", "something", "whatever", "anything",
        "ok", "fine", "good", "bad", "etc", "and so on",
        "blah", "yadda", "la la", "thingy", "whatnot"
    }

    # Minimum quality thresholds
    MIN_LENGTH = 5  # Minimum characters
    MIN_WORDS = 2   # Minimum number of words

    @staticmethod
    def is_vague(text: str) -> Tuple[bool, str]:
        """
        Evaluates if a prompt is too vague for meaningful model inference.
        Returns: (is_vague: bool, reason: str)
        """
        if not isinstance(text, str):
            return True, "Input must be a string"

        # Check 1: Empty or Whitespace Only
        if not text or not text.strip():
            return True, "Empty or whitespace-only input"

        # Check 2: Too Short
        if len(text.strip()) < PromptValidator.MIN_LENGTH:
            return True, f"Text too short (< {PromptValidator.MIN_LENGTH} characters)"

        # Check 3: Insufficient Word Count
        words = text.split()
        if len(words) < PromptValidator.MIN_WORDS:
            return True, f"Not enough words (< {PromptValidator.MIN_WORDS} words)"

        # Check 4: Mostly Vague Indicators
        words_lower = [w.lower().strip('.,!?;:') for w in words]
        vague_count = sum(1 for w in words_lower if w in PromptValidator.VAGUE_INDICATORS)
        vague_ratio = vague_count / len(words) if words else 0

        if vague_ratio > 0.4:  # If >40% of words are vague
            return True, f"High ratio of vague words ({vague_ratio:.1%})"

        # Check 5: Only Symbols/Numbers with No Text
        text_chars = re.sub(r'[^a-zA-Z\s]', '', text)
        if len(text_chars.strip()) < 3:
            return True, "Mostly symbols/numbers; insufficient meaningful text"

        # Check 6: Excessive Repetition (e.g., "aaaa" or "test test test")
        words_set = set(words_lower)
        if len(words) > 0 and len(words_set) / len(words) < 0.3:  # <30% unique words
            return True, "Excessive repetition detected"

        return False, "OK"

    @staticmethod
    def validate_batch(texts: List[str], raise_on_error: bool = False) -> dict:
        """
        Validates a batch of texts and returns detailed report.

        Args:
            texts: List of strings to validate
            raise_on_error: If True, raises ValueError on first vague input

        Returns:
            dict with validation results and statistics
        """
        results = {
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": len(texts),
            "failed_indices": [],
            "failed_reasons": [],
            "passing_rate": 0.0
        }

        for idx, text in enumerate(texts):
            is_vague, reason = PromptValidator.is_vague(text)

            if is_vague:
                results["invalid_count"] += 1
                results["failed_indices"].append(idx)
                results["failed_reasons"].append(reason)

                if raise_on_error:
                    raise ValueError(
                        f"Vague prompt detected at index {idx}: {reason}\n"
                        f"Input: '{text}'"
                    )
            else:
                results["valid_count"] += 1

        results["passing_rate"] = results["valid_count"] / len(texts) if texts else 0
        return results

# ==========================================
# 1. PROFESSIONAL CONFIGURATION INTERFACE
# ==========================================
@dataclass(frozen=True)
class ExperimentConfig:
    """
    Immutable configuration object for the ASF (Asymmetric Semantic Fragility) experiment.
    Using a dataclass ensures type safety and prevents accidental parameter overwrites.
    """
    # Meta Information
    experiment_name: str = "ASF_Phase1_Pilot"
    author: str = "Research_Candidate_01"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Computational Constraints
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use fp16 for efficiency on T4/A100

    # Model Parameters
    # We list multiple models to demonstrate comparative rigor immediately.
    target_models: List[str] = field(default_factory=lambda: [
        "distilbert-base-uncased-finetuned-sst-2-english", # Baseline (Fast)
        "openai-community/roberta-base-openai-detector",   # Robustness Check
        # "google/flan-t5-base"                            # Generative (Phase 2)
    ])

    # Dataset & Task Parameters
    task_type: str = "text-classification" # Can be switched to 'qa' or 'generation'
    batch_size: int = 32
    sample_limit: Optional[int] = None # Set to integer for quick debugging (e.g., 50)

    # Prompt Validation
    enforce_prompt_validation: bool = True  # Enable/disable vagueness filtering
    raise_on_vague_input: bool = False      # Raise exception or log warning

    # I/O Paths (Using Pathlib for OS independence)
    base_dir: Path = Path("./neuro_project_asf")
    log_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        # Auto-generate directory paths based on timestamp to avoid overwrites
        # This is a critical feature for experiment tracking.
        object.__setattr__(self, 'log_dir', self.base_dir / "logs" / self.timestamp)
        object.__setattr__(self, 'output_dir', self.base_dir / "results" / self.timestamp)

# ==========================================
# 2. THE ENVIRONMENT MANAGER (The "Engine Room")
# ==========================================
class ResearchEnvironment:
    """
    Singleton-style manager to handle hardware, logging, and reproducibility.
    """
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self._setup_directories()
        self.logger = self._setup_logging()
        self._set_seeds()
        self._log_hardware_specs()

    def _setup_directories(self):
        """Creates the directory tree. Fails loudly if permissions are denied."""
        try:
            self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Directories initialized at: {self.cfg.base_dir}")
        except OSError as e:
            sys.exit(f"CRITICAL ERROR: Could not create directories. {e}")

    def _setup_logging(self) -> logging.Logger:
        """Sets up a dual-stream logger (File + Console)."""
        logger = logging.getLogger(self.cfg.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers = [] # Clear existing handlers

        # Formatter: ISO8601 Time - Level - Message
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # Stream Handler (Console)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (Disk)
        log_file = self.cfg.log_dir / "experiment.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info("logger initialized. Audit trail started.")
        return logger

    def _set_seeds(self):
        """
        Enforces deterministic behavior across Python, NumPy, and Torch.
        Essential for publication reproducibility.
        """
        seed = self.cfg.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.logger.info(f"🔒 Global Random Seed set to: {seed}")

    def _log_hardware_specs(self):
        """Introspects the hardware and logs specific GPU details."""
        self.logger.info(f"⚙️ Computation Device: {self.cfg.device.upper()}")

        if self.cfg.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"   └── GPU Model: {gpu_name}")
            self.logger.info(f"   └── VRAM Available: {vram_total:.2f} GB")
        else:
            self.logger.warning("   └── ⚠️ Running on CPU. Performance will be degraded.")

# ==========================================
# 3. INITIALIZATION (The "Main" Entry)
# ==========================================
# Instantiate Configuration
config = ExperimentConfig(
    experiment_name="ASF_Review_Panel_Run",
    sample_limit=100,  # Remove this for full run
    enforce_prompt_validation=True
)

# Boot the Environment
env = ResearchEnvironment(config)

# Access the logger anywhere via env.logger
env.logger.info("✅ Module 1 (Configuration) loaded successfully.")
env.logger.info(f"🎯 Target Models: {config.target_models}")
env.logger.info(f"🛡️ Prompt Validation: {'ENABLED' if config.enforce_prompt_validation else 'DISABLED'}")

# ----------------------------------------------------------------------------------------------------------------------------------------------

from datasets import load_dataset, Dataset
import pandas as pd
from typing import Dict, Union, Literal
from IPython.display import display

# ==========================================
# 4. DATA INGESTION ENGINE (The "Supply Chain")
# ==========================================
class DataIngestionEngine:
    """
    Responsible for fetching, validating, and standardizing datasets.
    Supports both Academic Benchmarks (PAWS/MRPC) and Custom Synthetic Data.
    """
    def __init__(self, env: ResearchEnvironment):
        self.env = env
        self.raw_data = None
        self.processed_data = None

    def load_benchmark(self, dataset_name: str = "paws", subset: str = "labeled_final"):
        """
        Loads a standard academic benchmark from Hugging Face Hub.
        Recommended: 'paws' (Paraphrase Adversaries from Word Scrambling).
        """
        self.env.logger.info(f"📥 Downloading Benchmark: {dataset_name} ({subset})...")
        try:
            # Load from Hugging Face Hub
            hf_dataset = load_dataset(dataset_name, subset, split="test") # Using test set for evaluation

            # Convert to Pandas for easier manipulation during analysis
            self.raw_data = hf_dataset.to_pandas()

            self.env.logger.info(f"✅ Successfully loaded {len(self.raw_data)} rows from {dataset_name}.")
            self._standardize_schema(source_type=dataset_name)

        except Exception as e:
            self.env.logger.error(f"❌ Failed to load benchmark: {e}")
            raise e

    def load_custom_synthetic(self, file_path: Union[str, Path]):
        """
        Loads a local CSV file (e.g., generated by GPT-4 in Phase 2).
        """
        self.env.logger.info(f"📂 Loading Local Data: {file_path}...")
        try:
            self.raw_data = pd.read_csv(file_path)
            self._standardize_schema(source_type="custom")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to load local file: {e}")
            raise e

    def _standardize_schema(self, source_type: str):
        """
        Internal method to map various dataset column names to our Strict Schema.
        Target Schema: ['sentence_a', 'sentence_b', 'label', 'is_paraphrase']
        """
        self.env.logger.info("🔧 Standardizing Schema...")
        df = self.raw_data.copy()

        # Mapping logic based on source
        if source_type == "paws":
            # PAWS columns: 'sentence1', 'sentence2', 'label'
            df = df.rename(columns={
                "sentence1": "sentence_a",
                "sentence2": "sentence_b",
                "label": "is_paraphrase" # 1 = Paraphrase, 0 = Different
            })

        elif source_type == "mrpc":
            # MRPC columns vary, usually 'Sentence1', 'Sentence2'
            df = df.rename(columns={
                "Sentence1": "sentence_a",
                "Sentence2": "sentence_b",
                "Quality": "is_paraphrase"
            })

        elif source_type == "custom":
            # Expects our generation script format
            required = ['sentence_a', 'sentence_b']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Custom data missing required columns. Found: {df.columns}")
            if 'is_paraphrase' not in df.columns:
                if 'label' in df.columns:
                    # Rename existing label column to match our schema
                    df = df.rename(columns={"label": "is_paraphrase"})
                else:
                    df['is_paraphrase'] = 1 # Assume synthetic pairs are paraphrases unless stated otherwise

        # Final Validation
        required_cols = ["sentence_a", "sentence_b", "is_paraphrase"]
        df = df[required_cols] # Drop extra columns

        # Filter: In PAWS, we only want the "Positive" (Paraphrase) examples to test fragility
        # i.e., We want pairs that ARE synonymous, to see if the model fails.
        original_count = len(df)
        df_paraphrases = df[df["is_paraphrase"] == 1].reset_index(drop=True)
        filtered_count = len(df_paraphrases)

        self.env.logger.info(f"   └── Filtered Non-Paraphrases: {original_count} -> {filtered_count} pairs.")

        # Apply Sampling Limit (if set in Config for debugging)
        if self.env.cfg.sample_limit:
            df_paraphrases = df_paraphrases.head(self.env.cfg.sample_limit)
            self.env.logger.warning(f"   ⚠️ Debug Limit Applied: Reduced to {len(df_paraphrases)} rows.")

        self.processed_data = df_paraphrases
        self.env.logger.info("✅ Data Pipeline Complete. Ready for Inference.")

    def get_batch_iterator(self, batch_size: int = 32):
        """
        Yields batches for the experiment runner.
        """
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load_benchmark() first.")

        total = len(self.processed_data)
        for i in range(0, total, batch_size):
            yield self.processed_data.iloc[i : i + batch_size]

# ==========================================
# 5. EXECUTION TEST (Module 2)
# ==========================================
# Initialize Engine
data_engine = DataIngestionEngine(env)

# Load PAWS (Gold Standard for Paraphrasing)
data_engine.load_benchmark(dataset_name="paws")

# Inspect the standardized data
print("\n--- DATA SNAPSHOT (First 3 Rows) ---")
display(data_engine.processed_data.head(3))

#----------------------------------------------------------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from typing import List, Tuple

# ==========================================
# 6. INFERENCE ENGINE (The "Brain")
# ==========================================
class ModelInferenceWrapper:
    """
    A unified interface for different Transformer models.
    Handles tokenization, GPU transfer, and probability extraction.
    Includes built-in prompt validation to prevent vague inputs.
    """
    def __init__(self, env: ResearchEnvironment, model_name: str):
        self.env = env
        self.device = env.cfg.device
        self.model_name = model_name

        self.env.logger.info(f"🧠 Loading Model: {model_name}...")
        try:
            # AutoTokenizer handles the specific vocabulary for each model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load Model and move to GPU (if available)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set to Evaluation Mode (crucial for deterministic behavior)

            self.env.logger.info(f"✅ Model {model_name} loaded on {self.device}.")

        except OSError as e:
            self.env.logger.critical(f"❌ Could not load model {model_name}. Check internet or model name.")
            raise e

    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Runs inference on a batch of text with optional prompt validation.
        Returns:
            - labels: List of predicted class IDs (0 or 1)
            - confidences: List of probabilities for the predicted class

        Raises:
            ValueError: If prompt validation is enabled and inputs are too vague
        """
        # ⚡ DEFENSE MECHANISM: Validate inputs before inference
        if self.env.cfg.enforce_prompt_validation:
            validation_report = PromptValidator.validate_batch(
                texts,
                raise_on_error=self.env.cfg.raise_on_vague_input
            )

            if validation_report["invalid_count"] > 0:
                if self.env.cfg.raise_on_vague_input:
                    # Will raise in validate_batch already
                    pass
                else:
                    # Log warning and continue
                    self.env.logger.warning(
                        f"⚠️ Vague input(s) detected: {validation_report['invalid_count']}/{validation_report['total_count']} failed validation"
                    )
                    for idx, reason in zip(validation_report["failed_indices"], validation_report["failed_reasons"]):
                        self.env.logger.warning(f"   └── Index {idx}: {reason}")

        # 1. Tokenization (Padding & Truncation are critical for batching)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # 2. Move inputs to the correct device (CPU/GPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3. Inference (No Gradients = Faster & Low Memory)
        # Use automatic mixed precision if enabled in config
        amp_device = "cuda" if self.device == "cuda" else "cpu"
        autocast_ctx = torch.autocast(device_type=amp_device, enabled=self.env.cfg.mixed_precision and self.device == "cuda")
        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**inputs)
            logits = outputs.logits

            # 4. Convert Logits to Probabilities using Softmax
            probs = F.softmax(logits.float(), dim=-1)  # cast to float32 after AMP

            # 5. Extract Prediction and Confidence
            # We want the max probability and the corresponding label index
            max_probs, predicted_ids = torch.max(probs, dim=1)

        return predicted_ids.cpu().tolist(), max_probs.cpu().tolist()

    def get_full_distribution(self, texts: List[str]) -> torch.Tensor:
        """
        Returns the full probability distribution for advanced metrics (Jensen-Shannon Divergence).
        Useful for Phase 2 analysis.
        """
        # ⚡ Validate inputs here too
        if self.env.cfg.enforce_prompt_validation:
            validation_report = PromptValidator.validate_batch(texts, raise_on_error=False)
            if validation_report["invalid_count"] > 0:
                self.env.logger.warning(
                    f"⚠️ {validation_report['invalid_count']} vague input(s) in batch. Proceeding with caution."
                )

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        amp_device = "cuda" if self.device == "cuda" else "cpu"
        autocast_ctx = torch.autocast(device_type=amp_device, enabled=self.env.cfg.mixed_precision and self.device == "cuda")
        with torch.no_grad(), autocast_ctx:
            logits = self.model(**inputs).logits
            probs = F.softmax(logits.float(), dim=-1)
        return probs.cpu()

# ==========================================
# 7. EXECUTION TEST (Module 3)
# ==========================================
# Let's test with the first model in your config
test_model_name = config.target_models[0]
inference_engine = ModelInferenceWrapper(env, test_model_name)

# Sanity Check with Dummy Data
dummy_batch = [
    "This research is absolutely fascinating and novel.",
    "I am extremely disappointed with the results."
]

labels, confs = inference_engine.predict_batch(dummy_batch)

print("\n--- INFERENCE SANITY CHECK ---")
for text, label, conf in zip(dummy_batch, labels, confs):
    print(f"Text: '{text}'")
    print(f"   └── Predicted Label: {label} | Confidence: {conf:.4f}")

# Test the validation with intentionally vague inputs
print("\n--- VALIDATION TEST (Intentionally Vague Inputs) ---")
vague_inputs = [
    "stuff",
    "",
    "ok",
    "blah blah blah"
]

validation_results = PromptValidator.validate_batch(vague_inputs, raise_on_error=False)
print(f"Validation Results: {validation_results['valid_count']}/{validation_results['total_count']} passed")
for idx, reason in zip(validation_results["failed_indices"], validation_results["failed_reasons"]):
    print(f"  ❌ Input {idx}: {reason}")

# --------------------------------------------------------------------------------------------------------------
