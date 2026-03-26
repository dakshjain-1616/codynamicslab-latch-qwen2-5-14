"""
Perplexity evaluator for FP16 vs quantized model comparison.
Fails the build if perplexity delta exceeds the configured threshold.
"""

import os
import math
import json
import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

# Z-score for 95% confidence interval
_Z_95 = 1.96

logger = logging.getLogger(__name__)

# Configuration from environment
PERPLEXITY_DELTA_THRESHOLD = float(os.getenv("PERPLEXITY_DELTA_THRESHOLD", "0.05"))
DEFAULT_NUM_SAMPLES = int(os.getenv("NUM_PERPLEXITY_SAMPLES", "100"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Lazy torch availability flag — checked on first use to avoid import-time failures
_torch_available: Optional[bool] = None


def _get_torch():
    """Return torch module if available, else raise ImportError with a helpful message."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch as _torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    if not _torch_available:
        raise ImportError(
            "torch is required for real perplexity evaluation. "
            "Install it with: pip install torch. "
            "Set MOCK_MODE=true or USE_REAL_PROXY_MODEL=false for mock evaluation."
        )
    import torch
    return torch


WIKITEXT_SAMPLES = [
    "The history of natural language processing generally started in the 1950s, although work can be found from earlier periods.",
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "A transformer is a deep learning model that adopts the mechanism of self-attention.",
    "The attention mechanism allows the model to focus on different parts of the input sequence.",
    "Quantization is the process of constraining an input from a large set to output in a smaller set.",
    "Model compression techniques include pruning, quantization, knowledge distillation, and low-rank factorization.",
    "GGUF is a file format for storing models for inference with GGML and executors based on GGML.",
    "The perplexity of a language model on a dataset is the inverse probability of the dataset.",
    "Lower perplexity indicates that the probability distribution predicted by the model better fits the data.",
    "Weight quantization reduces memory requirements by storing weights in lower-precision formats.",
    "Q4_K_M is a 4-bit quantization scheme that uses k-means clustering for weight grouping.",
    "The Qwen2.5 model series represents a significant advancement in open-source language models.",
    "Fine-tuning pre-trained language models on domain-specific data improves task performance.",
    "The tokenizer converts raw text into a sequence of integer token identifiers.",
    "Autoregressive language models predict the next token given all previous tokens.",
    "Byte-pair encoding is a subword tokenization algorithm used in many modern language models.",
    "Inference speed is a critical metric for deploying language models in production environments.",
    "Memory bandwidth is often the bottleneck for large language model inference on consumer hardware.",
    "The KV cache stores key and value tensors to avoid recomputing attention for past tokens.",
    "Flash attention is an IO-aware exact attention algorithm that is both fast and memory-efficient.",
    "Speculative decoding uses a smaller draft model to accelerate inference of larger models.",
    "Rotary position embeddings encode positional information using rotation matrices.",
    "Grouped query attention reduces the memory footprint of the key-value cache during inference.",
    "The softmax function converts a vector of real numbers into a probability distribution.",
    "Gradient descent is an optimization algorithm used to minimize the loss function during training.",
    "Backpropagation computes the gradient of the loss with respect to each parameter.",
    "Batch normalization normalizes the inputs to each layer to stabilize training.",
    "Dropout is a regularization technique that randomly sets activations to zero during training.",
    "The cross-entropy loss measures the difference between predicted and true probability distributions.",
    "Transfer learning leverages knowledge from pre-trained models for new tasks.",
    "Zero-shot learning enables models to perform tasks without any task-specific training examples.",
    "Few-shot prompting provides a small number of examples in the model's context.",
    "Chain-of-thought prompting encourages models to show intermediate reasoning steps.",
    "Retrieval-augmented generation combines language models with external knowledge retrieval.",
    "Constitutional AI trains models to be helpful, harmless, and honest using AI feedback.",
    "Reinforcement learning from human feedback aligns language models with human preferences.",
    "The scaling laws of neural language models predict how performance improves with compute.",
    "Emergent abilities appear in large language models that are not present in smaller models.",
    "In-context learning allows language models to adapt to new tasks using only demonstrations.",
    "The context window determines the maximum sequence length a transformer can process.",
    "Positional encoding provides the model with information about the position of tokens.",
    "Multi-head attention allows the model to jointly attend to information from different subspaces.",
    "The feed-forward network in a transformer applies a two-layer MLP to each position independently.",
    "Layer normalization normalizes inputs across the feature dimension within each training example.",
    "Residual connections allow gradients to flow through the network more easily during training.",
    "The embedding layer maps discrete token identifiers to continuous vector representations.",
    "Vocabulary size determines the number of distinct tokens the model can represent.",
    "Temperature scaling controls the randomness of language model output distributions.",
    "Top-k sampling selects from the k most likely next tokens at each generation step.",
    "Nucleus sampling selects from the smallest set of tokens with cumulative probability above p.",
    "Beam search maintains multiple candidate sequences during decoding to find higher-quality outputs.",
    "Greedy decoding always selects the most likely next token at each step.",
    "Repetition penalties reduce the probability of generating tokens that have already appeared.",
    "Prompt engineering involves crafting inputs to elicit desired behaviors from language models.",
    "System prompts provide instructions that guide the model's overall behavior in a conversation.",
    "Function calling allows language models to invoke external tools and APIs.",
    "JSON mode constrains model output to valid JSON format for structured data extraction.",
    "Logprobs provide the log-probability of each generated token for downstream processing.",
    "The hidden dimension determines the size of the internal representations in a transformer.",
    "The number of attention heads determines how many parallel attention computations are performed.",
    "The number of layers determines the depth of the transformer architecture.",
    "Intermediate size refers to the dimension of the feed-forward network's hidden layer.",
    "Sliding window attention limits each token to attending only to a local context window.",
    "Sparse attention patterns reduce the quadratic complexity of full self-attention.",
    "Mixture of experts conditionally activates only a subset of parameters for each token.",
    "Expert routing determines which expert networks process each token in a mixture of experts.",
    "Load balancing ensures that all experts in a mixture of experts receive similar numbers of tokens.",
    "Activation functions introduce non-linearity into neural network computations.",
    "The SwiGLU activation function is commonly used in modern transformer feed-forward networks.",
    "RMSNorm is a simplified version of layer normalization that omits the mean centering step.",
    "Weight tying shares the embedding and output projection matrices to reduce parameter count.",
    "Model sharding distributes a large model across multiple devices for parallel inference.",
    "Tensor parallelism splits individual weight matrices across multiple GPUs for inference.",
    "Pipeline parallelism assigns different layers of a model to different GPUs.",
    "Data parallelism replicates the model across multiple GPUs and processes different batches.",
    "Mixed precision training uses float16 or bfloat16 for forward pass and float32 for gradients.",
    "Gradient checkpointing trades compute for memory by recomputing activations during backprop.",
    "DeepSpeed provides system optimizations for training and inference of large language models.",
    "VLLM is a high-throughput inference engine for large language models using PagedAttention.",
    "PagedAttention manages KV cache memory using virtual memory paging techniques.",
    "Continuous batching dynamically groups requests with different sequence lengths for inference.",
    "Quantization-aware training incorporates quantization into the training process.",
    "Post-training quantization applies quantization after training without any fine-tuning.",
    "The GPTQ algorithm quantizes weights by minimizing the quantization error layer by layer.",
    "AWQ identifies and protects salient weights before applying quantization.",
    "SmoothQuant migrates quantization difficulty from activations to weights.",
    "GGML is a tensor library for machine learning that enables efficient inference on CPU.",
    "llama.cpp implements efficient inference for transformer models using the GGML library.",
    "Metal performance shaders enable GPU acceleration on Apple Silicon for llama.cpp.",
    "CUDA enables GPU-accelerated computation for NVIDIA graphics cards.",
    "ROCm provides GPU computing support for AMD graphics cards.",
    "Vulkan is a cross-platform graphics and compute API that llama.cpp uses for GPU inference.",
    "The GGUF format stores both the model weights and metadata in a single binary file.",
    "Metadata in GGUF includes tokenizer configuration, model architecture, and quantization info.",
    "The F16 quantization format stores weights as 16-bit floating point numbers.",
    "The Q8_0 format quantizes weights to 8 bits with a single scale factor per block.",
    "The Q4_K_M format uses 4-bit quantization with k-quants for improved accuracy.",
]


class PerplexityEvaluator:
    """Evaluates model perplexity on standardized test prompts."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        mock_mode: bool = MOCK_MODE,
    ):
        self.model_name_or_path = model_name_or_path
        self.mock_mode = mock_mode

        if device is not None:
            self.device = device
        else:
            try:
                torch = _get_torch()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = _get_torch()

        logger.info(f"Loading model from: {self.model_name_or_path}")
        hf_token = os.getenv("HF_TOKEN")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            token=hf_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def get_test_samples(self, num_samples: int = DEFAULT_NUM_SAMPLES) -> List[str]:
        """Return standardized test samples for perplexity evaluation."""
        samples = WIKITEXT_SAMPLES * (num_samples // len(WIKITEXT_SAMPLES) + 1)
        return samples[:num_samples]

    def _compute_text_nll(self, text: str) -> Tuple[float, int]:
        """Compute negative log-likelihood for a single text."""
        torch = _get_torch()

        if self.tokenizer is None or self.model is None:
            self.load_model()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if inputs["input_ids"].shape[1] < 2:
            return 0.0, 0

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            nll = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1] - 1

        return nll * num_tokens, num_tokens

    def compute_perplexity(
        self, texts: Optional[List[str]] = None, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> float:
        """Compute perplexity over a list of texts."""
        if texts is None:
            texts = self.get_test_samples(num_samples)

        total_nll = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc=f"Evaluating {self.model_name_or_path}"):
            nll, tokens = self._compute_text_nll(text)
            total_nll += nll
            total_tokens += tokens

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        logger.info(f"Perplexity: {ppl:.4f} over {total_tokens} tokens")
        return ppl

    @staticmethod
    def compute_confidence_interval(
        ppl_values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute mean perplexity and 95% CI from a list of per-sample perplexities.

        Returns:
            (mean, ci_lower, ci_upper) — all rounded to 4 decimal places.
        """
        if not ppl_values:
            return float("nan"), float("nan"), float("nan")
        arr = np.array(ppl_values, dtype=float)
        mean = float(np.mean(arr))
        if len(arr) < 2:
            return round(mean, 4), round(mean, 4), round(mean, 4)
        std = float(np.std(arr, ddof=1))
        margin = _Z_95 * std / math.sqrt(len(arr))
        return round(mean, 4), round(mean - margin, 4), round(mean + margin, 4)

    def compute_perplexity_with_stats(
        self, texts: Optional[List[str]] = None, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> Dict[str, Any]:
        """
        Compute perplexity with per-sample stats (mean, std_dev, 95% CI).

        Returns a dict with keys: perplexity, std_dev, ci_lower, ci_upper,
        num_samples, num_tokens.
        """
        if texts is None:
            texts = self.get_test_samples(num_samples)

        per_sample_ppls: List[float] = []
        total_nll = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc=f"Evaluating {self.model_name_or_path}"):
            nll, tokens = self._compute_text_nll(text)
            if tokens > 0:
                total_nll += nll
                total_tokens += tokens
                sample_ppl = math.exp(nll / tokens)
                per_sample_ppls.append(sample_ppl)

        if total_tokens == 0:
            return {
                "perplexity": float("inf"),
                "std_dev": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "num_samples": len(texts),
                "num_tokens": 0,
            }

        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        mean_ppl, ci_lo, ci_hi = self.compute_confidence_interval(per_sample_ppls)
        std = float(np.std(per_sample_ppls, ddof=1)) if len(per_sample_ppls) > 1 else 0.0

        return {
            "perplexity": round(ppl, 4),
            "std_dev": round(std, 4),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "num_samples": len(texts),
            "num_tokens": total_tokens,
        }

    @staticmethod
    def compute_delta(fp16_ppl: float, quantized_ppl: float) -> float:
        """Relative perplexity delta between FP16 and quantized model."""
        if fp16_ppl == 0:
            return float("inf")
        return abs(quantized_ppl - fp16_ppl) / fp16_ppl

    @staticmethod
    def passes_threshold(
        fp16_ppl: float,
        quantized_ppl: float,
        threshold: float = PERPLEXITY_DELTA_THRESHOLD,
    ) -> bool:
        """Return True if perplexity delta is within acceptable threshold."""
        delta = PerplexityEvaluator.compute_delta(fp16_ppl, quantized_ppl)
        return delta <= threshold

    def evaluate_pair(
        self,
        fp16_perplexity: Optional[float] = None,
        texts: Optional[List[str]] = None,
        num_samples: int = DEFAULT_NUM_SAMPLES,
    ) -> Dict[str, Any]:
        """
        Evaluate the model and compare against a known FP16 perplexity.
        Returns a result dict with pass/fail status.
        """
        if texts is None:
            texts = self.get_test_samples(num_samples)

        quantized_ppl = self.compute_perplexity(texts)

        if fp16_perplexity is None:
            # If no FP16 reference, just return the quantized perplexity
            return {
                "quantized_perplexity": quantized_ppl,
                "fp16_perplexity": None,
                "delta": None,
                "passes": None,
                "num_samples": len(texts),
            }

        delta = self.compute_delta(fp16_perplexity, quantized_ppl)
        passes = delta <= PERPLEXITY_DELTA_THRESHOLD

        return {
            "quantized_perplexity": quantized_ppl,
            "fp16_perplexity": fp16_perplexity,
            "delta": delta,
            "passes": passes,
            "threshold": PERPLEXITY_DELTA_THRESHOLD,
            "num_samples": len(texts),
        }


class MockPerplexityEvaluator(PerplexityEvaluator):
    """
    Mock evaluator for local testing and CI.

    By default (USE_REAL_PROXY_MODEL=false) returns fully simulated perplexity
    values without loading any model or requiring torch/transformers.  This
    makes tests fast and dependency-free.

    Set USE_REAL_PROXY_MODEL=true to use Qwen/Qwen3-0.6B as a real
    proxy model (requires torch + transformers and internet access).

    Simulated Q4_K_M degradation is calibrated to realistic Qwen2.5-14B values.
    """

    # Qwen3-0.6B is tiny (≤1B) — ideal CPU proxy; configurable via env
    PROXY_MODEL = os.getenv("MOCK_PROXY_MODEL", "Qwen/Qwen3-0.6B")
    # Simulated Q4_K_M degradation factor (realistic for Qwen2.5-14B)
    SIMULATED_QUANT_DELTA = float(os.getenv("SIMULATED_QUANT_DELTA", "0.018"))
    # Simulated FP16 perplexity baseline (typical value for language models)
    SIMULATED_FP16_PPL = float(os.getenv("MOCK_FP16_PERPLEXITY", "100.0"))
    # Whether to load a real proxy model (defaults to pure-mock mode)
    USE_REAL_PROXY_MODEL = os.getenv("USE_REAL_PROXY_MODEL", "false").lower() == "true"

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        super().__init__(self.PROXY_MODEL, device, mock_mode=True)
        self.target_model = model_name_or_path
        logger.info(
            f"[MOCK] MockPerplexityEvaluator for {model_name_or_path} "
            f"(proxy={'real:' + self.PROXY_MODEL if self.USE_REAL_PROXY_MODEL else 'simulated'})"
        )

    def compute_fp16_perplexity(
        self, texts: Optional[List[str]] = None, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> float:
        """Compute perplexity using proxy model as FP16 stand-in."""
        if texts is None:
            texts = self.get_test_samples(num_samples)
        return self.compute_perplexity(texts)

    def compute_quantized_perplexity(
        self, fp16_ppl: float, noise_seed: int = 42
    ) -> float:
        """Simulate Q4_K_M quantized perplexity from FP16 baseline."""
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, self.SIMULATED_QUANT_DELTA * 0.1)
        delta = self.SIMULATED_QUANT_DELTA + noise
        return fp16_ppl * (1 + delta)

    def run_full_evaluation(
        self, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> Dict[str, Any]:
        """
        Run full mock evaluation pipeline.

        Uses pure simulation by default (no model loading required).
        Set USE_REAL_PROXY_MODEL=true to use the real gpt2 proxy model.
        """
        texts = self.get_test_samples(num_samples)

        if self.USE_REAL_PROXY_MODEL:
            logger.info("[MOCK] Computing FP16 perplexity via real proxy model...")
            fp16_ppl = self.compute_fp16_perplexity(texts)
        else:
            # Pure mock: use a realistic simulated baseline — no model download needed
            fp16_ppl = self.SIMULATED_FP16_PPL
            logger.info(
                f"[MOCK] Using simulated FP16 perplexity: {fp16_ppl:.4f} "
                "(set USE_REAL_PROXY_MODEL=true for real gpt2 proxy)"
            )

        logger.info("[MOCK] Simulating Q4_K_M quantized perplexity...")
        q4_ppl = self.compute_quantized_perplexity(fp16_ppl)

        delta = self.compute_delta(fp16_ppl, q4_ppl)
        passes = delta <= PERPLEXITY_DELTA_THRESHOLD

        # Simulate per-sample distribution for CI computation
        rng = np.random.default_rng(42)
        fp16_samples = rng.normal(fp16_ppl, fp16_ppl * 0.05, size=num_samples).tolist()
        q4_samples = rng.normal(q4_ppl, q4_ppl * 0.05, size=num_samples).tolist()

        _, fp16_ci_lo, fp16_ci_hi = self.compute_confidence_interval(fp16_samples)
        _, q4_ci_lo, q4_ci_hi = self.compute_confidence_interval(q4_samples)
        fp16_std = float(np.std(fp16_samples, ddof=1))
        q4_std = float(np.std(q4_samples, ddof=1))

        return {
            "model": self.target_model,
            "proxy_model": self.PROXY_MODEL,
            "fp16_perplexity": round(fp16_ppl, 4),
            "fp16_std_dev": round(fp16_std, 4),
            "fp16_ci_lower": fp16_ci_lo,
            "fp16_ci_upper": fp16_ci_hi,
            "quantized_perplexity": round(q4_ppl, 4),
            "quantized_std_dev": round(q4_std, 4),
            "quantized_ci_lower": q4_ci_lo,
            "quantized_ci_upper": q4_ci_hi,
            "delta": round(delta, 6),
            "delta_percent": round(delta * 100, 3),
            "threshold": PERPLEXITY_DELTA_THRESHOLD,
            "passes": passes,
            "num_samples": num_samples,
            "quantization_type": "Q4_K_M",
            "mock_mode": True,
        }
