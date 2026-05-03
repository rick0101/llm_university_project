"""
Taxonomy — Vocabolari controllati per i campi tassonomici di L1Summary.

Definisce gli enum legali per ``attack_vector``, ``defense_type``,
``evaluation_metrics`` e ``datasets`` nel dominio LLM security, più una
funzione di normalizzazione fuzzy (``normalize_taxonomy_value``) che
mappa l'output libero del LLM al valore canonico.

Quando il LLM produce un valore che non matcha nessun canonico viene
mantenuto come-is solo nel campo ``subcategory_freeform`` (gestito a
livello prompt), così il payload Qdrant resta filterable.

Uso::

    from paperhunter.summarization.taxonomy import (
        ATTACK_VECTORS, normalize_taxonomy_value, normalize_list,
    )

    canon = normalize_taxonomy_value("attack_vector", "Indirect Prompt-Injection")
    # → "indirect_prompt_injection"

    cleaned = normalize_list("attack_vector", ["Direct PI", "indirect", "weird"])
    # → ["direct_prompt_injection", "indirect_prompt_injection"]
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Vocabolari canonici
# ---------------------------------------------------------------------------

ATTACK_VECTORS: list[str] = [
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "jailbreak",
    "gradient_based_attack",
    "transfer_attack",
    "data_poisoning",
    "model_poisoning",
    "backdoor_attack",
    "membership_inference",
    "training_data_extraction",
    "adversarial_suffix",
    "multi_turn_attack",
    "tool_misuse",
    "rag_poisoning",
    "fine_tuning_attack",
    "prompt_leak",
    "denial_of_service",
    "hallucination_induction",
]

DEFENSE_TYPES: list[str] = [
    "input_filtering",
    "output_filtering",
    "prompt_hardening",
    "system_prompt_isolation",
    "rlhf_alignment",
    "constitutional_ai",
    "adversarial_training",
    "fine_tuning_defense",
    "ensemble_defense",
    "watermarking",
    "perplexity_filter",
    "self_reflection",
    "guardrails",
    "rag_grounding",
    "tool_sandboxing",
    "rate_limiting",
    "detection_classifier",
    "uncertainty_quantification",
]

EVALUATION_METRICS: list[str] = [
    "ASR",                # Attack Success Rate
    "BLEU",
    "ROUGE",
    "METEOR",
    "F1",
    "precision",
    "recall",
    "accuracy",
    "perplexity",
    "win_rate",
    "human_eval",
    "MMLU",
    "harmfulness_score",
    "robustness_score",
    "safety_score",
    "toxicity_score",
    "exact_match",
    "transfer_rate",
    "refusal_rate",
    "jailbreak_rate",
    "false_positive_rate",
    "false_negative_rate",
    "AUROC",
    "AUC",
]

DATASETS: list[str] = [
    "AdvBench",
    "HarmBench",
    "JailbreakBench",
    "MMLU",
    "TruthfulQA",
    "BeaverTails",
    "Anthropic_HH",
    "RealToxicityPrompts",
    "ToxicChat",
    "DecodingTrust",
    "PromptBench",
    "SafetyBench",
    "DoNotAnswer",
    "AlpacaEval",
    "MT-Bench",
    "Natural_Questions",
    "TriviaQA",
    "SQuAD",
    "GSM8K",
    "BBH",
    "WildJailbreak",
    "GCG",
    "ToolBench",
]

# Normalizzazione comune dei nomi modello (lowercase canonical)
TARGET_MODELS_HINTS: list[str] = [
    "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5", "chatgpt",
    "claude-3", "claude-2", "claude-3.5-sonnet", "claude-opus",
    "llama-2-7b", "llama-2-13b", "llama-2-70b", "llama-3",
    "mistral-7b", "mixtral-8x7b",
    "gemini-1.5-pro", "gemini-pro", "gemini-flash",
    "vicuna-7b", "vicuna-13b",
    "qwen-7b", "qwen-14b",
    "falcon-7b", "falcon-40b",
    "palm-2", "bard",
]


_VOCABS: dict[str, list[str]] = {
    "attack_vector": ATTACK_VECTORS,
    "defense_type": DEFENSE_TYPES,
    "evaluation_metrics": EVALUATION_METRICS,
    "datasets": DATASETS,
    "target_models": TARGET_MODELS_HINTS,
}


# ---------------------------------------------------------------------------
# Normalizzazione fuzzy
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD = 0.78  # similarità minima per accettare un match


def _slug(s: str) -> str:
    """Normalizza una stringa per il confronto fuzzy."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_taxonomy_value(field: str, raw: str) -> str | None:
    """Mappa un valore raw del LLM al canonico più simile.

    Ritorna ``None`` se nessun canonico raggiunge la soglia di similarità
    (il chiamante può decidere di scartare o mettere in ``subcategory_freeform``).

    Le metriche/dataset/target_models conservano il case originale del canonico
    (es. "ASR", "AdvBench"); attack_vector/defense_type sono lowercase snake.
    """
    if not raw or not raw.strip():
        return None
    vocab = _VOCABS.get(field)
    if vocab is None:
        return None

    raw_slug = _slug(raw)

    # Match esatto su slug
    for canon in vocab:
        if _slug(canon) == raw_slug:
            return canon

    # Match per substring (entrambi i versi)
    for canon in vocab:
        cs = _slug(canon)
        if cs in raw_slug or raw_slug in cs:
            # Richiedi che la parte condivisa sia ≥ metà del canonico
            if min(len(cs), len(raw_slug)) >= max(3, len(cs) // 2):
                return canon

    # Fuzzy fallback (SequenceMatcher su slug)
    best_canon: str | None = None
    best_score = 0.0
    for canon in vocab:
        score = SequenceMatcher(None, _slug(canon), raw_slug).ratio()
        if score > best_score:
            best_score = score
            best_canon = canon
    if best_score >= _FUZZY_THRESHOLD:
        return best_canon
    return None


def normalize_list(field: str, raw_values: list[str] | None) -> list[str]:
    """Normalizza una lista di valori raw, rimuovendo None/duplicati.

    Mantiene l'ordine di prima apparizione.
    """
    if not raw_values:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for v in raw_values:
        if not isinstance(v, str):
            continue
        canon = normalize_taxonomy_value(field, v)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def vocab_for_prompt(field: str, max_items: int | None = None) -> str:
    """Render della lista canonica per inclusione nel prompt LLM."""
    vocab = _VOCABS.get(field, [])
    if max_items is not None:
        vocab = vocab[:max_items]
    return ", ".join(vocab)
