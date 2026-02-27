"""
GraphRAG Entity & Relation Extractor — Estrae entità e relazioni dai summary L1.

Estrae un knowledge graph strutturato dai summary L1 per costruire un
PropertyGraph con LlamaIndex e/o popolare Neo4j.

Tipi di entità:
- **Attack** — tecnica di attacco (es. GCG, AutoDAN, prompt injection)
- **Defense** — meccanismo di difesa (es. perplexity filter, SmoothLLM)
- **Model** — modello LLM (es. GPT-4, LLaMA-2, Claude)
- **Dataset** — dataset usato (es. AdvBench, TruthfulQA, HaluEval)
- **Benchmark** — framework/benchmark di valutazione (es. GLUE, MT-Bench)
- **Metric** — metrica di valutazione (es. ASR, BLEU, F1, perplexity)
- **Author** — autore di rilievo (primo/ultimo autore di paper chiave)
- **Venue** — venue di pubblicazione (es. NeurIPS, ACL, ICML)

Tipi di relazione:
- **DEFENDS_AGAINST** — (Defense) → (Attack)
- **TESTED_ON** — (Attack|Defense) → (Dataset|Benchmark)
- **UPDATES** — (Attack|Defense) → (Attack|Defense) precedente
- **INTRODUCED_IN** — (Attack|Defense|Dataset) → (Venue) [con anno]
- **EXTENDS** — (Attack|Defense) → (Attack|Defense)
- **OUTPERFORMS** — (Attack|Defense) → (Attack|Defense) [su metrica]
- **EVALUATED_WITH** — (Paper|Attack|Defense) → (Metric)
- **TARGETS** — (Attack) → (Model)
- **USES_MODEL** — (Defense|Benchmark) → (Model)

Uso::

    from paperhunter.graphrag.entity_extractor import (
        extract_entities_and_relations,
        extract_batch,
        Entity,
        Relation,
        EntityType,
        RelationType,
    )
    from paperhunter.models import L1Summary

    entities, relations = extract_entities_and_relations(l1_summary)
    for e in entities:
        print(f"{e.type}: {e.canonical_name} (aliases: {e.aliases})")
    for r in relations:
        print(f"{r.source} --[{r.type}]--> {r.target}")

    # Batch extraction
    results = extract_batch(l1_summaries)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from paperhunter.models import L1Summary
from paperhunter.utils.llm_factory import (
    get_flash_llm,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)
from paperhunter.utils.rate_limiter import get_gemini_limiter

logger = get_logger(__name__)


# ============================================================================
# Enums
# ============================================================================

class EntityType(str, Enum):
    """Tipo di entità nel knowledge graph."""
    ATTACK = "Attack"
    DEFENSE = "Defense"
    MODEL = "Model"
    DATASET = "Dataset"
    BENCHMARK = "Benchmark"
    METRIC = "Metric"
    AUTHOR = "Author"
    VENUE = "Venue"


class RelationType(str, Enum):
    """Tipo di relazione nel knowledge graph."""
    DEFENDS_AGAINST = "DEFENDS_AGAINST"
    TESTED_ON = "TESTED_ON"
    UPDATES = "UPDATES"
    INTRODUCED_IN = "INTRODUCED_IN"
    EXTENDS = "EXTENDS"
    OUTPERFORMS = "OUTPERFORMS"
    EVALUATED_WITH = "EVALUATED_WITH"
    TARGETS = "TARGETS"
    USES_MODEL = "USES_MODEL"


# ============================================================================
# Data models
# ============================================================================

class Entity(BaseModel):
    """Entità estratta dal knowledge graph.

    Ogni entità ha un nome canonico normalizzato e una lista di alias.
    L'ID è deterministico basato su tipo + nome canonico.
    """
    id: str = Field("", description="ID univoco (hash di type::canonical_name)")
    type: EntityType = Field(..., description="Tipo dell'entità")
    canonical_name: str = Field(..., description="Nome canonico normalizzato")
    aliases: list[str] = Field(default_factory=list, description="Nomi alternativi / varianti")
    description: str = Field("", description="Breve descrizione dell'entità")
    year: Optional[int] = Field(None, description="Anno di introduzione (se applicabile)")
    source_paper_id: Optional[str] = Field(None, description="Paper ID in cui è stata estratta")
    source_bibtex_key: Optional[str] = Field(None, description="BibTeX key del paper sorgente")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Proprietà aggiuntive (es. n_params per modelli, ASR per attacchi)"
    )

    def model_post_init(self, __context: Any) -> None:
        """Genera l'ID deterministico se non fornito."""
        if not self.id:
            self.id = _make_entity_id(self.type, self.canonical_name)

    def to_dict(self) -> dict[str, Any]:
        """Serializza per export JSON/Neo4j."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, EntityType) else self.type,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "description": self.description,
            "year": self.year,
            "source_paper_id": self.source_paper_id,
            "source_bibtex_key": self.source_bibtex_key,
            "properties": self.properties,
        }


class Relation(BaseModel):
    """Relazione tra due entità nel knowledge graph.

    Connette una entità sorgente a una entità target con un tipo
    di relazione e proprietà opzionali (anno, metrica, ecc.).
    """
    id: str = Field("", description="ID univoco della relazione")
    type: RelationType = Field(..., description="Tipo della relazione")
    source: str = Field(..., description="Nome canonico dell'entità sorgente")
    source_type: EntityType = Field(..., description="Tipo dell'entità sorgente")
    target: str = Field(..., description="Nome canonico dell'entità target")
    target_type: EntityType = Field(..., description="Tipo dell'entità target")
    description: str = Field("", description="Descrizione della relazione")
    year: Optional[int] = Field(None, description="Anno della relazione (es. INTRODUCED_IN 2023)")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Proprietà aggiuntive (es. metrica per OUTPERFORMS, percentuale, ecc.)"
    )
    source_paper_id: Optional[str] = Field(None, description="Paper ID sorgente")
    source_bibtex_key: Optional[str] = Field(None, description="BibTeX key sorgente")

    def model_post_init(self, __context: Any) -> None:
        """Genera l'ID deterministico se non fornito."""
        if not self.id:
            self.id = _make_relation_id(self.type, self.source, self.target)

    def to_dict(self) -> dict[str, Any]:
        """Serializza per export JSON/Neo4j."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, RelationType) else self.type,
            "source": self.source,
            "source_type": self.source_type.value if isinstance(self.source_type, EntityType) else self.source_type,
            "target": self.target,
            "target_type": self.target_type.value if isinstance(self.target_type, EntityType) else self.target_type,
            "description": self.description,
            "year": self.year,
            "properties": self.properties,
            "source_paper_id": self.source_paper_id,
            "source_bibtex_key": self.source_bibtex_key,
        }


class ExtractionResult(BaseModel):
    """Risultato dell'estrazione per un singolo L1 summary."""
    paper_id: str = Field("", description="Paper ID")
    bibtex_key: Optional[str] = Field(None, description="BibTeX key")
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    extracted_at: Optional[datetime] = Field(None, description="Timestamp di estrazione")
    model_used: Optional[str] = Field(None, description="Modello usato")
    extraction_time_seconds: float = Field(0.0, description="Tempo di estrazione in secondi")
    error: Optional[str] = Field(None, description="Errore se estrazione fallita")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "bibtex_key": self.bibtex_key,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
            "model_used": self.model_used,
            "extraction_time_seconds": self.extraction_time_seconds,
            "error": self.error,
        }


# ============================================================================
# ID generators
# ============================================================================

def _make_entity_id(entity_type: EntityType | str, canonical_name: str) -> str:
    """Genera un ID deterministico per un'entità.

    ID = SHA-256(type::canonical_name) troncato a 16 hex chars.
    """
    t = entity_type.value if isinstance(entity_type, EntityType) else entity_type
    raw = f"{t}::{canonical_name.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_relation_id(
    rel_type: RelationType | str,
    source: str,
    target: str,
) -> str:
    """Genera un ID deterministico per una relazione.

    ID = SHA-256(type::source::target) troncato a 16 hex chars.
    """
    t = rel_type.value if isinstance(rel_type, RelationType) else rel_type
    raw = f"{t}::{source.lower().strip()}::{target.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================================
# Entity normalization
# ============================================================================

# Mapping entità note → canonical name.
# Copre le varianti più comuni nei paper LLM security.
_KNOWN_ENTITY_ALIASES: dict[str, str] = {
    # === Models ===
    "gpt4": "GPT-4",
    "gpt-4": "GPT-4",
    "gpt 4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gpt4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt3": "GPT-3",
    "gpt-3": "GPT-3",
    "gpt 3": "GPT-3",
    "gpt-3.5": "GPT-3.5",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt3.5": "GPT-3.5",
    "chatgpt": "ChatGPT",
    "chat gpt": "ChatGPT",
    "claude": "Claude",
    "claude-2": "Claude 2",
    "claude-3": "Claude 3",
    "claude 3 opus": "Claude 3 Opus",
    "claude 3 sonnet": "Claude 3 Sonnet",
    "claude 3 haiku": "Claude 3 Haiku",
    "llama": "LLaMA",
    "llama2": "LLaMA-2",
    "llama-2": "LLaMA-2",
    "llama 2": "LLaMA-2",
    "llama-2-7b": "LLaMA-2 7B",
    "llama-2-13b": "LLaMA-2 13B",
    "llama-2-70b": "LLaMA-2 70B",
    "llama3": "LLaMA-3",
    "llama-3": "LLaMA-3",
    "llama 3": "LLaMA-3",
    "vicuna": "Vicuna",
    "vicuna-7b": "Vicuna 7B",
    "vicuna-13b": "Vicuna 13B",
    "alpaca": "Alpaca",
    "mistral": "Mistral",
    "mistral-7b": "Mistral 7B",
    "mixtral": "Mixtral",
    "gemini": "Gemini",
    "gemini pro": "Gemini Pro",
    "gemini ultra": "Gemini Ultra",
    "gemma": "Gemma",
    "gemma-2": "Gemma 2",
    "palm": "PaLM",
    "palm2": "PaLM 2",
    "palm-2": "PaLM 2",
    "t5": "T5",
    "flan-t5": "Flan-T5",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "phi-2": "Phi-2",
    "phi-3": "Phi-3",
    "falcon": "Falcon",
    "falcon-7b": "Falcon 7B",
    "falcon-40b": "Falcon 40B",
    "opt": "OPT",
    "bloom": "BLOOM",
    "pythia": "Pythia",
    "qwen": "Qwen",
    "qwen2": "Qwen2",
    "yi": "Yi",
    "deepseek": "DeepSeek",
    "internlm": "InternLM",
    "baichuan": "Baichuan",
    "cohere": "Cohere Command",
    "command-r": "Cohere Command-R",

    # === Attacks ===
    "gcg": "GCG",
    "gcg attack": "GCG",
    "greedy coordinate gradient": "GCG",
    "autodan": "AutoDAN",
    "auto-dan": "AutoDAN",
    "pair": "PAIR",
    "tap": "TAP",
    "jailbreakchat": "JailbreakChat",
    "dan": "DAN",
    "do anything now": "DAN",
    "prompt injection": "Prompt Injection",
    "indirect prompt injection": "Indirect Prompt Injection",
    "direct prompt injection": "Direct Prompt Injection",
    "jailbreak": "Jailbreak",
    "jailbreaking": "Jailbreak",
    "adversarial suffix": "Adversarial Suffix",
    "adversarial suffixes": "Adversarial Suffix",
    "backdoor attack": "Backdoor Attack",
    "backdoor": "Backdoor Attack",
    "data poisoning": "Data Poisoning",
    "training data poisoning": "Data Poisoning",
    "instruction tuning poisoning": "Instruction Tuning Poisoning",
    "sleeper agent": "Sleeper Agent Attack",
    "trojan": "Trojan Attack",
    "trojan attack": "Trojan Attack",
    "model poisoning": "Model Poisoning",
    "context manipulation": "Context Manipulation",
    "goal hijacking": "Goal Hijacking",
    "prompt leaking": "Prompt Leaking",
    "prompt extraction": "Prompt Extraction",

    # === Defenses ===
    "perplexity filter": "Perplexity Filter",
    "perplexity filtering": "Perplexity Filter",
    "smoothllm": "SmoothLLM",
    "smooth llm": "SmoothLLM",
    "self-reminder": "Self-Reminder",
    "instructiondefense": "Instruction Defense",
    "llm guardrails": "LLM Guardrails",
    "guardrails": "Guardrails",
    "input sanitization": "Input Sanitization",
    "output filtering": "Output Filtering",
    "safety training": "Safety Training",
    "rlhf": "RLHF",
    "reinforcement learning from human feedback": "RLHF",
    "constitutional ai": "Constitutional AI",
    "red teaming": "Red Teaming",
    "safety alignment": "Safety Alignment",
    "system prompt hardening": "System Prompt Hardening",
    "prompt shield": "PromptShield",
    "promptshield": "PromptShield",
    "llama guard": "Llama Guard",
    "llamaguard": "Llama Guard",
    "nemo guardrails": "NeMo Guardrails",

    # === Datasets & Benchmarks ===
    "advbench": "AdvBench",
    "truthfulqa": "TruthfulQA",
    "truthful qa": "TruthfulQA",
    "halueval": "HaluEval",
    "halu eval": "HaluEval",
    "halubench": "HaluBench",
    "toxigen": "ToxiGen",
    "realtoxicityprompts": "RealToxicityPrompts",
    "jailbreakbench": "JailbreakBench",
    "harmbench": "HarmBench",
    "harm bench": "HarmBench",
    "strongreject": "StrongReject",
    "strong reject": "StrongReject",
    "mmlu": "MMLU",
    "mt-bench": "MT-Bench",
    "mt bench": "MT-Bench",
    "alpaca eval": "AlpacaEval",
    "alpacaeval": "AlpacaEval",
    "bigbench": "BIG-Bench",
    "big-bench": "BIG-Bench",
    "big bench": "BIG-Bench",
    "wikitext": "WikiText",
    "openai evals": "OpenAI Evals",
    "anthropic evals": "Anthropic Evals",
    "bamboogle": "BamBoogle",
    "felm": "FELM",
    "factscore": "FActScore",

    # === Metrics ===
    "asr": "ASR",
    "attack success rate": "ASR",
    "bleu": "BLEU",
    "rouge": "ROUGE",
    "rouge-l": "ROUGE-L",
    "meteor": "METEOR",
    "bertscore": "BERTScore",
    "bert score": "BERTScore",
    "perplexity": "Perplexity",
    "ppl": "Perplexity",
    "f1 score": "F1",
    "f1": "F1",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "fluency": "Fluency",
    "toxicity score": "Toxicity Score",
    "helpfulness": "Helpfulness",
    "harmlessness": "Harmlessness",
    "hallucination rate": "Hallucination Rate",
    "faithfulness": "Faithfulness",
    "clean accuracy": "Clean Accuracy",

    # === Venues ===
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "icml": "ICML",
    "iclr": "ICLR",
    "acl": "ACL",
    "emnlp": "EMNLP",
    "naacl": "NAACL",
    "aaai": "AAAI",
    "cvpr": "CVPR",
    "usenix security": "USENIX Security",
    "usenix": "USENIX Security",
    "ieee s&p": "IEEE S&P",
    "ieee sp": "IEEE S&P",
    "oakland": "IEEE S&P",
    "ccs": "ACM CCS",
    "acm ccs": "ACM CCS",
    "ndss": "NDSS",
    "arxiv": "arXiv",
    "arxiv.org": "arXiv",
}


def normalize_entity_name(raw_name: str) -> str:
    """Normalizza un nome di entità al suo canonical name.

    1. Strip + lowercase
    2. Rimuovi accenti (NFD → strip combining marks)
    3. Cerca nel dizionario _KNOWN_ENTITY_ALIASES
    4. Se non trovato, titolizza e ritorna

    Args:
        raw_name: Nome grezzo dal LLM.

    Returns:
        Nome canonico normalizzato.

    Examples:
        >>> normalize_entity_name("gpt4")
        'GPT-4'
        >>> normalize_entity_name("GPT-4")
        'GPT-4'
        >>> normalize_entity_name("smooth llm")
        'SmoothLLM'
        >>> normalize_entity_name("Some New Dataset")
        'Some New Dataset'
    """
    if not raw_name or not raw_name.strip():
        return raw_name.strip()

    # Normalizzazione Unicode → ASCII
    normalized = unicodedata.normalize("NFD", raw_name.strip())
    ascii_ish = "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    )

    # Lookup case-insensitive
    key = ascii_ish.lower().strip()
    if key in _KNOWN_ENTITY_ALIASES:
        return _KNOWN_ENTITY_ALIASES[key]

    # Prova rimuovendo trattini e underscore
    key_clean = re.sub(r"[-_]+", " ", key).strip()
    if key_clean in _KNOWN_ENTITY_ALIASES:
        return _KNOWN_ENTITY_ALIASES[key_clean]

    # Prova rimuovendo "attack", "defense", "method" trailing
    for suffix in (" attack", " defense", " method", " technique", " approach"):
        stripped = key_clean.rstrip().removesuffix(suffix).strip()
        if stripped and stripped in _KNOWN_ENTITY_ALIASES:
            return _KNOWN_ENTITY_ALIASES[stripped]

    # Non trovato: ritorna il nome originale con case preservato
    # ma con whitespace normalizzato
    return " ".join(raw_name.strip().split())


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """Merge entità con lo stesso canonical_name + type.

    Se due entità hanno lo stesso tipo e canonical_name, le unisce:
    - Unisce aliases (dedup)
    - Prende la descrizione più lunga
    - Unisce properties
    - Preserva il primo source_paper trovato

    Args:
        entities: Lista di entità potenzialmente duplicate.

    Returns:
        Lista di entità deduplicate.
    """
    merged: dict[str, Entity] = {}

    for e in entities:
        key = f"{e.type.value}::{e.canonical_name.lower()}"
        if key not in merged:
            merged[key] = e.model_copy(deep=True)
        else:
            existing = merged[key]
            # Unisci aliases
            all_aliases = set(existing.aliases) | set(e.aliases)
            all_aliases.discard(existing.canonical_name)
            existing.aliases = sorted(all_aliases)
            # Descrizione più lunga
            if len(e.description) > len(existing.description):
                existing.description = e.description
            # Unisci properties
            existing.properties.update(e.properties)
            # Anno: prendi il più vecchio
            if e.year and (not existing.year or e.year < existing.year):
                existing.year = e.year

    return list(merged.values())


def merge_relations(relations: list[Relation]) -> list[Relation]:
    """Merge relazioni duplicate (stessa source, target, type).

    Args:
        relations: Lista di relazioni potenzialmente duplicate.

    Returns:
        Lista di relazioni deduplicate.
    """
    merged: dict[str, Relation] = {}

    for r in relations:
        key = f"{r.type.value}::{r.source.lower()}::{r.target.lower()}"
        if key not in merged:
            merged[key] = r.model_copy(deep=True)
        else:
            existing = merged[key]
            # Descrizione più lunga
            if len(r.description) > len(existing.description):
                existing.description = r.description
            # Unisci properties
            existing.properties.update(r.properties)
            # Anno: preserva il primo
            if r.year and not existing.year:
                existing.year = r.year

    return list(merged.values())


# ============================================================================
# Prompt template
# ============================================================================

_EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert in LLM security research. Extract structured entities and \
relations from the following paper summary to build a knowledge graph.

## Paper Information
- **Paper ID:** {paper_id}
- **BibTeX Key:** {bibtex_key}
- **Title (from key):** {title_hint}
- **Category:** {category}
- **Year:** {year}

## Paper Summary (L1)
### Key Contributions
{key_contributions}

### Methodology
{methodology}

### Testing Methodology
{testing_methodology}

### Temporal Context
{temporal_context}

### Results Summary
{results_summary}

### Limitations
{limitations}

---

## Extraction Instructions

Extract **entities** and **relations** from the summary above. Be thorough \
but precise — only extract entities explicitly mentioned or clearly implied.

### Entity Types (use EXACTLY these labels):
- **Attack** — specific attack technique or class (e.g., GCG, AutoDAN, prompt injection, backdoor attack)
- **Defense** — specific defense mechanism or approach (e.g., perplexity filter, SmoothLLM, RLHF)
- **Model** — specific LLM or family (e.g., GPT-4, LLaMA-2, Vicuna)
- **Dataset** — specific dataset used (e.g., AdvBench, TruthfulQA, HaluEval)
- **Benchmark** — evaluation framework/benchmark (e.g., HarmBench, JailbreakBench, MMLU)
- **Metric** — evaluation metric (e.g., ASR, BLEU, F1, perplexity)
- **Author** — key author (only first/last author of the paper or frequently cited authors)
- **Venue** — publication venue (e.g., NeurIPS, ACL, arXiv)

### Relation Types (use EXACTLY these labels):
- **DEFENDS_AGAINST** — (Defense) defends against (Attack)
- **TESTED_ON** — (Attack|Defense) was tested on (Dataset|Benchmark)
- **UPDATES** — (Attack|Defense) updates/improves on a previous (Attack|Defense)
- **INTRODUCED_IN** — (Entity) was introduced at (Venue) in a specific year
- **EXTENDS** — (Attack|Defense) extends (Attack|Defense)
- **OUTPERFORMS** — (Attack|Defense) outperforms (Attack|Defense) on a specific metric
- **EVALUATED_WITH** — (Attack|Defense) was evaluated using (Metric)
- **TARGETS** — (Attack) targets/attacks (Model)
- **USES_MODEL** — (Defense|Benchmark) uses/builds upon (Model)

## Output Format
Return a JSON object with exactly two keys:

```json
{{
  "entities": [
    {{
      "type": "Attack|Defense|Model|Dataset|Benchmark|Metric|Author|Venue",
      "name": "canonical name of the entity",
      "aliases": ["alternative name 1", "alternative name 2"],
      "description": "brief one-sentence description",
      "year": 2023
    }}
  ],
  "relations": [
    {{
      "type": "DEFENDS_AGAINST|TESTED_ON|UPDATES|INTRODUCED_IN|EXTENDS|OUTPERFORMS|EVALUATED_WITH|TARGETS|USES_MODEL",
      "source": "canonical name of source entity",
      "source_type": "Attack|Defense|Model|Dataset|Benchmark|Metric|Author|Venue",
      "target": "canonical name of target entity",
      "target_type": "Attack|Defense|Model|Dataset|Benchmark|Metric|Author|Venue",
      "description": "brief description of the relationship",
      "year": 2023
    }}
  ]
}}
```

### Important Rules:
1. Use EXACT entity type labels from the list above. Do not invent new types.
2. Use EXACT relation type labels from the list above. Do not invent new types.
3. Entity names should be specific (e.g., "GCG" not "gradient-based attack")
4. Include year only if explicitly mentioned or inferable from the summary.
5. For OUTPERFORMS, include the metric in the description if available.
6. Every entity in a relation MUST also appear in the entities list.
7. Do NOT extract generic concepts (e.g., "machine learning", "neural network").
8. Extract 5-20 entities and 3-15 relations (be selective, not exhaustive).
9. If the summary lacks specific names, extract what you can — quality over quantity.
10. Return ONLY the JSON object, no additional text.
11. Keep your response concise: output the JSON directly without preamble or explanation.
"""


# ============================================================================
# Main extraction functions
# ============================================================================

def extract_entities_and_relations(
    l1_summary: L1Summary,
    *,
    llm=None,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
) -> tuple[list[Entity], list[Relation]]:
    """Estrae entità e relazioni da un singolo L1 summary.

    Usa Gemini Flash per estrarre un knowledge graph strutturato
    dal contenuto del summary L1.

    Args:
        l1_summary: Summary L1 del paper.
        llm: Istanza LLM override (default: Gemini Flash).
        max_retries: Tentativi massimi in caso di errore.
        base_delay: Delay base per backoff esponenziale.
        max_delay: Delay massimo per backoff.

    Returns:
        Tuple (entities, relations) con entità e relazioni estratte.
        Le entità sono normalizzate e deduplicate.
        Se l'estrazione fallisce, ritorna ([], []).
    """
    result = _extract_with_result(
        l1_summary,
        llm=llm,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    return result.entities, result.relations


def extract_with_details(
    l1_summary: L1Summary,
    *,
    llm=None,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
) -> ExtractionResult:
    """Come ``extract_entities_and_relations`` ma ritorna anche metadati.

    Returns:
        ExtractionResult con entità, relazioni, timestamps e statistiche.
    """
    return _extract_with_result(
        l1_summary,
        llm=llm,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def extract_batch(
    l1_summaries: list[L1Summary],
    *,
    llm=None,
    merge_results: bool = True,
    skip_errors: bool = True,
    show_progress: bool = True,
) -> tuple[list[Entity], list[Relation], list[ExtractionResult]]:
    """Estrae entità e relazioni da un batch di L1 summaries.

    Args:
        l1_summaries: Lista di summary L1.
        llm: Istanza LLM override (default: Gemini Flash).
        merge_results: Se True, merge entità/relazioni duplicate tra paper.
        skip_errors: Se True, continua in caso di errore su un singolo paper.
        show_progress: Se True, mostra progress bar.

    Returns:
        Tuple (all_entities, all_relations, extraction_results).
        Se merge_results=True, entità e relazioni sono deduplicate.
    """
    if not l1_summaries:
        return [], [], []

    if llm is None:
        # Thinking model usa token per ragionamento interno → servono 65K
        llm = get_flash_llm(timeout_seconds=120, max_tokens=65536)

    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    all_results: list[ExtractionResult] = []

    iterable = l1_summaries
    if show_progress:
        iterable = progress_bar(
            l1_summaries,
            description="Extracting entities & relations",
            total=len(l1_summaries),
        )

    success = 0
    failed = 0
    for summary in iterable:
        try:
            result = _extract_with_result(summary, llm=llm)
            all_results.append(result)

            if result.error:
                failed += 1
                if not skip_errors:
                    logger.error(f"Estrazione fallita per {summary.paper_id}: {result.error}")
                    raise RuntimeError(result.error)
            else:
                success += 1
                all_entities.extend(result.entities)
                all_relations.extend(result.relations)

        except Exception as exc:
            failed += 1
            logger.error(f"Errore estrazione per {summary.paper_id}: {exc}")
            all_results.append(ExtractionResult(
                paper_id=summary.paper_id,
                bibtex_key=summary.bibtex_key,
                error=str(exc),
                extracted_at=datetime.now(),
            ))
            if not skip_errors:
                raise

    # Merge se richiesto
    if merge_results:
        all_entities = merge_entities(all_entities)
        all_relations = merge_relations(all_relations)

    logger.info(
        f"Batch extraction completata: {success}/{len(l1_summaries)} successi, "
        f"{failed} falliti. "
        f"Totale: {len(all_entities)} entità, {len(all_relations)} relazioni"
        f"{' (merged)' if merge_results else ''}"
    )

    return all_entities, all_relations, all_results


# ============================================================================
# Internal implementation
# ============================================================================

def _extract_with_result(
    l1_summary: L1Summary,
    *,
    llm=None,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
) -> ExtractionResult:
    """Implementazione interna dell'estrazione."""
    start_time = time.monotonic()

    paper_id = l1_summary.paper_id
    bibtex_key = l1_summary.bibtex_key or "unknown"

    logger.info(f"Estrazione entità/relazioni per: {bibtex_key} ({paper_id})")

    # Lazy init LLM
    # Flash thinking model usa token per ragionamento interno → servono 65K
    if llm is None:
        llm = get_flash_llm(timeout_seconds=120, max_tokens=65536)

    # Costruisci prompt
    prompt = _build_prompt(l1_summary)

    # Chiamata LLM con retry
    try:
        raw_response = llm_call_with_retry(
            llm,
            prompt,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(f"LLM call fallita per {bibtex_key}: {exc}")
        return ExtractionResult(
            paper_id=paper_id,
            bibtex_key=bibtex_key,
            error=f"LLM call failed: {exc}",
            extracted_at=datetime.now(),
            extraction_time_seconds=elapsed,
        )

    # Parse JSON response
    parsed = _parse_json_response(raw_response, bibtex_key)
    if parsed is None:
        elapsed = time.monotonic() - start_time
        return ExtractionResult(
            paper_id=paper_id,
            bibtex_key=bibtex_key,
            error="Failed to parse JSON response",
            extracted_at=datetime.now(),
            extraction_time_seconds=elapsed,
            model_used=_get_model_name(llm),
        )

    # Costruisci oggetti Entity e Relation
    entities = _build_entities(parsed.get("entities", []), l1_summary)
    relations = _build_relations(parsed.get("relations", []), entities, l1_summary)

    # Validazione coerenza: ogni entità in una relazione deve esistere
    entity_names = {e.canonical_name.lower() for e in entities}
    valid_relations = []
    for rel in relations:
        if rel.source.lower() in entity_names and rel.target.lower() in entity_names:
            valid_relations.append(rel)
        else:
            missing = []
            if rel.source.lower() not in entity_names:
                missing.append(f"source={rel.source}")
            if rel.target.lower() not in entity_names:
                missing.append(f"target={rel.target}")
            logger.debug(
                f"Relazione scartata (entità mancanti: {', '.join(missing)}): "
                f"{rel.source} --[{rel.type}]--> {rel.target}"
            )

    elapsed = time.monotonic() - start_time
    logger.info(
        f"Estrazione completata per {bibtex_key}: "
        f"{len(entities)} entità, {len(valid_relations)} relazioni ({elapsed:.1f}s)"
    )

    return ExtractionResult(
        paper_id=paper_id,
        bibtex_key=bibtex_key,
        entities=entities,
        relations=valid_relations,
        extracted_at=datetime.now(),
        model_used=_get_model_name(llm),
        extraction_time_seconds=elapsed,
    )


# ============================================================================
# Prompt builder
# ============================================================================

def _build_prompt(l1: L1Summary) -> str:
    """Costruisce il prompt per l'estrazione da un L1 summary."""
    # Titolo: prova a inferirlo dal bibtex_key
    title_hint = ""
    if l1.bibtex_key:
        # "Zou_2023_GCG" → "Zou 2023 GCG"
        parts = l1.bibtex_key.replace("_", " ")
        title_hint = parts

    key_contributions = "\n".join(
        f"- {c}" for c in l1.key_contributions
    ) if l1.key_contributions else "Not available."

    return _EXTRACTION_PROMPT_TEMPLATE.format(
        paper_id=l1.paper_id,
        bibtex_key=l1.bibtex_key or "unknown",
        title_hint=title_hint,
        category=l1.category or "unknown",
        year=l1.year or "unknown",
        key_contributions=key_contributions,
        methodology=l1.methodology or "Not available.",
        testing_methodology=l1.testing_methodology or "Not available.",
        temporal_context=l1.temporal_context or "Not available.",
        results_summary=l1.results_summary or "Not available.",
        limitations=l1.limitations or "Not available.",
    )


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_json_response(raw_response: str, context: str = "") -> dict | None:
    """Parsing robusto della risposta JSON dal LLM.

    Gestisce:
    - JSON pulito
    - JSON con code fences (```json ... ```)
    - Trailing commas
    - Testo extra prima/dopo il JSON
    - Parentesi sbilanciate

    Args:
        raw_response: Risposta raw dal LLM.
        context: Contesto per logging (es. bibtex_key).

    Returns:
        Dict con "entities" e "relations", o None se parsing fallito.
    """
    if not raw_response or not raw_response.strip():
        logger.warning(f"Risposta vuota dal LLM ({context})")
        return None

    text = raw_response.strip()

    # 1. Rimuovi code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 2. Estrai il primo oggetto JSON {...}
    brace_start = text.find("{")
    if brace_start == -1:
        logger.warning(f"Nessun oggetto JSON trovato nella risposta ({context})")
        return None

    # Trova la parentesi di chiusura corrispondente
    depth = 0
    brace_end = -1
    in_string = False
    escape_next = False

    for i in range(brace_start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break

    if brace_end == -1:
        logger.warning(f"Parentesi JSON sbilanciate ({context})")
        # Prova ad aggiungere la chiusura
        text = text[brace_start:] + "}" * depth
    else:
        text = text[brace_start:brace_end + 1]

    # 3. Fix trailing commas prima di ] e }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 4. Prova parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        logger.warning(f"JSON non è un dict ({context}): {type(data)}")
        return None
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error ({context}): {exc}")

    # 5. Fallback: regex per estrarre array entities e relations
    return _regex_fallback_parse(raw_response, context)


def _regex_fallback_parse(raw_response: str, context: str) -> dict | None:
    """Fallback regex-based per estrarre entità e relazioni.

    Cerca pattern come "entities": [...] e "relations": [...].
    """
    result: dict[str, list] = {"entities": [], "relations": []}

    for key in ("entities", "relations"):
        pattern = rf'"{key}"\s*:\s*\['
        match = re.search(pattern, raw_response)
        if not match:
            continue

        # Trova la chiusura del bracket
        start = match.end() - 1  # include il [
        depth = 0
        end = -1
        in_str = False
        esc = False

        for i in range(start, len(raw_response)):
            c = raw_response[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end != -1:
            array_text = raw_response[start:end + 1]
            # Fix trailing commas
            array_text = re.sub(r",\s*([}\]])", r"\1", array_text)
            try:
                arr = json.loads(array_text)
                if isinstance(arr, list):
                    result[key] = arr
            except json.JSONDecodeError:
                logger.debug(f"Regex fallback: impossibile parsare {key} ({context})")

    if result["entities"] or result["relations"]:
        logger.info(f"Regex fallback riuscito ({context}): {len(result['entities'])} entities, {len(result['relations'])} relations")
        return result

    logger.warning(f"Tutti i metodi di parsing JSON falliti ({context})")
    return None


# ============================================================================
# Entity & Relation builders
# ============================================================================

_VALID_ENTITY_TYPES = {t.value for t in EntityType}
_VALID_RELATION_TYPES = {t.value for t in RelationType}


def _build_entities(
    raw_entities: list[dict],
    l1: L1Summary,
) -> list[Entity]:
    """Costruisce oggetti Entity normalizzati da una lista di dict raw."""
    entities: list[Entity] = []

    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue

        # Tipo
        raw_type = raw.get("type", "")
        if raw_type not in _VALID_ENTITY_TYPES:
            # Prova case-insensitive
            for vt in _VALID_ENTITY_TYPES:
                if raw_type.lower() == vt.lower():
                    raw_type = vt
                    break
            else:
                logger.debug(f"Entity type non valido: {raw_type}")
                continue

        # Nome
        raw_name = raw.get("name", "").strip()
        if not raw_name:
            continue

        # Normalizza
        canonical = normalize_entity_name(raw_name)
        if not canonical:
            continue

        # Aliases
        aliases = raw.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
        # Aggiungi il nome originale se diverso dal canonico
        if raw_name != canonical and raw_name not in aliases:
            aliases.append(raw_name)
        # Rimuovi il canonical dalle aliases
        aliases = [a for a in aliases if a != canonical]

        entity = Entity(
            type=EntityType(raw_type),
            canonical_name=canonical,
            aliases=aliases,
            description=raw.get("description", ""),
            year=_safe_int(raw.get("year")),
            source_paper_id=l1.paper_id,
            source_bibtex_key=l1.bibtex_key,
        )
        entities.append(entity)

    # Dedup interno al paper
    entities = merge_entities(entities)
    return entities


def _build_relations(
    raw_relations: list[dict],
    entities: list[Entity],
    l1: L1Summary,
) -> list[Relation]:
    """Costruisce oggetti Relation da una lista di dict raw.

    Normalizza i nomi delle entità sorgente/target per garantire
    che corrispondano alle entità estratte.
    """
    # Mappa entità per lookup rapido
    entity_lookup: dict[str, Entity] = {}
    for e in entities:
        entity_lookup[e.canonical_name.lower()] = e
        for alias in e.aliases:
            entity_lookup[alias.lower()] = e

    relations: list[Relation] = []

    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue

        # Tipo
        raw_type = raw.get("type", "")
        if raw_type not in _VALID_RELATION_TYPES:
            # Prova case-insensitive
            for vt in _VALID_RELATION_TYPES:
                if raw_type.lower() == vt.lower():
                    raw_type = vt
                    break
            else:
                logger.debug(f"Relation type non valido: {raw_type}")
                continue

        # Source e target
        raw_source = raw.get("source", "").strip()
        raw_target = raw.get("target", "").strip()
        if not raw_source or not raw_target:
            continue

        # Normalizza source e target
        source_name = _resolve_entity_name(raw_source, entity_lookup)
        target_name = _resolve_entity_name(raw_target, entity_lookup)

        # Source e target type
        raw_source_type = raw.get("source_type", "")
        raw_target_type = raw.get("target_type", "")

        # Risolvi source_type
        source_entity = entity_lookup.get(source_name.lower())
        if source_entity:
            source_type = source_entity.type
        elif raw_source_type in _VALID_ENTITY_TYPES:
            source_type = EntityType(raw_source_type)
        else:
            logger.debug(f"Source type non risolvibile per: {raw_source}")
            continue

        # Risolvi target_type
        target_entity = entity_lookup.get(target_name.lower())
        if target_entity:
            target_type = target_entity.type
        elif raw_target_type in _VALID_ENTITY_TYPES:
            target_type = EntityType(raw_target_type)
        else:
            logger.debug(f"Target type non risolvibile per: {raw_target}")
            continue

        # Properties opzionali
        props = {}
        if "metric" in raw:
            props["metric"] = raw["metric"]
        if "percentage" in raw:
            props["percentage"] = raw["percentage"]
        if "score" in raw:
            props["score"] = raw["score"]

        relation = Relation(
            type=RelationType(raw_type),
            source=source_name,
            source_type=source_type,
            target=target_name,
            target_type=target_type,
            description=raw.get("description", ""),
            year=_safe_int(raw.get("year")),
            properties=props,
            source_paper_id=l1.paper_id,
            source_bibtex_key=l1.bibtex_key,
        )
        relations.append(relation)

    # Dedup interno
    relations = merge_relations(relations)
    return relations


def _resolve_entity_name(raw_name: str, entity_lookup: dict[str, Entity]) -> str:
    """Risolvi un nome da relazione al canonical name dell'entità.

    Cerca prima nel lookup delle entità estratte, poi normalizza.
    """
    key = raw_name.lower().strip()

    # 1. Lookup diretto
    if key in entity_lookup:
        return entity_lookup[key].canonical_name

    # 2. Normalizza e riprova
    normalized = normalize_entity_name(raw_name)
    norm_key = normalized.lower()
    if norm_key in entity_lookup:
        return entity_lookup[norm_key].canonical_name

    # 3. Partial match: se il nome è contenuto in un'entità nota
    for entity_key, entity in entity_lookup.items():
        if key in entity_key or entity_key in key:
            return entity.canonical_name

    # 4. Ritorna il nome normalizzato come fallback
    return normalized


# ============================================================================
# Helpers
# ============================================================================

def _safe_int(value: Any) -> int | None:
    """Converte un valore in int, ritornando None se non possibile."""
    if value is None:
        return None
    try:
        n = int(value)
        # Sanity check per anni
        if 1900 <= n <= 2100:
            return n
        # Potrebbe essere un altro numero (non un anno)
        return n
    except (ValueError, TypeError):
        return None


def _get_model_name(llm) -> str:
    """Estrai il nome del modello da un'istanza LLM."""
    try:
        return getattr(llm, "model", "unknown")
    except Exception:
        return "unknown"


def get_extraction_stats(
    entities: list[Entity],
    relations: list[Relation],
) -> dict[str, Any]:
    """Calcola statistiche aggregate su entità e relazioni estratte.

    Args:
        entities: Lista di entità.
        relations: Lista di relazioni.

    Returns:
        Dict con statistiche.
    """
    entity_types: dict[str, int] = {}
    for e in entities:
        t = e.type.value if isinstance(e.type, EntityType) else e.type
        entity_types[t] = entity_types.get(t, 0) + 1

    relation_types: dict[str, int] = {}
    for r in relations:
        t = r.type.value if isinstance(r.type, RelationType) else r.type
        relation_types[t] = relation_types.get(t, 0) + 1

    # Papers sorgente unici
    source_papers = set()
    for e in entities:
        if e.source_paper_id:
            source_papers.add(e.source_paper_id)
    for r in relations:
        if r.source_paper_id:
            source_papers.add(r.source_paper_id)

    return {
        "total_entities": len(entities),
        "total_relations": len(relations),
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "unique_source_papers": len(source_papers),
        "avg_aliases_per_entity": (
            sum(len(e.aliases) for e in entities) / len(entities)
            if entities else 0.0
        ),
    }


def save_extraction_results(
    entities: list[Entity],
    relations: list[Relation],
    output_path: str | None = None,
) -> str:
    """Salva il risultato dell'estrazione in un file JSON.

    Args:
        entities: Entità estratte.
        relations: Relazioni estratte.
        output_path: Percorso file di output. Default: ``output/graphrag_extraction.json``.

    Returns:
        Percorso del file salvato.
    """
    from pathlib import Path

    if output_path is None:
        from paperhunter.config import settings
        output_path = str(Path(settings.output.output_dir) / "graphrag_extraction.json")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "extracted_at": datetime.now().isoformat(),
        "stats": get_extraction_stats(entities, relations),
        "entities": [e.to_dict() for e in entities],
        "relations": [r.to_dict() for r in relations],
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Risultati estrazione salvati in: {path}")
    return str(path)
