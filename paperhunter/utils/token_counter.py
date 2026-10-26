"""
Utility centralizzata per conteggio e gestione del budget token.

Fornisce funzioni per stimare i token di un testo, calcolare il budget
disponibile per le varie fasi della pipeline (L1, L2, Writer), e
supportare decisioni budget-aware nei moduli a valle.

Il conteggio usa un'approssimazione di ~4 caratteri per token (standard
per i modelli Gemini/GPT), con possibilità di calibrazione.

Uso::

    from paperhunter.utils.token_counter import (
        estimate_tokens,
        get_max_context_tokens,
        chars_budget_for_n_papers,
        compute_l1_target_chars,
        compute_l2_budget,
        compute_writer_budget,
        BudgetReport,
    )

    tokens = estimate_tokens("Testo di esempio lungo...")
    max_ctx = get_max_context_tokens()
    l1_chars = compute_l1_target_chars(n_papers=100)
    report = BudgetReport.from_config()
    report.print_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Rapporto medio caratteri-per-token per modelli Gemini/GPT.
# Gemini usa SentencePiece; ~4 char/token è un'approssimazione conservativa
# per testo inglese accademico (con formule, citazioni, ecc.).
CHARS_PER_TOKEN: float = 4.0

# Overhead stimato in token per i template di prompt (istruzioni,
# system message, formattazione JSON, ecc.) — riservati e non usabili
# per il contenuto effettivo dei paper.
PROMPT_OVERHEAD_TOKENS: int = 2_000

# Soglie di paper per categoria per determinare il livello di compressione L1.
# Derivate empiricamente: N ≤ 30 → standard, 30 < N ≤ 70 → concise, N > 70 → compressed.
L1_TIER_STANDARD_MAX: int = 30
L1_TIER_CONCISE_MAX: int = 70


# ---------------------------------------------------------------------------
# Funzioni base
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Stima il numero di token per un testo.

    Args:
        text: Il testo da stimare.
        chars_per_token: Rapporto caratteri/token (default: 4.0).

    Returns:
        Numero stimato di token (arrotondato per eccesso).
    """
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def estimate_tokens_for_messages(
    messages: list[dict[str, str]],
    chars_per_token: float = CHARS_PER_TOKEN,
    per_message_overhead: int = 4,
) -> int:
    """Stima i token per una lista di messaggi LLM (role + content).

    Ogni messaggio ha un overhead fisso (per role tag, separatori, ecc.)
    oltre al contenuto testuale.

    Args:
        messages: Lista di dict con chiave ``role`` e ``content``.
        chars_per_token: Rapporto caratteri/token.
        per_message_overhead: Token di overhead per ogni messaggio.

    Returns:
        Numero totale stimato di token.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content, chars_per_token) + per_message_overhead
    # Overhead finale per il reply priming
    total += 3
    return total


def tokens_to_chars(tokens: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Converte un budget in token nel corrispondente budget in caratteri.

    Args:
        tokens: Numero di token.
        chars_per_token: Rapporto caratteri/token.

    Returns:
        Numero di caratteri corrispondente.
    """
    return int(tokens * chars_per_token)


def chars_to_tokens(chars: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Converte un budget in caratteri nel corrispondente budget in token.

    Args:
        chars: Numero di caratteri.
        chars_per_token: Rapporto caratteri/token.

    Returns:
        Numero stimato di token (arrotondato per eccesso).
    """
    return math.ceil(chars / chars_per_token)


# ---------------------------------------------------------------------------
# Lettura config
# ---------------------------------------------------------------------------

def get_max_context_tokens() -> int:
    """Legge ``summarization.max_context_tokens`` da config.yaml.

    Returns:
        Budget massimo di token per il context window.
    """
    from paperhunter.config import settings
    return settings.summarization.max_context_tokens


def get_papers_per_category() -> int:
    """Legge ``ingestion.papers_per_category`` da config.yaml.

    Returns:
        Numero target di paper per categoria.
    """
    from paperhunter.config import settings
    return settings.ingestion.papers_per_category


def get_writer_max_context_chars() -> int:
    """Legge ``agents.writer_max_context_chars`` da config.yaml.

    Returns:
        Budget massimo in caratteri per il contesto del writer agent.
    """
    from paperhunter.config import settings
    return settings.agents.writer_max_context_chars


def get_writer_budget_mode() -> str:
    """Legge ``agents.writer_budget_mode`` da config.yaml.

    Returns:
        Modalità budget writer: ``"fixed"`` o ``"proportional"``.
    """
    from paperhunter.config import settings
    return settings.agents.writer_budget_mode


# ---------------------------------------------------------------------------
# Budget L1
# ---------------------------------------------------------------------------

def compute_l1_target_chars(
    n_papers: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    l1_min_chars: Optional[int] = None,
) -> int:
    """Calcola il target di caratteri per un singolo L1 summary.

    La logica: tutti gli L1 di una categoria vengono concatenati nel prompt
    L2. Per restare entro il budget token, dividiamo il budget disponibile
    (al netto dell'overhead prompt) per il numero di paper.

    Args:
        n_papers: Numero di paper nella categoria. Se None, legge
            ``ingestion.papers_per_category`` da config.
        max_context_tokens: Budget max token. Se None, legge da config.
        l1_min_chars: Floor in caratteri. Se None, legge da config.

    Returns:
        Target in caratteri per un singolo L1 summary.
    """
    from paperhunter.config import settings as cfg

    if n_papers is None:
        n_papers = cfg.ingestion.papers_per_category
    if max_context_tokens is None:
        max_context_tokens = cfg.summarization.max_context_tokens
    if l1_min_chars is None:
        l1_min_chars = cfg.summarization.l1_min_chars

    if n_papers <= 0:
        n_papers = 1

    # Budget disponibile in chars per il contenuto L1 aggregate
    available_tokens = max_context_tokens - PROMPT_OVERHEAD_TOKENS
    available_chars = tokens_to_chars(available_tokens)

    # Chars per paper
    target_chars = available_chars // n_papers

    # Applica il floor
    target_chars = max(target_chars, l1_min_chars)

    return target_chars


def get_l1_compression_tier(n_papers: Optional[int] = None) -> str:
    """Determina il livello di compressione per gli L1 summary.

    Args:
        n_papers: Numero di paper nella categoria. Se None, legge da config.

    Returns:
        Stringa: ``"standard"``, ``"concise"`` o ``"compressed"``.
    """
    if n_papers is None:
        n_papers = get_papers_per_category()

    if n_papers <= L1_TIER_STANDARD_MAX:
        return "standard"
    elif n_papers <= L1_TIER_CONCISE_MAX:
        return "concise"
    else:
        return "compressed"


# ---------------------------------------------------------------------------
# Budget L2
# ---------------------------------------------------------------------------

def compute_l2_budget(
    max_context_tokens: Optional[int] = None,
    prompt_overhead_tokens: int = PROMPT_OVERHEAD_TOKENS,
) -> dict[str, int]:
    """Calcola il budget disponibile per il prompt L2.

    Returns:
        Dict con:
        - ``max_context_tokens``: budget totale da config
        - ``prompt_overhead_tokens``: token riservati al template
        - ``available_tokens``: token per il contenuto (L1 summaries)
        - ``available_chars``: corrispettivo in caratteri
    """
    if max_context_tokens is None:
        max_context_tokens = get_max_context_tokens()

    available_tokens = max_context_tokens - prompt_overhead_tokens
    available_chars = tokens_to_chars(available_tokens)

    return {
        "max_context_tokens": max_context_tokens,
        "prompt_overhead_tokens": prompt_overhead_tokens,
        "available_tokens": available_tokens,
        "available_chars": available_chars,
    }


# ---------------------------------------------------------------------------
# Budget Writer
# ---------------------------------------------------------------------------

def compute_writer_budget(
    n_sections: int,
    writer_max_context_chars: Optional[int] = None,
    writer_budget_mode: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
) -> dict[str, int | str]:
    """Calcola il budget per sezione del writer agent.

    Args:
        n_sections: Numero di sezioni del paper.
        writer_max_context_chars: Limite fisso in chars. Se None, da config.
        writer_budget_mode: ``"fixed"`` o ``"proportional"``. Se None, da config.
        max_context_tokens: Budget token globale. Se None, da config.

    Returns:
        Dict con:
        - ``mode``: modalità budget applicata
        - ``total_budget_chars``: budget totale in chars
        - ``per_section_chars``: budget per singola sezione
        - ``n_sections``: numero di sezioni
    """
    if writer_max_context_chars is None:
        writer_max_context_chars = get_writer_max_context_chars()
    if writer_budget_mode is None:
        writer_budget_mode = get_writer_budget_mode()
    if max_context_tokens is None:
        max_context_tokens = get_max_context_tokens()

    if n_sections <= 0:
        n_sections = 1

    if writer_budget_mode == "proportional":
        # Divide il budget token globale (meno overhead) per il numero di sezioni
        available_tokens = max_context_tokens - PROMPT_OVERHEAD_TOKENS
        total_chars = tokens_to_chars(available_tokens)
        per_section_chars = total_chars // n_sections
        # Cap al max configurato
        per_section_chars = min(per_section_chars, writer_max_context_chars)
    else:
        # Modalità "fixed": ogni sezione ha lo stesso budget fisso
        total_chars = writer_max_context_chars * n_sections
        per_section_chars = writer_max_context_chars

    return {
        "mode": writer_budget_mode,
        "total_budget_chars": total_chars,
        "per_section_chars": per_section_chars,
        "n_sections": n_sections,
    }


# ---------------------------------------------------------------------------
# Utility generica: chars_budget_for_n_papers
# ---------------------------------------------------------------------------

def chars_budget_for_n_papers(
    n_papers: int,
    max_context_tokens: Optional[int] = None,
    prompt_overhead_tokens: int = PROMPT_OVERHEAD_TOKENS,
) -> int:
    """Calcola il budget totale in caratteri disponibile per N paper.

    Questo è il budget aggregato (non per-paper) per una fase che
    processa N paper in un singolo prompt (es. L2 summarization).

    Args:
        n_papers: Numero di paper.
        max_context_tokens: Budget max. Se None, da config.
        prompt_overhead_tokens: Token riservati.

    Returns:
        Budget totale in caratteri per l'insieme di N paper.
    """
    if max_context_tokens is None:
        max_context_tokens = get_max_context_tokens()

    available_tokens = max_context_tokens - prompt_overhead_tokens
    return tokens_to_chars(available_tokens)


# ---------------------------------------------------------------------------
# Budget Report (per la pipeline)
# ---------------------------------------------------------------------------

@dataclass
class BudgetReport:
    """Report di budget stimato per la pipeline agentica.

    Calcola e presenta una stima dei token necessari per ogni fase,
    basandosi sulla configurazione corrente e sul numero di paper.
    """

    n_categories: int = 0
    papers_per_category: int = 0
    total_papers: int = 0
    max_context_tokens: int = 0

    # Stime per fase
    l1_chars_per_paper: int = 0
    l2_total_chars_per_category: int = 0
    writer_chars_per_section: int = 0
    n_sections_estimate: int = 0

    # Token stimati
    l1_tokens_per_call: int = 0
    l2_tokens_per_call: int = 0
    writer_tokens_per_call: int = 0
    critic_tokens_per_call: int = 0

    # Stato
    within_budget: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        n_categories: int = 4,
        n_sections_estimate: int = 8,
    ) -> BudgetReport:
        """Costruisce un report di budget dalla configurazione corrente.

        Args:
            n_categories: Numero di categorie (default: 4).
            n_sections_estimate: Stima sezioni nel paper finale.

        Returns:
            Istanza di BudgetReport con tutte le stime.
        """
        from paperhunter.config import settings as cfg

        ppc = cfg.ingestion.papers_per_category
        max_ctx = cfg.summarization.max_context_tokens

        report = cls(
            n_categories=n_categories,
            papers_per_category=ppc,
            total_papers=ppc * n_categories,
            max_context_tokens=max_ctx,
            n_sections_estimate=n_sections_estimate,
        )

        # --- L1: ogni L1 processa 1 paper (non è un problema di budget) ---
        # Il paper originale viene troncato a ~50k chars se necessario.
        # Il summary L1 target dipende da N per la fase L2 a valle.
        report.l1_chars_per_paper = compute_l1_target_chars(n_papers=ppc)
        # Token per una singola chiamata L1 (paper chunks + overhead)
        report.l1_tokens_per_call = chars_to_tokens(50_000) + PROMPT_OVERHEAD_TOKENS

        # --- L2: concatena tutti gli L1 di una categoria ---
        report.l2_total_chars_per_category = report.l1_chars_per_paper * ppc
        report.l2_tokens_per_call = chars_to_tokens(
            report.l2_total_chars_per_category
        ) + PROMPT_OVERHEAD_TOKENS

        # --- Writer: un prompt per sezione ---
        writer_budget = compute_writer_budget(n_sections=n_sections_estimate)
        report.writer_chars_per_section = writer_budget["per_section_chars"]
        report.writer_tokens_per_call = chars_to_tokens(
            report.writer_chars_per_section
        ) + PROMPT_OVERHEAD_TOKENS

        # --- Critic: verifica la sezione (simile al writer) ---
        # Il critic riceve la sezione scritta + le fonti citate → ~stesso budget
        report.critic_tokens_per_call = report.writer_tokens_per_call

        # --- Verifica budget ---
        report.warnings = []

        if report.l2_tokens_per_call > max_ctx:
            report.within_budget = False
            report.warnings.append(
                f"L2 summary: {report.l2_tokens_per_call:,} token stimati > "
                f"budget {max_ctx:,}. Necessario ridurre papers_per_category "
                f"o abilitare compressione L1 più aggressiva."
            )

        if report.writer_tokens_per_call > max_ctx:
            report.within_budget = False
            report.warnings.append(
                f"Writer: {report.writer_tokens_per_call:,} token stimati > "
                f"budget {max_ctx:,}. Ridurre writer_max_context_chars o "
                f"il numero di sezioni."
            )

        return report

    def to_dict(self) -> dict:
        """Restituisce il report come dizionario."""
        return {
            "n_categories": self.n_categories,
            "papers_per_category": self.papers_per_category,
            "total_papers": self.total_papers,
            "max_context_tokens": self.max_context_tokens,
            "l1_chars_per_paper": self.l1_chars_per_paper,
            "l1_tokens_per_call": self.l1_tokens_per_call,
            "l2_total_chars_per_category": self.l2_total_chars_per_category,
            "l2_tokens_per_call": self.l2_tokens_per_call,
            "writer_chars_per_section": self.writer_chars_per_section,
            "writer_tokens_per_call": self.writer_tokens_per_call,
            "critic_tokens_per_call": self.critic_tokens_per_call,
            "n_sections_estimate": self.n_sections_estimate,
            "within_budget": self.within_budget,
            "warnings": self.warnings,
        }

    def _pct(self, tokens_per_call: int) -> str:
        """Calcola la percentuale del budget usata da una singola chiamata."""
        if self.max_context_tokens <= 0:
            return "N/A"
        pct = tokens_per_call / self.max_context_tokens * 100
        return f"{pct:.1f}%"

    def print_report(self) -> None:
        """Stampa il report formattato con rich."""
        from paperhunter.utils.logging_setup import (
            print_table, print_panel, print_success, print_warning,
        )

        # Tabella principale con colonna % Budget
        rows = [
            {
                "Fase": "L1 Summarization",
                "Token/call": f"{self.l1_tokens_per_call:,}",
                "% Budget": self._pct(self.l1_tokens_per_call),
                "Note": f"~{self.l1_chars_per_paper:,} chars/L1 target",
            },
            {
                "Fase": "L2 Summarization",
                "Token/call": f"{self.l2_tokens_per_call:,}",
                "% Budget": self._pct(self.l2_tokens_per_call),
                "Note": f"{self.papers_per_category} L1 × "
                        f"{self.l1_chars_per_paper:,} chars",
            },
            {
                "Fase": "Writer (per sezione)",
                "Token/call": f"{self.writer_tokens_per_call:,}",
                "% Budget": self._pct(self.writer_tokens_per_call),
                "Note": f"{self.writer_chars_per_section:,} chars/section, "
                        f"{self.n_sections_estimate} sezioni",
            },
            {
                "Fase": "Critic (per sezione)",
                "Token/call": f"{self.critic_tokens_per_call:,}",
                "% Budget": self._pct(self.critic_tokens_per_call),
                "Note": "~= Writer budget",
            },
        ]

        print_panel(
            "Budget Token Pipeline",
            f"Paper: {self.total_papers} totali "
            f"({self.n_categories} cat × {self.papers_per_category}/cat)\n"
            f"Context window max: {self.max_context_tokens:,} token\n"
            f"Stato: {'✅ ENTRO BUDGET' if self.within_budget else '⚠️ FUORI BUDGET'}",
        )
        print_table("Stima Token per Fase", rows)

        if self.within_budget and not self.warnings:
            print_success(
                "Tutte le fasi rientrano nel budget token. "
                "La pipeline può procedere."
            )

        for w in self.warnings:
            print_warning(w)

        # Suggerimenti se fuori budget
        if not self.within_budget:
            suggestions = self._get_suggestions()
            if suggestions:
                from paperhunter.utils.logging_setup import console
                console.print("\n[bold yellow]Suggerimenti:[/bold yellow]")
                for i, s in enumerate(suggestions, 1):
                    console.print(f"  {i}. {s}")

    def _get_suggestions(self) -> list[str]:
        """Genera suggerimenti per rientrare nel budget."""
        suggestions: list[str] = []
        if self.l2_tokens_per_call > self.max_context_tokens:
            suggestions.append(
                f"Ridurre papers_per_category (attuale: {self.papers_per_category}) "
                f"— con {self.max_context_tokens:,} token, il massimo consigliato è "
                f"~{(self.max_context_tokens - PROMPT_OVERHEAD_TOKENS) * int(CHARS_PER_TOKEN) // max(self.l1_chars_per_paper, 1)}"
            )
            suggestions.append(
                "Abilitare compressione L1 più aggressiva "
                "(l1_target_chars_auto: true in config.yaml)"
            )
        if self.writer_tokens_per_call > self.max_context_tokens:
            suggestions.append(
                f"Ridurre writer_max_context_chars (attuale: dalle config) "
                f"o aumentare il numero di sezioni"
            )
        return suggestions
