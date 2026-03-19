"""
VenueRanker — Tier dictionary e boost score per venue accademiche.

Assegna un tier di autorevolezza a ciascuna venue/journal e restituisce
un boost (positivo o negativo) da sommare allo score RRF nel HybridRetriever.

Tier:
  1 — Conferenze e journal top (NeurIPS, ICML, ICLR, ACL, IEEE S&P, ...)
  2 — Workshop ufficiali, conferenze di secondo livello (AAAI, IJCAI, findings of ACL, ...)
  3 — Preprint non revisionati (arXiv, biorXiv, ...)
  0 — Sconosciuta / non classificata

Uso:
    from paperhunter.indexing.venue_ranker import VenueRanker

    ranker = VenueRanker()
    tier  = ranker.get_tier("NeurIPS 2023")   # → 1
    boost = ranker.get_boost("NeurIPS 2023")  # → +0.15
    boost = ranker.get_boost("arXiv")         # → -0.05
    boost = ranker.get_boost(None)            # → 0.0
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier sets — matching case-insensitive e substring-based
# ---------------------------------------------------------------------------

# Tier 1 — Conferenze flagship e journal top nel campo ML/NLP/Security
TIER1_VENUES: frozenset[str] = frozenset({
    # Machine Learning
    "neurips", "nips", "icml", "iclr",
    # NLP
    "acl", "emnlp", "naacl", "coling", "tacl",
    # Computer Vision
    "cvpr", "iccv", "eccv",
    # Security
    "ieee s&p", "usenix security", "usenix", "ccs", "acm ccs", "ndss", "sp",
    # Data Mining / AI
    "kdd", "sigkdd", "aaai",
    # Journals top
    "nature", "science", "cell", "jmlr", "ieee tifs",
    "transactions on information forensics",
    "ieee transactions on neural networks",
    "journal of machine learning research",
})

# Tier 2 — Workshop ufficiali, conferenze di secondo livello, findings
TIER2_VENUES: frozenset[str] = frozenset({
    # Findings
    "findings of acl", "findings of emnlp", "findings of naacl",
    # Conferenze secondarie
    "ijcai", "www", "wsdm", "aistats", "ecml", "pakdd", "sigir",
    # Workshop qualificati
    "workshop", "trustworthy ml", "socially responsible ml",
    "red teaming genai", "genbench", "gem",
    # Journals secondari
    "pattern recognition", "information sciences", "expert systems",
    "neurocomputing", "ieee access",
})

# Tier 3 — Preprint non revisionati
TIER3_VENUES: frozenset[str] = frozenset({
    "arxiv", "biorxiv", "medrxiv", "preprint", "ssrn", "techrxiv",
    "openreview", "unpublished",
})


class VenueRanker:
    """Assegna tier e boost per venue/journal accademiche.

    I boost possono essere configurati tramite il costruttore oppure
    caricati automaticamente da ``paperhunter.config.settings`` se
    disponibile.

    Args:
        tier1_boost:    Boost per Tier 1 (default +0.15)
        tier2_boost:    Boost per Tier 2 (default +0.05)
        tier3_penalty:  Penalità per Tier 3 (valore positivo, sarà sottratto; default 0.05)
    """

    def __init__(
        self,
        tier1_boost: float = 0.15,
        tier2_boost: float = 0.05,
        tier3_penalty: float = 0.05,
    ) -> None:
        self._tier1_boost = tier1_boost
        self._tier2_boost = tier2_boost
        self._tier3_penalty = tier3_penalty

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def get_tier(self, venue: Optional[str]) -> int:
        """Restituisce il tier di autorevolezza della venue.

        Args:
            venue: Stringa della venue (es. "NeurIPS 2023", "arXiv", None).

        Returns:
            1 = top, 2 = medium, 3 = preprint, 0 = sconosciuta/None.
        """
        if not venue:
            return 0

        normalized = venue.lower().strip()

        # Tier 2 si controlla PRIMA di Tier 1: venue come "Findings of ACL" o
        # "Workshop at ICML" contengono alias Tier 1 ma sono Tier 2 per loro natura.
        if self._matches_tier(normalized, TIER2_VENUES):
            return 2
        if self._matches_tier(normalized, TIER1_VENUES):
            return 1
        if self._matches_tier(normalized, TIER3_VENUES):
            return 3
        return 0

    def get_boost(self, venue: Optional[str]) -> float:
        """Restituisce il boost da sommare allo score RRF.

        Args:
            venue: Stringa della venue (es. "NeurIPS 2023", "arXiv", None).

        Returns:
            Float boost: positivo per venue autorevoli, negativo per preprint,
            0.0 per venue sconosciute o None.
        """
        tier = self.get_tier(venue)
        if tier == 1:
            return self._tier1_boost
        if tier == 2:
            return self._tier2_boost
        if tier == 3:
            return -self._tier3_penalty
        return 0.0

    # ------------------------------------------------------------------
    # Helper privato
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_tier(normalized_venue: str, tier_set: frozenset[str]) -> bool:
        """Verifica se la venue normalizzata matcha (substring) uno degli alias del tier."""
        for alias in tier_set:
            if alias in normalized_venue:
                return True
        return False


# ---------------------------------------------------------------------------
# Factory function — carica la configurazione da settings se disponibile
# ---------------------------------------------------------------------------

def create_venue_ranker() -> VenueRanker:
    """Crea un VenueRanker con parametri dal config.yaml (se disponibile).

    Legge ``settings.retrieval.venue_tier1_boost``, ``venue_tier2_boost``
    e ``venue_tier3_penalty`` se i campi esistono; altrimenti usa i default.

    Returns:
        Istanza configurata di VenueRanker.
    """
    try:
        from paperhunter.config import settings  # import locale per evitare circolarità

        retrieval = settings.retrieval
        tier1_boost = getattr(retrieval, "venue_tier1_boost", 0.15)
        tier2_boost = getattr(retrieval, "venue_tier2_boost", 0.05)
        tier3_penalty = getattr(retrieval, "venue_tier3_penalty", 0.05)
        return VenueRanker(
            tier1_boost=tier1_boost,
            tier2_boost=tier2_boost,
            tier3_penalty=tier3_penalty,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Impossibile caricare VenueRanker da settings: %s. Uso default.", exc)
        return VenueRanker()
