"""
Test completo per VenueRanker — Task 4.1 / 4.3 TODO6_MetadataRankingCitations.md

Copre 12 test unitari su get_tier() e get_boost() + invarianti di score.
"""

from __future__ import annotations

import pytest

from paperhunter.indexing.venue_ranker import VenueRanker, create_venue_ranker


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def ranker() -> VenueRanker:
    """VenueRanker con boost di default (T1=+0.15, T2=+0.05, T3=-0.05)."""
    return VenueRanker(tier1_boost=0.15, tier2_boost=0.05, tier3_penalty=0.05)


# ---------------------------------------------------------------------------
# Test get_tier()
# ---------------------------------------------------------------------------

def test_get_tier_neurips(ranker: VenueRanker) -> None:
    """NeurIPS 2023 deve essere Tier 1."""
    assert ranker.get_tier("NeurIPS 2023") == 1


def test_get_tier_iclr(ranker: VenueRanker) -> None:
    """ICLR 2024 deve essere Tier 1."""
    assert ranker.get_tier("ICLR 2024") == 1


def test_get_tier_arxiv_lowercase(ranker: VenueRanker) -> None:
    """'arxiv' in minuscolo deve essere Tier 3."""
    assert ranker.get_tier("arxiv") == 3


def test_get_tier_arxiv_preprint(ranker: VenueRanker) -> None:
    """'arXiv preprint' deve essere Tier 3 (substring match case-insensitive)."""
    assert ranker.get_tier("arXiv preprint") == 3


def test_get_tier_workshop_trustworthy_ml(ranker: VenueRanker) -> None:
    """Workshop on Trustworthy ML at ICML deve essere Tier 2."""
    assert ranker.get_tier("Workshop on Trustworthy ML at ICML") == 2


def test_get_tier_none(ranker: VenueRanker) -> None:
    """None deve ritornare 0 (sconosciuta)."""
    assert ranker.get_tier(None) == 0


def test_get_tier_unknown_university(ranker: VenueRanker) -> None:
    """Una università sconosciuta non deve matchare nessun tier."""
    assert ranker.get_tier("Università di Bologna Tech Report") == 0


def test_get_tier_acl(ranker: VenueRanker) -> None:
    """ACL deve essere Tier 1."""
    assert ranker.get_tier("Proceedings of the 61st Annual Meeting of ACL") == 1


def test_get_tier_findings_emnlp(ranker: VenueRanker) -> None:
    """'Findings of EMNLP' deve essere Tier 2."""
    assert ranker.get_tier("Findings of EMNLP 2023") == 2


# ---------------------------------------------------------------------------
# Test get_boost()
# ---------------------------------------------------------------------------

def test_get_boost_iclr(ranker: VenueRanker) -> None:
    """ICLR 2024 deve avere boost +0.15."""
    assert ranker.get_boost("ICLR 2024") == pytest.approx(0.15)


def test_get_boost_arxiv(ranker: VenueRanker) -> None:
    """arXiv deve avere boost -0.05 (penalità)."""
    assert ranker.get_boost("arXiv") == pytest.approx(-0.05)


def test_get_boost_none(ranker: VenueRanker) -> None:
    """None deve avere boost 0.0."""
    assert ranker.get_boost(None) == pytest.approx(0.0)


def test_get_boost_findings_acl(ranker: VenueRanker) -> None:
    """'findings of acl' deve avere boost +0.05."""
    assert ranker.get_boost("findings of acl") == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Invarianti di score
# ---------------------------------------------------------------------------

def test_score_after_boost_does_not_exceed_1(ranker: VenueRanker) -> None:
    """Score 0.9 + boost T1 (+0.15) clampato a 1.0."""
    base_score = 0.9
    boost = ranker.get_boost("NeurIPS 2022")
    clamped = min(1.0, max(0.0, base_score + boost))
    assert clamped <= 1.0


def test_score_after_boost_does_not_go_below_0(ranker: VenueRanker) -> None:
    """Score 0.02 + penalità T3 (-0.05) clampato a 0.0."""
    base_score = 0.02
    boost = ranker.get_boost("arXiv preprint")
    clamped = min(1.0, max(0.0, base_score + boost))
    assert clamped >= 0.0


# ---------------------------------------------------------------------------
# Test factory function e configurazione custom
# ---------------------------------------------------------------------------

def test_create_venue_ranker_factory() -> None:
    """create_venue_ranker() deve restituire un VenueRanker funzionante."""
    ranker = create_venue_ranker()
    assert isinstance(ranker, VenueRanker)
    assert ranker.get_tier("NeurIPS") == 1


def test_custom_boost_values() -> None:
    """VenueRanker con boost custom deve rispettarli."""
    ranker = VenueRanker(tier1_boost=0.20, tier2_boost=0.10, tier3_penalty=0.10)
    assert ranker.get_boost("NeurIPS 2023") == pytest.approx(0.20)
    assert ranker.get_boost("workshop") == pytest.approx(0.10)
    assert ranker.get_boost("arXiv") == pytest.approx(-0.10)


def test_case_insensitive_matching(ranker: VenueRanker) -> None:
    """Il matching deve essere case-insensitive."""
    assert ranker.get_tier("NEURIPS 2023") == ranker.get_tier("neurips 2023")
    assert ranker.get_tier("ARXIV") == ranker.get_tier("arxiv")


def test_substring_matching(ranker: VenueRanker) -> None:
    """Il matching deve essere substring-based."""
    # "neurips" è substring di "Advances in Neural Information Processing Systems (NeurIPS)"
    assert ranker.get_tier("Advances in Neural Information Processing Systems (NeurIPS) 2023") == 1
    # "arxiv" è substring di "arXiv:2310.12345"
    assert ranker.get_tier("arXiv:2310.12345") == 3


def test_empty_string_returns_tier_0(ranker: VenueRanker) -> None:
    """Stringa vuota deve restituire tier 0 e boost 0.0."""
    assert ranker.get_tier("") == 0
    assert ranker.get_boost("") == pytest.approx(0.0)


def test_ordering_after_boost() -> None:
    """I paper con venue autorevoli devono avere score più alto dopo il boost."""
    ranker = VenueRanker()
    base = 0.5

    score_neurips = base + ranker.get_boost("NeurIPS 2023")    # Tier 1
    score_workshop = base + ranker.get_boost("Workshop ICLR")  # Tier 2
    score_unknown = base + ranker.get_boost(None)               # Tier 0
    score_arxiv = base + ranker.get_boost("arXiv preprint")    # Tier 3

    assert score_neurips > score_workshop >= score_unknown > score_arxiv
