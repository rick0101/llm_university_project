"""Regression test per DeepSeek thinking e gate citazionale DBLP strict."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from paperhunter.ingestion.dblp_lookup import DblpLookup, DblpResult
from paperhunter.ingestion.bibtex_generator import enrich_paper_bibtex, save_master_bib
from paperhunter.config import _build_settings
from paperhunter.models import Paper
from paperhunter.postprocessing import latex_exporter
from paperhunter.postprocessing.latex_exporter import (
    CitationIntegrityError,
    validate_citation_integrity,
)
from paperhunter.utils.llm_factory import DeepSeekLLM
from paperhunter.utils.rate_limiter import NoOpLimiter, get_embedding_limiter, get_llm_limiter
from paperhunter.agents.critic_agent import (
    _extract_citations_from_text as extract_critic_citations,
    _extract_claims_for_citation,
)


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 20}


class _FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="verified response")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=_FakeUsage())


def test_deepseek_v4_pro_uses_native_thinking_max_without_temperature():
    """La richiesta deve usare esattamente il profilo qualità massimo richiesto."""
    llm = DeepSeekLLM(
        api_key="test-key",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="max",
    )
    fake = _FakeCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    response = llm.complete("Check this source")

    assert response.text == "verified response"
    assert fake.request["model"] == "deepseek-v4-pro"
    assert fake.request["reasoning_effort"] == "max"
    assert fake.request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in fake.request
    assert "top_p" not in fake.request


def test_deepseek_and_local_embeddings_do_not_use_gemini_rate_limits():
    assert isinstance(get_llm_limiter("deepseek"), NoOpLimiter)
    assert isinstance(get_embedding_limiter("huggingface"), NoOpLimiter)


def test_llm_provider_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert _build_settings().llm_provider == "google"


def test_grouped_citations_are_individually_verified():
    first = "Smith_2024_First_abcd1234"
    second = "Jones_2025_Second_efgh5678"
    text = f"The methods differ substantially [{first}, {second}]."
    assert extract_critic_citations(text) == [first, second]
    assert text in _extract_claims_for_citation(text, first)
    assert text in _extract_claims_for_citation(text, second)


def test_dblp_strict_match_requires_title_author_and_exact_year(tmp_path):
    lookup = DblpLookup(cache_dir=tmp_path)
    good = {
        "title": "Attention Is All You Need.",
        "authors": {"author": [{"text": "Ashish Vaswani"}, {"text": "Noam Shazeer"}]},
        "year": "2017",
        "type": "Conference and Workshop Papers",
        "key": "conf/nips/VaswaniSPUJGKP17",
    }

    hit, similarity = lookup._select_best_hit(
        [good],
        "Attention Is All You Need",
        2017,
        threshold=0.90,
        target_authors=["Ashish Vaswani"],
        require_author_match=True,
        require_exact_year=True,
    )
    assert hit == good
    assert similarity > 0.99

    wrong_author = {**good, "authors": {"author": [{"text": "Jane Doe"}]}}
    hit, _ = lookup._select_best_hit(
        [wrong_author],
        "Attention Is All You Need",
        2017,
        threshold=0.90,
        target_authors=["Ashish Vaswani"],
        require_author_match=True,
        require_exact_year=True,
    )
    assert hit is None

    wrong_year = {**good, "year": "2018"}
    hit, _ = lookup._select_best_hit(
        [wrong_year],
        "Attention Is All You Need",
        2017,
        threshold=0.90,
        target_authors=["Ashish Vaswani"],
        require_author_match=True,
        require_exact_year=True,
    )
    assert hit is None
    lookup.close()


def _write_verified_fixture(tmp_path):
    key = "Vaswani_2017_Attention_0123abcd"
    entry = (
        f"@inproceedings{{{key},\n"
        "  author = {Ashish Vaswani and Noam Shazeer},\n"
        "  title = {Attention is All you Need},\n"
        "  booktitle = {Advances in Neural Information Processing Systems 30},\n"
        "  year = {2017}\n"
        "}"
    )
    bib_path = tmp_path / "references_final.bib"
    bib_path.write_text(entry + "\n", encoding="utf-8")
    manifest = {
        key: {
            "dblp_key": "conf/nips/VaswaniSPUJGKP17",
            "dblp_url": "https://dblp.org/rec/conf/nips/VaswaniSPUJGKP17",
            "dblp_title": "Attention is All you Need",
            "title_similarity": 1.0,
            "bibtex_sha256": hashlib.sha256(entry.encode("utf-8")).hexdigest(),
            "verified": True,
        }
    }
    manifest_path = tmp_path / "references_dblp_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return key, entry, bib_path, manifest_path


def test_final_integrity_gate_accepts_only_matching_dblp_manifest(tmp_path):
    key, _, bib_path, _ = _write_verified_fixture(tmp_path)
    report = validate_citation_integrity(
        f"This result is established by [{key}].",
        bib_path,
        require_dblp=True,
    )
    assert report["passed"] is True
    assert report["cited_keys"] == [key]


def test_final_integrity_gate_blocks_bibtex_drift(tmp_path):
    key, entry, bib_path, _ = _write_verified_fixture(tmp_path)
    bib_path.write_text(
        entry.replace("year = {2017}", "year = {2018}") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CitationIntegrityError, match="diversa da quella verificata"):
        validate_citation_integrity(
            f"This result is established by [{key}].",
            bib_path,
            require_dblp=True,
        )


def test_final_integrity_gate_blocks_missing_manifest(tmp_path, monkeypatch):
    key, _, bib_path, manifest_path = _write_verified_fixture(tmp_path)
    manifest_path.unlink()
    # Evita che un manifest di una run reale funga da fallback per il fixture.
    monkeypatch.setattr(latex_exporter.settings.output, "base_dir", tmp_path)

    with pytest.raises(CitationIntegrityError, match="manifest DBLP assente"):
        validate_citation_integrity(
            f"This result is established by [{key}].",
            bib_path,
            require_dblp=True,
        )


def test_ingestion_builds_a_valid_manifest_from_official_dblp(monkeypatch, tmp_path):
    official = """@inproceedings{conf/nips/VaswaniSPUJGKP17,
  author = {Ashish Vaswani and Noam Shazeer},
  title = {Attention is All you Need},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year = {2017}
}"""
    result = DblpResult(
        dblp_key="conf/nips/VaswaniSPUJGKP17",
        bibtex=official,
        dblp_type="Conference and Workshop Papers",
        title="Attention is All you Need",
        similarity=1.0,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        record_url="https://dblp.org/rec/conf/nips/VaswaniSPUJGKP17",
    )
    fake_lookup = SimpleNamespace(lookup_bibtex=lambda **kwargs: result)
    monkeypatch.setattr(
        "paperhunter.ingestion.bibtex_generator.get_dblp_lookup",
        lambda: fake_lookup,
    )
    paper = Paper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
        url="https://arxiv.org/abs/1706.03762",
    )

    enriched = enrich_paper_bibtex(paper)
    bib_path = save_master_bib([], tmp_path / "references.bib", papers=[enriched])

    assert enriched.is_citable is True
    assert enriched.dblp_verified is True
    assert enriched.bibtex_source == "dblp"
    assert enriched.bibtex_key.endswith("_7d858ca9")
    assert "@inproceedings" in bib_path.read_text(encoding="utf-8")
    report = validate_citation_integrity(
        f"Transformers use self-attention [{enriched.bibtex_key}].",
        bib_path,
        require_dblp=True,
    )
    assert report["passed"] is True
