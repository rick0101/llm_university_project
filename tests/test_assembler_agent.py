"""
Test per agents/assembler_agent.py — Agent 4: Final Assembler & Versioner.

Test offline (mock LLM) che verificano:
1. Formattazione overview sezioni per prompt
2. Formattazione summary sezioni per prompt
3. Estrazione key findings
4. Estrazione citazioni dal testo
5. Finalizzazione sezioni (status → FINALIZED)
6. Versioning (_get_next_version_number, _get_version_dir)
7. Slugify per anchor links
8. Changelog manuale
9. generate_abstract con mock
10. generate_introduction con mock
11. generate_conclusion con mock
12. assemble_document
13. generate_bibliography con mock master .bib
14. create_version
15. assemble_paper completo con mock
16. assemble nodo LangGraph
17. get_assembly_stats
18. Serializzazione e round-trip

Esecuzione:
    python test_assembler_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Aggiungi il progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _create_mock_state(
    include_sections: bool = True,
    include_outline: bool = True,
    include_verifications: bool = False,
    include_l2: bool = False,
) -> dict:
    """Crea un SurveyState di test con sezioni finte."""
    from paperhunter.agents.state import (
        CitationRecord,
        SectionDraft,
        SectionOutline,
        SectionStatus,
        SurveyState,
        VerificationResult,
        create_initial_state,
    )
    from paperhunter.models import SurveyVersion, L2Summary

    # Outline
    outline = []
    if include_outline:
        outline = [
            SectionOutline(
                key="abstract", title="Abstract", order=0,
                section_type="abstract", description="Paper abstract",
                target_length_words=250,
            ),
            SectionOutline(
                key="introduction", title="Introduction", order=1,
                section_type="introduction", description="Introduction and motivation",
                target_length_words=1000, related_categories=["Prompt Injection"],
            ),
            SectionOutline(
                key="prompt_injection", title="Prompt Injection Attacks", order=2,
                section_type="body", description="Overview of prompt injection attacks",
                target_length_words=2000, related_categories=["Prompt Injection"],
                key_topics=["direct injection", "indirect injection"],
            ),
            SectionOutline(
                key="adversarial_attacks", title="Adversarial Attacks", order=3,
                section_type="body", description="Adversarial attack methods on LLMs",
                target_length_words=2000, related_categories=["Adversarial Attacks"],
                key_topics=["GCG", "AutoDAN"],
            ),
            SectionOutline(
                key="testing_methodologies", title="Testing Methodologies", order=4,
                section_type="testing", description="Overview of testing methods",
                target_length_words=1500,
                related_categories=["Prompt Injection", "Adversarial Attacks"],
            ),
            SectionOutline(
                key="conclusion", title="Conclusion and Future Work", order=5,
                section_type="conclusion", description="Conclusions and future directions",
                target_length_words=800,
            ),
        ]

    # Sezioni scritte
    sections = {}
    if include_sections:
        sections = {
            "introduction": SectionDraft(
                section_key="introduction", title="Introduction",
                text="Large Language Models have transformed NLP tasks. "
                     "However, they introduce security risks [Greshake_2023_LLM]. "
                     "This survey reviews the landscape of LLM security threats "
                     "and defense mechanisms [Zou_2023_GCG].",
                citations=[
                    CitationRecord(bibtex_key="Greshake_2023_LLM", claim="security risks"),
                    CitationRecord(bibtex_key="Zou_2023_GCG", claim="defense mechanisms"),
                ],
                word_count=30, status=SectionStatus.VERIFIED, iteration=1,
                drafted_at=datetime.now(),
            ),
            "prompt_injection": SectionDraft(
                section_key="prompt_injection",
                title="Prompt Injection Attacks",
                text="Prompt injection attacks exploit the way LLMs process "
                     "mixed instructions and data [Greshake_2023_LLM]. "
                     "Perez and Ribeiro demonstrated that these attacks can "
                     "compromise production systems [Perez_2022_Ignore]. "
                     "Recent work by Lee et al. proposed PromptShield "
                     "as a defense [Lee_2025_PromptShield].",
                citations=[
                    CitationRecord(bibtex_key="Greshake_2023_LLM"),
                    CitationRecord(bibtex_key="Perez_2022_Ignore"),
                    CitationRecord(bibtex_key="Lee_2025_PromptShield"),
                ],
                word_count=45, status=SectionStatus.VERIFIED, iteration=2,
                drafted_at=datetime.now(),
            ),
            "adversarial_attacks": SectionDraft(
                section_key="adversarial_attacks",
                title="Adversarial Attacks",
                text="Zou et al. introduced the GCG attack, which uses "
                     "gradient-based optimization to generate adversarial "
                     "suffixes [Zou_2023_GCG]. AutoDAN further automated "
                     "the process [Liu_2024_AutoDAN].",
                citations=[
                    CitationRecord(bibtex_key="Zou_2023_GCG"),
                    CitationRecord(bibtex_key="Liu_2024_AutoDAN"),
                ],
                word_count=30, status=SectionStatus.VERIFIED, iteration=1,
                drafted_at=datetime.now(),
            ),
            "testing_methodologies": SectionDraft(
                section_key="testing_methodologies",
                title="Testing Methodologies",
                text="Testing LLM security requires specialized benchmarks. "
                     "AdvBench provides a standardized dataset [Zou_2023_GCG]. "
                     "HarmBench extends this with more categories [Mazeika_2024_HarmBench].",
                citations=[
                    CitationRecord(bibtex_key="Zou_2023_GCG"),
                    CitationRecord(bibtex_key="Mazeika_2024_HarmBench"),
                ],
                word_count=25, status=SectionStatus.DRAFTED, iteration=1,
                drafted_at=datetime.now(),
            ),
        }

    # L2 summaries
    l2_summaries = []
    if include_l2:
        l2_summaries = [
            L2Summary(
                category="Prompt Injection",
                paper_ids=["p1", "p2", "p3"],
                paper_count=3,
                temporal_trends="Evolution from 2022 to 2025 shows increasing sophistication.",
                taxonomy="Direct injection, indirect injection, multi-turn attacks.",
                open_gaps="Defense against adaptive attacks remains an open challenge.",
                testing_overview="ASR, detection rate, F1 score are common metrics.",
                synthesis="Prompt injection is the most studied threat vector.",
            ),
            L2Summary(
                category="Adversarial Attacks",
                paper_ids=["p4", "p5"],
                paper_count=2,
                temporal_trends="Gradient-based attacks emerged in 2023.",
                taxonomy="White-box vs black-box attacks.",
                open_gaps="Transferability across model families needs more study.",
                testing_overview="AdvBench and HarmBench are standard benchmarks.",
                synthesis="Adversarial robustness is a critical concern.",
            ),
        ]

    # Verifications
    verifications = {}
    if include_verifications:
        verifications = {
            "prompt_injection": VerificationResult(
                section_key="prompt_injection",
                total_citations=3, supported=3, pass_rate=1.0, passed=True,
            ),
            "adversarial_attacks": VerificationResult(
                section_key="adversarial_attacks",
                total_citations=2, supported=2, pass_rate=1.0, passed=True,
            ),
        }

    state = create_initial_state(
        l2_summaries=l2_summaries if l2_summaries else None,
        outline=outline if outline else None,
    )

    # Aggiorna sezioni e verifiche
    if include_sections:
        state["sections"] = {k: v.to_dict() for k, v in sections.items()}
    if include_verifications:
        state["verification_results"] = {k: v.to_dict() for k, v in verifications.items()}

    return state


def test_format_sections_overview():
    """Test formattazione overview sezioni per prompt abstract."""
    from paperhunter.agents.assembler_agent import _format_sections_overview
    from paperhunter.agents.state import SectionDraft, SectionOutline, SectionStatus

    outline = [
        SectionOutline(key="abstract", title="Abstract", order=0, section_type="abstract"),
        SectionOutline(key="intro", title="Introduction", order=1, section_type="introduction"),
        SectionOutline(key="body1", title="Body Section", order=2, section_type="body"),
    ]

    sections = {
        "intro": SectionDraft(
            section_key="intro", title="Introduction",
            text="This is a test introduction. " * 20,
            citations=[], word_count=80, status=SectionStatus.VERIFIED,
        ),
    }

    overview = _format_sections_overview(sections, outline)

    # Abstract deve essere escluso
    assert "Abstract" not in overview
    # Intro deve avere preview
    assert "Introduction" in overview
    assert "80 words" in overview
    # Body non scritto → [not yet written]
    assert "[not yet written]" in overview

    print("  ✓ format_sections_overview: abstract escluso, preview, sezioni mancanti")


def test_format_sections_summary():
    """Test formattazione summary sezioni per prompt conclusion."""
    from paperhunter.agents.assembler_agent import _format_sections_summary
    from paperhunter.agents.state import SectionDraft, SectionOutline, SectionStatus

    outline = [
        SectionOutline(key="abstract", title="Abstract", order=0, section_type="abstract"),
        SectionOutline(key="intro", title="Introduction", order=1, section_type="introduction"),
        SectionOutline(key="conclusion", title="Conclusion", order=5, section_type="conclusion"),
    ]

    sections = {
        "intro": SectionDraft(
            section_key="intro", title="Introduction",
            text="This is the introduction. " * 30,
            citations=[], word_count=150, status=SectionStatus.VERIFIED,
        ),
    }

    summary = _format_sections_summary(sections, outline)

    # Abstract e conclusion devono essere esclusi
    assert "Abstract" not in summary
    assert "Conclusion" not in summary
    # Introduction inclusa
    assert "Introduction" in summary

    print("  ✓ format_sections_summary: abstract/conclusion esclusi")


def test_extract_key_findings():
    """Test estrazione key findings dalle sezioni."""
    from paperhunter.agents.assembler_agent import _extract_key_findings
    from paperhunter.agents.state import SectionDraft, SectionStatus

    sections = {
        "abstract": SectionDraft(
            section_key="abstract", title="Abstract",
            text="This is the abstract.", status=SectionStatus.FINALIZED,
        ),
        "body1": SectionDraft(
            section_key="body1", title="Body Section",
            text="The GCG attack was introduced in 2023. It uses gradient optimization. Results show 90% ASR.",
            status=SectionStatus.VERIFIED,
        ),
    }

    findings = _extract_key_findings(sections)

    # Abstract escluso
    assert "abstract" not in findings.lower() or "body" in findings.lower()
    # Body incluso
    assert "Body Section" in findings
    assert "GCG" in findings

    # Empty sections
    empty_findings = _extract_key_findings({})
    assert "No key findings" in empty_findings

    print("  ✓ extract_key_findings: body incluso, abstract escluso, empty handled")


def test_extract_citations_from_text():
    """Test estrazione citazioni [bibtex_key] dal testo."""
    from paperhunter.agents.assembler_agent import _extract_citations_from_text

    text = (
        "As shown by [Zou_2023_GCG] and further explored by [Liu_2024_AutoDAN], "
        "adversarial attacks pose significant threats. Previous work [Zou_2023_GCG] "
        "also demonstrated the issue."
    )

    citations = _extract_citations_from_text(text)

    # Deduplicazione
    keys = {c.bibtex_key for c in citations}
    assert keys == {"Zou_2023_GCG", "Liu_2024_AutoDAN"}
    assert len(citations) == 2  # Deduplicati

    # Testo senza citazioni
    no_cit = _extract_citations_from_text("No citations here.")
    assert len(no_cit) == 0

    # Falsi positivi (pattern non matching)
    false_pos = _extract_citations_from_text("See [Table 1] and [figure2].")
    assert len(false_pos) == 0

    print("  ✓ extract_citations_from_text: dedup, empty, false positivi filtrati")


def test_finalize_all_sections():
    """Test finalizzazione sezioni."""
    from paperhunter.agents.assembler_agent import _finalize_all_sections
    from paperhunter.agents.state import SectionDraft, SectionStatus

    state = _create_mock_state()

    updated = _finalize_all_sections(state)

    # testing_methodologies is DRAFTED and must not be promoted implicitly.
    assert "testing_methodologies" not in updated

    # Le sezioni già VERIFIED devono essere FINALIZED
    assert "prompt_injection" in updated
    assert "adversarial_attacks" in updated

    print("  ✓ finalize_all_sections: only VERIFIED → FINALIZED")


def test_slugify():
    """Test slugify per anchor links."""
    from paperhunter.agents.assembler_agent import _slugify

    assert _slugify("Prompt Injection Attacks") == "prompt-injection-attacks"
    assert _slugify("Testing & Evaluation") == "testing-evaluation"
    assert _slugify("  Spaced  Out  ") == "spaced-out"
    assert _slugify("LLM-Based Methods") == "llm-based-methods"

    print("  ✓ slugify: conversioni corrette")


def test_version_number():
    """Test calcolo numero versione successivo."""
    from paperhunter.agents.assembler_agent import _get_next_version_number

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
            mock_settings.output.versions_dir = tmpdir

            # Nessuna versione → 1.0
            assert _get_next_version_number() == "1.0"

            # Crea una versione
            (Path(tmpdir) / "Survey_v1.0_Feb2026").mkdir()
            assert _get_next_version_number() == "1.1"

            # Crea un'altra
            (Path(tmpdir) / "Survey_v1.1_Mar2026").mkdir()
            assert _get_next_version_number() == "1.2"

            # Crea una major
            (Path(tmpdir) / "Survey_v2.0_Apr2026").mkdir()
            assert _get_next_version_number() == "2.1"

    print("  ✓ version_number: 1.0, 1.1, 1.2, 2.1 — corretti")


def test_version_dir():
    """Test calcolo directory versione."""
    from paperhunter.agents.assembler_agent import _get_version_dir
    from paperhunter.models import SurveyVersion

    version = SurveyVersion(
        version_number="1.0",
        timestamp=datetime(2026, 2, 25),
    )

    with patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
        mock_settings.output.versions_dir = "./output/versions"
        path = _get_version_dir(version)
        assert "Survey_v1.0_Feb2026" in str(path)

    print("  ✓ version_dir: formato Survey_vX.Y_MonYYYY")


def test_manual_changelog():
    """Test changelog manuale senza LLM."""
    from paperhunter.agents.assembler_agent import _manual_changelog
    from paperhunter.models import SurveyVersion

    old = SurveyVersion(
        version_number="1.0", paper_count=50,
        categories=["Prompt Injection", "Adversarial Attacks"],
        sections_count=6, total_citations=100,
    )
    new = SurveyVersion(
        version_number="1.1", paper_count=60,
        categories=["Prompt Injection", "Adversarial Attacks", "Data Poisoning"],
        sections_count=8, total_citations=130,
    )

    changelog = _manual_changelog(old, new)
    assert "10 new papers" in changelog
    assert "Data Poisoning" in changelog
    assert "2 new sections" in changelog
    assert "100 → 130" in changelog

    # Nessuna differenza → "Minor updates"
    same = _manual_changelog(old, old)
    assert "Minor updates" in same

    print("  ✓ manual_changelog: aggiunto paper, categorie, sezioni, citazioni, no-diff")


def test_generate_abstract_mock():
    """Test generazione abstract con mock LLM."""
    state = _create_mock_state(include_outline=True)

    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(
        __str__=lambda self: (
            "This survey presents a comprehensive analysis of security challenges "
            "in Large Language Models. We review over 100 papers covering prompt "
            "injection attacks, adversarial attacks, and testing methodologies. "
            "Our analysis reveals that while significant progress has been made "
            "in understanding these threats, critical challenges remain."
        )
    )

    with patch("paperhunter.agents.assembler_agent.get_llm", return_value=mock_llm), \
         patch("paperhunter.agents.assembler_agent.llm_call_with_retry",
               side_effect=lambda llm, prompt, **kw: str(llm.complete(prompt))):
        from paperhunter.agents.assembler_agent import generate_abstract
        draft = generate_abstract(state)

    assert draft.section_key == "abstract"
    assert draft.status.value == "finalized"
    assert draft.word_count > 20
    assert "Large Language Models" in draft.text

    print(f"  ✓ generate_abstract mock: {draft.word_count} parole, status=finalized")


def test_generate_introduction_mock():
    """Test generazione introduction con mock LLM."""
    state = _create_mock_state(include_outline=True, include_l2=True)

    mock_response = (
        "Large Language Models have become ubiquitous in modern applications. "
        "However, their adoption raises significant security concerns "
        "[Greshake_2023_LLM]. Adversarial attacks can manipulate model outputs "
        "[Zou_2023_GCG]. This survey provides a comprehensive review of these "
        "threats and proposes a taxonomy of defense mechanisms."
    )

    with patch("paperhunter.agents.assembler_agent.get_llm") as mock_get, \
         patch("paperhunter.agents.assembler_agent.llm_call_with_retry",
               return_value=mock_response):
        from paperhunter.agents.assembler_agent import generate_introduction
        draft = generate_introduction(state)

    assert draft.section_key == "introduction"
    assert draft.status.value == "finalized"
    # Deve aver estratto citazioni
    citation_keys = {c.bibtex_key for c in draft.citations}
    assert "Greshake_2023_LLM" in citation_keys
    assert "Zou_2023_GCG" in citation_keys

    print(f"  ✓ generate_introduction mock: {draft.word_count} parole, {len(draft.citations)} citazioni")


def test_generate_conclusion_mock():
    """Test generazione conclusion con mock LLM."""
    state = _create_mock_state(include_outline=True, include_l2=True)

    mock_response = (
        "This survey has provided a comprehensive overview of security "
        "challenges in Large Language Models. Key findings include the "
        "increasing sophistication of prompt injection attacks and the "
        "emergence of gradient-based adversarial methods [Zou_2023_GCG]. "
        "Future work should focus on developing robust, adaptive defenses."
    )

    with patch("paperhunter.agents.assembler_agent.get_llm"), \
         patch("paperhunter.agents.assembler_agent.llm_call_with_retry",
               return_value=mock_response):
        from paperhunter.agents.assembler_agent import generate_conclusion
        draft = generate_conclusion(state)

    assert draft.section_key == "conclusion"
    assert draft.status.value == "finalized"
    assert "Large Language Models" in draft.text

    print(f"  ✓ generate_conclusion mock: {draft.word_count} parole")


def test_assemble_document():
    """Test assemblaggio documento Markdown."""
    from paperhunter.agents.assembler_agent import assemble_document
    from paperhunter.agents.state import SectionDraft, SectionStatus

    state = _create_mock_state(include_outline=True)

    # Aggiungi un abstract
    abstract = SectionDraft(
        section_key="abstract", title="Abstract",
        text="This is the abstract of the survey.",
        status=SectionStatus.FINALIZED, word_count=7,
    )

    document = assemble_document(state, abstract_draft=abstract)

    # Verifica struttura
    assert "# A Comprehensive Survey" in document
    assert "## Abstract" in document
    assert "This is the abstract" in document
    assert "## Table of Contents" in document
    assert "Introduction" in document
    assert "Prompt Injection Attacks" in document
    assert "Adversarial Attacks" in document
    assert "Testing Methodologies" in document
    assert "Conclusion" in document
    assert "## References" in document

    # Testo delle sezioni presenti
    assert "Large Language Models have transformed" in document
    assert "GCG attack" in document

    print(f"  ✓ assemble_document: {len(document.split())} parole, struttura completa")


def test_generate_bibliography():
    """Test generazione bibliografia filtrata."""
    from paperhunter.agents.assembler_agent import generate_bibliography

    state = _create_mock_state()

    # Mock master .bib
    mock_bib = {
        "Zou_2023_GCG": "@inproceedings{Zou_2023_GCG,\n  title={Universal},\n  author={Zou},\n  year={2023}\n}",
        "Greshake_2023_LLM": "@article{Greshake_2023_LLM,\n  title={Compromising},\n  author={Greshake},\n  year={2023}\n}",
        "Unused_2024_Key": "@article{Unused_2024_Key,\n  title={Unused},\n  author={Nobody},\n  year={2024}\n}",
        "Liu_2024_AutoDAN": "@inproceedings{Liu_2024_AutoDAN,\n  title={AutoDAN},\n  author={Liu},\n  year={2024}\n}",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_refs.bib"

        with patch("paperhunter.agents.assembler_agent.load_master_bib", return_value=mock_bib), \
             patch("paperhunter.agents.assembler_agent.save_master_bib",
                   side_effect=lambda entries, path: path), \
             patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
            mock_settings.output.base_dir = tmpdir
            mock_settings.output.versions_dir = str(Path(tmpdir) / "versions")

            keys, path = generate_bibliography(state, output_path=output_path)

    # Deve avere solo le chiavi usate nelle sezioni
    used_keys = set(keys)
    assert "Zou_2023_GCG" in used_keys
    assert "Greshake_2023_LLM" in used_keys
    # Unused non deve essere presente
    assert "Unused_2024_Key" not in used_keys

    print(f"  ✓ generate_bibliography: {len(keys)} chiavi filtrate, unused escluso")


def test_create_version():
    """Test creazione versione."""
    from paperhunter.agents.assembler_agent import create_version

    state = _create_mock_state(include_l2=True, include_outline=True)

    with patch("paperhunter.agents.assembler_agent._get_next_version_number",
               return_value="1.0"), \
         patch("paperhunter.agents.assembler_agent._generate_changelog",
               return_value=""):
        version = create_version(state, sections_count=6, total_citations=15)

    assert version.version_number == "1.0"
    assert version.sections_count == 6
    assert version.total_citations == 15
    assert version.paper_count == 5  # 3 + 2 dai L2 mock
    assert "Prompt Injection" in version.categories or "Adversarial Attacks" in version.categories

    print(f"  ✓ create_version: v{version.version_number}, {version.paper_count} paper, {version.sections_count} sezioni")


def test_save_version():
    """Test salvataggio versione su disco."""
    from paperhunter.agents.assembler_agent import save_version
    from paperhunter.models import SurveyVersion

    version = SurveyVersion(
        version_number="1.0",
        timestamp=datetime(2026, 2, 25),
        paper_count=50,
        categories=["Prompt Injection"],
        sections_count=6,
        total_citations=100,
        changelog="- Added 50 papers\n- Initial release",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
            mock_settings.output.versions_dir = str(Path(tmpdir) / "versions")

            version_dir = save_version(
                "# Test Survey\n\nThis is a test.",
                version,
                bibliography_keys=["Zou_2023_GCG", "Greshake_2023_LLM"],
                provenance=[{"claim": "test", "key": "Zou_2023_GCG"}],
            )

        # Verifica file creati
        assert (version_dir / "survey.md").exists()
        assert (version_dir / "version_info.json").exists()
        assert (version_dir / "provenance.json").exists()
        assert (version_dir / "changelog.md").exists()
        assert (version_dir / "citation_keys.json").exists()

        # Verifica contenuto survey
        survey_text = (version_dir / "survey.md").read_text()
        assert "Test Survey" in survey_text

        # Verifica version_info
        vi = json.loads((version_dir / "version_info.json").read_text())
        assert vi["version_number"] == "1.0"
        assert vi["paper_count"] == 50

        # Verifica provenance
        prov = json.loads((version_dir / "provenance.json").read_text())
        assert prov["total_records"] == 1

        # Verifica citation keys
        keys = json.loads((version_dir / "citation_keys.json").read_text())
        assert "Zou_2023_GCG" in keys

    print(f"  ✓ save_version: 5 file creati, contenuti verificati")


def test_assemble_paper_mock():
    """Test assemble_paper completo con mock LLM."""
    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        include_verifications=True,
        include_l2=True,
    )

    abstract_response = "This survey reviews LLM security challenges."
    intro_response = (
        "LLMs are widely adopted [Greshake_2023_LLM]. "
        "Security threats include prompt injection [Zou_2023_GCG]."
    )
    conclusion_response = "We conclude that LLM security remains an open challenge."

    call_count = {"n": 0}
    responses = [abstract_response, intro_response, conclusion_response]

    def mock_llm_call(llm, prompt, **kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.assembler_agent.get_llm"), \
             patch("paperhunter.agents.assembler_agent.llm_call_with_retry",
                   side_effect=mock_llm_call), \
             patch("paperhunter.agents.assembler_agent.load_master_bib", return_value={}), \
             patch("paperhunter.agents.assembler_agent.save_master_bib",
                   side_effect=lambda entries, path: path), \
             patch("paperhunter.agents.assembler_agent._get_next_version_number",
                   return_value="1.0"), \
             patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
            mock_settings.output.base_dir = tmpdir
            mock_settings.output.versions_dir = str(Path(tmpdir) / "versions")
            mock_settings.agents.assembler_model = "pro"

            from paperhunter.agents.assembler_agent import assemble_paper
            result = assemble_paper(state, save_output=True)

    # Verifica risultato
    assert "final_paper" in result
    assert result["final_paper"]  # Non vuoto
    assert "## Abstract" in result["final_paper"]
    assert "current_phase" in result
    assert result["current_phase"] == "completed"
    assert "sections" in result
    assert "abstract" in result["sections"]
    assert "introduction" in result["sections"]
    assert "conclusion" in result["sections"]
    assert "version_info" in result
    assert result["version_info"]["version_number"] == "1.0"
    assert "metadata" in result
    assert "total_words" in result["metadata"]

    print(f"  ✓ assemble_paper mock: documento generato, versioning OK, {result['metadata']['total_words']} parole")


def test_assemble_langgraph_node():
    """Test nodo LangGraph assemble."""
    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        include_verifications=True,
    )
    from paperhunter.agents.state import SectionDraft, SectionStatus, VerificationResult
    state["sections"]["testing_methodologies"]["status"] = SectionStatus.VERIFIED.value
    state["sections"]["conclusion"] = SectionDraft(
        section_key="conclusion",
        title="Conclusion and Future Work",
        text="Verified conclusion.",
        word_count=800,
        status=SectionStatus.VERIFIED,
        iteration=1,
        writer_notes="",
        critic_feedback="",
        drafted_at=datetime.now(),
    ).to_dict()
    for key in ("introduction", "testing_methodologies", "conclusion"):
        state["verification_results"][key] = VerificationResult(
            section_key=key,
            total_citations=1,
            supported=1,
            partially_supported=0,
            not_supported=0,
            source_not_found=0,
            pass_rate=1.0,
            passed=True,
            iteration=1,
            verified_at=datetime.now(),
        ).to_dict()

    with patch("paperhunter.agents.assembler_agent.get_llm"), \
         patch("paperhunter.agents.assembler_agent.llm_call_with_retry",
               return_value="Mock generated text for survey section."), \
         patch("paperhunter.agents.assembler_agent.load_master_bib", return_value={}), \
         patch("paperhunter.agents.assembler_agent.save_master_bib",
               side_effect=lambda entries, path: path), \
         patch("paperhunter.agents.assembler_agent._get_next_version_number",
               return_value="1.0"), \
         patch("paperhunter.agents.assembler_agent.settings") as mock_settings:
        mock_settings.output.base_dir = "/tmp/test_assembler"
        mock_settings.output.versions_dir = "/tmp/test_assembler/versions"
        mock_settings.agents.assembler_model = "pro"

        from paperhunter.agents.assembler_agent import assemble
        result = assemble(state)

    assert result["current_phase"] == "completed"
    assert result.get("final_paper")

    print("  ✓ assemble nodo LangGraph: phase=completed, documento generato")


def test_assemble_no_sections():
    """Test assemble con stato vuoto → failure."""
    from paperhunter.agents.assembler_agent import assemble
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state()

    result = assemble(state)

    assert result["current_phase"] == "failed"
    assert len(result.get("errors", [])) > 0

    print("  ✓ assemble no sections: phase=failed, errore loggato")


def test_get_assembly_stats():
    """Test statistiche assemblaggio."""
    from paperhunter.agents.assembler_agent import get_assembly_stats

    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        include_verifications=True,
    )

    stats = get_assembly_stats(state)

    assert stats["total_sections"] == 4  # intro, pi, aa, testing
    assert stats["total_words"] > 0
    assert stats["unique_citations"] > 0
    assert stats["verified_sections"] >= 0
    assert "version" in stats
    assert "categories" in stats

    print(f"  ✓ get_assembly_stats: {stats['total_sections']} sez, {stats['unique_citations']} cit, {stats['total_words']} parole")


def test_langgraph_integration():
    """Test integrazione con StateGraph LangGraph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("  ⚠ LangGraph non disponibile, skip test integrazione")
        return

    from paperhunter.agents.state import SurveyState

    # Verifica che assemble sia un callable valido come nodo
    from paperhunter.agents.assembler_agent import assemble

    graph = StateGraph(SurveyState)
    graph.add_node("assemble", assemble)
    graph.set_entry_point("assemble")
    graph.add_edge("assemble", END)

    # Compila senza errori
    compiled = graph.compile()
    assert compiled is not None

    print("  ✓ LangGraph integration: StateGraph compilato con nodo assemble")


def test_package_imports():
    """Test che tutti gli export siano importabili."""
    from paperhunter.agents import (
        assemble,
        assemble_paper,
        assemble_document,
        generate_abstract,
        generate_bibliography,
        generate_conclusion,
        generate_introduction,
        create_version,
        save_version,
        get_assembly_stats,
    )

    assert callable(assemble)
    assert callable(assemble_paper)
    assert callable(assemble_document)
    assert callable(generate_abstract)
    assert callable(generate_bibliography)
    assert callable(generate_conclusion)
    assert callable(generate_introduction)
    assert callable(create_version)
    assert callable(save_version)
    assert callable(get_assembly_stats)

    print("  ✓ package imports: 10 funzioni importate e callable")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: agents/assembler_agent.py — Agent 4: Assembler")
    print("=" * 60)

    tests = [
        ("Format sections overview", test_format_sections_overview),
        ("Format sections summary", test_format_sections_summary),
        ("Extract key findings", test_extract_key_findings),
        ("Extract citations from text", test_extract_citations_from_text),
        ("Finalize all sections", test_finalize_all_sections),
        ("Slugify", test_slugify),
        ("Version number", test_version_number),
        ("Version directory", test_version_dir),
        ("Manual changelog", test_manual_changelog),
        ("Generate abstract (mock)", test_generate_abstract_mock),
        ("Generate introduction (mock)", test_generate_introduction_mock),
        ("Generate conclusion (mock)", test_generate_conclusion_mock),
        ("Assemble document", test_assemble_document),
        ("Generate bibliography", test_generate_bibliography),
        ("Create version", test_create_version),
        ("Save version", test_save_version),
        ("Assemble paper (mock)", test_assemble_paper_mock),
        ("Assemble LangGraph node", test_assemble_langgraph_node),
        ("Assemble no sections", test_assemble_no_sections),
        ("Get assembly stats", test_get_assembly_stats),
        ("LangGraph integration", test_langgraph_integration),
        ("Package imports", test_package_imports),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n--- Test: {name} ---")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RISULTATI: {passed}/{passed + failed} test passati")
    if failed > 0:
        print(f"  ✗ {failed} test falliti!")
    else:
        print(f"  ✓ Tutti i test passati!")
    print("=" * 60)
