"""Extract lessons from pipeline results (verification, summarization, errors).

Each extractor function inspects a specific pipeline output and returns
zero or more :class:`Lesson` instances that the :class:`LessonStore` can
persist for future prompt-overlay injection.
"""

from __future__ import annotations

from datetime import datetime, timezone

from paperhunter.evolution.lesson_store import (
    Lesson,
    LessonCategory,
    LessonSeverity,
    auto_classify,
)
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_run_id(stage: str) -> str:
    """Generate a run identifier like ``write_20260322_130000``."""
    now = datetime.now(timezone.utc)
    return f"{stage}_{now.strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# 1. Verification results  (critic_agent)
# ---------------------------------------------------------------------------

def extract_from_verification(
    section_key: str,
    verification_result: dict | object,
    *,
    threshold: float = 0.80,
    run_id: str = "",
) -> list[Lesson]:
    """Extract lessons from a single section's verification result.

    Parameters
    ----------
    section_key:
        The section that was verified (e.g. ``"prompt_injection"``).
    verification_result:
        A :class:`VerificationResult` instance **or** its ``.to_dict()``
        representation.  Both are accepted for flexibility.
    threshold:
        Pass-rate threshold below which a lesson is generated.
    run_id:
        Pipeline run identifier (auto-generated if empty).

    Returns
    -------
    list[Lesson]
        One lesson per failing section, plus one per NOT_SUPPORTED /
        SOURCE_NOT_FOUND citation.
    """
    run_id = run_id or _make_run_id("write")

    # Normalise to dict
    vr: dict
    if hasattr(verification_result, "to_dict"):
        vr = verification_result.to_dict()
    elif hasattr(verification_result, "model_dump"):
        vr = verification_result.model_dump()
    elif isinstance(verification_result, dict):
        vr = verification_result
    else:
        logger.warning("Unsupported verification_result type: %s", type(verification_result))
        return []

    pass_rate: float = vr.get("pass_rate", 1.0)
    passed: bool = vr.get("passed", pass_rate >= threshold)

    if passed:
        return []

    lessons: list[Lesson] = []

    # -- Section-level lesson --
    total = vr.get("total_citations", 0)
    supported = vr.get("supported", 0)
    partially = vr.get("partially_supported", 0)
    not_supported = vr.get("not_supported", 0)
    source_not_found = vr.get("source_not_found", 0)
    iteration = vr.get("iteration", 1)

    desc = (
        f"Section '{section_key}' failed verification "
        f"(pass_rate={pass_rate:.0%}, threshold={threshold:.0%}, "
        f"iteration={iteration}). "
        f"Citations: {supported} SUPPORTED, {partially} PARTIALLY, "
        f"{not_supported} NOT_SUPPORTED, {source_not_found} SOURCE_NOT_FOUND "
        f"out of {total} total."
    )

    # Collect failing bibtex keys
    failing_keys: list[str] = []
    details: list[dict] = vr.get("details", [])
    for d in details:
        status = d.get("status", "")
        if isinstance(status, str):
            status_val = status
        else:
            status_val = status.value if hasattr(status, "value") else str(status)

        if status_val in ("NOT_SUPPORTED", "SOURCE_NOT_FOUND"):
            key = d.get("citation_key", "")
            if key and key not in failing_keys:
                failing_keys.append(key)

    severity = LessonSeverity.ERROR if pass_rate < 0.5 else LessonSeverity.WARNING

    lessons.append(Lesson(
        stage_name="write",
        node_name="verify",
        category=LessonCategory.WRITING,
        severity=severity,
        description=desc,
        section_key=section_key,
        bibtex_keys_involved=failing_keys,
        run_id=run_id,
    ))

    # -- Per-citation lessons for NOT_SUPPORTED (with claim context) --
    for d in details:
        status = d.get("status", "")
        if isinstance(status, str):
            status_val = status
        else:
            status_val = status.value if hasattr(status, "value") else str(status)

        if status_val == "NOT_SUPPORTED":
            claim = d.get("claim_text", "")[:120]
            cit_key = d.get("citation_key", "unknown")
            reasoning = d.get("reasoning", "")[:150]
            lessons.append(Lesson(
                stage_name="write",
                node_name="verify",
                category=LessonCategory.WRITING,
                severity=LessonSeverity.ERROR,
                description=(
                    f"Citation {cit_key} NOT_SUPPORTED in '{section_key}'. "
                    f"Claim: \"{claim}\". Critic: \"{reasoning}\""
                ),
                section_key=section_key,
                bibtex_keys_involved=[cit_key],
                run_id=run_id,
            ))

    return lessons


# ---------------------------------------------------------------------------
# 2. Summarization failures
# ---------------------------------------------------------------------------

def extract_from_summarization(
    paper_id: str,
    error: Exception | str,
    *,
    level: str = "L1",
    category: str | None = None,
    run_id: str = "",
) -> Lesson:
    """Create a lesson from a failed L1 or L2 summarization.

    Parameters
    ----------
    paper_id:
        Paper identifier (for L1) or category name (for L2).
    error:
        The exception or error message string.
    level:
        ``"L1"`` or ``"L2"``.
    category:
        Paper category (e.g. ``"Prompt Injection"``).
    run_id:
        Pipeline run identifier (auto-generated if empty).
    """
    run_id = run_id or _make_run_id("summarize")
    error_str = str(error)

    severity = LessonSeverity.ERROR
    cat = auto_classify(error_str)

    if level == "L2":
        desc = (
            f"L2 summarization failed for category '{paper_id}'. "
            f"Error: {error_str[:200]}"
        )
    else:
        desc = (
            f"L1 summarization failed for paper '{paper_id}'"
            + (f" (category: {category})" if category else "")
            + f". Error: {error_str[:200]}"
        )

    return Lesson(
        stage_name="summarize",
        node_name=f"generate_{level.lower()}",
        category=cat,
        severity=severity,
        description=desc,
        section_key=category,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# 3. Generic pipeline errors
# ---------------------------------------------------------------------------

def extract_from_pipeline(
    node_name: str,
    error: Exception | str,
    *,
    stage_name: str = "write",
    section_key: str | None = None,
    run_id: str = "",
) -> Lesson:
    """Create a lesson from a generic pipeline error.

    Covers loop exhaustion, timeouts, LangGraph crashes, etc.

    Parameters
    ----------
    node_name:
        LangGraph node where the error occurred (e.g. ``"plan"``,
        ``"write"``, ``"assemble"``).
    error:
        The exception or error message.
    stage_name:
        Top-level pipeline stage (default ``"write"``).
    section_key:
        Survey section key, if applicable.
    run_id:
        Pipeline run identifier (auto-generated if empty).
    """
    run_id = run_id or _make_run_id(stage_name)
    error_str = str(error)

    return Lesson(
        stage_name=stage_name,
        node_name=node_name,
        category=auto_classify(error_str),
        severity=LessonSeverity.ERROR,
        description=f"Pipeline error in {stage_name}/{node_name}: {error_str[:250]}",
        section_key=section_key,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# 4. Batch helper — extract from full verification state
# ---------------------------------------------------------------------------

def extract_from_all_verifications(
    verification_results: dict[str, dict | object],
    *,
    threshold: float = 0.80,
    run_id: str = "",
) -> list[Lesson]:
    """Extract lessons from **all** section verification results at once.

    Parameters
    ----------
    verification_results:
        ``state["verification_results"]`` — mapping of section_key to
        either a ``VerificationResult`` object or its dict representation.
    threshold:
        Pass-rate threshold.
    run_id:
        Shared run identifier.

    Returns
    -------
    list[Lesson]
        Aggregated lessons from every failing section.
    """
    run_id = run_id or _make_run_id("write")
    all_lessons: list[Lesson] = []

    for section_key, vr in verification_results.items():
        lessons = extract_from_verification(
            section_key=section_key,
            verification_result=vr,
            threshold=threshold,
            run_id=run_id,
        )
        all_lessons.extend(lessons)

    if all_lessons:
        logger.info(
            "Extracted %d lesson(s) from %d section(s)",
            len(all_lessons),
            len(verification_results),
        )

    return all_lessons
