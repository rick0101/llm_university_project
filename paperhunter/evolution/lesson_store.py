"""Append-only JSONL store for lessons learned across pipeline runs.

Each lesson records an error, warning, or decision from a specific pipeline
stage. Lessons are ranked by temporal weight (exponential decay) so that
recent failures carry more influence when injected into LLM prompts.
"""

from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LessonSeverity(str, Enum):
    """How critical the lesson is."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LessonCategory(str, Enum):
    """Broad domain the lesson belongs to."""
    WRITING = "writing"
    LITERATURE = "literature"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Keyword → category mapping (used by auto_classify)
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], LessonCategory]] = [
    (re.compile(r"timeout|connection|memory|oom|disk", re.IGNORECASE), LessonCategory.SYSTEM),
    (re.compile(r"NOT_SUPPORTED|SOURCE_NOT_FOUND|citation|hallucin|fabricat|invent", re.IGNORECASE), LessonCategory.WRITING),
    (re.compile(r"JSON|parse|format|schema|decode|malformed", re.IGNORECASE), LessonCategory.PIPELINE),
    (re.compile(r"stale|invalidat|outdated|retract", re.IGNORECASE), LessonCategory.LITERATURE),
]


def auto_classify(description: str) -> LessonCategory:
    """Return a :class:`LessonCategory` inferred from *description* keywords.

    Falls back to :attr:`LessonCategory.PIPELINE` if no pattern matches.
    """
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(description):
            return category
    return LessonCategory.PIPELINE


# ---------------------------------------------------------------------------
# Lesson model
# ---------------------------------------------------------------------------

class Lesson(BaseModel):
    """A single lesson extracted from a pipeline run."""

    stage_name: str = Field(..., description="Pipeline stage (e.g. write, summarize, ingest)")
    node_name: str | None = Field(None, description="LangGraph node (e.g. verify, plan)")
    category: LessonCategory = Field(LessonCategory.PIPELINE, description="Broad classification")
    severity: LessonSeverity = Field(LessonSeverity.WARNING, description="info / warning / error")
    description: str = Field(..., description="Human-readable explanation of the lesson")
    section_key: str | None = Field(None, description="Survey section key if applicable")
    bibtex_keys_involved: list[str] = Field(default_factory=list, description="BibTeX keys related to this lesson")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the lesson was recorded")
    run_id: str = Field("", description="Identifier of the pipeline run (e.g. write_20260322_110000)")

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: str | LessonCategory) -> LessonCategory:
        if isinstance(v, str):
            return LessonCategory(v)
        return v

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: str | LessonSeverity) -> LessonSeverity:
        if isinstance(v, str):
            return LessonSeverity(v)
        return v


# ---------------------------------------------------------------------------
# Scored wrapper (returned by query)
# ---------------------------------------------------------------------------

class ScoredLesson(BaseModel):
    """A lesson annotated with its temporal weight."""
    lesson: Lesson
    weight: float = Field(..., ge=0.0, le=1.0, description="Temporal relevance weight")


# ---------------------------------------------------------------------------
# LessonStore
# ---------------------------------------------------------------------------

class LessonStore:
    """Append-only JSONL store with temporal-decay query.

    Parameters
    ----------
    path : str | Path
        Path to the ``.jsonl`` file (created on first :meth:`record`).
    half_life_days : float
        Half-life in days for exponential decay (default 30).
    expiry_days : float
        Lessons older than this are ignored by :meth:`query` and
        removed by :meth:`prune` (default 90).
    """

    def __init__(
        self,
        path: str | Path = "data/evolution/lessons.jsonl",
        half_life_days: float = 30.0,
        expiry_days: float = 90.0,
    ) -> None:
        self.path = Path(path)
        self.half_life_days = half_life_days
        self.expiry_days = expiry_days
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, lesson: Lesson) -> None:
        """Append *lesson* to the JSONL file (thread-safe)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = lesson.model_dump_json() + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
        logger.debug("Recorded lesson [%s/%s]: %s", lesson.stage_name, lesson.node_name, lesson.description[:80])

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> list[Lesson]:
        """Read every lesson from disk."""
        if not self.path.exists():
            return []
        lessons: list[Lesson] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lessons.append(Lesson.model_validate_json(raw))
                except Exception:
                    logger.warning("Skipping malformed lesson at line %d", lineno)
        return lessons

    def _temporal_weight(self, lesson: Lesson, now: datetime | None = None) -> float:
        """Compute exponential-decay weight: ``exp(-age * ln2 / half_life)``."""
        now = now or datetime.now(timezone.utc)
        ts = lesson.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (now - ts).total_seconds() / 86_400.0
        if age_days < 0:
            age_days = 0.0
        if age_days >= self.expiry_days:
            return 0.0
        return math.exp(-age_days * math.log(2) / self.half_life_days)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        stage: str | None = None,
        node: str | None = None,
        section_key: str | None = None,
        category: LessonCategory | None = None,
        top_k: int = 5,
    ) -> list[ScoredLesson]:
        """Return the *top_k* most relevant (non-expired) lessons.

        Lessons are filtered by the provided criteria, then scored by
        temporal weight and severity, and returned in descending order.
        """
        now = datetime.now(timezone.utc)
        candidates: list[ScoredLesson] = []

        for lesson in self._load_all():
            weight = self._temporal_weight(lesson, now)
            if weight <= 0.0:
                continue

            # Filters
            if stage is not None and lesson.stage_name != stage:
                continue
            if node is not None and lesson.node_name != node:
                continue
            if section_key is not None and lesson.section_key != section_key:
                continue
            if category is not None and lesson.category != category:
                continue

            # Severity bonus: errors rank higher than warnings/info
            severity_mult = {"error": 1.0, "warning": 0.7, "info": 0.4}
            final_weight = weight * severity_mult.get(lesson.severity.value, 0.5)

            candidates.append(ScoredLesson(lesson=lesson, weight=round(final_weight, 4)))

        candidates.sort(key=lambda s: s.weight, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """Remove expired lessons (older than *expiry_days*) from disk.

        Returns the number of lessons removed.
        """
        if not self.path.exists():
            return 0

        now = datetime.now(timezone.utc)
        all_lessons = self._load_all()
        kept = [l for l in all_lessons if self._temporal_weight(l, now) > 0.0]
        removed = len(all_lessons) - len(kept)

        if removed > 0:
            with self._lock:
                with open(self.path, "w", encoding="utf-8") as fh:
                    for lesson in kept:
                        fh.write(lesson.model_dump_json() + "\n")
            logger.info("Pruned %d expired lesson(s) (%d remaining)", removed, len(kept))

        return removed

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of lessons on disk."""
        return len(self._load_all())

    def clear(self) -> None:
        """Delete the JSONL file entirely."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Cleared lesson store at %s", self.path)
