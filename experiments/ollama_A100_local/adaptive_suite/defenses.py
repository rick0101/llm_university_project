"""
Compact defense set for the adaptive suite.

Selected from the previous full matrix:
- keep no-defense to preserve the strongest observed baseline signal;
- keep the defenses that still had non-zero bypasses or are important
  holdouts for promotion decisions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_base_defenses", PARENT / "defenses.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Cannot load parent defenses.py")

_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)

DEFENSES = {
    "no-defense": _base.no_defense,
    "safety-prompt": _base.safety_prompt_defense,
    "in-context": _base.in_context_defense,
    "goal-priority": _base.goal_priority_defense,
    "constitutional": _base.constitutional_critique_defense,
}
