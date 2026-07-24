"""Regression coverage for DeepSeek planner output-token budgeting."""

import pytest

from paperhunter.agents import planner_agent
from paperhunter.models import L2Summary


class _StopAfterLlmConstruction(Exception):
    """Stops the planner after observing the requested LLM token budget."""


def test_deepseek_planner_uses_configured_pro_token_budget(monkeypatch):
    """The planner must not cap DeepSeek reasoning output at the legacy 16k limit."""
    captured = {}

    def fake_get_llm(tier, *, max_tokens):
        captured["tier"] = tier
        captured["max_tokens"] = max_tokens
        raise _StopAfterLlmConstruction

    monkeypatch.setattr(planner_agent.settings, "llm_provider", "deepseek")
    monkeypatch.setattr(planner_agent, "generate_full_overlay", lambda *args, **kwargs: "")
    monkeypatch.setattr(planner_agent, "get_llm", fake_get_llm)

    l2_summary = L2Summary(
        category="Prompt Injection",
        paper_ids=["paper-1"],
        paper_count=1,
        synthesis="A compact L2 summary.",
    )

    with pytest.raises(_StopAfterLlmConstruction):
        planner_agent.generate_outline([l2_summary], min_sections=1, target_sections=1)

    assert captured == {
        "tier": planner_agent.settings.agents.planner_model,
        "max_tokens": planner_agent.settings.deepseek.max_output_tokens_pro,
    }
