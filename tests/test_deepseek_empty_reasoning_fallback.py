"""Regression coverage for DeepSeek responses exhausted by hidden reasoning."""

from types import SimpleNamespace

from paperhunter.utils import llm_factory


class _FakeDeepSeek:
    """Minimal DeepSeek-like double with a deterministic completion queue."""

    provider = "deepseek"
    model = "deepseek-v4-pro"
    thinking_enabled = True

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self._responses.pop(0)


def _response(text: str, finish_reason: str):
    return SimpleNamespace(
        text=text,
        additional_kwargs={"finish_reason": finish_reason},
    )


def test_deepseek_retries_without_thinking_when_reasoning_uses_entire_output_budget(monkeypatch):
    """A blank length-limited thinking response must fall back before continuation."""
    thinking_llm = _FakeDeepSeek([_response("", "length")])
    no_thinking_llm = _FakeDeepSeek([_response('[{"key":"intro"}]', "stop")])

    monkeypatch.setattr(
        llm_factory,
        "_clone_deepseek_without_thinking",
        lambda llm: no_thinking_llm,
        raising=False,
    )

    result = llm_factory._llm_call_deepseek(
        thinking_llm,
        "produce a JSON outline",
        retries=0,
        base_delay=0,
        max_delay=0,
        autonomous=False,
    )

    assert result == '[{"key":"intro"}]'
    assert thinking_llm.calls == 1
    assert no_thinking_llm.calls == 1
