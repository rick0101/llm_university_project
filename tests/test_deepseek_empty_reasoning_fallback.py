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


def test_retry_wrapper_does_not_duplicate_model_level_empty_reasoning_recovery():
    """Il wrapper accetta il testo già recuperato da DeepSeekLLM.complete()."""
    llm = _FakeDeepSeek([_response('[{"key":"intro"}]', "stop")])

    result = llm_factory._llm_call_deepseek(
        llm,
        "produce a JSON outline",
        retries=0,
        base_delay=0,
        max_delay=0,
    )

    assert result == '[{"key":"intro"}]'
    assert llm.calls == 1
