"""Regression tests for fail-closed writer export gating."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_failed_pipeline_never_runs_auto_export():
    from paperhunter.main import _auto_export_if_ready

    state = {"current_phase": "failed", "final_paper": ""}
    args = SimpleNamespace()
    logger = MagicMock()

    with patch("paperhunter.main._auto_export") as auto_export:
        exported = _auto_export_if_ready(state, args, logger)

    assert exported is False
    auto_export.assert_not_called()


def test_completed_manuscript_runs_auto_export():
    from paperhunter.main import _auto_export_if_ready

    state = {"current_phase": "assembling", "final_paper": "# Survey"}
    args = SimpleNamespace()
    logger = MagicMock()

    with patch("paperhunter.main._auto_export") as auto_export:
        exported = _auto_export_if_ready(state, args, logger)

    assert exported is True
    auto_export.assert_called_once_with(state, args, logger)
