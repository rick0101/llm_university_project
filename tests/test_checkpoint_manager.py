"""
Test suite per postprocessing/checkpoint_manager.py

Copre tutte le funzionalità di CheckpointManager:
  - CheckpointInfo dataclass e from_file
  - Salvataggio/caricamento checkpoint pipeline
  - Salvataggio/caricamento checkpoint ingestion
  - Listing, filtraggio, ricerca
  - Resume recommendation
  - Diff tra checkpoint
  - Cleanup con retention policy
  - Export/Import
  - Statistiche
  - Convenience functions
"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ============================================================================
# Setup: assicura che paperhunter sia importabile
# ============================================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Import modulo sotto test
# ============================================================================

from paperhunter.postprocessing.checkpoint_manager import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE_INGESTION,
    CHECKPOINT_TYPE_PIPELINE,
    PIPELINE_NODES,
    CheckpointInfo,
    CheckpointManager,
    get_checkpoint_manager,
    get_resume_recommendation,
    list_all_checkpoints,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dir():
    """Directory temporanea per i test."""
    d = tempfile.mkdtemp(prefix="test_checkpoint_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mgr(tmp_dir):
    """CheckpointManager con directory temporanea."""
    return CheckpointManager(checkpoint_dir=tmp_dir)


def _make_dummy_state(
    phase: str = "writing",
    iteration: int = 1,
    n_outline: int = 5,
    sections: dict | None = None,
    errors: list | None = None,
    final_paper: str = "",
) -> dict:
    """Stato SurveyState fittizio per test."""
    return {
        "l2_summaries": [{"category": "Prompt Injection", "version": 1}],
        "outline": [{"key": f"sec_{i}", "title": f"Section {i}"} for i in range(n_outline)],
        "sections": sections or {},
        "verification_results": {},
        "bibliography": [],
        "provenance": [],
        "final_paper": final_paper,
        "version_info": {"version_number": "1.0"},
        "current_phase": phase,
        "current_section_key": "",
        "iteration_count": iteration,
        "errors": errors or [],
        "metadata": {},
    }


# ============================================================================
# Test: CheckpointInfo dataclass
# ============================================================================

class TestCheckpointInfo:

    def test_to_dict(self):
        info = CheckpointInfo(
            path="/tmp/test.json",
            filename="test.json",
            checkpoint_type=CHECKPOINT_TYPE_PIPELINE,
            node="write",
            timestamp=datetime(2026, 2, 25, 10, 0, 0),
            timestamp_str="20260225_100000",
            size_kb=42.5,
            is_latest=False,
        )
        d = info.to_dict()
        assert d["path"] == "/tmp/test.json"
        assert d["node"] == "write"
        assert d["checkpoint_type"] == CHECKPOINT_TYPE_PIPELINE
        assert d["size_kb"] == 42.5
        assert d["timestamp"] == "2026-02-25T10:00:00"
        assert d["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_repr(self):
        info = CheckpointInfo(
            path="/tmp/test.json",
            filename="test.json",
            node="plan",
            timestamp_str="20260225_100000",
            size_kb=10.0,
        )
        r = repr(info)
        assert "plan" in r
        assert "10.0" in r

    def test_from_file_pipeline(self, tmp_dir):
        fp = tmp_dir / "checkpoint_write_20260225_100000.json"
        data = {
            "schema_version": "1.0",
            "checkpoint_type": CHECKPOINT_TYPE_PIPELINE,
            "node": "write",
            "timestamp": "20260225_100000",
            "state": {
                "current_phase": "writing",
                "iteration_count": 2,
                "outline": [{"key": "s1"}, {"key": "s2"}, {"key": "s3"}],
                "sections": {"s1": {}, "s2": {}},
                "errors": ["err1"],
                "final_paper": "",
            },
        }
        fp.write_text(json.dumps(data), encoding="utf-8")

        info = CheckpointInfo.from_file(fp)
        assert info.checkpoint_type == CHECKPOINT_TYPE_PIPELINE
        assert info.node == "write"
        assert info.timestamp is not None
        assert info.timestamp.year == 2026
        assert info.metadata["current_phase"] == "writing"
        assert info.metadata["iteration_count"] == 2
        assert info.metadata["outline_sections"] == 3
        assert info.metadata["drafted_sections"] == 2
        assert info.metadata["errors_count"] == 1
        assert info.metadata["has_final_paper"] is False

    def test_from_file_ingestion(self, tmp_dir):
        fp = tmp_dir / "ingestion_checkpoint.json"
        data = {
            "checkpoint_type": CHECKPOINT_TYPE_INGESTION,
            "started_at": "2026-02-25T10:00:00",
            "last_updated": "2026-02-25T10:15:00",
            "completed_pdfs": ["a.pdf", "b.pdf", "c.pdf"],
            "total_pdfs_to_process": 10,
        }
        fp.write_text(json.dumps(data), encoding="utf-8")

        info = CheckpointInfo.from_file(fp)
        assert info.checkpoint_type == CHECKPOINT_TYPE_INGESTION
        assert info.node == "ingestion"
        assert info.metadata["completed_pdfs"] == 3
        assert info.metadata["total_to_process"] == 10

    def test_from_file_corrupted(self, tmp_dir):
        fp = tmp_dir / "bad.json"
        fp.write_text("not json at all", encoding="utf-8")

        info = CheckpointInfo.from_file(fp)
        assert info.node == "corrupted"

    def test_from_file_unknown_type(self, tmp_dir):
        fp = tmp_dir / "mystery.json"
        fp.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        info = CheckpointInfo.from_file(fp)
        assert info.checkpoint_type == "unknown"

    def test_from_file_iso_timestamp(self, tmp_dir):
        fp = tmp_dir / "ingestion_checkpoint.json"
        data = {
            "completed_pdfs": [],
            "total_pdfs_to_process": 0,
            "last_updated": "2026-02-25T10:15:00",
        }
        fp.write_text(json.dumps(data), encoding="utf-8")

        info = CheckpointInfo.from_file(fp)
        assert info.timestamp is not None
        assert info.timestamp.year == 2026
        assert info.timestamp.month == 2


# ============================================================================
# Test: Pipeline checkpoints
# ============================================================================

class TestPipelineCheckpoints:

    def test_save_and_load(self, mgr, tmp_dir):
        state = _make_dummy_state(phase="writing", iteration=2)
        info = mgr.save_pipeline_checkpoint(state, "write")

        assert info.checkpoint_type == CHECKPOINT_TYPE_PIPELINE
        assert info.node == "write"
        assert info.size_kb > 0

        # Verifica file timestamped e latest
        files = list(tmp_dir.glob("checkpoint_write_*.json"))
        assert len(files) == 2  # timestamped + latest
        latest = tmp_dir / "checkpoint_write_latest.json"
        assert latest.exists()

        # Carica
        node, loaded = mgr.load_pipeline_checkpoint(info.path)
        assert node == "write"
        assert loaded["current_phase"] == "writing"
        assert loaded["iteration_count"] == 2

    def test_load_pipeline_state(self, mgr):
        state = _make_dummy_state(phase="planning", n_outline=3)
        info = mgr.save_pipeline_checkpoint(state, "plan")

        node, loaded_state = mgr.load_pipeline_state(info.path)
        assert node == "plan"
        # state_from_dict restituisce un dict-like
        assert loaded_state["current_phase"] == "planning"

    def test_load_nonexistent(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.load_pipeline_checkpoint("/nonexistent/path.json")

    def test_load_invalid(self, mgr, tmp_dir):
        fp = tmp_dir / "invalid.json"
        fp.write_text(json.dumps({"no_state": True}), encoding="utf-8")
        with pytest.raises(ValueError, match="non è un checkpoint pipeline valido"):
            mgr.load_pipeline_checkpoint(fp)

    def test_save_with_extra_metadata(self, mgr):
        state = _make_dummy_state()
        info = mgr.save_pipeline_checkpoint(
            state, "verify",
            extra_metadata={"custom_key": "custom_value"},
        )
        # Rileggi il file direttamente per verificare
        data = json.loads(Path(info.path).read_text(encoding="utf-8"))
        assert data["extra_metadata"]["custom_key"] == "custom_value"

    def test_multiple_saves_same_node(self, mgr, tmp_dir):
        """Salva 3 checkpoint dello stesso nodo: 3 timestamped + 1 latest."""
        for i in range(3):
            state = _make_dummy_state(iteration=i)
            mgr.save_pipeline_checkpoint(state, "write")
            time.sleep(0.05)  # Timestamp diversi

        files = list(tmp_dir.glob("checkpoint_write_*.json"))
        # 3 timestamped + 1 latest = 4 file
        assert len(files) == 4

        latest = tmp_dir / "checkpoint_write_latest.json"
        data = json.loads(latest.read_text(encoding="utf-8"))
        # Il latest è l'ultimo salvato
        assert data["state"]["iteration_count"] == 2


# ============================================================================
# Test: Ingestion checkpoints
# ============================================================================

class TestIngestionCheckpoints:

    def test_save_and_load(self, mgr):
        info = mgr.save_ingestion_checkpoint(
            completed_pdfs=["a.pdf", "b.pdf"],
            total_to_process=10,
            started_at="2026-02-25T10:00:00",
        )
        assert info.checkpoint_type == CHECKPOINT_TYPE_INGESTION
        assert info.node == "ingestion"

        data = mgr.load_ingestion_checkpoint()
        assert len(data["completed_pdfs"]) == 2
        assert data["total_pdfs_to_process"] == 10
        assert data["started_at"] == "2026-02-25T10:00:00"

    def test_has_checkpoint(self, mgr):
        assert not mgr.has_ingestion_checkpoint()

        mgr.save_ingestion_checkpoint(["a.pdf"], 5)
        assert mgr.has_ingestion_checkpoint()

    def test_clear_ingestion(self, mgr):
        mgr.save_ingestion_checkpoint(["a.pdf"], 5)
        assert mgr.has_ingestion_checkpoint()

        result = mgr.clear_ingestion_checkpoint()
        assert result is True
        assert not mgr.has_ingestion_checkpoint()

        # Clear di nuovo → False
        result = mgr.clear_ingestion_checkpoint()
        assert result is False

    def test_load_empty(self, mgr):
        data = mgr.load_ingestion_checkpoint()
        assert data == {}

    def test_load_corrupted(self, mgr, tmp_dir):
        fp = tmp_dir / "ingestion_checkpoint.json"
        fp.write_text("NOT JSON", encoding="utf-8")
        data = mgr.load_ingestion_checkpoint()
        assert data == {}

    def test_save_with_extra_metadata(self, mgr, tmp_dir):
        mgr.save_ingestion_checkpoint(
            ["a.pdf"], 5,
            extra_metadata={"batch_size": 10},
        )
        data = json.loads(
            (tmp_dir / "ingestion_checkpoint.json").read_text(encoding="utf-8")
        )
        assert data["extra_metadata"]["batch_size"] == 10


# ============================================================================
# Test: Listing & search
# ============================================================================

class TestListing:

    def _populate(self, mgr):
        """Popola con checkpoint misti."""
        mgr.save_pipeline_checkpoint(_make_dummy_state(phase="planning"), "plan")
        time.sleep(0.05)
        mgr.save_pipeline_checkpoint(
            _make_dummy_state(phase="writing", iteration=1), "write"
        )
        time.sleep(0.05)
        mgr.save_pipeline_checkpoint(
            _make_dummy_state(phase="verifying", iteration=1), "verify"
        )
        time.sleep(0.05)
        mgr.save_ingestion_checkpoint(["a.pdf", "b.pdf"], 10)

    def test_list_all(self, mgr):
        self._populate(mgr)
        cps = mgr.list_checkpoints()
        # 3 pipeline (timestamped) + 3 pipeline (latest) + 1 ingestion = 7
        assert len(cps) == 7

    def test_list_no_latest(self, mgr):
        self._populate(mgr)
        cps = mgr.list_checkpoints(include_latest=False)
        # 3 pipeline (timestamped) + 1 ingestion = 4
        assert len(cps) == 4

    def test_filter_by_type(self, mgr):
        self._populate(mgr)
        pipeline_cps = mgr.list_checkpoints(checkpoint_type=CHECKPOINT_TYPE_PIPELINE)
        ingestion_cps = mgr.list_checkpoints(checkpoint_type=CHECKPOINT_TYPE_INGESTION)
        assert all(cp.checkpoint_type == CHECKPOINT_TYPE_PIPELINE for cp in pipeline_cps)
        assert all(cp.checkpoint_type == CHECKPOINT_TYPE_INGESTION for cp in ingestion_cps)

    def test_filter_by_node(self, mgr):
        self._populate(mgr)
        write_cps = mgr.list_checkpoints(node="write")
        assert len(write_cps) == 2  # timestamped + latest
        assert all(cp.node == "write" for cp in write_cps)

    def test_sort_by_size(self, mgr):
        self._populate(mgr)
        cps = mgr.list_checkpoints(sort_by="size")
        for i in range(len(cps) - 1):
            assert cps[i].size_kb >= cps[i + 1].size_kb

    def test_sort_by_node(self, mgr):
        self._populate(mgr)
        cps = mgr.list_checkpoints(sort_by="node")
        nodes = [cp.node for cp in cps]
        assert nodes == sorted(nodes)

    def test_get_latest_pipeline(self, mgr):
        self._populate(mgr)

        # Specifico nodo
        cp = mgr.get_latest_pipeline_checkpoint("write")
        assert cp is not None
        assert cp.node == "write"

        # Nodo inesistente
        cp = mgr.get_latest_pipeline_checkpoint("assemble")
        assert cp is None

        # Più recente in assoluto (verify è l'ultimo salvato)
        cp = mgr.get_latest_pipeline_checkpoint()
        assert cp is not None
        assert cp.node == "verify"

    def test_get_latest_empty(self, mgr):
        cp = mgr.get_latest_pipeline_checkpoint()
        assert cp is None

    def test_find_checkpoints_by_size(self, mgr):
        self._populate(mgr)
        cps = mgr.find_checkpoints(min_size_kb=0.1)
        assert len(cps) > 0

    def test_find_checkpoints_by_date(self, mgr):
        self._populate(mgr)
        # Tutti dopo un'ora fa
        cps = mgr.find_checkpoints(after=datetime.now() - timedelta(hours=1))
        assert len(cps) > 0

        # Nessuno prima di ieri
        cps = mgr.find_checkpoints(before=datetime.now() - timedelta(days=1))
        assert len(cps) == 0

    def test_list_empty_dir(self, mgr, tmp_dir):
        # Rimuovi la directory
        shutil.rmtree(tmp_dir)
        cps = mgr.list_checkpoints()
        assert cps == []


# ============================================================================
# Test: Resume recommendation
# ============================================================================

class TestResumeRecommendation:

    def test_no_checkpoints(self, mgr):
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is False
        assert "Nessun checkpoint" in rec["recommendation"]

    def test_ingestion_pending(self, mgr):
        mgr.save_ingestion_checkpoint(
            completed_pdfs=["a.pdf", "b.pdf", "c.pdf"],
            total_to_process=10,
        )
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is True
        assert rec["checkpoint_type"] == CHECKPOINT_TYPE_INGESTION
        assert rec["resume_node"] == "ingestion"
        assert rec["details"]["completed_pdfs"] == 3
        assert rec["details"]["remaining_pdfs"] == 7

    def test_ingestion_completed(self, mgr):
        """Ingestion con tutti i PDF completati → non è una raccomandazione."""
        mgr.save_ingestion_checkpoint(
            completed_pdfs=["a.pdf", "b.pdf", "c.pdf"],
            total_to_process=3,
        )
        rec = mgr.get_resume_recommendation()
        # remaining = 0, non è una raccomandazione di ingestion
        assert rec["checkpoint_type"] != CHECKPOINT_TYPE_INGESTION or not rec["has_recommendation"]

    def test_plan_completed(self, mgr):
        mgr.save_pipeline_checkpoint(_make_dummy_state(phase="planning"), "plan")
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is True
        assert rec["resume_node"] == "write"
        assert "plan" in rec["recommendation"]

    def test_write_completed(self, mgr):
        mgr.save_pipeline_checkpoint(
            _make_dummy_state(phase="writing", iteration=1), "write"
        )
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is True
        assert rec["resume_node"] == "verify"

    def test_verify_completed_all_ok(self, mgr):
        """Verify completato senza sezioni da riscrivere → assemble."""
        state = _make_dummy_state(
            phase="verifying",
            sections={"s1": {"status": "verified"}, "s2": {"status": "verified"}},
        )
        mgr.save_pipeline_checkpoint(state, "verify")
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is True
        assert rec["resume_node"] == "assemble"

    def test_verify_needs_revision(self, mgr):
        """Verify completato con sezioni da riscrivere → write."""
        state = _make_dummy_state(
            phase="verifying",
            iteration=1,
            sections={
                "s1": {"status": "verified"},
                "s2": {"status": "needs_revision"},
            },
        )
        mgr.save_pipeline_checkpoint(state, "verify")
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is True
        assert rec["resume_node"] == "write"
        assert "needs_revision" in rec["details"]

    def test_assemble_completed(self, mgr):
        """Pipeline completa (assemble) → nessuna ripresa necessaria."""
        state = _make_dummy_state(phase="completed", final_paper="# Survey")
        mgr.save_pipeline_checkpoint(state, "assemble")
        rec = mgr.get_resume_recommendation()
        assert rec["has_recommendation"] is False
        assert "completata" in rec["recommendation"]


# ============================================================================
# Test: Diff between checkpoints
# ============================================================================

class TestDiffCheckpoints:

    def test_basic_diff(self, mgr):
        state_a = _make_dummy_state(phase="writing", iteration=1, n_outline=5)
        state_b = _make_dummy_state(
            phase="verifying",
            iteration=2,
            n_outline=5,
            sections={"s1": {"status": "verified"}, "s2": {"status": "needs_revision"}},
        )
        info_a = mgr.save_pipeline_checkpoint(state_a, "write")
        time.sleep(0.05)
        info_b = mgr.save_pipeline_checkpoint(state_b, "verify")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert diff["node_progression"] == "write → verify"
        assert diff["total_changes"] > 0
        changes = diff["changes"]
        assert "current_phase" in changes
        assert changes["current_phase"]["from"] == "writing"
        assert changes["current_phase"]["to"] == "verifying"
        assert "iteration_count" in changes
        assert changes["iteration_count"]["from"] == 1
        assert changes["iteration_count"]["to"] == 2

    def test_diff_sections_added(self, mgr):
        state_a = _make_dummy_state(sections={})
        state_b = _make_dummy_state(sections={
            "intro": {"status": "drafted"},
            "body": {"status": "drafted"},
        })
        info_a = mgr.save_pipeline_checkpoint(state_a, "plan")
        info_b = mgr.save_pipeline_checkpoint(state_b, "write")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert "sections" in diff["changes"]
        assert "intro" in diff["changes"]["sections"]["added"]
        assert "body" in diff["changes"]["sections"]["added"]

    def test_diff_no_changes(self, mgr):
        state = _make_dummy_state()
        info_a = mgr.save_pipeline_checkpoint(state, "write")
        info_b = mgr.save_pipeline_checkpoint(state, "write")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert diff["total_changes"] == 0

    def test_diff_nonexistent(self, mgr):
        state = _make_dummy_state()
        info_a = mgr.save_pipeline_checkpoint(state, "write")
        with pytest.raises(FileNotFoundError):
            mgr.diff_checkpoints(info_a.path, "/nonexistent.json")

    def test_diff_final_paper(self, mgr):
        state_a = _make_dummy_state(final_paper="")
        state_b = _make_dummy_state(final_paper="# My Survey\n\nContent here.")
        info_a = mgr.save_pipeline_checkpoint(state_a, "verify")
        info_b = mgr.save_pipeline_checkpoint(state_b, "assemble")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert "final_paper" in diff["changes"]
        assert diff["changes"]["final_paper"]["from"] is False
        assert diff["changes"]["final_paper"]["to"] is True

    def test_diff_errors(self, mgr):
        state_a = _make_dummy_state(errors=[])
        state_b = _make_dummy_state(errors=["Error in section s1"])
        info_a = mgr.save_pipeline_checkpoint(state_a, "write")
        info_b = mgr.save_pipeline_checkpoint(state_b, "verify")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert "new_errors" in diff["changes"]
        assert "Error in section s1" in diff["changes"]["new_errors"]

    def test_diff_section_status_change(self, mgr):
        state_a = _make_dummy_state(sections={
            "s1": {"status": "drafted"},
            "s2": {"status": "drafted"},
        })
        state_b = _make_dummy_state(sections={
            "s1": {"status": "verified"},
            "s2": {"status": "needs_revision"},
        })
        info_a = mgr.save_pipeline_checkpoint(state_a, "write")
        info_b = mgr.save_pipeline_checkpoint(state_b, "verify")

        diff = mgr.diff_checkpoints(info_a.path, info_b.path)
        assert "section_status_changes" in diff["changes"]
        sc = diff["changes"]["section_status_changes"]
        assert sc["s1"]["from"] == "drafted"
        assert sc["s1"]["to"] == "verified"


# ============================================================================
# Test: Cleanup
# ============================================================================

class TestCleanup:

    def test_clean_by_age(self, mgr, tmp_dir):
        # Crea un checkpoint "vecchio" simulando un mtime antico
        info = mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        old_path = Path(info.path)
        # Imposta mtime a 100 ore fa
        old_time = time.time() - (100 * 3600)
        os.utime(old_path, (old_time, old_time))

        # Crea un checkpoint recente
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "write")

        removed = mgr.clean(max_age_hours=72, keep_latest_per_node=False)
        assert removed >= 1
        assert not old_path.exists()

    def test_clean_keep_latest(self, mgr, tmp_dir):
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        time.sleep(0.05)
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")

        latest = tmp_dir / "checkpoint_plan_latest.json"
        assert latest.exists()

        # Pulisci tutto vecchio ma mantieni latest
        removed = mgr.clean(max_age_hours=0, keep_latest_per_node=True)
        assert latest.exists()  # Latest sopravvive

    def test_clean_max_checkpoints(self, mgr, tmp_dir):
        for i in range(5):
            mgr.save_pipeline_checkpoint(
                _make_dummy_state(iteration=i), f"write"
            )
            time.sleep(0.05)

        # Tieni solo 2 checkpoint (escludendo latest)
        removed = mgr.clean(max_checkpoints=2, keep_latest_per_node=True)
        assert removed >= 1

        # Verifica che ci sono al massimo 2 + i latest
        non_latest = [
            f for f in tmp_dir.glob("checkpoint_write_*.json")
            if "_latest" not in f.name
        ]
        assert len(non_latest) <= 2

    def test_clean_dry_run(self, mgr, tmp_dir):
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        before = len(list(tmp_dir.glob("*.json")))

        removed = mgr.clean(max_age_hours=0, dry_run=True, keep_latest_per_node=False)
        after = len(list(tmp_dir.glob("*.json")))

        assert before == after  # Nessun file rimosso
        assert removed >= 1  # Ma il conteggio dice che li avrebbe rimossi

    def test_clean_empty_dir(self, mgr, tmp_dir):
        shutil.rmtree(tmp_dir)
        removed = mgr.clean(max_age_hours=0)
        assert removed == 0

    def test_clear_all(self, mgr, tmp_dir):
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "write")
        mgr.save_ingestion_checkpoint(["a.pdf"], 5)

        before = len(list(tmp_dir.glob("*.json")))
        assert before > 0

        removed = mgr.clear_all()
        assert removed == before
        assert len(list(tmp_dir.glob("*.json"))) == 0

    def test_clear_all_empty(self, mgr, tmp_dir):
        shutil.rmtree(tmp_dir)
        removed = mgr.clear_all()
        assert removed == 0


# ============================================================================
# Test: Export / Import
# ============================================================================

class TestExportImport:

    def test_export(self, mgr, tmp_dir):
        info = mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        export_dir = tmp_dir / "backup"
        exported = mgr.export_checkpoint(info.path, export_dir)

        assert exported.exists()
        # Verifica che il contenuto è identico
        original = json.loads(Path(info.path).read_text(encoding="utf-8"))
        exported_data = json.loads(exported.read_text(encoding="utf-8"))
        assert original == exported_data

    def test_export_nonexistent(self, mgr, tmp_dir):
        with pytest.raises(FileNotFoundError):
            mgr.export_checkpoint("/nonexistent.json", tmp_dir / "backup")

    def test_import(self, mgr, tmp_dir):
        # Crea un file checkpoint esterno
        external_dir = tmp_dir / "external"
        external_dir.mkdir()
        fp = external_dir / "checkpoint_plan_20260225_100000.json"
        data = {
            "schema_version": "1.0",
            "checkpoint_type": CHECKPOINT_TYPE_PIPELINE,
            "node": "plan",
            "timestamp": "20260225_100000",
            "state": _make_dummy_state(),
        }
        fp.write_text(json.dumps(data), encoding="utf-8")

        info = mgr.import_checkpoint(fp)
        assert info.node == "plan"
        # Verifica che il file è nella checkpoint_dir
        assert (tmp_dir / fp.name).exists()

    def test_import_nonexistent(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.import_checkpoint("/nonexistent.json")

    def test_import_invalid_json(self, mgr, tmp_dir):
        fp = tmp_dir / "bad_external.json"
        fp.write_text("NOT JSON", encoding="utf-8")
        with pytest.raises(ValueError, match="non è un JSON valido"):
            mgr.import_checkpoint(fp)

    def test_export_import_roundtrip(self, mgr, tmp_dir):
        """Export + import = stato identico."""
        state = _make_dummy_state(phase="verifying", iteration=3)
        info = mgr.save_pipeline_checkpoint(state, "verify")

        export_dir = tmp_dir / "backup"
        exported = mgr.export_checkpoint(info.path, export_dir)

        # Pulisci i checkpoint
        mgr.clear_all()
        assert len(mgr.list_checkpoints()) == 0

        # Import
        imported = mgr.import_checkpoint(exported)
        node, loaded = mgr.load_pipeline_checkpoint(imported.path)
        assert node == "verify"
        assert loaded["iteration_count"] == 3


# ============================================================================
# Test: Statistics
# ============================================================================

class TestStatistics:

    def test_stats_empty(self, mgr):
        stats = mgr.get_stats()
        assert stats["total_checkpoints"] == 0
        assert stats["pipeline_checkpoints"] == 0
        assert stats["ingestion_checkpoints"] == 0
        assert stats["latest_checkpoint"] is None

    def test_stats_populated(self, mgr):
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        time.sleep(0.05)
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "write")
        mgr.save_ingestion_checkpoint(["a.pdf"], 5)

        stats = mgr.get_stats()
        assert stats["total_checkpoints"] > 0
        assert stats["pipeline_checkpoints"] > 0
        assert stats["ingestion_checkpoints"] == 1
        assert stats["total_size_kb"] > 0
        assert "write" in stats["nodes"] or "plan" in stats["nodes"]
        assert stats["has_ingestion_pending"] is True

    def test_print_summary_smoke(self, mgr):
        """Smoke test: no crash."""
        mgr.print_summary()

        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        mgr.print_summary()

    def test_repr(self, mgr):
        r = repr(mgr)
        assert "CheckpointManager" in r
        assert "total=" in r

    def test_len(self, mgr):
        assert len(mgr) == 0
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "plan")
        assert len(mgr) > 0


# ============================================================================
# Test: Convenience functions
# ============================================================================

class TestConvenienceFunctions:

    def test_get_checkpoint_manager(self, tmp_dir):
        mgr = get_checkpoint_manager(tmp_dir)
        assert isinstance(mgr, CheckpointManager)

    def test_get_resume_recommendation_standalone(self, tmp_dir):
        rec = get_resume_recommendation(tmp_dir)
        assert rec["has_recommendation"] is False

    def test_list_all_checkpoints_standalone(self, tmp_dir):
        cps = list_all_checkpoints(tmp_dir)
        assert cps == []

    def test_convenience_with_data(self, tmp_dir):
        mgr = get_checkpoint_manager(tmp_dir)
        mgr.save_pipeline_checkpoint(_make_dummy_state(), "write")

        cps = list_all_checkpoints(tmp_dir)
        assert len(cps) > 0

        rec = get_resume_recommendation(tmp_dir)
        assert rec["has_recommendation"] is True


# ============================================================================
# Test: Package imports
# ============================================================================

class TestPackageImports:

    def test_import_from_module(self):
        from paperhunter.postprocessing.checkpoint_manager import (
            CheckpointInfo,
            CheckpointManager,
            get_checkpoint_manager,
            get_resume_recommendation,
            list_all_checkpoints,
        )
        assert CheckpointManager is not None
        assert CheckpointInfo is not None

    def test_import_from_package(self):
        from paperhunter.postprocessing import (
            CheckpointManager,
            get_checkpoint_manager,
            get_resume_recommendation,
            list_all_checkpoints,
        )
        assert CheckpointManager is not None


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
