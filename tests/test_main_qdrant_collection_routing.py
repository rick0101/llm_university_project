"""Regression tests for configured Qdrant collection routing in CLI pipeline phases."""
from types import SimpleNamespace

from paperhunter import main
import paperhunter.summarization as summarization


class _Point:
    def __init__(self, payload):
        self.payload = payload


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _StoreForConfiguredCollections:
    COLLECTION_CHUNKS = "paper_chunks_bge_m3"
    COLLECTION_L1 = "l1_summaries_bge_m3"

    def __init__(self):
        self.scroll_calls = []
        self.chunk_calls = []
        self.client = self

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append(collection_name)
        if collection_name == self.COLLECTION_CHUNKS and offset is None:
            return [_Point({"bibtex_key": "Doe_2026_Test", "paper_id": "paper-1"})], None
        return [], None

    def get_paper_chunks(self, bibtex_key, *, collection_name):
        self.chunk_calls.append((bibtex_key, collection_name))
        if collection_name != self.COLLECTION_CHUNKS:
            return []
        return [
            {
                "chunk_id": "paper-1-0",
                "text": "A valid paper chunk.",
                "metadata": {
                    "paper_id": "paper-1",
                    "paper_title": "Test Paper",
                    "category": "Prompt Injection",
                    "year": 2026,
                },
            }
        ]


def test_l1_summarization_reads_and_writes_configured_collections(monkeypatch):
    """A BGE-M3 configuration must not fall back to legacy collection names."""
    store = _StoreForConfiguredCollections()
    generated_for = []

    def fake_generate_l1(paper, chunks, *, save_to_store, store):
        generated_for.append((paper.bibtex_key, len(chunks), save_to_store, store))
        return object()

    monkeypatch.setattr(summarization, "generate_l1_summary", fake_generate_l1)

    result = main._run_l1_summarization(
        store,
        SimpleNamespace(force=True),
        _Logger(),
    )

    assert result == 0
    assert store.scroll_calls[:2] == [store.COLLECTION_CHUNKS, store.COLLECTION_L1]
    assert store.chunk_calls == [("Doe_2026_Test", store.COLLECTION_CHUNKS)]
    assert generated_for == [("Doe_2026_Test", 1, True, store)]
