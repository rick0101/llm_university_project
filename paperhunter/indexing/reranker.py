"""
Reranker con Gemini Flash per PaperHunter.

Usa Gemini Flash come cross-encoder per riordinare i risultati del
retriever ibrido. Il modello valuta la rilevanza di ogni candidato
rispetto alla query e assegna uno score di pertinenza.

Workflow:
1. Riceve query + candidati dal HybridRetriever
2. Se candidati > ``batch_size``, li divide in batch
3. Per ogni batch, prompt Gemini Flash con query + testi candidati
4. Gemini ritorna un ranking JSON con score per ogni candidato
5. Merge dei batch, ri-ordina per score, tronca a top_k
6. Fallback: se Gemini fallisce, mantiene ordine RRF originale

Parametri configurabili da ``config.yaml`` → sezione ``retrieval``:
- ``rerank_batch_size``: max candidati per batch (default: 20)
- ``top_k_final``: risultati finali dopo reranking (default: 10)

Uso::

    from paperhunter.indexing.reranker import rerank, GeminiReranker

    # Funzione standalone
    reranked = rerank(query="prompt injection defenses", candidates=results, top_k=10)

    # Classe (riuso LLM)
    reranker = GeminiReranker()
    reranked = reranker.rerank(query="...", candidates=results, top_k=10)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Prompt Template
# ===========================================================================

_RERANK_PROMPT_TEMPLATE = """\
You are an expert academic search reranker specializing in LLM security research.

Given a QUERY and a list of CANDIDATE text passages, evaluate how relevant each candidate \
is to the query. Assign a relevance score from 0.0 (completely irrelevant) to 1.0 \
(perfectly relevant).

Scoring guidelines:
- 1.0: Directly answers or addresses the query with specific, detailed information
- 0.8-0.9: Highly relevant, covers the main topic with useful details
- 0.6-0.7: Moderately relevant, covers related aspects but not the core question
- 0.4-0.5: Tangentially relevant, mentions related concepts but not the query topic
- 0.1-0.3: Barely relevant, only superficially related
- 0.0: Completely irrelevant

QUERY: {query}

CANDIDATES:
{candidates_text}

Respond with ONLY a JSON array. Each element has "id" (the candidate number, 1-based \
integer) and "score" (float 0.0-1.0). Order by score descending.

Example response format:
[{{"id": 3, "score": 0.95}}, {{"id": 1, "score": 0.82}}, {{"id": 2, "score": 0.41}}]

JSON array:"""


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class RerankResult:
    """Risultato di una singola operazione di reranking.

    Attributes:
        reranked: Lista di candidati riordinati per rilevanza.
        original_count: Numero di candidati in input.
        output_count: Numero di candidati in output (dopo top_k).
        model_used: Modello usato per il reranking.
        elapsed_seconds: Tempo impiegato.
        fallback_used: True se è stato usato il fallback (ordine originale).
        batches_processed: Numero di batch processati.
    """
    reranked: list[dict[str, Any]]
    original_count: int
    output_count: int
    model_used: str
    elapsed_seconds: float
    fallback_used: bool = False
    batches_processed: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serializza in dizionario."""
        return {
            "original_count": self.original_count,
            "output_count": self.output_count,
            "model_used": self.model_used,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "fallback_used": self.fallback_used,
            "batches_processed": self.batches_processed,
        }


# ===========================================================================
# GeminiReranker
# ===========================================================================

class GeminiReranker:
    """Reranker basato su Gemini Flash come cross-encoder.

    Valuta la rilevanza di ogni candidato rispetto alla query e
    riordina i risultati. Supporta batching per liste grandi e
    fallback automatico all'ordine originale in caso di errore.

    Args:
        batch_size: Massimo candidati per batch. Se None, da config.yaml.
        top_k: Numero di risultati finali. Se None, da config.yaml.
        max_text_chars: Massimo caratteri per candidato nel prompt (troncamento).
        llm: Istanza LLM da usare. Se None, crea Gemini Flash automaticamente.

    Uso::

        reranker = GeminiReranker()
        results = reranker.rerank("prompt injection defense", candidates)
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        max_text_chars: int = 500,
        llm: Optional[Any] = None,
    ):
        from paperhunter.config import settings

        self._batch_size = batch_size or settings.retrieval.rerank_batch_size
        self._top_k = top_k or settings.retrieval.top_k_final
        self._max_text_chars = max_text_chars
        self._llm = llm

    @property
    def llm(self):
        """Lazy initialization del modello LLM.

        Usa un timeout di 90s (molto più corto del default 600s per Pro)
        perché il reranking con Flash è un task breve.
        Reranking is a short deterministic JSON task, so thinking is disabled
        to reserve the completion budget for the ranking itself.
        """
        if self._llm is None:
            from paperhunter.utils.llm_factory import get_flash_llm
            self._llm = get_flash_llm(
                temperature=0.0,
                timeout_seconds=90,
                max_tokens=4096,
                thinking_enabled=False,
            )
            logger.debug(
                "Reranker: LLM Flash inizializzato "
                "(thinking=off, timeout=90s, max_tokens=4096)"
            )
        return self._llm

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        top_k: Optional[int] = None,
    ) -> list[Any]:
        """Riordina i candidati per rilevanza rispetto alla query.

        Accetta sia ``list[RetrievedChunk]`` (dataclass dal hybrid_retriever)
        sia ``list[dict]`` con almeno ``chunk_id`` e ``text``.

        Se i candidati superano ``batch_size``, vengono divisi in batch,
        ciascuno reranked separatamente, poi i risultati sono fusi
        per score decrescente.

        Se Gemini fallisce (errore API, JSON non parsabile, ecc.),
        ritorna i candidati nell'ordine originale (fallback RRF).

        Args:
            query: Query testuale dell'utente.
            candidates: Lista di candidati da riordinare.
            top_k: Override del numero di risultati (default: self._top_k).

        Returns:
            Lista di candidati riordinati, dello stesso tipo dell'input
            (RetrievedChunk o dict), troncata a top_k.
        """
        t0 = time.monotonic()
        effective_top_k = top_k or self._top_k

        if not query or not query.strip():
            logger.warning("Reranker: query vuota, ritorno candidati invariati")
            return candidates[:effective_top_k]

        if not candidates:
            logger.debug("Reranker: nessun candidato da riordinare")
            return []

        # Normalizzare i candidati in formato uniforme (list[dict])
        normalized, is_dataclass_input = self._normalize_candidates(candidates)

        if len(normalized) <= 1:
            logger.debug("Reranker: ≤1 candidato, nessun reranking necessario")
            return candidates[:effective_top_k]

        # Dividere in batch se necessario
        batches = self._split_into_batches(normalized)
        logger.info(
            f"Reranker: {len(normalized)} candidati, "
            f"{len(batches)} batch (batch_size={self._batch_size}), "
            f"top_k={effective_top_k}"
        )

        # Processare ogni batch
        all_scored: list[dict[str, Any]] = []
        fallback_used = False

        for batch_idx, batch in enumerate(batches):
            scored = self._rerank_batch(query, batch, batch_idx, len(batches))
            if scored is None:
                # Fallback: mantieni ordine originale con score decrescente fittizio
                logger.warning(
                    f"Reranker: batch {batch_idx + 1}/{len(batches)} fallito, "
                    f"mantengo ordine originale per questo batch"
                )
                fallback_used = True
                for i, item in enumerate(batch):
                    item["rerank_score"] = 1.0 - (i * 0.001)  # Preserva ordine
                all_scored.extend(batch)
            else:
                all_scored.extend(scored)

        # Ordinare per rerank_score decrescente e troncare
        all_scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_results = all_scored[:effective_top_k]

        elapsed = time.monotonic() - t0

        # Log risultati
        if fallback_used:
            logger.warning(
                f"Reranker completato con FALLBACK in {elapsed:.1f}s: "
                f"{len(normalized)} → {len(top_results)} candidati"
            )
        else:
            logger.info(
                f"Reranker completato in {elapsed:.1f}s: "
                f"{len(normalized)} → {len(top_results)} candidati, "
                f"top score={top_results[0].get('rerank_score', 0):.3f}"
            )

        # Ricostruire output nel formato originale
        output = self._denormalize_results(
            top_results, candidates, is_dataclass_input
        )

        return output

    def rerank_with_details(
        self,
        query: str,
        candidates: list[Any],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Come ``rerank()``, ma ritorna anche metadati sull'operazione.

        Args:
            query: Query testuale.
            candidates: Candidati da riordinare.
            top_k: Override top_k.

        Returns:
            ``RerankResult`` con candidati reranked e statistiche.
        """
        t0 = time.monotonic()
        effective_top_k = top_k or self._top_k

        if not candidates:
            return RerankResult(
                reranked=[],
                original_count=0,
                output_count=0,
                model_used=self._get_model_name(),
                elapsed_seconds=0.0,
            )

        normalized, is_dataclass_input = self._normalize_candidates(candidates)
        batches = self._split_into_batches(normalized)

        all_scored: list[dict[str, Any]] = []
        fallback_used = False

        for batch_idx, batch in enumerate(batches):
            scored = self._rerank_batch(query, batch, batch_idx, len(batches))
            if scored is None:
                fallback_used = True
                for i, item in enumerate(batch):
                    item["rerank_score"] = 1.0 - (i * 0.001)
                all_scored.extend(batch)
            else:
                all_scored.extend(scored)

        all_scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_results = all_scored[:effective_top_k]

        elapsed = time.monotonic() - t0

        output = self._denormalize_results(
            top_results, candidates, is_dataclass_input
        )

        return RerankResult(
            reranked=output,
            original_count=len(normalized),
            output_count=len(output),
            model_used=self._get_model_name(),
            elapsed_seconds=elapsed,
            fallback_used=fallback_used,
            batches_processed=len(batches),
        )

    # ------------------------------------------------------------------
    # Batch Reranking
    # ------------------------------------------------------------------

    def _rerank_batch(
        self,
        query: str,
        batch: list[dict[str, Any]],
        batch_idx: int,
        total_batches: int,
    ) -> Optional[list[dict[str, Any]]]:
        """Rerank di un singolo batch con Gemini Flash.

        Args:
            query: Query dell'utente.
            batch: Lista di candidati normalizzati (dict).
            batch_idx: Indice del batch corrente (0-based).
            total_batches: Numero totale di batch.

        Returns:
            Lista di candidati con ``rerank_score`` assegnato, oppure
            None se il reranking fallisce (trigger fallback).
        """
        from paperhunter.utils.llm_factory import llm_call_with_retry

        # Costruire il prompt con i candidati
        candidates_text = self._format_candidates_for_prompt(batch)
        prompt = _RERANK_PROMPT_TEMPLATE.format(
            query=query,
            candidates_text=candidates_text,
        )

        try:
            raw_response = llm_call_with_retry(
                self.llm,
                prompt,
                max_retries=2,  # Max 3 tentativi per il reranking
                base_delay=5.0,
                max_delay=30.0,
            )
        except Exception as e:
            logger.error(
                f"Reranker: chiamata LLM fallita per batch "
                f"{batch_idx + 1}/{total_batches}: {e}"
            )
            return None

        # Parsare la risposta JSON
        scores = self._parse_rerank_response(raw_response, len(batch))
        if scores is None:
            logger.warning(
                f"Reranker: parsing fallito per batch "
                f"{batch_idx + 1}/{total_batches}"
            )
            return None

        # Assegnare gli score ai candidati
        for item in batch:
            item_idx = item["_batch_index"]  # 1-based index nel batch
            item["rerank_score"] = scores.get(item_idx, 0.0)

        logger.debug(
            f"Reranker batch {batch_idx + 1}/{total_batches}: "
            f"{len(batch)} candidati scored, "
            f"top={max(scores.values()):.3f}, "
            f"min={min(scores.values()):.3f}"
        )

        return batch

    # ------------------------------------------------------------------
    # Prompt Formatting
    # ------------------------------------------------------------------

    def _format_candidates_for_prompt(
        self,
        batch: list[dict[str, Any]],
    ) -> str:
        """Formatta i candidati per il prompt di reranking.

        Ogni candidato mostra: numero, metadata chiave (categoria, anno,
        sezione, tipo), testo troncato.

        Args:
            batch: Lista di candidati normalizzati.

        Returns:
            Testo formattato per il prompt.
        """
        lines: list[str] = []

        for i, item in enumerate(batch):
            idx = i + 1  # 1-based per il prompt
            item["_batch_index"] = idx  # Salva per il mapping successivo

            text = item.get("text", "")
            if len(text) > self._max_text_chars:
                text = text[:self._max_text_chars] + "..."

            # Metadata utili per il contesto
            meta = item.get("metadata", {})
            meta_parts = []
            if meta.get("category"):
                meta_parts.append(f"Category: {meta['category']}")
            if meta.get("year"):
                meta_parts.append(f"Year: {meta['year']}")
            if meta.get("section_name"):
                meta_parts.append(f"Section: {meta['section_name']}")
            if meta.get("chunk_type"):
                meta_parts.append(f"Type: {meta['chunk_type']}")
            if meta.get("bibtex_key"):
                meta_parts.append(f"Paper: {meta['bibtex_key']}")

            meta_str = " | ".join(meta_parts) if meta_parts else "N/A"

            lines.append(
                f"[Candidate {idx}] ({meta_str})\n{text}\n"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_rerank_response(
        self,
        raw_response: str,
        expected_count: int,
    ) -> Optional[dict[int, float]]:
        """Parsa la risposta JSON di Gemini con gli score di reranking.

        Gestisce:
        - Code fences (```json ... ```)
        - Array JSON puro
        - Risposte con testo extra prima/dopo il JSON
        - Score fuori range (clampati a 0.0-1.0)
        - Candidati mancanti nella risposta (assegnato score 0.0)

        Args:
            raw_response: Risposta raw di Gemini.
            expected_count: Numero di candidati attesi.

        Returns:
            Dict ``{candidate_id_1based: score}`` oppure None se il
            parsing fallisce completamente.
        """
        if not raw_response or not raw_response.strip():
            logger.warning("Reranker: risposta vuota")
            return None

        text = raw_response.strip()

        # Rimuovere code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        # Tentativo 1: parsing JSON diretto
        parsed = self._try_parse_json_array(text)

        # Tentativo 2: estrarre array JSON dal testo
        if parsed is None:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = self._try_parse_json_array(match.group(0))

        # Tentativo 3: regex per estrarre coppie id/score
        if parsed is None:
            parsed = self._regex_extract_scores(text, expected_count)

        if parsed is None:
            logger.warning(
                f"Reranker: impossibile parsare risposta "
                f"(lunghezza={len(raw_response)} chars)"
            )
            return None

        # Convertire in dict id→score con validazione
        scores: dict[int, float] = {}
        for entry in parsed:
            try:
                cid = int(entry.get("id", 0))
                score = float(entry.get("score", 0.0))
                # Validazione range
                score = max(0.0, min(1.0, score))
                if 1 <= cid <= expected_count:
                    scores[cid] = score
            except (TypeError, ValueError):
                continue

        if not scores:
            logger.warning("Reranker: nessuno score valido estratto dalla risposta")
            return None

        # Assegnare score 0.0 ai candidati mancanti
        for i in range(1, expected_count + 1):
            if i not in scores:
                scores[i] = 0.0
                logger.debug(f"Reranker: candidato {i} assente nella risposta, score=0.0")

        return scores

    def _try_parse_json_array(self, text: str) -> Optional[list[dict]]:
        """Tenta il parsing di un array JSON.

        Args:
            text: Testo da parsare.

        Returns:
            Lista di dict oppure None.
        """
        try:
            # Fix trailing commas
            text = re.sub(r",\s*([}\]])", r"\1", text)
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _regex_extract_scores(
        self,
        text: str,
        expected_count: int,
    ) -> Optional[list[dict]]:
        """Fallback: estrai coppie id/score con regex.

        Cerca pattern come:
        - "id": 3, "score": 0.95
        - id: 3, score: 0.95
        - Candidate 3: 0.95
        - 3 → 0.95

        Args:
            text: Testo della risposta.
            expected_count: Numero di candidati attesi.

        Returns:
            Lista di dict ``[{"id": N, "score": X.X}]`` oppure None.
        """
        results = []

        # Pattern 1: "id": N, "score": X.X
        pattern1 = re.findall(
            r'"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([\d.]+)', text
        )
        if pattern1:
            for cid, score in pattern1:
                results.append({"id": int(cid), "score": float(score)})

        # Pattern 2: id: N, score: X.X (senza virgolette)
        if not results:
            pattern2 = re.findall(
                r'id\s*:\s*(\d+)\s*,?\s*score\s*:\s*([\d.]+)', text,
                re.IGNORECASE,
            )
            if pattern2:
                for cid, score in pattern2:
                    results.append({"id": int(cid), "score": float(score)})

        # Pattern 3: Candidate N: X.X o N → X.X
        if not results:
            pattern3 = re.findall(
                r'(?:Candidate\s+)?(\d+)\s*[:\-→=]\s*(0?\.\d+|1\.0+|0\.0+)',
                text,
                re.IGNORECASE,
            )
            if pattern3:
                for cid, score in pattern3:
                    results.append({"id": int(cid), "score": float(score)})

        return results if results else None

    # ------------------------------------------------------------------
    # Normalization & Denormalization
    # ------------------------------------------------------------------

    def _normalize_candidates(
        self,
        candidates: list[Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Normalizza i candidati in formato uniforme (list[dict]).

        Accetta sia RetrievedChunk (dataclass) che dict.

        Args:
            candidates: Lista eterogenea di candidati.

        Returns:
            Tupla (lista_normalizzata, is_dataclass_input).
        """
        from paperhunter.indexing.hybrid_retriever import RetrievedChunk

        is_dataclass = False
        normalized: list[dict[str, Any]] = []

        for i, c in enumerate(candidates):
            if isinstance(c, RetrievedChunk):
                is_dataclass = True
                normalized.append({
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": c.score,
                    "metadata": c.metadata,
                    "retrieval_method": c.retrieval_method,
                    "vector_score": c.vector_score,
                    "bm25_score": c.bm25_score,
                    "vector_rank": c.vector_rank,
                    "bm25_rank": c.bm25_rank,
                    "_original_index": i,
                })
            elif isinstance(c, dict):
                entry = dict(c)  # Copia per non mutare l'originale
                entry["_original_index"] = i
                if "chunk_id" not in entry:
                    entry["chunk_id"] = f"candidate_{i}"
                if "text" not in entry:
                    entry["text"] = ""
                normalized.append(entry)
            else:
                logger.warning(f"Reranker: candidato tipo sconosciuto ({type(c)}), skip")

        return normalized, is_dataclass

    def _denormalize_results(
        self,
        scored: list[dict[str, Any]],
        original_candidates: list[Any],
        is_dataclass_input: bool,
    ) -> list[Any]:
        """Ricostruisce l'output nel formato originale.

        Se l'input erano RetrievedChunk, l'output sarà RetrievedChunk
        con lo score aggiornato. Se l'input erano dict, aggiunge
        ``rerank_score`` al dict originale.

        Args:
            scored: Candidati con rerank_score, ordinati.
            original_candidates: Candidati originali (per riferiemento).
            is_dataclass_input: True se input era RetrievedChunk.

        Returns:
            Lista di candidati nel formato originale, riordinati.
        """
        from paperhunter.indexing.hybrid_retriever import RetrievedChunk

        output = []

        for item in scored:
            orig_idx = item.get("_original_index")
            rerank_score = item.get("rerank_score", 0.0)

            if is_dataclass_input and orig_idx is not None:
                orig = original_candidates[orig_idx]
                if isinstance(orig, RetrievedChunk):
                    # Crea una copia con lo score aggiornato
                    reranked = RetrievedChunk(
                        chunk_id=orig.chunk_id,
                        text=orig.text,
                        score=rerank_score,
                        retrieval_method=orig.retrieval_method,
                        metadata=orig.metadata,
                        vector_score=orig.vector_score,
                        bm25_score=orig.bm25_score,
                        vector_rank=orig.vector_rank,
                        bm25_rank=orig.bm25_rank,
                    )
                    output.append(reranked)
                    continue

            # Fallback: dict output
            clean = {k: v for k, v in item.items()
                     if not k.startswith("_")}
            clean["rerank_score"] = rerank_score
            output.append(clean)

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_into_batches(
        self,
        items: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Divide i candidati in batch di dimensione ``batch_size``.

        Args:
            items: Lista di candidati.

        Returns:
            Lista di batch (liste di candidati).
        """
        if len(items) <= self._batch_size:
            return [items]

        batches = []
        for i in range(0, len(items), self._batch_size):
            batches.append(items[i:i + self._batch_size])
        return batches

    def _get_model_name(self) -> str:
        """Ritorna il nome del modello in uso."""
        try:
            return getattr(self.llm, "model", "gemini-flash")
        except Exception:
            return "gemini-flash"


# ===========================================================================
# Funzione standalone
# ===========================================================================

def rerank(
    query: str,
    candidates: list[Any],
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    llm: Optional[Any] = None,
) -> list[Any]:
    """Funzione standalone per reranking con Gemini Flash.

    Convenience wrapper attorno a ``GeminiReranker.rerank()``.

    Args:
        query: Query testuale.
        candidates: Lista di candidati (RetrievedChunk o dict).
        top_k: Numero di risultati finali.
        batch_size: Dimensione batch.
        llm: Istanza LLM da usare (se None, crea Gemini Flash).

    Returns:
        Lista riordinata di candidati, troncata a top_k.
    """
    reranker = GeminiReranker(
        batch_size=batch_size,
        top_k=top_k or 10,
        llm=llm,
    )
    return reranker.rerank(query=query, candidates=candidates, top_k=top_k)


def rerank_with_details(
    query: str,
    candidates: list[Any],
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    llm: Optional[Any] = None,
) -> RerankResult:
    """Funzione standalone per reranking con dettagli.

    Args:
        query: Query testuale.
        candidates: Candidati da riordinare.
        top_k: Numero di risultati finali.
        batch_size: Dimensione batch.
        llm: Istanza LLM da usare.

    Returns:
        ``RerankResult`` con candidati reranked e statistiche.
    """
    reranker = GeminiReranker(
        batch_size=batch_size,
        top_k=top_k or 10,
        llm=llm,
    )
    return reranker.rerank_with_details(
        query=query, candidates=candidates, top_k=top_k
    )
