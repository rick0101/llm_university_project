"""
Modulo Ingestion — Pipeline di processing incrementale dei PDF scientifici.

Componenti:
- ``grobid_client.GrobidClient`` — Client GROBID Docker (PDF → TEI-XML → strutturato)
- ``pdf_hasher`` — Hash SHA-256 per upsert incrementale
- ``semantic_scholar.SemanticScholarEnricher`` — Enrichment metadati S2 API
- ``bibtex_generator`` — Generazione BibTeX automatica
- ``chunker`` — Chunking GROBID-aware per sezione
- ``ingestion_pipeline.IngestionPipeline`` — Orchestratore ingestion completo
"""

from paperhunter.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    IngestionReport,
    sort_pdfs_by_recency,
)

__all__ = ["IngestionPipeline", "IngestionReport", "sort_pdfs_by_recency"]
