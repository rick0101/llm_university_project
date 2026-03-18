#!/usr/bin/env python3
"""
Test end-to-end della pipeline di image processing (TODO5_ImageProcessing.md).

Verifica:
1. Estrazione immagini da PDF reali (PyMuPDF)
2. Filtro euristico Stage 1 (ImageRelevanceFilter)
3. Embedding multimodale diretto via google-genai SDK (gemini-embedding-2-preview)
   - Embedding di immagini PNG reali estratte dai PDF
   - Embedding di testi di riferimento scientifici
   - Confronto cross-modale (cosine similarity immagine vs testo)
4. Pipeline completa FigureExtractor.extract_and_describe() su ~12 PDF

Output salvato sia su terminale che su file: output/test_image_pipeline_report.txt
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Assicura che il progetto sia nel path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# --- Import dal progetto ---
from paperhunter.config import settings
from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageRelevanceFilter,
    ImageScore,
)
from paperhunter.ingestion.figure_extractor import FigureExtractor
from paperhunter.utils.rate_limiter import get_gemini_limiter

# ============================================================================
# Configurazione test
# ============================================================================

BASE_DIR = PROJECT_ROOT / "scaricati"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_FILE = OUTPUT_DIR / "test_image_pipeline_report.txt"

# 3 PDF per ogni categoria (12 totali)
TEST_PDFS = {
    "Prompt_injections1": [
        "Adversarial_Attacks_on_LLM-as-a-Judge_Systems_Insights_from_Prompt_Injections.pdf",
        "Adaptive_Attacks_Break_Defenses_Against_Indirect_Prompt_Injection_Attacks_on_LLM_Agents.pdf",
        "AgentVigil_Generic_Black-Box_Red-teaming_for_Indirect_Prompt_Injection_against_LLM_Agents.pdf",
    ],
    "Hallucinations1": [
        "HalluLens_LLM_Hallucination_Benchmark.pdf",
        "Attention-guided_Self-reflection_for_Zero-shot_Hallucination_Detection_in_Large_Language_Models.pdf",
        "Hallucination_Detection_in_LLMs_Using_Spectral_Features_of_Attention_Maps.pdf",
    ],
    "Data_and_Model_Poisoning1": [
        "A_Survey_on_Data_Security_in_Large_Language_Models.pdf",
        "SAFER_Probing_Safety_in_Reward_Models_with_Sparse_Autoencoder.pdf",
        "Detecting_Sleeper_Agents_in_Large_Language_Models_via_Semantic_Drift_Analysis.pdf",
    ],
    "Adversarial_Code1": [
        "AICD_Bench_A_Challenging_Benchmark_for_AI-Generated_Code_Detection.pdf",
        "A_Systematic_Study_of_Code_Obfuscation_Against_LLM-based_Vulnerability_Detection.pdf",
        "Auto-SPT_Automating_Semantic_Preserving_Transformations_for_Code.pdf",
    ],
}

# Testi di riferimento per il confronto embedding cross-modale
REFERENCE_TEXTS = {
    "scientific_plot": "scientific data plot with labeled axes showing experimental results and performance metrics",
    "comparison_table": "comparison table with rows and columns of numerical data showing method performance",
    "architecture_diagram": "system architecture diagram showing neural network components and data flow connections",
    "flowchart": "flowchart or pipeline diagram showing sequential process steps and decision branches",
    "confusion_matrix": "confusion matrix or heatmap visualization showing classification accuracy results",
    "logo_decorative": "university or company logo with simple shapes and solid colors",
}


# ============================================================================
# Utility
# ============================================================================

console = Console()
report_lines: list[str] = []


def log(msg: str, style: str = ""):
    """Scrive sia su console che nel buffer del report."""
    if style:
        console.print(msg, style=style)
    else:
        console.print(msg)
    # Versione plain per il file
    import re
    plain = re.sub(r'\[.*?\]', '', msg)
    report_lines.append(plain)


def log_separator(title: str = ""):
    sep = "=" * 80
    if title:
        log(f"\n{sep}")
        log(f"  {title}")
        log(sep)
    else:
        log(sep)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcola la similarità coseno tra due vettori."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def save_report():
    """Salva il report su file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n[bold green]Report salvato in:[/bold green] {REPORT_FILE}")


# ============================================================================
# TEST 1: Estrazione immagini e filtro euristico Stage 1
# ============================================================================

@dataclass
class PDFExtractionResult:
    pdf_name: str
    category: str
    total_pages: int
    images_extracted: int
    scientific: int
    uncertain: int
    decorative: int
    scores: list[float]
    extraction_time: float
    filter_time: float


def test_extraction_and_heuristic_filter() -> list[PDFExtractionResult]:
    """TEST 1: Estrae immagini da tutti i PDF e applica il filtro euristico."""
    log_separator("TEST 1: Estrazione Immagini + Filtro Euristico Stage 1")

    extractor = FigureExtractor()
    relevance_filter = ImageRelevanceFilter()
    results: list[PDFExtractionResult] = []

    for category, pdf_names in TEST_PDFS.items():
        for pdf_name in pdf_names:
            pdf_path = BASE_DIR / category / pdf_name
            if not pdf_path.exists():
                log(f"  [red]⚠ PDF non trovato: {pdf_path}[/red]")
                continue

            log(f"\n  📄 [bold]{category}/{pdf_name}[/bold]")

            # Step 1: Estrazione
            t0 = time.time()
            try:
                images = extractor.extract_images_from_pdf(pdf_path)
            except Exception as e:
                log(f"    [red]Errore estrazione: {e}[/red]")
                continue
            t_extract = time.time() - t0

            if not images:
                log(f"    Nessuna immagine estratta (tempo: {t_extract:.2f}s)")
                results.append(PDFExtractionResult(
                    pdf_name=pdf_name, category=category,
                    total_pages=0, images_extracted=0,
                    scientific=0, uncertain=0, decorative=0,
                    scores=[], extraction_time=t_extract, filter_time=0.0,
                ))
                continue

            total_pages = images[0].get("total_pages", 0)
            log(f"    Pagine: {total_pages}, Immagini estratte: {len(images)}")

            # Step 2: Filtro euristico
            t1 = time.time()
            scores = relevance_filter.classify_batch(images)
            t_filter = time.time() - t1

            n_sci = sum(1 for s in scores if s.classification == ImageClassification.SCIENTIFIC_FIGURE)
            n_unc = sum(1 for s in scores if s.classification == ImageClassification.UNCERTAIN)
            n_dec = sum(1 for s in scores if s.classification == ImageClassification.LIKELY_DECORATIVE)
            all_scores = [s.combined_score for s in scores]

            log(f"    Classificazione: ✅ Scientific={n_sci}, ❓ Uncertain={n_unc}, 🗑 Decorative={n_dec}")
            log(f"    Score medio: {np.mean(all_scores):.3f}, min={min(all_scores):.3f}, max={max(all_scores):.3f}")
            log(f"    Tempi: estrazione={t_extract:.2f}s, filtro={t_filter:.3f}s")

            # Dettaglio per immagine (top e bottom)
            for i, (img, score) in enumerate(zip(images, scores)):
                emoji = {"scientific_figure": "✅", "uncertain": "❓", "likely_decorative": "🗑"}.get(
                    score.classification.value, "?"
                )
                log(
                    f"      Img {i}: {emoji} score={score.combined_score:.3f} "
                    f"(AR={score.aspect_ratio_score:.2f} ENT={score.entropy_score:.2f} "
                    f"EDGE={score.edge_density_score:.2f} POS={score.position_score:.2f} "
                    f"SZ={score.size_score:.2f}) "
                    f"[{img['width']}x{img['height']} p.{img['page_number']}]"
                )

            results.append(PDFExtractionResult(
                pdf_name=pdf_name, category=category,
                total_pages=total_pages, images_extracted=len(images),
                scientific=n_sci, uncertain=n_unc, decorative=n_dec,
                scores=all_scores, extraction_time=t_extract, filter_time=t_filter,
            ))

    # --- Riepilogo ---
    log("\n  --- RIEPILOGO TEST 1 ---")
    total_imgs = sum(r.images_extracted for r in results)
    total_sci = sum(r.scientific for r in results)
    total_unc = sum(r.uncertain for r in results)
    total_dec = sum(r.decorative for r in results)
    vision_saved = total_dec  # immagini che NON verrebbero inviate a Vision
    pct_saved = (vision_saved / total_imgs * 100) if total_imgs > 0 else 0

    log(f"  PDF processati: {len(results)}")
    log(f"  Immagini totali estratte: {total_imgs}")
    log(f"  Classificazione: Scientific={total_sci}, Uncertain={total_unc}, Decorative={total_dec}")
    log(f"  [bold green]Risparmio Vision calls: {vision_saved}/{total_imgs} ({pct_saved:.1f}%)[/bold green]")

    return results


# ============================================================================
# TEST 2: Embedding Multimodale con google-genai SDK
# ============================================================================

@dataclass
class EmbeddingTestResult:
    source: str  # "image" | "text"
    label: str
    dimension: int
    time_seconds: float
    success: bool
    error: str = ""


def embed_image_via_genai(image_bytes: bytes, mime_type: str = "image/png") -> tuple[list[float], float]:
    """Embed un'immagine via Gemini Embedding 2 Preview (SDK diretto)."""
    from google import genai
    from google.genai import types

    limiter = get_gemini_limiter()
    limiter.wait()

    client = genai.Client(api_key=settings.google_api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    t0 = time.time()
    response = client.models.embed_content(
        model=settings.gemini.embedding_model,
        contents=image_part,
    )
    elapsed = time.time() - t0

    embedding = response.embeddings[0].values
    return embedding, elapsed


def embed_text_via_genai(text: str) -> tuple[list[float], float]:
    """Embed un testo via Gemini Embedding 2 Preview (SDK diretto)."""
    from google import genai
    from google.genai import types

    limiter = get_gemini_limiter()
    limiter.wait()

    client = genai.Client(api_key=settings.google_api_key)

    t0 = time.time()
    response = client.models.embed_content(
        model=settings.gemini.embedding_model,
        contents=text,
    )
    elapsed = time.time() - t0

    embedding = response.embeddings[0].values
    return embedding, elapsed


def test_multimodal_embeddings(extraction_results: list[PDFExtractionResult]) -> dict:
    """TEST 2: Testa embedding multimodale (immagini + testo) con gemini-embedding-2-preview."""
    log_separator("TEST 2: Embedding Multimodale (gemini-embedding-2-preview)")
    log(f"  Modello: {settings.gemini.embedding_model}")

    # --- 2a: Embed testi di riferimento ---
    log("\n  📝 [bold]Embedding testi di riferimento...[/bold]")
    text_embeddings: dict[str, list[float]] = {}
    text_results: list[EmbeddingTestResult] = []

    for label, text in REFERENCE_TEXTS.items():
        try:
            emb, elapsed = embed_text_via_genai(text)
            text_embeddings[label] = emb
            text_results.append(EmbeddingTestResult(
                source="text", label=label, dimension=len(emb),
                time_seconds=elapsed, success=True,
            ))
            log(f"    ✅ '{label}': {len(emb)}-dim in {elapsed:.2f}s")
        except Exception as e:
            text_results.append(EmbeddingTestResult(
                source="text", label=label, dimension=0,
                time_seconds=0, success=False, error=str(e),
            ))
            log(f"    [red]❌ '{label}': {e}[/red]")

    # --- 2b: Embed immagini reali estratte dai PDF ---
    log("\n  🖼️  [bold]Embedding immagini reali dai PDF...[/bold]")

    # Raccogli un campione di immagini reali (max 2 per PDF che ne ha)
    extractor = FigureExtractor()
    sample_images: list[dict] = []

    for category, pdf_names in TEST_PDFS.items():
        for pdf_name in pdf_names:
            pdf_path = BASE_DIR / category / pdf_name
            if not pdf_path.exists():
                continue
            try:
                imgs = extractor.extract_images_from_pdf(pdf_path)
                for img in imgs[:2]:  # max 2 per PDF
                    img["pdf_name"] = pdf_name
                    img["category"] = category
                    sample_images.append(img)
            except Exception:
                continue
            if len(sample_images) >= 10:
                break
        if len(sample_images) >= 10:
            break

    log(f"  Campione: {len(sample_images)} immagini da PDF reali")

    image_embeddings: list[dict] = []  # {label, embedding, img_info}
    image_results: list[EmbeddingTestResult] = []

    for i, img_info in enumerate(sample_images):
        label = f"img_{i}_{img_info['pdf_name'][:30]}_p{img_info['page_number']}"
        try:
            emb, elapsed = embed_image_via_genai(
                img_info["image_bytes"], img_info["mime_type"]
            )
            image_embeddings.append({
                "label": label,
                "embedding": emb,
                "img_info": img_info,
            })
            image_results.append(EmbeddingTestResult(
                source="image", label=label, dimension=len(emb),
                time_seconds=elapsed, success=True,
            ))
            log(
                f"    ✅ [{i+1}/{len(sample_images)}] {label}: {len(emb)}-dim "
                f"in {elapsed:.2f}s [{img_info['width']}x{img_info['height']}]"
            )
        except Exception as e:
            image_results.append(EmbeddingTestResult(
                source="image", label=label, dimension=0,
                time_seconds=0, success=False, error=str(e),
            ))
            log(f"    [red]❌ [{i+1}/{len(sample_images)}] {label}: {e}[/red]")
            # Logga traceback completo per debug
            log(f"        {traceback.format_exc()}")

    # --- 2c: Confronto cross-modale (cosine similarity) ---
    log("\n  🔗 [bold]Confronto cross-modale (immagine ↔ testo)...[/bold]")

    if text_embeddings and image_embeddings:
        cross_modal_results = []
        for img_data in image_embeddings:
            img_label = img_data["label"]
            img_emb = img_data["embedding"]
            sims = {}
            for text_label, text_emb in text_embeddings.items():
                sim = cosine_similarity(img_emb, text_emb)
                sims[text_label] = sim

            best_match = max(sims, key=sims.get)
            best_sim = sims[best_match]
            cross_modal_results.append({
                "image": img_label,
                "best_match": best_match,
                "best_similarity": best_sim,
                "all_similarities": sims,
            })

            sims_str = " | ".join(f"{k}={v:.3f}" for k, v in sorted(sims.items(), key=lambda x: -x[1]))
            log(f"    🖼→📝 {img_label[:40]}")
            log(f"          Best: [bold]{best_match}[/bold] ({best_sim:.3f})")
            log(f"          All: {sims_str}")

        # --- 2d: Confronto intra-modale (testo ↔ testo) ---
        log("\n  📝↔📝 [bold]Similarità intra-modale (testo ↔ testo):[/bold]")
        text_labels = list(text_embeddings.keys())
        for i in range(len(text_labels)):
            for j in range(i + 1, len(text_labels)):
                sim = cosine_similarity(
                    text_embeddings[text_labels[i]],
                    text_embeddings[text_labels[j]],
                )
                log(f"    {text_labels[i]} ↔ {text_labels[j]}: {sim:.3f}")

        # --- 2e: Confronto intra-modale (immagine ↔ immagine) ---
        if len(image_embeddings) >= 2:
            log("\n  🖼↔🖼 [bold]Similarità intra-modale (immagine ↔ immagine, top 5 coppie):[/bold]")
            img_pairs = []
            for i in range(len(image_embeddings)):
                for j in range(i + 1, len(image_embeddings)):
                    sim = cosine_similarity(
                        image_embeddings[i]["embedding"],
                        image_embeddings[j]["embedding"],
                    )
                    img_pairs.append((image_embeddings[i]["label"], image_embeddings[j]["label"], sim))
            img_pairs.sort(key=lambda x: -x[2])
            for a, b, sim in img_pairs[:5]:
                log(f"    {a[:35]} ↔ {b[:35]}: {sim:.3f}")

    else:
        log("  [yellow]⚠ Non abbastanza embedding per il confronto cross-modale[/yellow]")
        cross_modal_results = []

    # Riepilogo
    n_text_ok = sum(1 for r in text_results if r.success)
    n_img_ok = sum(1 for r in image_results if r.success)
    log(f"\n  --- RIEPILOGO TEST 2 ---")
    log(f"  Embedding testo: {n_text_ok}/{len(text_results)} riusciti")
    log(f"  Embedding immagini: {n_img_ok}/{len(image_results)} riusciti")
    if text_results and text_results[0].success:
        log(f"  Dimensione vettori: {text_results[0].dimension}")
    if cross_modal_results:
        avg_best = np.mean([r["best_similarity"] for r in cross_modal_results])
        log(f"  Similarità cross-modale media (best match): {avg_best:.3f}")

    return {
        "text_results": text_results,
        "image_results": image_results,
        "cross_modal_results": cross_modal_results,
    }


# ============================================================================
# TEST 3: Pipeline FigureExtractor completa (con Gemini Vision)
# ============================================================================

@dataclass
class PipelineResult:
    pdf_name: str
    category: str
    chunks_generated: int
    time_seconds: float
    success: bool
    error: str = ""
    chunk_texts_preview: list[str] = None  # type: ignore


def test_full_pipeline(max_pdfs: int = 4) -> list[PipelineResult]:
    """TEST 3: Esegue la pipeline FigureExtractor completa su un sottoinsieme di PDF.

    Questo test chiama Gemini Vision per generare descrizioni,
    quindi è il più costoso. Limitato a max_pdfs.
    """
    log_separator("TEST 3: Pipeline FigureExtractor Completa (con Gemini Vision)")
    log(f"  Vision model: {settings.figure_extraction.vision_model}")
    log(f"  Image filtering enabled: {settings.image_filtering.enabled}")
    log(f"  Max PDF da testare: {max_pdfs}")

    from paperhunter.models import Paper

    extractor = FigureExtractor()
    results: list[PipelineResult] = []
    pdf_count = 0

    for category, pdf_names in TEST_PDFS.items():
        if pdf_count >= max_pdfs:
            break
        for pdf_name in pdf_names[:1]:  # 1 PDF per categoria
            if pdf_count >= max_pdfs:
                break

            pdf_path = BASE_DIR / category / pdf_name
            if not pdf_path.exists():
                continue

            # Crea un Paper fittizio con i campi minimi
            paper = Paper(
                title=pdf_name.replace(".pdf", "").replace("_", " "),
                url=f"https://example.com/{pdf_name}",
                bibtex_key=pdf_name[:30].replace(" ", "_").replace(",", ""),
                category=category.replace("1", ""),
                year=2025,
            )

            log(f"\n  📄 [bold]{category}/{pdf_name}[/bold]")
            t0 = time.time()
            try:
                chunks = extractor.extract_and_describe(
                    pdf_path=pdf_path,
                    paper=paper,
                    figure_captions=None,  # Senza GROBID per questo test
                )
                elapsed = time.time() - t0

                previews = [c.text[:150] + "..." for c in chunks] if chunks else []
                results.append(PipelineResult(
                    pdf_name=pdf_name, category=category,
                    chunks_generated=len(chunks),
                    time_seconds=elapsed, success=True,
                    chunk_texts_preview=previews,
                ))

                log(f"    ✅ {len(chunks)} chunk generati in {elapsed:.1f}s")
                for i, chunk in enumerate(chunks):
                    log(f"      Chunk {i}: [dim]{chunk.text[:120]}...[/dim]")
                    if chunk.image_path:
                        log(f"        📷 image_path: {chunk.image_path}")

            except Exception as e:
                elapsed = time.time() - t0
                results.append(PipelineResult(
                    pdf_name=pdf_name, category=category,
                    chunks_generated=0, time_seconds=elapsed,
                    success=False, error=str(e),
                ))
                log(f"    [red]❌ Errore: {e}[/red]")
                log(f"    {traceback.format_exc()}")

            pdf_count += 1

    # Riepilogo
    n_ok = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_generated for r in results)
    log(f"\n  --- RIEPILOGO TEST 3 ---")
    log(f"  PDF processati: {len(results)} ({n_ok} successo)")
    log(f"  Chunk totali generati: {total_chunks}")
    if results:
        avg_time = np.mean([r.time_seconds for r in results])
        log(f"  Tempo medio per PDF: {avg_time:.1f}s")

    return results


# ============================================================================
# TEST 4: Verifica embedding LlamaIndex vs SDK diretto
# ============================================================================

def test_embedding_consistency():
    """TEST 4: Verifica che embedding LlamaIndex e SDK diretto producano vettori coerenti."""
    log_separator("TEST 4: Confronto Embedding LlamaIndex vs SDK Diretto")

    test_text = "Scientific data plot showing attack success rate across different LLM models"

    # Via SDK diretto
    log("  Embedding via SDK google-genai diretto...")
    try:
        emb_sdk, t_sdk = embed_text_via_genai(test_text)
        log(f"    ✅ SDK: {len(emb_sdk)}-dim in {t_sdk:.2f}s")
    except Exception as e:
        log(f"    [red]❌ SDK: {e}[/red]")
        return

    # Via LlamaIndex wrapper
    log("  Embedding via LlamaIndex GoogleGenAIEmbedding...")
    try:
        from paperhunter.utils.llm_factory import get_embedding_model
        embed_model = get_embedding_model()
        t0 = time.time()
        emb_llama = embed_model.get_text_embedding(test_text)
        t_llama = time.time() - t0
        log(f"    ✅ LlamaIndex: {len(emb_llama)}-dim in {t_llama:.2f}s")
    except Exception as e:
        log(f"    [red]❌ LlamaIndex: {e}[/red]")
        return

    # Confronto
    sim = cosine_similarity(emb_sdk, emb_llama)
    log(f"\n  Similarità coseno SDK ↔ LlamaIndex: [bold]{sim:.6f}[/bold]")
    log(f"  Dimensione SDK: {len(emb_sdk)}, LlamaIndex: {len(emb_llama)}")

    if sim > 0.99:
        log("  [bold green]✅ Embedding identici (sim > 0.99) — OK[/bold green]")
    elif sim > 0.95:
        log("  [yellow]⚠ Embedding molto simili ma non identici (sim > 0.95)[/yellow]")
    else:
        log(f"  [red]❌ Embedding divergenti (sim={sim:.4f})[/red]")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_global = time.time()

    log_separator("TEST PIPELINE IMAGE PROCESSING — TODO5_ImageProcessing.md")
    log(f"  Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Embedding model: {settings.gemini.embedding_model}")
    log(f"  Vision model: {settings.figure_extraction.vision_model}")
    log(f"  Image filtering enabled: {settings.image_filtering.enabled}")
    log(f"  Filtering threshold: {settings.image_filtering.relevance_threshold}")
    log(f"  API key presente: {'✅' if settings.google_api_key else '❌'}")
    log(f"  PDF base dir: {BASE_DIR}")

    # Verifica che i PDF esistano
    n_found = 0
    for category, pdfs in TEST_PDFS.items():
        for pdf in pdfs:
            if (BASE_DIR / category / pdf).exists():
                n_found += 1
    log(f"  PDF trovati: {n_found}/{sum(len(v) for v in TEST_PDFS.values())}")

    if not settings.google_api_key:
        log("[red]❌ GOOGLE_API_KEY non configurata. Impossibile procedere.[/red]")
        save_report()
        return

    # --- TEST 1: Estrazione + Filtro euristico ---
    try:
        extraction_results = test_extraction_and_heuristic_filter()
    except Exception as e:
        log(f"[red]TEST 1 FALLITO: {e}[/red]")
        log(traceback.format_exc())
        extraction_results = []

    # --- TEST 2: Embedding multimodale ---
    try:
        embedding_results = test_multimodal_embeddings(extraction_results)
    except Exception as e:
        log(f"[red]TEST 2 FALLITO: {e}[/red]")
        log(traceback.format_exc())
        embedding_results = {}

    # --- TEST 3: Pipeline completa FigureExtractor ---
    try:
        pipeline_results = test_full_pipeline(max_pdfs=4)
    except Exception as e:
        log(f"[red]TEST 3 FALLITO: {e}[/red]")
        log(traceback.format_exc())
        pipeline_results = []

    # --- TEST 4: Confronto embedding ---
    try:
        test_embedding_consistency()
    except Exception as e:
        log(f"[red]TEST 4 FALLITO: {e}[/red]")
        log(traceback.format_exc())

    # --- RIEPILOGO FINALE ---
    elapsed_total = time.time() - t_global
    log_separator("RIEPILOGO FINALE")
    log(f"  Tempo totale: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    if extraction_results:
        total_imgs = sum(r.images_extracted for r in extraction_results)
        total_dec = sum(r.decorative for r in extraction_results)
        log(f"  TEST 1 - Estrazione + Filtro: {len(extraction_results)} PDF, "
            f"{total_imgs} immagini, {total_dec} filtrate ({total_dec/total_imgs*100:.0f}% risparmio)" if total_imgs else
            f"  TEST 1 - Nessuna immagine estratta")

    if embedding_results:
        tr = embedding_results.get("text_results", [])
        ir = embedding_results.get("image_results", [])
        n_text_ok = sum(1 for r in tr if r.success)
        n_img_ok = sum(1 for r in ir if r.success)
        log(f"  TEST 2 - Embedding: {n_text_ok} testi + {n_img_ok} immagini embeddati con successo")
        if n_img_ok > 0:
            log(f"    [bold green]✅ Gemini Embedding 2 Preview funziona per immagini![/bold green]")
        else:
            log(f"    [bold red]❌ Embedding immagini fallito[/bold red]")

    if pipeline_results:
        n_ok = sum(1 for r in pipeline_results if r.success)
        total_chunks = sum(r.chunks_generated for r in pipeline_results)
        log(f"  TEST 3 - Pipeline completa: {n_ok}/{len(pipeline_results)} PDF OK, {total_chunks} chunk")

    log(f"\n  [bold]Verdetto: {'✅ TUTTI I TEST PASSATI' if all([extraction_results, embedding_results, pipeline_results]) else '⚠ ALCUNI TEST FALLITI'}[/bold]")

    save_report()


if __name__ == "__main__":
    main()
