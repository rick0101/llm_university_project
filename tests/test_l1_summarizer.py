#!/usr/bin/env python3
"""
Test script per Task 2.1: Level 1 Summarizer.

Testa la generazione di L1 summary per 3 paper da 3 categorie diverse.
Verifica che testing_methodology sia non-vuoto.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paperhunter.summarization.level1_summarizer import (
    generate_l1_summary,
    generate_l1_batch,
    load_paper_and_chunks_from_ingestion,
)
from paperhunter.utils.logging_setup import setup_logging, print_success, print_error, print_table, print_panel

setup_logging("INFO")

# --- 3 PDF da 3 categorie diverse ---
# NOTA: alcuni filename hanno \n letterale nei nomi (artefatto download)
TEST_PDFS = [
    "scaricati/Prompt_injections1/Adaptive_Attacks_Break_Defenses_Against_Indirect_Prompt_Injection_Attacks_on_LLM_Agents.pdf",
    "scaricati/Data_and_Model_Poisoning1/BackdoorLLM_A_Comprehensive_Benchmark_for_Backdoor_Attacks_on_Large\n__LanguageModels.pdf",
    "scaricati/Adversarial_Code1/AdversarialAttacksagainst_Closed-Source_MLLMs_via_Feature_Optimal\n__Alignment.pdf",
]


def main():
    total_start = time.time()
    
    papers_with_chunks = []
    
    # === STEP 1: Carica paper e chunk (GROBID + S2 + BibTeX + chunking) ===
    print_panel("Step 1: Loading papers via GROBID + S2 enrichment", "")
    
    for pdf_path in TEST_PDFS:
        p = Path(pdf_path)
        if not p.exists():
            print_error(f"PDF non trovato: {pdf_path}")
            continue
        
        try:
            paper, chunks = load_paper_and_chunks_from_ingestion(
                pdf_path,
                skip_s2=False,  # Enrich con S2
            )
            papers_with_chunks.append((paper, chunks))
            print_success(
                f"Caricato: '{paper.title[:60]}...' "
                f"({len(chunks)} chunk, bibtex_key={paper.bibtex_key}, "
                f"category={paper.category}, year={paper.year})"
            )
        except Exception as e:
            print_error(f"Errore caricamento {p.name}: {e}")
    
    if not papers_with_chunks:
        print_error("Nessun paper caricato! Interrompo.")
        return
    
    # === STEP 2: Genera L1 per ciascun paper ===
    print_panel("Step 2: Generating L1 summaries", "")
    
    summaries = []
    for paper, chunks in papers_with_chunks:
        try:
            summary = generate_l1_summary(
                paper, chunks,
                save_to_store=False,  # Non salva nel Qdrant per questo test
            )
            summaries.append(summary)
            
            # Stampa risultato
            print_success(f"\n=== L1 Summary: {paper.title[:60]}... ===")
            print(f"  Paper ID: {summary.paper_id}")
            print(f"  Category: {summary.category}")
            print(f"  Year: {summary.year}")
            print(f"  Model: {summary.model_used}")
            print(f"  Key Contributions ({len(summary.key_contributions)}):")
            for i, c in enumerate(summary.key_contributions, 1):
                print(f"    {i}. {c[:100]}{'...' if len(c) > 100 else ''}")
            print(f"  Methodology ({len(summary.methodology)} chars):")
            print(f"    {summary.methodology[:200]}...")
            print(f"  Testing Methodology ({len(summary.testing_methodology)} chars):")
            print(f"    {summary.testing_methodology[:300]}...")
            print(f"  Temporal Context ({len(summary.temporal_context)} chars):")
            print(f"    {summary.temporal_context[:200]}...")
            print(f"  Results ({len(summary.results_summary)} chars):")
            print(f"    {summary.results_summary[:200]}...")
            print(f"  Limitations ({len(summary.limitations)} chars):")
            print(f"    {summary.limitations[:200]}...")
            
        except Exception as e:
            print_error(f"Errore L1 per '{paper.title[:40]}...': {e}")
            import traceback
            traceback.print_exc()
    
    # === STEP 3: Verifica testing_methodology non-vuoto ===
    print_panel("Step 3: Validation", "")
    
    all_ok = True
    for s in summaries:
        tm_len = len(s.testing_methodology)
        has_tm = tm_len > 30
        status = "OK" if has_tm else "FAIL"
        if not has_tm:
            all_ok = False
        print(f"  [{status}] {s.paper_id}: testing_methodology = {tm_len} chars")
    
    # === Report finale ===
    total_elapsed = time.time() - total_start
    print_panel("Final Report", "")
    print_table("L1 Summary Test Results", {
        "Paper testati": len(papers_with_chunks),
        "L1 generati": len(summaries),
        "testing_methodology non-vuoto": sum(1 for s in summaries if len(s.testing_methodology) > 30),
        "Tempo totale": f"{total_elapsed:.1f}s",
    })
    
    if all_ok and len(summaries) == len(papers_with_chunks):
        print_success("TUTTI I TEST PASSATI!")
    else:
        print_error("Alcuni test falliti.")


if __name__ == "__main__":
    main()
