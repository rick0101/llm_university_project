"""
Test per Task C.1 — Ordinamento per data nella ingestion pipeline.

Test delle funzioni di ordinamento per recency:
1. _extract_year_from_filename — estrazione anno da filename
2. _get_pdf_sort_key — chiave di ordinamento composita
3. sort_pdfs_by_recency — ordinamento decrescente per data
4. Integrazione con scan_pdf_directory (campo mtime)
5. Annotazioni sort_year e sort_source

Esecuzione:
    python test_recency_sort.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Aggiungi il progetto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Test 1: _extract_year_from_filename
# ============================================================================

def test_extract_year_from_filename():
    """Test estrazione anno dal nome del file PDF."""
    from paperhunter.ingestion.ingestion_pipeline import _extract_year_from_filename

    passed = 0
    total = 0

    cases = [
        # (filename, expected_year)
        ("paper_2024_prompt_injection.pdf", 2024),
        ("Survey_on_LLM_Security_2023.pdf", 2023),
        ("2025_adversarial_attacks.pdf", 2025),
        ("GCG_attack_method_2023_v2.pdf", 2023),
        ("HaluEval_benchmark.pdf", None),  # Nessun anno
        ("Prompt_Injection_Attacks.pdf", None),
        ("paper_with_two_years_2020_and_2024.pdf", 2024),  # Prende il più recente
        ("review2023.pdf", 2023),  # Anno attaccato a parola
        ("arxiv_2305.12345.pdf", None),  # 2305 non è un anno (>2029)
        ("paper_2000_old.pdf", 2000),  # Anno al limite inferiore
        ("paper_2029_future.pdf", 2029),  # Anno al limite superiore
        ("paper_1999_too_old.pdf", None),  # Fuori range 2000-2029
        ("", None),  # Stringa vuota
        ("paper.pdf", None),  # Nessun numero
        ("paper_123_test.pdf", None),  # Numero troppo corto
        ("file_20240101_report.pdf", None),  # 20240101 è troppo lungo (8 cifre)
    ]

    for filename, expected in cases:
        total += 1
        result = _extract_year_from_filename(filename)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: _extract_year_from_filename('{filename}') = {result}, expected {expected}")

    print(f"Test 1 (_extract_year_from_filename): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 2: _get_pdf_sort_key
# ============================================================================

def test_get_pdf_sort_key():
    """Test chiave di ordinamento composita per PDF."""
    from paperhunter.ingestion.ingestion_pipeline import _get_pdf_sort_key

    passed = 0
    total = 0

    # Test con anno nel filename
    total += 1
    pdf_info = {
        "filename": "paper_2024_method.pdf",
        "absolute_path": Path("/tmp/fake.pdf"),
        "mtime": datetime(2020, 6, 15).timestamp(),  # mtime dice 2020
    }
    key = _get_pdf_sort_key(pdf_info)
    # L'anno dal filename (2024) deve vincere su mtime (2020)
    if key[0] == 2024:
        passed += 1
    else:
        print(f"  FAIL: anno from filename dovrebbe essere 2024, got {key[0]}")

    # Test senza anno nel filename → fallback a mtime
    total += 1
    pdf_info2 = {
        "filename": "HaluEval_benchmark.pdf",
        "absolute_path": Path("/tmp/fake.pdf"),
        "mtime": datetime(2023, 3, 20).timestamp(),
    }
    key2 = _get_pdf_sort_key(pdf_info2)
    if key2[0] == 2023 and key2[1] == 3:
        passed += 1
    else:
        print(f"  FAIL: anno/mese da mtime dovrebbe essere (2023, 3), got ({key2[0]}, {key2[1]})")

    # Test senza mtime → fallback a anno 2000
    total += 1
    pdf_info3 = {
        "filename": "old_paper.pdf",
        "absolute_path": None,
        "mtime": None,
    }
    key3 = _get_pdf_sort_key(pdf_info3)
    if key3[0] == 2000:
        passed += 1
    else:
        print(f"  FAIL: anno fallback dovrebbe essere 2000, got {key3[0]}")

    # Test che la tupla permette ordinamento decrescente corretto
    total += 1
    key_2024 = _get_pdf_sort_key({"filename": "paper_2024.pdf", "mtime": 0})
    key_2022 = _get_pdf_sort_key({"filename": "paper_2022.pdf", "mtime": 0})
    key_old = _get_pdf_sort_key({"filename": "old.pdf", "mtime": datetime(2019, 1, 1).timestamp()})
    if key_2024 > key_2022 > key_old:
        passed += 1
    else:
        print(f"  FAIL: ordinamento decrescente non rispettato: {key_2024} > {key_2022} > {key_old}")

    print(f"Test 2 (_get_pdf_sort_key): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 3: sort_pdfs_by_recency — ordinamento base
# ============================================================================

def test_sort_pdfs_by_recency_basic():
    """Test ordinamento base: paper più recenti prima."""
    from paperhunter.ingestion.ingestion_pipeline import sort_pdfs_by_recency

    passed = 0
    total = 0

    # Crea lista di PDF con anni diversi
    pdfs = [
        {"filename": "old_paper.pdf", "absolute_path": Path("/tmp/a.pdf"),
         "relative_path": "cat1/old.pdf", "mtime": datetime(2019, 6, 1).timestamp()},
        {"filename": "paper_2024.pdf", "absolute_path": Path("/tmp/b.pdf"),
         "relative_path": "cat1/2024.pdf", "mtime": datetime(2024, 1, 1).timestamp()},
        {"filename": "paper_2023_review.pdf", "absolute_path": Path("/tmp/c.pdf"),
         "relative_path": "cat1/2023.pdf", "mtime": datetime(2023, 6, 1).timestamp()},
        {"filename": "recent_no_year.pdf", "absolute_path": Path("/tmp/d.pdf"),
         "relative_path": "cat2/recent.pdf", "mtime": datetime(2025, 2, 1).timestamp()},
        {"filename": "paper_2020_attack.pdf", "absolute_path": Path("/tmp/e.pdf"),
         "relative_path": "cat2/2020.pdf", "mtime": datetime(2020, 11, 1).timestamp()},
    ]

    sorted_pdfs = sort_pdfs_by_recency(pdfs)

    # Test: lista non vuota
    total += 1
    if len(sorted_pdfs) == 5:
        passed += 1
    else:
        print(f"  FAIL: attesi 5 PDF ordinati, ottenuti {len(sorted_pdfs)}")

    # Test: ordinamento decrescente
    total += 1
    years = [p["sort_year"] for p in sorted_pdfs]
    if years == sorted(years, reverse=True):
        passed += 1
    else:
        print(f"  FAIL: anni non in ordine decrescente: {years}")

    # Test: il primo deve essere il più recente (2025 da mtime o 2024 da filename)
    total += 1
    first_year = sorted_pdfs[0]["sort_year"]
    if first_year >= 2024:
        passed += 1
    else:
        print(f"  FAIL: primo paper dovrebbe essere ≥2024, got {first_year}")

    # Test: l'ultimo deve essere il più vecchio (2019)
    total += 1
    last_year = sorted_pdfs[-1]["sort_year"]
    if last_year <= 2020:
        passed += 1
    else:
        print(f"  FAIL: ultimo paper dovrebbe essere ≤2020, got {last_year}")

    # Test: la lista originale non è modificata
    total += 1
    if pdfs[0]["filename"] == "old_paper.pdf":
        passed += 1
    else:
        print(f"  FAIL: lista originale modificata!")

    print(f"Test 3 (sort_pdfs_by_recency base): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 4: sort_pdfs_by_recency — annotazioni
# ============================================================================

def test_sort_pdfs_annotazioni():
    """Test che sort_pdfs_by_recency annota correttamente sort_year e sort_source."""
    from paperhunter.ingestion.ingestion_pipeline import sort_pdfs_by_recency

    passed = 0
    total = 0

    pdfs = [
        {"filename": "paper_2024.pdf", "absolute_path": Path("/tmp/a.pdf"),
         "relative_path": "cat/a.pdf", "mtime": datetime(2020, 1, 1).timestamp()},
        {"filename": "no_year_paper.pdf", "absolute_path": Path("/tmp/b.pdf"),
         "relative_path": "cat/b.pdf", "mtime": datetime(2023, 5, 15).timestamp()},
    ]

    sorted_pdfs = sort_pdfs_by_recency(pdfs)

    # Test: paper con anno nel filename → sort_source = "filename"
    total += 1
    paper_with_year = [p for p in sorted_pdfs if p["filename"] == "paper_2024.pdf"][0]
    if paper_with_year["sort_source"] == "filename" and paper_with_year["sort_year"] == 2024:
        passed += 1
    else:
        print(f"  FAIL: paper_2024.pdf dovrebbe avere source=filename year=2024, "
              f"got source={paper_with_year['sort_source']} year={paper_with_year['sort_year']}")

    # Test: paper senza anno → sort_source = "mtime"
    total += 1
    paper_no_year = [p for p in sorted_pdfs if p["filename"] == "no_year_paper.pdf"][0]
    if paper_no_year["sort_source"] == "mtime" and paper_no_year["sort_year"] == 2023:
        passed += 1
    else:
        print(f"  FAIL: no_year_paper.pdf dovrebbe avere source=mtime year=2023, "
              f"got source={paper_no_year['sort_source']} year={paper_no_year['sort_year']}")

    # Test: campi originali preservati
    total += 1
    if all("relative_path" in p for p in sorted_pdfs):
        passed += 1
    else:
        print(f"  FAIL: campi originali non preservati")

    print(f"Test 4 (annotazioni sort_year/sort_source): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 5: sort_pdfs_by_recency — edge cases
# ============================================================================

def test_sort_edge_cases():
    """Test edge cases: lista vuota, singolo elemento, tutti stessi anno."""
    from paperhunter.ingestion.ingestion_pipeline import sort_pdfs_by_recency

    passed = 0
    total = 0

    # Test: lista vuota
    total += 1
    result = sort_pdfs_by_recency([])
    if result == []:
        passed += 1
    else:
        print(f"  FAIL: lista vuota dovrebbe restituire [], got {result}")

    # Test: singolo elemento
    total += 1
    single = [{"filename": "paper_2024.pdf", "absolute_path": Path("/tmp/a.pdf"),
                "relative_path": "cat/a.pdf", "mtime": datetime(2024, 1, 1).timestamp()}]
    result = sort_pdfs_by_recency(single)
    if len(result) == 1 and result[0]["sort_year"] == 2024:
        passed += 1
    else:
        print(f"  FAIL: singolo elemento non gestito correttamente")

    # Test: tutti stessi anno (ordine stabile per mtime)
    total += 1
    same_year = [
        {"filename": f"paper_2024_{i}.pdf", "absolute_path": Path(f"/tmp/{i}.pdf"),
         "relative_path": f"cat/{i}.pdf", "mtime": datetime(2024, i + 1, 1).timestamp()}
        for i in range(5)
    ]
    result = sort_pdfs_by_recency(same_year)
    years = [p["sort_year"] for p in result]
    if all(y == 2024 for y in years):
        passed += 1
    else:
        print(f"  FAIL: tutti 2024 ma anni diversi trovati: {years}")

    # Test: stesso anno → ordinati per mese (mtime) decrescente
    total += 1
    months = [datetime.fromtimestamp(p["mtime"]).month for p in result]
    if months == sorted(months, reverse=True):
        passed += 1
    else:
        print(f"  FAIL: mesi non decrescenti: {months}")

    # Test: nessun mtime, nessun anno → non crasha
    total += 1
    no_info = [
        {"filename": "paper.pdf", "absolute_path": None, "relative_path": "a.pdf", "mtime": None},
        {"filename": "paper2.pdf", "absolute_path": None, "relative_path": "b.pdf", "mtime": None},
    ]
    result = sort_pdfs_by_recency(no_info)
    if len(result) == 2:
        passed += 1
    else:
        print(f"  FAIL: PDF senza info temporali non gestiti")

    print(f"Test 5 (edge cases): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 6: scan_pdf_directory include mtime
# ============================================================================

def test_scan_pdf_directory_mtime():
    """Test che scan_pdf_directory include il campo mtime."""
    from paperhunter.ingestion.pdf_hasher import scan_pdf_directory

    passed = 0
    total = 0

    # Crea una directory temporanea con PDF finti
    with tempfile.TemporaryDirectory() as tmpdir:
        cat_dir = Path(tmpdir) / "TestCategory"
        cat_dir.mkdir()

        # Crea file PDF finti
        (cat_dir / "paper_a.pdf").write_bytes(b"%PDF-1.4 test content A")
        time.sleep(0.05)  # Piccolo delay per avere mtime diversi
        (cat_dir / "paper_b.pdf").write_bytes(b"%PDF-1.4 test content B different")

        pdfs = scan_pdf_directory(tmpdir)

        # Test: i PDF sono trovati
        total += 1
        if len(pdfs) == 2:
            passed += 1
        else:
            print(f"  FAIL: attesi 2 PDF, trovati {len(pdfs)}")

        # Test: ogni pdf_info ha il campo mtime
        total += 1
        if all("mtime" in p for p in pdfs):
            passed += 1
        else:
            print(f"  FAIL: campo 'mtime' mancante in alcuni PDF")

        # Test: mtime è un float positivo
        total += 1
        if all(isinstance(p["mtime"], float) and p["mtime"] > 0 for p in pdfs):
            passed += 1
        else:
            print(f"  FAIL: mtime non è float positivo")

        # Test: mtime di paper_b >= mtime di paper_a (creato dopo)
        total += 1
        a = [p for p in pdfs if "paper_a" in p["filename"]][0]
        b = [p for p in pdfs if "paper_b" in p["filename"]][0]
        if b["mtime"] >= a["mtime"]:
            passed += 1
        else:
            print(f"  FAIL: mtime di paper_b dovrebbe essere >= paper_a")

    print(f"Test 6 (scan_pdf_directory mtime): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 7: Integrazione sort con dati reali (scaricati/)
# ============================================================================

def test_sort_real_pdfs():
    """Test ordinamento sui PDF reali in scaricati/ (se la directory esiste)."""
    from paperhunter.ingestion.pdf_hasher import scan_pdf_directory
    from paperhunter.ingestion.ingestion_pipeline import sort_pdfs_by_recency

    scaricati_dir = Path("scaricati")
    if not scaricati_dir.exists():
        print("Test 7 (real PDFs): SKIP — directory 'scaricati/' non trovata")
        return True

    passed = 0
    total = 0

    pdfs = scan_pdf_directory(scaricati_dir)

    if not pdfs:
        print("Test 7 (real PDFs): SKIP — nessun PDF trovato")
        return True

    sorted_pdfs = sort_pdfs_by_recency(pdfs)

    # Test: stesso numero di elementi
    total += 1
    if len(sorted_pdfs) == len(pdfs):
        passed += 1
    else:
        print(f"  FAIL: {len(sorted_pdfs)} ordinati vs {len(pdfs)} originali")

    # Test: tutti hanno sort_year e sort_source
    total += 1
    if all("sort_year" in p and "sort_source" in p for p in sorted_pdfs):
        passed += 1
    else:
        print(f"  FAIL: annotazioni mancanti")

    # Test: sort_year è ragionevole (2000-2030)
    total += 1
    years = [p["sort_year"] for p in sorted_pdfs]
    if all(2000 <= y <= 2030 for y in years):
        passed += 1
    else:
        out_of_range = [y for y in years if y < 2000 or y > 2030]
        print(f"  FAIL: anni fuori range: {out_of_range}")

    # Test: ordinamento non è crescente (dovrebbe essere decrescente o piatto)
    total += 1
    if years[0] >= years[-1]:
        passed += 1
    else:
        print(f"  FAIL: primo anno ({years[0]}) < ultimo anno ({years[-1]}) — non decrescente")

    # Statistiche
    from_filename = sum(1 for p in sorted_pdfs if p["sort_source"] == "filename")
    from_mtime = sum(1 for p in sorted_pdfs if p["sort_source"] == "mtime")
    unique_years = sorted(set(years), reverse=True)
    print(f"  Info: {len(sorted_pdfs)} PDF, range {min(years)}-{max(years)}, "
          f"da filename: {from_filename}, da mtime: {from_mtime}")
    print(f"  Anni unici: {unique_years}")

    # Test: tutti i relative_path sono preservati
    total += 1
    orig_paths = {p["relative_path"] for p in pdfs}
    sorted_paths = {p["relative_path"] for p in sorted_pdfs}
    if orig_paths == sorted_paths:
        passed += 1
    else:
        print(f"  FAIL: relative_path non preservati")

    print(f"Test 7 (real PDFs sort): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Test 8: _extract_year_from_filename — numeri embedded in parole
# ============================================================================

def test_year_regex_boundaries():
    """Test che il regex rispetta i confini: non matcha parti di numeri più lunghi."""
    from paperhunter.ingestion.ingestion_pipeline import _extract_year_from_filename

    passed = 0
    total = 0

    cases = [
        # Numeri lunghi che contengono un anno — NON devono matchare
        ("file_20240101_report.pdf", None),  # 8 cifre: data YYYYMMDD
        ("paper_1234520231.pdf", None),  # Anno embedded in numero lungo
        # Ma separato da non-cifre va bene
        ("file_2024-01-01_report.pdf", 2024),  # Anno separato da trattino
        ("paper v2024 final.pdf", 2024),  # Anno separato da spazi
        ("paper(2023).pdf", 2023),  # Anno tra parentesi
        ("2024.pdf", 2024),  # Solo anno
    ]

    for filename, expected in cases:
        total += 1
        result = _extract_year_from_filename(filename)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: _extract_year_from_filename('{filename}') = {result}, expected {expected}")

    print(f"Test 8 (year regex boundaries): {passed}/{total} passati")
    return passed == total


# ============================================================================
# Main
# ============================================================================

def main():
    """Esecue tutti i test e stampa il riepilogo."""
    print("=" * 60)
    print("Test Task C.1: Ordinamento per recency nella ingestion pipeline")
    print("=" * 60)
    print()

    results = []

    results.append(("_extract_year_from_filename", test_extract_year_from_filename()))
    results.append(("_get_pdf_sort_key", test_get_pdf_sort_key()))
    results.append(("sort_pdfs_by_recency base", test_sort_pdfs_by_recency_basic()))
    results.append(("annotazioni sort_year/source", test_sort_pdfs_annotazioni()))
    results.append(("edge cases", test_sort_edge_cases()))
    results.append(("scan_pdf_directory mtime", test_scan_pdf_directory_mtime()))
    results.append(("real PDFs sort", test_sort_real_pdfs()))
    results.append(("year regex boundaries", test_year_regex_boundaries()))

    print()
    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"✅ Tutti {len(results)} test passati!")
    else:
        failed = sum(1 for _, p in results if not p)
        print(f"❌ {failed}/{len(results)} test falliti!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
