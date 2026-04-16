"""Script di verifica DBLP per tutte le entry nel references.bib.

Testa ogni paper nel .bib corrente contro DBLP usando lo stesso
modulo dblp_lookup.py usato nella pipeline.
"""
import re
import sys
import json
import time
from pathlib import Path

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).parent))

from paperhunter.ingestion.dblp_lookup import DblpLookup, _normalize_title


def parse_bib_entries(bib_path: str) -> list[dict]:
    """Parsa il .bib e restituisce lista di dict con i metadati."""
    content = Path(bib_path).read_text(encoding="utf-8")
    entries = []

    # Split per entry
    raw_entries = re.findall(r"(@\w+\{[^@]+\})", content, re.DOTALL)
    for raw in raw_entries:
        entry = {}
        # Tipo e chiave
        m = re.match(r"@(\w+)\{([^,]+),", raw)
        if not m:
            continue
        entry["type"] = m.group(1).lower()
        entry["key"] = m.group(2).strip()

        # Campi
        for field in ["title", "author", "year", "booktitle", "journal",
                       "doi", "eprint", "howpublished", "url"]:
            pat = rf"\b{field}\s*=\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}"
            fm = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
            if fm:
                entry[field] = fm.group(1).strip()
            else:
                # Senza parentesi
                pat2 = rf"\b{field}\s*=\s*([^,\}}]+)"
                fm2 = re.search(pat2, raw, re.IGNORECASE)
                if fm2:
                    entry[field] = fm2.group(1).strip().strip("{}")

        # Estrai lista autori
        if "author" in entry:
            authors_raw = entry["author"]
            entry["authors_list"] = [a.strip() for a in re.split(r"\s+and\s+", authors_raw)]
        else:
            entry["authors_list"] = []

        # Anno come int
        if "year" in entry:
            ym = re.search(r"(\d{4})", entry["year"])
            entry["year_int"] = int(ym.group(1)) if ym else None
        else:
            entry["year_int"] = None

        entries.append(entry)

    return entries


def main():
    bib_path = "output/latest/references.bib"
    print(f"{'='*80}")
    print(f"VERIFICA DBLP — {bib_path}")
    print(f"{'='*80}\n")

    entries = parse_bib_entries(bib_path)
    print(f"Totale entry nel .bib: {len(entries)}\n")

    # Usa una cache temporanea per questo test
    dblp = DblpLookup(cache_dir="cache/dblp_verify", enabled=True)

    results = []
    for i, entry in enumerate(entries):
        title = entry.get("title", "")
        authors = entry.get("authors_list", [])
        year = entry.get("year_int")
        doi = entry.get("doi")
        bib_key = entry.get("key", "?")

        print(f"[{i+1}/{len(entries)}] {bib_key}")
        print(f"  Titolo: {title[:80]}...")

        result = dblp.lookup_bibtex(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
        )

        entry_result = {
            "key": bib_key,
            "title": title,
            "current_type": entry.get("type", "?"),
        }

        if result:
            entry_result["dblp_found"] = True
            entry_result["dblp_key"] = result.dblp_key
            entry_result["dblp_type"] = result.dblp_type
            entry_result["similarity"] = result.similarity

            # Che tipo di entry è sul DBLP?
            bib_type_match = re.match(r"@(\w+)\{", result.bibtex)
            entry_result["dblp_bib_type"] = bib_type_match.group(1) if bib_type_match else "?"

            status = "✅ CONFERENZA" if "Conference" in result.dblp_type else (
                "📄 JOURNAL" if "Journal" in result.dblp_type else "📋 ALTRO"
            )
            print(f"  → {status} | DBLP: {result.dblp_key} (sim={result.similarity:.2f})")
            print(f"    Tipo DBLP: {result.dblp_type}")
            print(f"    BibTeX @{entry_result['dblp_bib_type']}")
        else:
            entry_result["dblp_found"] = False
            print(f"  → ❌ NON TROVATO su DBLP")

        results.append(entry_result)
        print()

    # --- Riepilogo ---
    print(f"\n{'='*80}")
    print("RIEPILOGO")
    print(f"{'='*80}\n")

    found = [r for r in results if r["dblp_found"]]
    not_found = [r for r in results if not r["dblp_found"]]

    conf = [r for r in found if "Conference" in r.get("dblp_type", "")]
    journal = [r for r in found if "Journal" in r.get("dblp_type", "")]
    informal = [r for r in found if "Informal" in r.get("dblp_type", "")]

    print(f"Trovati su DBLP:         {len(found)}/{len(results)}")
    print(f"  ✅ Conferenza/Workshop: {len(conf)}")
    print(f"  📄 Journal:            {len(journal)}")
    print(f"  📋 Informale (arXiv):  {len(informal)}")
    print(f"  ❌ Non trovati:        {len(not_found)}")

    if conf:
        print(f"\n--- Paper di CONFERENZA trovati ---")
        for r in conf:
            print(f"  {r['key']} → {r['dblp_key']} (sim={r['similarity']:.2f})")

    if journal:
        print(f"\n--- Paper di JOURNAL trovati ---")
        for r in journal:
            print(f"  {r['key']} → {r['dblp_key']} (sim={r['similarity']:.2f})")

    if informal:
        print(f"\n--- Paper INFORMALI (preprint) trovati ---")
        for r in informal:
            print(f"  {r['key']} → {r['dblp_key']} (sim={r['similarity']:.2f})")

    if not_found:
        print(f"\n--- Paper NON TROVATI su DBLP ---")
        for r in not_found:
            print(f"  {r['key']}: {r['title'][:70]}")

    # Salva report JSON
    report_path = "cache/dblp_verify/verification_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport salvato: {report_path}")

    print(f"\nStatistiche DBLP client: {dblp.stats}")
    dblp.close()


if __name__ == "__main__":
    main()
