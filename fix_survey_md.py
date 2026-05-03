"""
Patcha l'ultimo survey.md generato per correggere:
1. Heading duplicato all'inizio di una sezione (writer ripete il titolo)
2. Sotto-sezioni come **bold** o ## H2 → diventano ### H3
3. Heading H1 "fuori posto" subito dopo un H2 → rimosso
4. Inverte titoli "7." e "8." per allinearli all'ordine outline
"""
import re
import sys
from pathlib import Path

SRC = Path("output/versions/Survey_v1.7_Apr2026/survey.md")
if not SRC.exists():
    sys.exit(f"Not found: {SRC}")

text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")

# --- Pass 1: rimuovi heading duplicato (## X seguito da #/##/**X** ravvicinato) ---
out = []
i = 0
while i < len(lines):
    line = lines[i]
    out.append(line)
    m = re.match(r"^## (.+)$", line)
    if m:
        # Look ahead up to 4 lines for a duplicate of the same title
        title_words = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m.group(1)).strip().lower()
        j = i + 1
        skipped = 0
        while j < len(lines) and j < i + 5:
            l = lines[j]
            stripped = l.strip()
            if not stripped:
                j += 1
                continue
            # Match either "# ...", "## ...", "### ...", or "**...**"
            m2 = re.match(r"^(#{1,3}\s+|\*\*)(.+?)(\*\*)?$", stripped)
            if m2:
                cand = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m2.group(2)).strip().lower().rstrip("*")
                if cand == title_words or cand.startswith(title_words[:30]):
                    # skip the duplicate (and the blank between)
                    i = j + 1
                    skipped = 1
                    break
            break
        if skipped:
            continue
    i += 1

text = "\n".join(out)

# --- Pass 1b: rimuovi un secondo ## ravvicinato (writer-generated)
# subito dopo un ## numerato (assembler-generated). Esempio:
#   ## 4. Adversarial Attacks on LLMs
#   ## Adversarial Attacks and Jailbreaking Mechanisms     ← rimuovere
lines = text.split("\n")
out = []
i = 0
while i < len(lines):
    line = lines[i]
    out.append(line)
    m = re.match(r"^## (\d+\.[^\n]+)$", line)
    if m:
        # Look ahead at most 4 non-blank lines
        j = i + 1
        while j < len(lines) and j < i + 5 and not lines[j].strip():
            j += 1
        if j < len(lines):
            m2 = re.match(r"^## (?!\d)([^\n]+)$", lines[j])
            if m2:
                # skip lines i+1..j (incl) — drop blank+H2
                # but keep one blank separator
                out.append("")
                i = j + 1
                continue
    i += 1
text = "\n".join(out)
text = re.sub(
    r"^\*\*(\d+\.\d+\s+[^*]+?)\*\*\s*$",
    r"### \1",
    text,
    flags=re.MULTILINE,
)

# Bold "Conclusion" / "Introduction" su riga propria → ### heading
text = re.sub(
    r"^\*\*([A-Z][A-Za-z][^*\n]{2,60})\*\*\s*$",
    r"### \1",
    text,
    flags=re.MULTILINE,
)

# --- Pass 3: H2 con pattern "N.X Title" → H3 (es. ## 8.1 ... → ### 8.1 ...) ---
text = re.sub(
    r"^## (\d+\.\d+\s+.+)$",
    r"### \1",
    text,
    flags=re.MULTILINE,
)

# --- Pass 4: rimuovi heading "# N. Title" o "# Title" su riga propria
# che riproduce un H2 immediatamente sopra (writer ha messo H1 dentro la sezione) ---
lines = text.split("\n")
out = []
for idx, line in enumerate(lines):
    m = re.match(r"^# (.+)$", line)
    if m and idx > 0:
        # Look back to find the most recent ## heading
        for k in range(idx - 1, max(0, idx - 5), -1):
            mp = re.match(r"^## (.+)$", lines[k])
            if mp:
                a = re.sub(r"^\d+(\.\d+)*\.?\s*", "", mp.group(1)).strip().lower()
                b = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m.group(1)).strip().lower()
                if a == b or a.startswith(b[:30]) or b.startswith(a[:30]):
                    line = ""  # drop the duplicate H1
                    break
    out.append(line)
text = "\n".join(out)

# --- Pass 5: inverti numerazione 7./8. nei titoli per allinearli all'ordine fisico ---
# La sezione "8. Cross-Cutting Defenses" appare prima di "7. Testing Methodologies".
# Scambio i numeri perché l'ordine del documento è quello che il lettore vede.
text = text.replace(
    "## 8. Cross-Cutting Defenses and Mitigation Strategies",
    "## 7. Cross-Cutting Defenses and Mitigation Strategies",
)
# Sotto-sezioni 8.1..8.5 → 7.1..7.5 (per coerenza)
for n in range(1, 9):
    text = text.replace(f"### 8.{n} ", f"### 7.{n} ")
text = text.replace(
    "## 7. Testing Methodologies and Evaluation",
    "## 8. Testing Methodologies and Evaluation",
)

# --- Pass 6: aggiorna anche il TOC con i nuovi numeri ---
text = text.replace(
    "8. Cross-Cutting Defenses and Mitigation Strategies",
    "8._TMP_Cross-Cutting Defenses and Mitigation Strategies",
)  # placeholder per evitare doppio-replace
# (già sostituito sopra in heading, qui serve solo per TOC)
text = text.replace(
    "8._TMP_Cross-Cutting Defenses and Mitigation Strategies",
    "7. Cross-Cutting Defenses and Mitigation Strategies",
)
text = re.sub(r"\b7\. Testing Methodologies and Evaluation\b",
              "8. Testing Methodologies and Evaluation", text)

SRC.write_text(text, encoding="utf-8")
print(f"OK: {SRC} patched ({len(text):,} chars)")

# Stampa nuovo elenco heading
print("\n--- Headings dopo il fix ---")
for ln in text.split("\n"):
    if re.match(r"^#{1,3} ", ln):
        print(ln)
