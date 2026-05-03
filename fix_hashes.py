import json
import logging

c = 0
with open('data/hash_registry.json', 'r') as f:
    data = json.load(f)

keywords = ['kumar', 'nie', 'muntasir', 'paudel', 'wang', 'urlana']
new_data = {}

for k, v in data.items():
    # v contains 'authors', 'bibtex_key' 
    authors_str = str(v.get('authors', [])).lower()
    bibtex_key = str(v.get('bibtex_key', '')).lower()
    match = any(kw in authors_str or kw in bibtex_key for kw in keywords)
    if not match:
        new_data[k] = v
    else:
        print(f"Rimosso: {bibtex_key} ({k})")
        c += 1

with open('data/hash_registry.json', 'w') as f:
    json.dump(new_data, f, indent=2)

print(f"{c} entries rimosse.")
