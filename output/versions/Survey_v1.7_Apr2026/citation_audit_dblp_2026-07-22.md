# Audit citazioni DBLP — Survey v1.7

Data audit: 22 luglio 2026

Documento controllato: `survey.md` in questa directory

Fonte autoritativa: record e BibTeX ufficiali DBLP

## Esito

- 156 occorrenze citazionali nel testo.
- 66 chiavi bibliografiche uniche controllate individualmente.
- 37 entry presenti in `references_final.bib`; 29 entry mancanti.
- 6 entry con metadati corrispondenti a DBLP.
- 26 entry presenti ma non identiche ai metadati/BibTeX DBLP.
- 5 entry non verificate in modo rigoroso su DBLP.
- Manifest DBLP assente: nessuna entry della v1.7 supera il nuovo gate finale.
- Tre placeholder interni `l2_*` sono rimasti nel testo.

Le voci `NON_VERIFICATA` non sono dichiarate inesistenti: l'audit è fail-closed,
quindi una fonte non confermata senza ambiguità viene esclusa dal documento finale.

## Controllo individuale delle 66 chiavi

| Chiave | Esito |
|---|---|
| `Agarwal_2024_Zero` | `METADATI_ERRATI` |
| `Aghaebrahimian_2025_Aligncheck` | `METADATI_ERRATI` |
| `Baumgartner_2024_RLHF` | `BIB_MANCANTE` |
| `Bender_2021_Stochastic` | `BIB_MANCANTE` |
| `Bommasani_2021_Foundation` | `BIB_MANCANTE` |
| `Carlini_2023_Poisoning` | `BIB_MANCANTE` |
| `Chang_2024_Transactions` | `BIB_MANCANTE` |
| `Chaudhari_2025_Cascading` | `METADATI_ERRATI` |
| `Chen_2024_Rmcbench` | `METADATI_ERRATI` |
| `Cheng_2024_LLM` | `NON_VERIFICATA` |
| `Cheng_2024_Unit` | `BIB_MANCANTE` |
| `Duan_2025_RLHF` | `NON_VERIFICATA` |
| `Dutta_2025_RLHF` | `BIB_MANCANTE` |
| `Fang_2024_LLM` | `METADATI_DBLP_CORRISPONDENTI` |
| `Foerster_2025_Reasoning` | `METADATI_ERRATI` |
| `Goodfellow_2014_Explaining` | `BIB_MANCANTE` |
| `Greshake_2023_NotWhat` | `BIB_MANCANTE` |
| `Guo_2024_Redcode` | `METADATI_ERRATI` |
| `Hao_2024_Exploring` | `METADATI_DBLP_CORRISPONDENTI` |
| `Harary_2025_Prefixnli` | `METADATI_ERRATI` |
| `He_2024_LLMs` | `BIB_MANCANTE` |
| `Hussain_2024_Measuring` | `METADATI_DBLP_CORRISPONDENTI` |
| `Ji_2023_Survey` | `BIB_MANCANTE` |
| `Jin_2025_Guard` | `BIB_MANCANTE` |
| `Kusaka_2025_LLM` | `BIB_MANCANTE` |
| `Li_2025_Autobackdoor` | `BIB_MANCANTE` |
| `Li_2025_CoT` | `METADATI_ERRATI` |
| `Li_2025_LLM` | `METADATI_ERRATI` |
| `Ling_2026_Correct` | `BIB_MANCANTE` |
| `Liu_2024_Autodan` | `METADATI_ERRATI` |
| `Liu_2024_Flipattack` | `METADATI_ERRATI` |
| `Liu_2025_AI` | `METADATI_ERRATI` |
| `Liu_2025_Purpcode` | `METADATI_ERRATI` |
| `Maloyan_2026_Prompt` | `METADATI_ERRATI` |
| `Mamun_2025_LLMs` | `METADATI_ERRATI` |
| `Mo_2025_LLMs` | `METADATI_ERRATI` |
| `Morbiato_2025_Grounded` | `METADATI_ERRATI` |
| `Mu_2024_Codepurify` | `BIB_MANCANTE` |
| `Muntasir_2025_Mocha` | `METADATI_ERRATI` |
| `Nie_2025_Secodeplt` | `NON_VERIFICATA` |
| `Nika_2025_Policy` | `BIB_MANCANTE` |
| `Noever_2024_AI` | `NON_VERIFICATA` |
| `Ouyang_2022_Training` | `BIB_MANCANTE` |
| `Ouyang_2025_LLM` | `METADATI_ERRATI` |
| `Pathmanathan_2024_LLM` | `BIB_MANCANTE` |
| `Peng_2025_LLM` | `METADATI_ERRATI` |
| `Raghuram_2024_Study` | `BIB_MANCANTE` |
| `Rubel_2024_LLM` | `METADATI_ERRATI` |
| `Shen_2025_Decepchain` | `METADATI_ERRATI` |
| `Sun_2025_Show` | `BIB_MANCANTE` |
| `Tang_2024_LLMs` | `BIB_MANCANTE` |
| `Triedman_2025_Multi` | `BIB_MANCANTE` |
| `Tu_2025_Attention` | `BIB_MANCANTE` |
| `Wang_2024_LLM` | `METADATI_DBLP_CORRISPONDENTI` |
| `Wang_2024_Mllm` | `BIB_MANCANTE` |
| `Wang_2024_Security` | `BIB_MANCANTE` |
| `Wang_2025_Draincode` | `BIB_MANCANTE` |
| `Yu_2025_LLMs` | `METADATI_DBLP_CORRISPONDENTI` |
| `Zhang_2024_Persistent` | `METADATI_ERRATI` |
| `Zhao_2024_Models` | `BIB_MANCANTE` |
| `Zhao_2024_SQL` | `METADATI_ERRATI` |
| `Zheng_2024_Improved` | `METADATI_ERRATI` |
| `Zhu_2025_AI` | `METADATI_ERRATI` |
| `Zhu_2025_LLM` | `METADATI_DBLP_CORRISPONDENTI` |
| `Zhu_2025_Think` | `NON_VERIFICATA` |
| `Zou_2023_Universal` | `BIB_MANCANTE` |

## Corruzioni più evidenti

- `Foerster_2025_Reasoning`: il titolo include rumore da PDF ("A PREPRINT") e
  l'affiliazione "Google DeepMind" è stata trasformata in autore.
- `Harary_2025_Prefixnli`: la lista autori contiene nomi aggiuntivi non presenti
  nel record DBLP.
- `Li_2025_LLM`: manca l'autore Gang Tu.
- `Liu_2024_Autodan`: titolo spezzato (`STRAT-EGY`) e autore non canonico.
- `Mo_2025_LLMs`: lista autori spezzata/corrotta.
- `Muntasir_2025_Mocha`: lista autori gravemente corrotta, con nomi di modelli e
  affiliazioni interpretati come persone; il record DBLP canonico è
  `conf/emnlp/WahedZNYDWHL25`.
- `Peng_2025_LLM`: manca Qi Liu e sono presenti autori estranei.
- Diverse entry locali sono `@misc` mentre DBLP le registra come
  `@article`/`@inproceedings`; anche queste non sono accettate come identiche.

## Placeholder interni da eliminare

- `[l2_Adversarial Attacks_v5, Triedman_2025_Multi]`
- `[l2_Adversarial Attacks_v5]`
- `[l2_Data and Model Poisoning_v4]`

## Conclusione

Il LaTeX v1.7 esistente non è idoneo come documento finale. Deve essere
rigenerato dopo una nuova ingestion DBLP rigorosa; il nuovo export fail-closed
blocca automaticamente un futuro `.tex` se anche una sola fonte citata non è
presente nel BibTeX ufficiale verificato e nel relativo manifest DBLP.
