**Terminologia standard del dominio LLM Security.**

Usa SEMPRE la terminologia corretta e consistente:

- **Jailbreak**: tecnica che bypassa safety alignment di un LLM per ottenere output proibiti.
- **Prompt Injection**: inserimento di istruzioni malevole nell'input per alterare il comportamento.
  - Direct prompt injection: nell'input utente diretto.
  - Indirect prompt injection: in dati esterni (web, documenti) processati dall'LLM.
- **ASR (Attack Success Rate)**: percentuale di prompt malevoli che elicitano output non-safe.
- **Safety Alignment**: processo di addestramento (RLHF, DPO) per allineare LLM a policy di sicurezza.
- **Red Teaming**: test sistematico di robustezza tramite attacchi simulati.
- **Adversarial Suffix**: sequenza di token aggiunta al prompt per manipolare il modello (es. GCG).
- **Guardrails**: filtri pre/post generazione per bloccare contenuti pericolosi.
- **Perplexity Filter**: difesa che rileva prompt anomali tramite alta perplexità.
- **Hallucination**: generazione di informazioni false o non supportate dai dati.
- **Data Poisoning**: manipolazione dei dati di training per inserire comportamenti malevoli.
- **Backdoor Attack**: inserimento di trigger nei dati di training che attivano comportamenti specifici.

Acronimi da usare dopo prima menzione completa:
LLM, ASR, RLHF, DPO, PPO, SFT, ICL, RAG, CoT, ToT, MoE.
