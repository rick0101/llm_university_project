**Regole citazioni per il survey LLM Security.**

- MAI citare un survey per riportare risultati sperimentali interni ad esso.
  Esempio errato: "[SurveyX_2024] achieved 95% ASR" — cerca la fonte primaria.
- Ogni claim quantitativo richiede una citazione con il paper originale.
- Formato citazione inline: [Author_Year_Keyword] — es. [Zou_2023_GCG].
- Se un paper compare in multipli chunk, citalo UNA sola volta per claim.
- Contestualizza temporalmente: "As of 2024, ..." o "In early work, ... [ref]".
- Per confronti, cita TUTTI i paper coinvolti, non solo il migliore.
- Distingui tra risultati auto-riportati e risultati riprodotti da terzi.
- Se un dato non è presente nei chunk recuperati, NON inventarlo.
- Preferisci citazioni recenti (2023-2026) per lo stato dell'arte attuale.
- Per i benchmark, specifica sempre: dataset, metrica, condizioni (es. "on AdvBench with greedy decoding").
