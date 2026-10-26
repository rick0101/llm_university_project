"""
Clustering semantico per RAPTOR — Raggruppamento gerarchico degli L1 Summary.

Raggruppa semanticamente gli L1 Summary usando i loro embedding
(dal Qdrant store), creando cluster tematici di dimensione controllata.
Questi cluster vengono poi usati dal livello L1.5 per generare
meta-summary compatti, riducendo da ~1.000 L1 a ~50 cluster summary.

Algoritmo:
- ``AgglomerativeClustering`` (scikit-learn) con metrica coseno e linkage average
- ``distance_threshold`` derivato da ``1 - similarity_threshold`` (da config.yaml)
- Se un cluster supera ``max_cluster_size``, viene ri-clusterizzato ricorsivamente
- Outlier (cluster di dimensione 1) vengono mantenuti come cluster singoli

Uso::

    from paperhunter.summarization.clustering_utils import cluster_summaries

    # summaries: list[L1Summary], embeddings: list[list[float]]
    clusters = cluster_summaries(summaries, embeddings)
    # → {0: [L1Summary, ...], 1: [L1Summary, ...], ...}

Configurazione (config.yaml → clustering):
- ``max_cluster_size``: Massimo numero di paper per cluster (default: 20)
- ``similarity_threshold``: Soglia similarità coseno (default: 0.75)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from paperhunter.models import L1Summary
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

def _get_clustering_config() -> tuple[int, float]:
    """Carica i parametri di clustering dalla configurazione.

    Returns:
        Tupla ``(max_cluster_size, similarity_threshold)``.
    """
    from paperhunter.config import settings

    max_cluster_size = settings.clustering.max_cluster_size
    similarity_threshold = settings.clustering.similarity_threshold
    return max_cluster_size, similarity_threshold


# ---------------------------------------------------------------------------
# Funzione principale: clustering degli L1 Summary
# ---------------------------------------------------------------------------

def cluster_summaries(
    l1_summaries: list[L1Summary],
    embeddings: list[list[float]],
    max_cluster_size: int | None = None,
    similarity_threshold: float | None = None,
) -> dict[int, list[L1Summary]]:
    """Raggruppa semanticamente gli L1 Summary usando Agglomerative Clustering.

    Usa la distanza coseno (``1 - cosine_similarity``) per misurare la
    distanza tra gli embedding. I cluster che superano ``max_cluster_size``
    vengono ri-clusterizzati ricorsivamente.

    Args:
        l1_summaries: Lista di L1 Summary da raggruppare.
        embeddings: Lista di vettori embedding corrispondenti (stessa
            lunghezza di ``l1_summaries``).
        max_cluster_size: Massimo numero di paper per cluster. Se None,
            usa il valore da ``config.yaml``.
        similarity_threshold: Soglia di similarità coseno. Se None,
            usa il valore da ``config.yaml``.

    Returns:
        Dizionario ``{cluster_id: [L1Summary, ...]}`` con i cluster
        risultanti. I cluster ID sono interi sequenziali da 0.

    Raises:
        ValueError: Se il numero di summary non corrisponde al numero
            di embedding, o se le liste sono vuote.
    """
    # Validazione input
    if len(l1_summaries) != len(embeddings):
        raise ValueError(
            f"Numero di summary ({len(l1_summaries)}) diverso dal "
            f"numero di embedding ({len(embeddings)})"
        )
    if not l1_summaries:
        raise ValueError("La lista di L1 Summary è vuota")

    # Carica config se non specificata
    cfg_max_size, cfg_sim_threshold = _get_clustering_config()
    max_cluster_size = max_cluster_size or cfg_max_size
    similarity_threshold = similarity_threshold or cfg_sim_threshold

    logger.info(
        f"Clustering {len(l1_summaries)} L1 Summary "
        f"(max_cluster_size={max_cluster_size}, "
        f"similarity_threshold={similarity_threshold})"
    )

    # Caso banale: pochi summary, un solo cluster
    if len(l1_summaries) <= max_cluster_size:
        logger.info(
            f"Solo {len(l1_summaries)} summary (≤ max_cluster_size={max_cluster_size}): "
            f"creato 1 cluster unico"
        )
        return {0: list(l1_summaries)}

    # Converti in numpy array e normalizza per cosine distance
    X = np.array(embeddings, dtype=np.float64)
    X = _normalize_embeddings(X)

    # Calcola le label di cluster
    distance_threshold = 1.0 - similarity_threshold
    labels = _agglomerative_cluster(X, distance_threshold)

    # Raggruppa i summary per cluster
    clusters = _group_by_labels(l1_summaries, labels)

    # Ri-clusterizza ricorsivamente i cluster troppo grandi
    embeddings_array = X  # già normalizzato
    clusters = _split_oversized_clusters(
        clusters=clusters,
        embeddings_map={
            i: embeddings_array[i] for i in range(len(embeddings_array))
        },
        summaries_index={
            id(s): idx for idx, s in enumerate(l1_summaries)
        },
        max_cluster_size=max_cluster_size,
        distance_threshold=distance_threshold,
    )

    # Rinumera i cluster da 0 in sequenza
    clusters = _renumber_clusters(clusters)

    # Statistiche e logging
    _log_cluster_stats(clusters)

    return clusters


# ---------------------------------------------------------------------------
# Funzione leggera: solo label di cluster
# ---------------------------------------------------------------------------

def get_cluster_labels(
    embeddings: list[list[float]],
    similarity_threshold: float | None = None,
) -> list[int]:
    """Wrapper leggero che restituisce solo le label di cluster.

    Args:
        embeddings: Lista di vettori embedding.
        similarity_threshold: Soglia similarità coseno. Se None,
            usa il valore da ``config.yaml``.

    Returns:
        Lista di label di cluster (un int per ogni embedding).

    Raises:
        ValueError: Se la lista di embedding è vuota.
    """
    if not embeddings:
        raise ValueError("La lista di embedding è vuota")

    _, cfg_sim_threshold = _get_clustering_config()
    similarity_threshold = similarity_threshold or cfg_sim_threshold

    X = np.array(embeddings, dtype=np.float64)
    X = _normalize_embeddings(X)

    distance_threshold = 1.0 - similarity_threshold
    labels = _agglomerative_cluster(X, distance_threshold)

    return labels.tolist()


# ---------------------------------------------------------------------------
# Funzione per recuperare gli embedding degli L1 dal Qdrant store
# ---------------------------------------------------------------------------

def fetch_l1_embeddings_from_qdrant(
    l1_summaries: list[L1Summary],
    qdrant_store: object | None = None,
) -> list[list[float]]:
    """Recupera i vettori embedding degli L1 Summary dalla collection Qdrant.

    Esegue uno scroll sulla collection ``l1_summaries`` con ``with_vectors=True``
    e associa ogni vettore al corrispondente L1 Summary tramite ``paper_id``.

    Args:
        l1_summaries: Lista di L1 Summary di cui recuperare gli embedding.
        qdrant_store: Istanza ``QdrantStore``. Se None, ne crea una nuova.

    Returns:
        Lista di vettori embedding (stessa lunghezza e ordine di ``l1_summaries``).

    Raises:
        ValueError: Se non vengono trovati embedding per uno o più summary.
    """
    from qdrant_client import models as qdrant_models

    if qdrant_store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        qdrant_store = QdrantStore()

    from paperhunter.config import settings
    collection = settings.qdrant.collection_l1_summaries

    # Mappa paper_id → indice nell'array di L1 summary
    paper_id_to_index: dict[str, int] = {}
    for idx, s in enumerate(l1_summaries):
        paper_id_to_index[s.paper_id] = idx

    # Scroll tutti i punti con vettori dalla collection L1
    all_vectors: dict[str, list[float]] = {}
    offset = None

    while True:
        try:
            points, next_offset = qdrant_store.client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=["paper_id"],
                with_vectors=True,
            )
        except Exception as e:
            logger.error(f"Errore durante scroll L1 embeddings: {e}")
            break

        for point in points:
            payload = point.payload or {}
            pid = payload.get("paper_id", "")
            if pid and pid in paper_id_to_index:
                vector = point.vector
                if isinstance(vector, list):
                    all_vectors[pid] = vector
                elif isinstance(vector, dict):
                    # Named vectors: prendi il primo disponibile
                    for v in vector.values():
                        all_vectors[pid] = v
                        break

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        f"Recuperati {len(all_vectors)}/{len(l1_summaries)} embedding L1 da Qdrant"
    )

    # Assembla il risultato nell'ordine originale
    embeddings: list[list[float]] = []
    missing: list[str] = []

    for s in l1_summaries:
        if s.paper_id in all_vectors:
            embeddings.append(all_vectors[s.paper_id])
        else:
            missing.append(s.paper_id)
            # Vettore zero come fallback (verrà potenzialmente isolato come outlier)
            dim = len(next(iter(all_vectors.values()))) if all_vectors else 3072
            embeddings.append([0.0] * dim)

    if missing:
        logger.warning(
            f"{len(missing)} L1 Summary senza embedding: "
            f"verranno trattati come outlier. Primi 5: {missing[:5]}"
        )

    return embeddings


# ===========================================================================
# Funzioni interne
# ===========================================================================

def _normalize_embeddings(X: np.ndarray) -> np.ndarray:
    """Normalizza L2 i vettori per rendere la distanza euclidea ≈ coseno.

    Dopo la normalizzazione, ``||a - b||² = 2(1 - cos(a,b))``, quindi
    la distanza euclidea su vettori normalizzati è monotona rispetto
    alla cosine distance.

    Args:
        X: Matrice (n_samples, n_features).

    Returns:
        Matrice normalizzata (stessa shape).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Evita divisione per zero
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def _agglomerative_cluster(
    X: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """Esegue Agglomerative Clustering con distanza euclidea su vettori normalizzati.

    Poiché i vettori sono normalizzati L2, la distanza euclidea è
    monotonamente legata alla cosine distance:
    ``euclidean_dist = sqrt(2 * (1 - cosine_similarity))``

    Si converte la ``distance_threshold`` (cosine) in soglia euclidea.

    Args:
        X: Matrice normalizzata (n_samples, n_features).
        distance_threshold: Soglia cosine distance (``1 - similarity_threshold``).

    Returns:
        Array di label per ogni campione.
    """
    # Converti cosine distance threshold in euclidean distance threshold
    # euclidean_dist = sqrt(2 * cosine_distance)
    euclidean_threshold = np.sqrt(2.0 * distance_threshold)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=euclidean_threshold,
        metric="euclidean",
        linkage="average",
    )
    labels = model.fit_predict(X)
    return labels


def _group_by_labels(
    summaries: list[L1Summary],
    labels: np.ndarray,
) -> dict[int, list[L1Summary]]:
    """Raggruppa i summary per label di cluster.

    Args:
        summaries: Lista di L1 Summary.
        labels: Array di label (un int per ogni summary).

    Returns:
        Dizionario ``{label: [L1Summary, ...]}``.
    """
    clusters: dict[int, list[L1Summary]] = {}
    for summary, label in zip(summaries, labels):
        label_int = int(label)
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append(summary)
    return clusters


def _split_oversized_clusters(
    clusters: dict[int, list[L1Summary]],
    embeddings_map: dict[int, np.ndarray],
    summaries_index: dict[int, int],
    max_cluster_size: int,
    distance_threshold: float,
) -> dict[int, list[L1Summary]]:
    """Ri-clusterizza ricorsivamente i cluster che superano max_cluster_size.

    Per ogni cluster troppo grande:
    1. Estrai i sotto-embedding corrispondenti
    2. Riduci la soglia di distanza del 20% per ottenere cluster più piccoli
    3. Applica Agglomerative Clustering al sotto-insieme
    4. Se il risultato contiene ancora cluster troppo grandi, ricorsione

    Args:
        clusters: Dizionario cluster attuali.
        embeddings_map: Mappa indice globale → vettore normalizzato.
        summaries_index: Mappa id(summary) → indice globale.
        max_cluster_size: Dimensione massima per cluster.
        distance_threshold: Soglia cosine distance corrente.

    Returns:
        Dizionario aggiornato con tutti i cluster entro il limite.
    """
    result: dict[int, list[L1Summary]] = {}
    next_id = 0

    for label, members in clusters.items():
        if len(members) <= max_cluster_size:
            result[next_id] = members
            next_id += 1
        else:
            # Recupera gli embedding dei membri di questo cluster
            sub_indices = []
            sub_embeddings = []
            for s in members:
                idx = summaries_index.get(id(s))
                if idx is not None and idx in embeddings_map:
                    sub_indices.append(idx)
                    sub_embeddings.append(embeddings_map[idx])

            if len(sub_embeddings) < 2:
                # Non abbastanza dati per ri-clusterizzare
                result[next_id] = members
                next_id += 1
                continue

            X_sub = np.array(sub_embeddings, dtype=np.float64)

            # Riduci la soglia per ottenere cluster più piccoli
            reduced_threshold = distance_threshold * 0.8

            # Soglia minima per evitare loop infiniti
            if reduced_threshold < 0.01:
                logger.warning(
                    f"Soglia minima raggiunta per cluster con {len(members)} "
                    f"membri: verrà suddiviso in blocchi di {max_cluster_size}"
                )
                # Fallback: split sequenziale
                for i in range(0, len(members), max_cluster_size):
                    result[next_id] = members[i:i + max_cluster_size]
                    next_id += 1
                continue

            sub_labels = _agglomerative_cluster(X_sub, reduced_threshold)
            sub_clusters = _group_by_labels(members, sub_labels)

            # Ricorsione: controlla se i sotto-cluster sono ancora troppo grandi
            for sub_label, sub_members in sub_clusters.items():
                if len(sub_members) <= max_cluster_size:
                    result[next_id] = sub_members
                    next_id += 1
                else:
                    # Ricorsione con soglia ulteriormente ridotta
                    sub_sub = _split_oversized_clusters(
                        clusters={0: sub_members},
                        embeddings_map=embeddings_map,
                        summaries_index=summaries_index,
                        max_cluster_size=max_cluster_size,
                        distance_threshold=reduced_threshold,
                    )
                    for _, sub_sub_members in sub_sub.items():
                        result[next_id] = sub_sub_members
                        next_id += 1

    return result


def _renumber_clusters(
    clusters: dict[int, list[L1Summary]],
) -> dict[int, list[L1Summary]]:
    """Rinumera i cluster con ID sequenziali da 0.

    Args:
        clusters: Dizionario con ID potenzialmente non sequenziali.

    Returns:
        Dizionario con ID 0, 1, 2, ... in ordine decrescente di dimensione.
    """
    # Ordina per dimensione decrescente (cluster più grandi prima)
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    return {i: members for i, members in enumerate(sorted_clusters)}


def _log_cluster_stats(clusters: dict[int, list[L1Summary]]) -> None:
    """Logga statistiche sui cluster con Rich.

    Riporta: numero di cluster, distribuzione dimensioni, outlier count.

    Args:
        clusters: Dizionario dei cluster risultanti.
    """
    n_clusters = len(clusters)
    sizes = [len(members) for members in clusters.values()]
    total_papers = sum(sizes)
    outlier_count = sum(1 for s in sizes if s == 1)

    if not sizes:
        logger.info("Nessun cluster creato")
        return

    avg_size = total_papers / n_clusters
    min_size = min(sizes)
    max_size = max(sizes)

    logger.info(
        f"Clustering completato: {n_clusters} cluster, "
        f"{total_papers} paper totali"
    )
    logger.info(
        f"Dimensioni cluster — "
        f"min: {min_size}, max: {max_size}, avg: {avg_size:.1f}, "
        f"outlier (size=1): {outlier_count}"
    )

    # Top-5 cluster per dimensione
    top_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for cid, members in top_clusters:
        categories = set()
        years = set()
        for s in members:
            if s.category:
                categories.add(s.category)
            if s.year:
                years.add(s.year)
        year_range = f"{min(years)}-{max(years)}" if years else "N/A"
        logger.debug(
            f"  Cluster {cid}: {len(members)} paper, "
            f"categorie={categories}, anni={year_range}"
        )
