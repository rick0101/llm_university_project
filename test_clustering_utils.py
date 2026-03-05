"""
Test per paperhunter.summarization.clustering_utils — Clustering semantico RAPTOR.

Verifica il raggruppamento degli L1 Summary usando embedding mock
con struttura nota (5 gruppi da 20 con piccola varianza intra-gruppo).
"""

from __future__ import annotations

import numpy as np
import pytest

from paperhunter.models import L1Summary
from paperhunter.summarization.clustering_utils import (
    cluster_summaries,
    get_cluster_labels,
    _normalize_embeddings,
    _agglomerative_cluster,
    _group_by_labels,
    _renumber_clusters,
    _split_oversized_clusters,
)


# ===========================================================================
# Helpers per generazione dati mock
# ===========================================================================

def _make_l1_summary(paper_id: str, category: str = "test_cat", year: int = 2024) -> L1Summary:
    """Crea un L1Summary mock minimale."""
    return L1Summary(
        paper_id=paper_id,
        bibtex_key=paper_id,
        category=category,
        year=year,
        key_contributions=["Contribution 1"],
        methodology="Test methodology",
        testing_methodology="Test testing methodology",
        temporal_context="2024",
        results_summary="Results",
        limitations="Limitations",
    )


def _make_clustered_embeddings(
    n_groups: int = 5,
    group_size: int = 20,
    dim: int = 128,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> tuple[list[L1Summary], list[list[float]]]:
    """Genera N gruppi di embedding con struttura nota.

    Ogni gruppo ha un centroide casuale e i membri hanno piccola
    varianza intra-gruppo (noise_scale).

    Returns:
        (summaries, embeddings) — liste allineate.
    """
    rng = np.random.RandomState(seed)
    summaries: list[L1Summary] = []
    embeddings: list[list[float]] = []

    for g in range(n_groups):
        # Centroide casuale normalizzato
        centroid = rng.randn(dim)
        centroid = centroid / np.linalg.norm(centroid)

        for i in range(group_size):
            paper_id = f"paper_g{g}_i{i}"
            summaries.append(_make_l1_summary(paper_id, category=f"cat_{g}", year=2020 + g))

            # Embedding = centroide + rumore piccolo
            noise = rng.randn(dim) * noise_scale
            vec = centroid + noise
            embeddings.append(vec.tolist())

    return summaries, embeddings


# ===========================================================================
# Test cluster_summaries
# ===========================================================================


class TestClusterSummaries:
    """Test per la funzione principale cluster_summaries."""

    def test_basic_clustering_5_groups(self) -> None:
        """5 gruppi da 20 → ~5 cluster, nessun cluster > max_cluster_size."""
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=5, group_size=20, dim=128, noise_scale=0.05,
        )
        clusters = cluster_summaries(
            summaries, embeddings,
            max_cluster_size=25,
            similarity_threshold=0.75,
        )

        # Deve creare cluster (il numero esatto dipende dalla soglia e dal rumore)
        assert len(clusters) >= 3, f"Troppo pochi cluster: {len(clusters)}"
        assert len(clusters) <= 30, f"Troppi cluster: {len(clusters)}"

        # Tutti i paper devono essere assegnati
        total = sum(len(members) for members in clusters.values())
        assert total == 100, f"Paper totali: {total}, attesi: 100"

        # Nessun cluster deve superare max_cluster_size
        for cid, members in clusters.items():
            assert len(members) <= 25, (
                f"Cluster {cid} ha {len(members)} membri (max 25)"
            )

    def test_cluster_ids_sequential(self) -> None:
        """I cluster ID devono essere 0, 1, 2, ... senza buchi."""
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=3, group_size=10, dim=64,
        )
        clusters = cluster_summaries(
            summaries, embeddings,
            max_cluster_size=15,
            similarity_threshold=0.75,
        )
        assert set(clusters.keys()) == set(range(len(clusters)))

    def test_few_summaries_single_cluster(self) -> None:
        """Se ci sono ≤ max_cluster_size summary, un solo cluster."""
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=1, group_size=10, dim=64,
        )
        clusters = cluster_summaries(
            summaries, embeddings,
            max_cluster_size=20,
            similarity_threshold=0.75,
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == 10

    def test_validation_mismatched_lengths(self) -> None:
        """Deve sollevare ValueError se summaries e embeddings hanno length diversa."""
        summaries = [_make_l1_summary(f"p{i}") for i in range(5)]
        embeddings = [[0.0] * 64 for _ in range(3)]
        with pytest.raises(ValueError, match="Numero di summary"):
            cluster_summaries(summaries, embeddings)

    def test_validation_empty(self) -> None:
        """Deve sollevare ValueError con lista vuota."""
        with pytest.raises(ValueError, match="vuota"):
            cluster_summaries([], [])

    def test_recursive_split_large_cluster(self) -> None:
        """Un singolo gruppo grande deve essere suddiviso ricorsivamente."""
        # 1 gruppo da 50 paper, max_cluster_size = 10
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=1, group_size=50, dim=128, noise_scale=0.3,
        )
        clusters = cluster_summaries(
            summaries, embeddings,
            max_cluster_size=10,
            similarity_threshold=0.5,  # soglia bassa per forzare split
        )
        # Tutti i cluster devono essere ≤ 10
        for cid, members in clusters.items():
            assert len(members) <= 10, (
                f"Cluster {cid} ha {len(members)} membri (max 10)"
            )
        # Tutti i paper devono essere presenti
        total = sum(len(m) for m in clusters.values())
        assert total == 50

    def test_deterministic(self) -> None:
        """Due chiamate con gli stessi dati devono dare lo stesso risultato."""
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=3, group_size=15, dim=64,
        )
        c1 = cluster_summaries(summaries, embeddings, max_cluster_size=20, similarity_threshold=0.75)
        c2 = cluster_summaries(summaries, embeddings, max_cluster_size=20, similarity_threshold=0.75)

        assert len(c1) == len(c2)
        for cid in c1:
            ids_1 = sorted(s.paper_id for s in c1[cid])
            ids_2 = sorted(s.paper_id for s in c2[cid])
            assert ids_1 == ids_2


# ===========================================================================
# Test get_cluster_labels
# ===========================================================================

class TestGetClusterLabels:
    """Test per la funzione get_cluster_labels."""

    def test_returns_correct_length(self) -> None:
        """La lista di label deve avere la stessa lunghezza degli embedding."""
        _, embeddings = _make_clustered_embeddings(n_groups=3, group_size=10, dim=64)
        labels = get_cluster_labels(embeddings, similarity_threshold=0.75)
        assert len(labels) == 30

    def test_returns_ints(self) -> None:
        """Le label devono essere int."""
        _, embeddings = _make_clustered_embeddings(n_groups=2, group_size=5, dim=32)
        labels = get_cluster_labels(embeddings, similarity_threshold=0.75)
        assert all(isinstance(lbl, int) for lbl in labels)

    def test_empty_raises(self) -> None:
        """Deve sollevare ValueError con lista vuota."""
        with pytest.raises(ValueError, match="vuota"):
            get_cluster_labels([])


# ===========================================================================
# Test funzioni interne
# ===========================================================================

class TestNormalizeEmbeddings:
    """Test per _normalize_embeddings."""

    def test_unit_norm(self) -> None:
        """Dopo la normalizzazione, ogni vettore ha norma L2 ≈ 1."""
        X = np.random.RandomState(0).randn(20, 64)
        X_norm = _normalize_embeddings(X)
        norms = np.linalg.norm(X_norm, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_zero_vector(self) -> None:
        """Un vettore zero non causa divisione per zero."""
        X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        X_norm = _normalize_embeddings(X)
        # Il vettore zero resta zero (normalizzato con divisore 1.0)
        assert np.allclose(X_norm[0], [0.0, 0.0, 0.0])
        assert np.allclose(X_norm[1], [1.0, 0.0, 0.0])


class TestAgglomerativeCluster:
    """Test per _agglomerative_cluster."""

    def test_two_well_separated_groups(self) -> None:
        """Due gruppi ben separati → 2 cluster."""
        rng = np.random.RandomState(42)
        # Gruppo A: centrato in [1, 0, 0, ...]
        A = rng.randn(10, 32) * 0.01
        A[:, 0] += 5.0
        # Gruppo B: centrato in [-1, 0, 0, ...]
        B = rng.randn(10, 32) * 0.01
        B[:, 0] -= 5.0

        X = np.vstack([A, B])
        X = _normalize_embeddings(X)

        labels = _agglomerative_cluster(X, distance_threshold=0.3)
        unique = set(labels.tolist())
        assert len(unique) == 2, f"Attesi 2 cluster, trovati {len(unique)}"


class TestGroupByLabels:
    """Test per _group_by_labels."""

    def test_basic_grouping(self) -> None:
        summaries = [_make_l1_summary(f"p{i}") for i in range(6)]
        labels = np.array([0, 0, 1, 1, 2, 2])
        groups = _group_by_labels(summaries, labels)
        assert len(groups) == 3
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2
        assert len(groups[2]) == 2


class TestRenumberClusters:
    """Test per _renumber_clusters."""

    def test_sequential_sorted_by_size(self) -> None:
        clusters = {
            10: [_make_l1_summary("a")],
            20: [_make_l1_summary("b"), _make_l1_summary("c"), _make_l1_summary("d")],
            30: [_make_l1_summary("e"), _make_l1_summary("f")],
        }
        renumbered = _renumber_clusters(clusters)
        assert set(renumbered.keys()) == {0, 1, 2}
        # Ordine per dimensione decrescente
        assert len(renumbered[0]) == 3  # il più grande
        assert len(renumbered[1]) == 2
        assert len(renumbered[2]) == 1


# ===========================================================================
# Test di integrazione: struttura cluster coerente
# ===========================================================================

class TestClusteringIntegration:
    """Test di integrazione: verifica la coerenza dei cluster."""

    def test_cluster_purity(self) -> None:
        """I paper dello stesso gruppo mock dovrebbero finire nello stesso cluster
        (con alta probabilità, dato il basso rumore)."""
        summaries, embeddings = _make_clustered_embeddings(
            n_groups=5, group_size=20, dim=128, noise_scale=0.02,
        )
        clusters = cluster_summaries(
            summaries, embeddings,
            max_cluster_size=25,
            similarity_threshold=0.80,
        )

        # Per ogni cluster, calcola il gruppo di appartenenza dominante
        for cid, members in clusters.items():
            groups = [s.paper_id.split("_")[0] + "_" + s.paper_id.split("_")[1]
                      for s in members]
            # Almeno l'80% dei membri dovrebbe appartenere allo stesso gruppo
            from collections import Counter
            group_counts = Counter(groups)
            dominant_count = group_counts.most_common(1)[0][1]
            purity = dominant_count / len(members)
            assert purity >= 0.8, (
                f"Cluster {cid}: purity={purity:.2f}, "
                f"distribuzione={dict(group_counts)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
