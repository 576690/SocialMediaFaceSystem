import unittest

import numpy as np

from core.clustering import approximate_purity
from core.clustering import build_full_cluster_result
from core.clustering import cluster_embeddings
from core.clustering import evaluate_embedding_clusters
from core.clustering import evaluate_embedding_retrieval
from core.clustering import inherit_person_names
from core.clustering import perform_clustering


class FakeClusterDB:
    def __init__(self, records, person_name_map=None, max_person_id=40):
        self.records = records
        self.person_name_map = person_name_map or {}
        self.max_person_id = max_person_id
        self.snapshot_config = None
        self.updated_person_ids = []
        self.replaced_person_ids = None

    def get_unassigned_faces_with_embeddings(self):
        return [record for record in self.records if int(record.get("person_id", -1)) == -1]

    def get_all_faces_with_embeddings(self):
        return list(self.records)

    def get_max_person_id(self):
        return self.max_person_id

    def update_person_ids(self, id_label_map):
        self.updated_person_ids = list(id_label_map)

    def get_person_name_map(self):
        return dict(self.person_name_map)

    def save_cluster_snapshot(self, cluster_config=None):
        self.snapshot_config = dict(cluster_config or {})
        return len(self.records)

    def replace_all_person_ids(self, id_label_map, people_name_map=None):
        self.replaced_person_ids = {
            "assignments": list(id_label_map),
            "people_name_map": dict(people_name_map or {}),
        }


class ClusteringTests(unittest.TestCase):
    def test_dbscan_clusters_two_groups(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.02, 0.98],
            ],
            dtype=np.float32,
        )
        labels = cluster_embeddings(
            embeddings,
            algorithm="dbscan",
            metric="euclidean",
            eps=0.1,
            min_samples=2,
        )
        self.assertEqual(len(set(labels.tolist())) - (1 if -1 in labels else 0), 2)

    def test_approximate_purity(self):
        labels = np.array([0, 0, 1, 1, -1])
        person_ids = [10, 10, 20, 21, 30]
        self.assertEqual(approximate_purity(labels, person_ids), 0.75)

    def test_evaluate_embedding_clusters_returns_external_metrics(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        results = evaluate_embedding_clusters(
            embeddings,
            person_ids=[0, 0, 1, 1],
            algorithms=["dbscan"],
            metrics=["euclidean"],
            eps_values=[0.1],
            min_samples_values=[2],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["clusters_count"], 2)
        self.assertEqual(results[0]["purity"], 1.0)
        self.assertEqual(results[0]["nmi"], 1.0)
        self.assertEqual(results[0]["ari"], 1.0)

    def test_evaluate_embedding_retrieval_returns_top1_and_top5(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        results = evaluate_embedding_retrieval(
            embeddings,
            person_ids=[0, 0, 1, 1],
            metrics=["cosine"],
            top_k=2,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["top1"], 1.0)
        self.assertEqual(results[0]["top5"], 1.0)

    def test_inherit_person_names_uses_max_overlap_and_unique_name(self):
        records = [
            {"id": 1, "person_id": 10},
            {"id": 2, "person_id": 10},
            {"id": 3, "person_id": 20},
            {"id": 4, "person_id": 20},
            {"id": 5, "person_id": 20},
        ]
        labels = np.array([0, 0, 0, 1, 1], dtype=int)
        people_map = inherit_person_names(
            records,
            labels,
            {0: 1, 1: 2},
            {10: "Alice", 20: "Bob"},
        )
        self.assertEqual(people_map, {1: "Alice", 2: "Bob"})

    def test_build_full_cluster_result_creates_continuous_ids(self):
        records = [
            {"id": 11, "person_id": 101},
            {"id": 12, "person_id": 101},
            {"id": 21, "person_id": 202},
            {"id": 22, "person_id": -1},
        ]
        labels = np.array([5, 5, 9, -1], dtype=int)
        result = build_full_cluster_result(
            records,
            labels,
            {101: "Alpha", 202: "Beta"},
        )
        self.assertEqual(result["updates"], [(1, 11), (1, 12), (2, 21), (-1, 22)])
        self.assertEqual(result["people_name_map"], {1: "Alpha", 2: "Beta"})
        self.assertEqual(result["reassigned_faces"], 3)

    def test_perform_clustering_incremental_only_updates_unassigned(self):
        records = [
            {"id": 1, "person_id": -1, "embedding": np.array([1.0, 0.0], dtype=np.float32)},
            {"id": 2, "person_id": -1, "embedding": np.array([0.98, 0.02], dtype=np.float32)},
            {"id": 3, "person_id": 8, "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        ]
        db = FakeClusterDB(records, max_person_id=8)
        result = perform_clustering(
            db,
            algorithm="dbscan",
            metric="euclidean",
            eps=0.1,
            min_samples=2,
            mode="incremental",
        )
        self.assertEqual(result["mode"], "incremental")
        self.assertFalse(result["snapshot_created"])
        self.assertEqual(db.updated_person_ids, [(9, 1), (9, 2)])
        self.assertIsNone(db.replaced_person_ids)

    def test_perform_clustering_full_replaces_people_and_saves_snapshot(self):
        records = [
            {"id": 1, "person_id": 10, "embedding": np.array([1.0, 0.0], dtype=np.float32)},
            {"id": 2, "person_id": 10, "embedding": np.array([0.98, 0.02], dtype=np.float32)},
            {"id": 3, "person_id": 20, "embedding": np.array([0.0, 1.0], dtype=np.float32)},
            {"id": 4, "person_id": 20, "embedding": np.array([0.02, 0.98], dtype=np.float32)},
        ]
        db = FakeClusterDB(records, person_name_map={10: "Alice", 20: "Bob"})
        result = perform_clustering(
            db,
            algorithm="dbscan",
            metric="euclidean",
            eps=0.1,
            min_samples=2,
            mode="full",
            snapshot_config={"algorithm": "optics", "metric": "cosine", "eps": 0.7, "min_samples": 3},
        )
        self.assertEqual(result["mode"], "full")
        self.assertTrue(result["snapshot_created"])
        self.assertEqual(db.snapshot_config["algorithm"], "optics")
        self.assertEqual(
            db.replaced_person_ids["assignments"],
            [(1, 1), (1, 2), (2, 3), (2, 4)],
        )
        self.assertEqual(db.replaced_person_ids["people_name_map"], {1: "Alice", 2: "Bob"})


if __name__ == "__main__":
    unittest.main()
