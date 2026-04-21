from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


try:
    import hdbscan
except ImportError:
    hdbscan = None


SUPPORTED_ALGORITHMS = ("dbscan", "hdbscan", "optics")
SUPPORTED_METRICS = ("cosine", "euclidean")
SUPPORTED_CLUSTER_MODES = ("incremental", "full")


def _normalize_options(algorithm, metric):
    algorithm = (algorithm or "dbscan").lower()
    metric = (metric or "cosine").lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return algorithm, metric


def normalize_cluster_mode(mode):
    mode = (mode or "incremental").lower()
    if mode not in SUPPORTED_CLUSTER_MODES:
        raise ValueError(f"Unsupported clustering mode: {mode}")
    return mode


def cluster_embeddings(embeddings, algorithm="dbscan", metric="cosine", eps=0.4, min_samples=2):
    algorithm, metric = _normalize_options(algorithm, metric)
    if len(embeddings) == 0:
        return np.array([], dtype=int)

    matrix = np.asarray(embeddings, dtype=np.float32)
    if algorithm == "dbscan":
        model = DBSCAN(
            eps=float(eps),
            min_samples=max(int(min_samples), 2),
            metric=metric,
            n_jobs=-1,
        )
    elif algorithm == "optics":
        model = OPTICS(
            min_samples=max(int(min_samples), 2),
            metric=metric,
            max_eps=float(eps),
            n_jobs=-1,
        )
    else:
        if hdbscan is None:
            raise RuntimeError("HDBSCAN is unavailable. Install the hdbscan package.")
        model = hdbscan.HDBSCAN(
            min_cluster_size=max(int(min_samples), 2),
            min_samples=max(int(min_samples), 2),
            metric=metric,
            cluster_selection_epsilon=float(eps),
        )

    return np.asarray(model.fit_predict(matrix))


def cluster_summary(labels):
    labels = np.asarray(labels, dtype=int)
    unique_labels = set(int(label) for label in labels.tolist()) if len(labels) else set()
    clusters_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = int(np.sum(labels == -1)) if len(labels) else 0
    return {
        "clusters_count": clusters_count,
        "noise_count": noise_count,
        "noise_ratio": round(noise_count / len(labels), 4) if len(labels) else 0.0,
    }


def approximate_purity(labels, person_ids):
    buckets = defaultdict(list)
    labeled_total = 0
    dominant_total = 0

    for cluster_label, person_id in zip(labels, person_ids):
        if int(cluster_label) == -1 or int(person_id) == -1:
            continue
        buckets[int(cluster_label)].append(int(person_id))
        labeled_total += 1

    for members in buckets.values():
        dominant_total += Counter(members).most_common(1)[0][1]

    if labeled_total == 0:
        return None
    return round(dominant_total / labeled_total, 4)


def external_cluster_metrics(labels, person_ids):
    labels = np.asarray(labels, dtype=int)
    person_ids = np.asarray(person_ids, dtype=int)
    valid_mask = person_ids != -1
    if int(np.sum(valid_mask)) < 2:
        return {"purity": None, "nmi": None, "ari": None}

    filtered_labels = labels[valid_mask]
    filtered_person_ids = person_ids[valid_mask]
    if len(set(filtered_person_ids.tolist())) < 2:
        return {
            "purity": approximate_purity(filtered_labels, filtered_person_ids),
            "nmi": None,
            "ari": None,
        }

    return {
        "purity": approximate_purity(filtered_labels, filtered_person_ids),
        "nmi": round(
            float(normalized_mutual_info_score(filtered_person_ids, filtered_labels)),
            4,
        ),
        "ari": round(
            float(adjusted_rand_score(filtered_person_ids, filtered_labels)),
            4,
        ),
    }


def evaluate_embedding_clusters(
    embeddings,
    person_ids,
    algorithms=None,
    metrics=None,
    eps_values=None,
    min_samples_values=None,
):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) < 2:
        return []

    algorithms = algorithms or list(SUPPORTED_ALGORITHMS)
    metrics = metrics or list(SUPPORTED_METRICS)
    eps_values = [float(value) for value in (eps_values or [0.4])]
    min_samples_values = [max(int(value), 2) for value in (min_samples_values or [2])]
    results = []

    for algorithm in algorithms:
        for metric in metrics:
            for eps in eps_values:
                for min_samples in min_samples_values:
                    try:
                        labels = cluster_embeddings(
                            embeddings,
                            algorithm=algorithm,
                            metric=metric,
                            eps=eps,
                            min_samples=min_samples,
                        )
                        summary = cluster_summary(labels)
                        metrics_summary = external_cluster_metrics(labels, person_ids)
                        results.append(
                            {
                                "algorithm": algorithm,
                                "metric": metric,
                                "eps": float(eps),
                                "min_samples": int(min_samples),
                                "clusters_count": summary["clusters_count"],
                                "noise_count": summary["noise_count"],
                                "noise_ratio": summary["noise_ratio"],
                                "purity": metrics_summary["purity"],
                                "nmi": metrics_summary["nmi"],
                                "ari": metrics_summary["ari"],
                            }
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "algorithm": algorithm,
                                "metric": metric,
                                "eps": float(eps),
                                "min_samples": int(min_samples),
                                "error": str(exc),
                            }
                        )

    results.sort(
        key=lambda item: (
            bool(item.get("error")),
            item.get("purity") is None,
            -(item.get("purity") or 0.0),
            -(item.get("nmi") or -1.0),
            -(item.get("ari") or -1.0),
            item.get("noise_ratio", 1.0),
        )
    )
    return results


def evaluate_embedding_retrieval(embeddings, person_ids, metrics=None, top_k=5):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) < 3:
        return []

    metrics = metrics or list(SUPPORTED_METRICS)
    person_ids = [int(person_id) for person_id in person_ids]
    normalized_matrix = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        a_min=1e-12,
        a_max=None,
    )
    metrics_summary = []

    for metric in metrics:
        top1_hits = 0
        topk_hits = 0
        valid_queries = 0

        for index, query in enumerate(normalized_matrix):
            target_person = person_ids[index]
            if person_ids.count(target_person) < 2:
                continue

            valid_queries += 1
            candidates = []
            for candidate_index, candidate in enumerate(normalized_matrix):
                if index == candidate_index:
                    continue
                if metric == "euclidean":
                    distance = float(np.linalg.norm(query - candidate))
                    score = 1.0 / (1.0 + distance)
                    candidates.append((candidate_index, score, distance))
                else:
                    score = float(np.dot(query, candidate))
                    candidates.append((candidate_index, score, None))

            if metric == "euclidean":
                candidates.sort(key=lambda item: item[2])
            else:
                candidates.sort(key=lambda item: item[1], reverse=True)

            top_candidates = candidates[:top_k]
            if top_candidates and person_ids[top_candidates[0][0]] == target_person:
                top1_hits += 1
            if any(person_ids[item[0]] == target_person for item in top_candidates):
                topk_hits += 1

        metrics_summary.append(
            {
                "metric": metric,
                "top1": round(top1_hits / valid_queries, 4) if valid_queries else None,
                "top5": round(topk_hits / valid_queries, 4) if valid_queries else None,
                "topk": round(topk_hits / valid_queries, 4) if valid_queries else None,
                "queries": valid_queries,
            }
        )

    return metrics_summary


def inherit_person_names(records, labels, label_to_person_id, old_name_map):
    cluster_candidates = []
    for label in sorted(label for label in set(labels.tolist()) if int(label) != -1):
        old_counts = Counter(
            int(record.get("person_id", -1))
            for record, cluster_label in zip(records, labels)
            if int(cluster_label) == int(label) and int(record.get("person_id", -1)) != -1
        )
        for old_person_id, overlap in old_counts.items():
            name = str(old_name_map.get(old_person_id, "") or "").strip()
            if not name:
                continue
            cluster_candidates.append(
                {
                    "new_person_id": int(label_to_person_id[label]),
                    "old_person_id": int(old_person_id),
                    "name": name,
                    "overlap": int(overlap),
                }
            )

    cluster_candidates.sort(
        key=lambda item: (
            -item["overlap"],
            item["new_person_id"],
            item["old_person_id"],
        )
    )

    assigned_people = {}
    used_names = set()
    assigned_new_persons = set()
    for item in cluster_candidates:
        if item["new_person_id"] in assigned_new_persons:
            continue
        if item["name"] in used_names:
            continue
        assigned_people[item["new_person_id"]] = item["name"]
        used_names.add(item["name"])
        assigned_new_persons.add(item["new_person_id"])
    return assigned_people


def build_full_cluster_result(records, labels, old_name_map):
    labels = np.asarray(labels, dtype=int)
    active_labels = sorted(label for label in set(labels.tolist()) if int(label) != -1)
    label_to_person_id = {
        int(label): index + 1 for index, label in enumerate(active_labels)
    }

    updates = []
    reassigned_faces = 0
    for record, label in zip(records, labels):
        new_person_id = label_to_person_id.get(int(label), -1)
        updates.append((int(new_person_id), int(record["id"])))
        if int(record.get("person_id", -1)) != int(new_person_id):
            reassigned_faces += 1

    people_name_map = inherit_person_names(
        records,
        labels,
        label_to_person_id,
        old_name_map,
    )
    summary = cluster_summary(labels)
    return {
        "updates": updates,
        "people_name_map": people_name_map,
        "reassigned_faces": reassigned_faces,
        "summary": summary,
    }


def perform_clustering(
    db_manager,
    algorithm="dbscan",
    metric="cosine",
    eps=0.4,
    min_samples=2,
    mode="incremental",
    snapshot_config=None,
):
    algorithm, metric = _normalize_options(algorithm, metric)
    mode = normalize_cluster_mode(mode)

    if mode == "full":
        records = db_manager.get_all_faces_with_embeddings()
    else:
        records = db_manager.get_unassigned_faces_with_embeddings()

    if not records:
        return {
            "status": "success",
            "message": "No face embeddings need clustering.",
            "clusters_count": 0,
            "total_faces": 0,
            "reassigned_faces": 0,
            "mode": mode,
            "snapshot_created": False,
            "config": {
                "algorithm": algorithm,
                "metric": metric,
                "eps": float(eps),
                "min_samples": int(min_samples),
            },
        }

    ids = [face["id"] for face in records]
    embeddings = [face["embedding"] for face in records]
    labels = cluster_embeddings(
        embeddings,
        algorithm=algorithm,
        metric=metric,
        eps=eps,
        min_samples=min_samples,
    )

    snapshot_created = False
    reassigned_faces = 0
    if mode == "full":
        old_name_map = db_manager.get_person_name_map()
        snapshot_created = bool(
            db_manager.save_cluster_snapshot(cluster_config=snapshot_config or {})
        )
        full_result = build_full_cluster_result(records, labels, old_name_map)
        db_manager.replace_all_person_ids(
            full_result["updates"],
            people_name_map=full_result["people_name_map"],
        )
        reassigned_faces = full_result["reassigned_faces"]
        summary = full_result["summary"]
    else:
        max_pid = db_manager.get_max_person_id()
        updates = []
        for face_id, label in zip(ids, labels):
            if int(label) == -1:
                continue
            new_person_id = max_pid + 1 + int(label)
            updates.append((new_person_id, face_id))

        if updates:
            db_manager.update_person_ids(updates)
        reassigned_faces = len(updates)
        summary = cluster_summary(labels)

    return {
        "status": "success",
        "clusters_count": summary["clusters_count"],
        "total_faces": len(records),
        "noise_count": summary["noise_count"],
        "noise_ratio": summary["noise_ratio"],
        "reassigned_faces": reassigned_faces,
        "mode": mode,
        "snapshot_created": snapshot_created,
        "message": f"Clustering finished with {summary['clusters_count']} clusters.",
        "config": {
            "algorithm": algorithm,
            "metric": metric,
            "eps": float(eps),
            "min_samples": int(min_samples),
        },
    }


def evaluate_clustering(
    db_manager,
    algorithms=None,
    metrics=None,
    eps=0.4,
    min_samples=2,
):
    records = db_manager.get_all_faces_with_embeddings()
    if len(records) < 2:
        return {
            "status": "success",
            "results": [],
            "message": "Not enough face embeddings for evaluation.",
        }

    embeddings = [record["embedding"] for record in records]
    person_ids = [record.get("person_id", -1) for record in records]
    results = evaluate_embedding_clusters(
        embeddings,
        person_ids,
        algorithms=algorithms,
        metrics=metrics,
        eps_values=[eps],
        min_samples_values=[min_samples],
    )
    return {"status": "success", "results": results}


def compare_image_search_metrics(db_manager, top_k=5):
    records = db_manager.get_labeled_faces_with_embeddings()
    if len(records) < 3:
        return {
            "status": "success",
            "results": [],
            "message": "Not enough labeled data for image search evaluation.",
        }

    embeddings = [record["embedding"] for record in records]
    person_ids = [record["person_id"] for record in records]
    return {
        "status": "success",
        "results": evaluate_embedding_retrieval(
            embeddings,
            person_ids,
            metrics=list(SUPPORTED_METRICS),
            top_k=top_k,
        ),
    }
