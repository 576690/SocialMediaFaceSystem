import numpy as np
from sklearn.cluster import DBSCAN


def perform_clustering(db_manager):
    print("开始执行增量聚类分析...")

    # 1. 仅获取尚未归类的人脸 (person_id = -1)
    unassigned_faces = db_manager.get_unassigned_faces_with_embeddings()
    if not unassigned_faces:
        return {
            "status": "success",
            "message": "没有需要聚类的新数据",
            "clusters_count": 0,
            "total_faces": 0,
        }

    ids = [f["id"] for f in unassigned_faces]
    embeddings = [f["embedding"] for f in unassigned_faces]
    X = np.array(embeddings)

    # 2. 执行 DBSCAN
    clustering = DBSCAN(eps=0.4, min_samples=2, metric="cosine", n_jobs=-1).fit(X)
    labels = clustering.labels_

    # 3. 获取当前数据库中最大的 person_id，以避免覆盖已存在的人物
    max_pid = db_manager.get_max_person_id()

    updates = []
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    for face_id, label in zip(ids, labels):
        if label == -1:
            continue  # DBSCAN 认为它是杂讯，保持 -1 留给人工处理

        # 给新发现的人物分配全新的编号
        new_person_id = max_pid + 1 + int(label)
        updates.append((new_person_id, face_id))

    # 4. 更新数据库
    if updates:
        db_manager.update_person_ids(updates)

    return {
        "status": "success",
        "clusters_count": n_clusters,
        "total_faces": len(unassigned_faces),
        "message": f"增量聚类完成，为您新增了 {n_clusters} 个人物组",
    }
