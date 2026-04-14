from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import shutil
import os
import uuid
from core.collector import VideoCollector
from core.analyzer import AIProcessor
from core.database import DatabaseManager
from core.clustering import perform_clustering

app = FastAPI(title="FaceRetriever 2026")

# 初始化组件
db = DatabaseManager()
# 延迟加载 AI 模块以免启动过慢，或者在启动时加载
ai_engine = AIProcessor()
collector = VideoCollector()

# 挂载静态文件
os.makedirs("storage/faces", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/faces", StaticFiles(directory="storage/faces"), name="faces")


@app.get("/")
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# 1. 修改后台处理任务：保存两张图
def process_video_task(video_info):
    cap = cv2.VideoCapture(video_info["path"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # 现在 results 是一个列表，包含了这一帧里的所有人脸
            results = ai_engine.process_frame(frame)

            if results and len(results) > 0:
                # 1. 完整原画面只需要保存一次 (取第一张脸里携带的 full_frame 即可)
                full_filename = f"{video_info['id']}_{frame_id}_full.jpg"
                full_frame = results[0]["full_frame"]

                # 限制全图最大宽度为 1280，节省硬盘空间
                if full_frame.shape[1] > 1280:

                    full_frame = cv2.resize(
                        full_frame,
                        (1280, int(1280 * full_frame.shape[0] / full_frame.shape[1])),
                    )
                cv2.imwrite(os.path.join("storage/faces", full_filename), full_frame)

                # 2. 遍历这一帧里的每一个人脸，单独存库
                for i, face_data in enumerate(results):
                    # 文件名加上序号 i，防止同帧的多张脸互相覆盖
                    face_filename = f"{video_info['id']}_{frame_id}_face_{i}.jpg"

                    # 保存人脸大头照
                    cv2.imwrite(
                        os.path.join("storage/faces", face_filename),
                        face_data["face_img"],
                    )

                    # 写入数据库 (多人共享同一个 timestamp, description 和 full_image_path)
                    db.add_face(
                        video_id=video_info["id"],
                        timestamp=count / fps,
                        image_path=f"/faces/{face_filename}",
                        full_image_path=f"/faces/{full_filename}",
                        source_url=video_info.get("url", ""),
                        embedding=face_data["embedding"],
                        description=face_data["description"],
                    )
        count += 1
        frame_id += 1
    cap.release()


@app.post("/api/collect")
async def collect_video(url: str, background_tasks: BackgroundTasks):
    try:
        # 1. 下载
        info = collector.download(url)
        # 2. 后台分析
        background_tasks.add_task(process_video_task, info)
        return {
            "status": "success",
            "msg": "Video downloaded, processing in background",
            "video": info["title"],
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# 2. 修复文本搜索字段映射
@app.get("/api/search/text")
async def search_by_text(q: str):
    all_records = db.get_all_faces()
    if not all_records:
        return {"status": "success", "results": []}
    search_results = []
    for record in all_records:
        desc = record.get("description", "")
        similarity = ai_engine.compute_text_similarity(q, desc)
        if similarity > 0.15:
            search_results.append(
                {
                    "id": record.get("id"),
                    "image": record.get("image_path"),
                    "full_image": record.get("full_image_path"),  # 全图
                    "source_url": record.get("source_url"),  # 链接
                    "video_id": record.get("video_id"),  # 修复
                    "timestamp": record.get("timestamp"),  # 修复
                    "desc": desc,
                    "score": round(float(similarity), 4),
                }
            )
    search_results.sort(key=lambda x: x["score"], reverse=True)
    return {"status": "success", "results": search_results[:20]}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "无法识别图片"}

        # 1. 获取归一化的特征向量 (核心修改)
        target_embedding, _ = ai_engine.get_face_embedding(img)

        if target_embedding is None:
            return {"status": "error", "message": "未检测到人脸"}

        # 2. 从数据库获取所有向量进行比对
        # 假设 db.get_all_embeddings() 返回列表: [{'id':1, 'embedding': [0.1, ...], 'desc': '...'}]
        # 即使你用 db.search_by_image_vector，也要确保数据库里的向量也是归一化过的

        all_faces = db.get_all_faces_with_embeddings()
        matches = []
        for face in all_faces:
            db_emb = np.array(face["embedding"], dtype=np.float32)
            db_norm = np.linalg.norm(db_emb)
            if db_norm > 0:
                db_emb = db_emb / db_norm
            score = np.dot(target_embedding, db_emb)
            if score > 0.4:
                matches.append(
                    {
                        "id": face["id"],
                        "video_id": face["video_id"],  # 修复
                        "timestamp": face["timestamp"],  # 修复
                        "image": face["image_path"],
                        "full_image": face["full_image_path"],  # 全图
                        "source_url": face["source_url"],  # 链接
                        "desc": face["description"],
                        "score": float(score),
                    }
                )
        matches.sort(key=lambda x: x["score"], reverse=True)
        return {"status": "success", "results": matches[:10]}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/api/cluster/run")
async def run_clustering():
    """触发后台聚类任务"""
    try:
        result = perform_clustering(db)
        return result
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/people")
async def get_people():
    """获取人物列表（相册封面）"""
    people = db.get_clustered_people()
    return {"status": "success", "people": people}


@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    """获取某个人物的时间轴详情"""
    timeline = db.get_person_timeline(person_id)
    return {"status": "success", "timeline": timeline}


# ==========================================
# 人物管理与聚类相关接口
# ==========================================


@app.get("/api/people")
async def get_people():
    """获取人物列表（相册封面）和未归类的人脸"""
    people = db.get_clustered_people()
    unassigned = db.get_unassigned_faces()  # 获取未归类数据
    return {"status": "success", "people": people, "unassigned": unassigned}


@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    """获取某个人物的时间轴详情"""
    timeline = db.get_person_timeline(person_id)
    return {"status": "success", "timeline": timeline}


@app.post("/api/person/rename")
async def rename_person(data: dict):
    """给聚类出的人物命名"""
    person_id = data.get("person_id")
    new_name = data.get("name")
    if person_id is None or not new_name:
        return {"status": "error", "message": "参数缺失"}
    db.rename_person(person_id, new_name)
    return {"status": "success"}


@app.post("/api/face/reassign")
async def reassign_face(data: dict):
    """手动拖拽改变人脸归属"""
    face_id = data.get("face_id")
    target_person_id = data.get("target_person_id")
    if face_id is None or target_person_id is None:
        return {"status": "error", "message": "参数缺失"}
    db.reassign_face(face_id, target_person_id)
    return {"status": "success"}


# 5. 新增接口：合并人物
@app.post("/api/person/merge")
async def merge_person(data: dict):
    """合并两个已聚类的人物"""
    source_id = data.get("source_person_id")
    target_id = data.get("target_person_id")
    if source_id is None or target_id is None:
        return {"status": "error", "message": "参数缺失"}

    try:
        db.merge_persons(source_id, target_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
