import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from sentence_transformers import SentenceTransformer, util


class AIProcessor:
    def __init__(self):
        print("正在初始化 AI 模型 (RTX 3060)...")

        # 1. 初始化人脸识别 (InsightFace)
        # 统一变量名为 self.face_app
        self.face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. 初始化语义理解模型 (Florence-2-base)
        self.vlm_model_id = "microsoft/Florence-2-base"
        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_model_id, trust_remote_code=True
        )

        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.vlm_model_id,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",  # 保持 eager 模式避免报错
        ).to("cuda")

        # 3. 初始化语义搜索模型
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        print("所有模型加载完毕。")

    # 辅助函数：归一化向量
    def _normalize(self, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_face_embedding(self, img_bgr):
        """返回最大人脸的特征向量 (已归一化)"""
        faces = self.face_app.get(img_bgr)
        if not faces:
            return None, None

        # 找最大的人脸
        max_face = max(
            faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        # 【关键修改】立即归一化特征向量
        norm_embedding = self._normalize(max_face.embedding)

        return norm_embedding, max_face.bbox

    def generate_description(self, pil_image):
        prompt = "<DETAILED_CAPTION>"

        inputs = self.vlm_processor(
            text=prompt, images=pil_image, return_tensors="pt"
        ).to("cuda", torch.float16)

        generated_ids = self.vlm_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            use_cache=False,  # 禁用缓存以匹配 eager 模式
        )

        text = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.vlm_processor.post_process_generation(
            text, task=prompt, image_size=(pil_image.width, pil_image.height)
        )

        # 确保返回的是字符串
        return str(parsed_answer.get(prompt, ""))

    def compute_text_similarity(self, query, target_text):
        """计算文本相似度"""
        if not target_text or not isinstance(target_text, str):
            return 0.0

        # 编码
        embedding_1 = self.text_encoder.encode(query, convert_to_tensor=True)
        embedding_2 = self.text_encoder.encode(target_text, convert_to_tensor=True)

        # 计算余弦相似度
        score = util.pytorch_cos_sim(embedding_1, embedding_2)
        final_score = score.item()

        # 调试：在控制台打印分数，方便你观察为什么 man 搜不到
        # 如果分数低于 0.25，你需要降低 app.py 里的阈值
        # print(f"Query: [{query}] vs Target: [{target_text[:20]}...] -> Score: {final_score:.4f}")

        return final_score

    def analyze_uploaded_image(self, img_array):
        """处理上传图片，返回归一化特征"""
        faces = self.face_app.get(img_array)
        if not faces:
            return None

        target_face = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )[0]

        # 【关键修改】立即归一化
        return self._normalize(target_face.embedding)

    def process_frame(self, frame_bgr):
        # 1. 使用 InsightFace 检测画面中的所有人脸
        faces = self.face_app.get(frame_bgr)

        # 如果没检测到人脸，直接返回空列表
        if not faces:
            return []

        # 2. 生成完整画面的语义描述 (一帧画面只需生成一次，节省算力)
        import cv2
        from PIL import Image

        pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        description = self.generate_description(pil_frame)

        results = []
        height, width = frame_bgr.shape[:2]

        # 3. 遍历检测到的每一张脸
        for face in faces:
            embedding = face.embedding
            bbox = face.bbox.astype(int)

            # 安全裁切人脸边界，防止坐标越界导致程序崩溃
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(width, bbox[2])
            y2 = min(height, bbox[3])

            face_img = frame_bgr[y1:y2, x1:x2]

            # 过滤掉太小的脸（比如背景里的路人，避免杂讯干扰聚类）
            if face_img.shape[0] < 40 or face_img.shape[1] < 40:
                continue

            results.append(
                {
                    "embedding": embedding,
                    "face_img": face_img,
                    "full_frame": frame_bgr,
                    "description": description,
                }
            )

        return results
