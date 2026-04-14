import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoProcessor


class AIProcessor:
    def __init__(self):
        print("Initializing AI models (RTX 3060)...")

        self.face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.vlm_model_id = "microsoft/Florence-2-base"
        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_model_id, trust_remote_code=True
        )
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.vlm_model_id,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to("cuda")

        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        print("All models loaded.")

    def _normalize(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_face_embedding(self, img_bgr):
        faces = self.face_app.get(img_bgr)
        if not faces:
            return None, None

        max_face = max(
            faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
        return self._normalize(max_face.embedding), max_face.bbox

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
            use_cache=False,
        )

        text = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.vlm_processor.post_process_generation(
            text, task=prompt, image_size=(pil_image.width, pil_image.height)
        )
        return str(parsed_answer.get(prompt, ""))

    def compute_text_similarity(self, query, target_text):
        if not target_text or not isinstance(target_text, str):
            return 0.0

        embedding_1 = self.text_encoder.encode(query, convert_to_tensor=True)
        embedding_2 = self.text_encoder.encode(target_text, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

    def rank_texts_by_similarity(self, query, texts):
        valid_items = [
            (idx, text)
            for idx, text in enumerate(texts)
            if text and isinstance(text, str) and text.strip()
        ]
        if not valid_items:
            return []

        text_indices = [idx for idx, _ in valid_items]
        text_values = [text for _, text in valid_items]
        query_embedding = self.text_encoder.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        text_embeddings = self.text_encoder.encode(
            text_values, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(query_embedding, text_embeddings)[0]

        return [
            (text_indices[position], float(score))
            for position, score in enumerate(scores)
        ]

    def analyze_uploaded_image(self, img_array):
        faces = self.face_app.get(img_array)
        if not faces:
            return None

        target_face = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )[0]
        return self._normalize(target_face.embedding)

    def process_frame(self, frame_bgr):
        faces = self.face_app.get(frame_bgr)
        if not faces:
            return []

        from PIL import Image

        pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        description = self.generate_description(pil_frame)

        results = []
        height, width = frame_bgr.shape[:2]
        for face in faces:
            bbox = face.bbox.astype(int)
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(width, bbox[2])
            y2 = min(height, bbox[3])

            face_img = frame_bgr[y1:y2, x1:x2]
            if face_img.shape[0] < 40 or face_img.shape[1] < 40:
                continue

            results.append(
                {
                    "embedding": self._normalize(face.embedding),
                    "face_img": face_img,
                    "full_frame": frame_bgr,
                    "description": description,
                }
            )

        return results
