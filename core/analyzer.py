import importlib.util
import logging
import re
import unicodedata
from threading import Lock

import cv2
import numpy as np

from core.config import app_config


logger = logging.getLogger("core.analyzer")


class AIProcessor:
    def __init__(self):
        self.device = self._detect_device()
        self.face_app = None
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_model_id = None
        self.text_encoder = None
        self.text_encoder_model_id = None
        self.text_encoder_mode = None
        self.text_encoder_prompt_name = None
        self.asr_model = None
        self.asr_backend = None
        self.asr_model_size = None
        self._lock = Lock()

    def _detect_device(self):
        if not importlib.util.find_spec("torch"):
            return "cpu"

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _normalize(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _ensure_face_app(self):
        if self.face_app is not None:
            return self.face_app

        with self._lock:
            if self.face_app is not None:
                return self.face_app

            from insightface.app import FaceAnalysis

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self.face_app = FaceAnalysis(providers=providers)
            self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        return self.face_app

    def _ensure_vlm(self):
        desired_model_id = app_config.vlm_model_id
        if (
            self.vlm_model is not None
            and self.vlm_processor is not None
            and self.vlm_model_id == desired_model_id
        ):
            return self.vlm_processor, self.vlm_model

        with self._lock:
            if (
                self.vlm_model is not None
                and self.vlm_processor is not None
                and self.vlm_model_id == desired_model_id
            ):
                return self.vlm_processor, self.vlm_model
            if self.vlm_model_id != desired_model_id:
                self._reset_vlm()
            self._prepare_for_vlm()

            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.vlm_processor = AutoProcessor.from_pretrained(
                desired_model_id,
                trust_remote_code=True,
            )
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                desired_model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)
            self.vlm_model_id = desired_model_id
        return self.vlm_processor, self.vlm_model

    def _ensure_text_encoder(self):
        desired_model_id = app_config.semantic_model_id
        desired_prompt_name = app_config.semantic_model_prompt_name or None
        desired_mode = app_config.semantic_model_mode
        if (
            self.text_encoder is not None
            and self.text_encoder_model_id == desired_model_id
            and self.text_encoder_mode == desired_mode
            and self.text_encoder_prompt_name == desired_prompt_name
        ):
            return self.text_encoder

        with self._lock:
            if (
                self.text_encoder is not None
                and self.text_encoder_model_id == desired_model_id
                and self.text_encoder_mode == desired_mode
                and self.text_encoder_prompt_name == desired_prompt_name
            ):
                return self.text_encoder

            from sentence_transformers import SentenceTransformer

            if desired_mode == "qwen3_4b_int4_experimental":
                raise RuntimeError(
                    "Qwen3-Embedding-4B int4 实验模式已预留，但当前运行环境尚未启用。"
                )

            self._reset_text_encoder()
            self._reset_vlm()
            self.text_encoder = SentenceTransformer(
                desired_model_id,
                device=self.device,
            )
            self.text_encoder_model_id = desired_model_id
            self.text_encoder_mode = desired_mode
            self.text_encoder_prompt_name = desired_prompt_name
        return self.text_encoder

    def _ensure_asr_model(self):
        if self.asr_model is not None:
            return self.asr_model

        if not app_config.transcription_enabled:
            return None

        with self._lock:
            if self.asr_model is not None:
                return self.asr_model
            return self._load_asr_with_fallback(
                self._build_asr_model_candidates(app_config.transcription_model_size)
            )

    def _clear_cuda_cache(self):
        if not importlib.util.find_spec("torch"):
            return

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_asr_model(self):
        self.asr_model = None
        self.asr_backend = None
        self.asr_model_size = None
        self._clear_cuda_cache()

    def _reset_vlm(self):
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_model_id = None
        self._clear_cuda_cache()

    def _reset_text_encoder(self):
        self.text_encoder = None
        self.text_encoder_model_id = None
        self.text_encoder_mode = None
        self.text_encoder_prompt_name = None
        self._clear_cuda_cache()

    def _prepare_for_vlm(self):
        if self.asr_model is not None:
            self._reset_asr_model()
        if app_config.release_text_encoder_before_vlm:
            self._reset_text_encoder()

    def _query_encode_kwargs(self):
        prompt_name = app_config.semantic_model_prompt_name or ""
        if not prompt_name:
            return {}
        return {"prompt_name": prompt_name}

    def _build_caption_prompt(self):
        if app_config.caption_style == "retrieval_keywords":
            return "<MORE_DETAILED_CAPTION>"
        return "<DETAILED_CAPTION>"

    def _append_unique(self, items, value):
        cleaned = self.clean_text(value)
        if cleaned and cleaned not in items:
            items.append(cleaned)

    def _trim_text(self, text, limit):
        cleaned = self.clean_text(text)
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip(" ,.;:，；。！？、")

    def _build_asr_model_candidates(self, preferred_size):
        preferred = (preferred_size or "tiny").lower()
        fallback_map = {
            "large-v3": ["large-v3", "medium", "small", "tiny"],
            "large-v2": ["large-v2", "medium", "small", "tiny"],
            "large": ["large", "medium", "small", "tiny"],
            "medium": ["medium", "small", "tiny"],
            "small": ["small", "tiny"],
            "base": ["base", "tiny"],
            "tiny": ["tiny"],
        }
        candidates = fallback_map.get(preferred, [preferred, "small", "tiny"])
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _load_asr_with_fallback(self, candidates):
        candidates = [candidate for candidate in candidates if candidate]

        if (
            app_config.transcription_backend == "faster_whisper"
            and importlib.util.find_spec("faster_whisper")
        ):
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "int8"
            for model_size in candidates:
                try:
                    model = WhisperModel(
                        model_size,
                        device=self.device,
                        compute_type=compute_type,
                    )
                    self.asr_model = model
                    self.asr_backend = "faster_whisper"
                    self.asr_model_size = model_size
                    logger.info("ASR 模型加载完成：backend=faster_whisper model=%s", model_size)
                    return model
                except Exception as exc:
                    logger.warning(
                        "ASR 候选模型加载失败：backend=faster_whisper model=%s error=%s",
                        model_size,
                        exc,
                    )
                    self._reset_asr_model()

        if importlib.util.find_spec("whisper"):
            import whisper

            for model_size in candidates:
                try:
                    model = whisper.load_model(model_size, device=self.device)
                    self.asr_model = model
                    self.asr_backend = "whisper"
                    self.asr_model_size = model_size
                    logger.info("ASR 模型加载完成：backend=whisper model=%s", model_size)
                    return model
                except Exception as exc:
                    logger.warning(
                        "ASR 候选模型加载失败：backend=whisper model=%s error=%s",
                        model_size,
                        exc,
                    )
                    self._reset_asr_model()

        logger.warning("ASR 模型不可用：未找到可加载的候选模型。")
        return None

    def clean_text(self, text):
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        normalized = normalized.replace("\r", " ").replace("\n", " ")
        return " ".join(normalized.split()).strip()

    def _normalize_search_query(self, query):
        cleaned = self.clean_text(query)
        cleaned = cleaned.translate(
            str.maketrans(
                {
                    "，": ",",
                    "。": ".",
                    "；": ";",
                    "：": ":",
                    "！": "!",
                    "？": "?",
                    "（": "(",
                    "）": ")",
                    "【": "[",
                    "】": "]",
                    "“": '"',
                    "”": '"',
                    "‘": "'",
                    "’": "'",
                    "、": " ",
                }
            )
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"([,.;:!?]){2,}", r"\1", cleaned)
        return cleaned.strip()

    def _replace_caption_terms(self, text):
        replacements = [
            (r"\bin this image we can see\b", ""),
            (r"\bin this image\b", ""),
            (r"\bthis image shows\b", ""),
            (r"\bthe image shows\b", ""),
            (r"\ba photo of\b", ""),
            (r"\ba picture of\b", ""),
            (r"\ba close[- ]up of\b", "特写"),
            (r"\ba woman\b", "女性"),
            (r"\ba man\b", "男性"),
            (r"\ba girl\b", "女孩"),
            (r"\ba boy\b", "男孩"),
            (r"\ba person\b", "人物"),
            (r"\btwo people\b", "两人"),
            (r"\bthree people\b", "三人"),
            (r"\bgroup of people\b", "多人"),
            (r"\bpeople\b", "多人"),
            (r"\bperson\b", "人物"),
            (r"\bwoman\b", "女性"),
            (r"\bman\b", "男性"),
            (r"\bgirl\b", "女孩"),
            (r"\bboy\b", "男孩"),
            (r"\byoung\b", "年轻"),
            (r"\badult\b", "成年"),
            (r"\bwearing\b", "穿着"),
            (r"\band\b", "；"),
            (r"\bwith\b", "，"),
            (r"\bon\b", "在"),
            (r"\bin\b", "在"),
            (r"\bat\b", "在"),
            (r"\bholding\b", "手持"),
            (r"\busing\b", "使用"),
            (r"\bstanding\b", "站立"),
            (r"\bsitting\b", "坐着"),
            (r"\bwalking\b", "行走"),
            (r"\brunning\b", "跑动"),
            (r"\bspeaking\b", "说话"),
            (r"\btalking\b", "交谈"),
            (r"\bsinging\b", "唱歌"),
            (r"\bdancing\b", "跳舞"),
            (r"\bsmiling\b", "微笑"),
            (r"\blaughing\b", "笑"),
            (r"\bstage\b", "舞台"),
            (r"\bstreet\b", "街道"),
            (r"\broad\b", "道路"),
            (r"\broom\b", "房间"),
            (r"\bindoor(?:s)?\b", "室内"),
            (r"\boutdoor(?:s)?\b", "户外"),
            (r"\bbackground\b", "背景"),
            (r"\baudience\b", "观众"),
            (r"\bmicrophone\b", "麦克风"),
            (r"\bphone\b", "手机"),
            (r"\bcamera\b", "镜头"),
            (r"\bsign\b", "标识"),
            (r"\bvisible text\b", "画面文字"),
            (r"\btext\b", "文字"),
            (r"\bposter\b", "海报"),
            (r"\bglasses\b", "眼镜"),
            (r"\bhat\b", "帽子"),
            (r"\blong hair\b", "长发"),
            (r"\bshort hair\b", "短发"),
            (r"\bjacket\b", "外套"),
            (r"\bshirt\b", "上衣"),
            (r"\bdress\b", "连衣裙"),
            (r"\bblack\b", "黑色"),
            (r"\bwhite\b", "白色"),
            (r"\bred\b", "红色"),
            (r"\bblue\b", "蓝色"),
            (r"\bgreen\b", "绿色"),
            (r"\ba\b", ""),
            (r"\ban\b", ""),
            (r"\bthe\b", ""),
        ]
        translated = text
        for pattern, replacement in replacements:
            translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)
        translated = re.sub(r"\s+", " ", translated)
        translated = re.sub(r"\s*[,/|]+\s*", "；", translated)
        translated = re.sub(r"\s*[.]+\s*", "；", translated)
        return translated.strip(" ；,.;:-")

    def _split_semantic_segments(self, text):
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        raw_segments = re.split(r"[；;，,。.!?！？\n]+", cleaned)
        segments = []
        for segment in raw_segments:
            normalized = self.clean_text(segment.strip(" ,，;；。.!?！？"))
            if normalized and normalized not in segments:
                segments.append(normalized)
        return segments

    def _normalize_visual_caption(self, text):
        cleaned = self.clean_text(text)
        if app_config.caption_language == "zh":
            cleaned = self._replace_caption_terms(cleaned)
        if not cleaned:
            return ""
        segments = []
        for segment in self._split_semantic_segments(cleaned):
            normalized = re.sub(r"\s+", " ", segment).strip(" ,，;；。.!?！？")
            if normalized:
                segments.append(normalized)
        return "；".join(segments[:6])

    def _classify_visual_segments(self, visual_text):
        person_segments = []
        action_segments = []
        ocr_segments = []
        person_keywords = (
            "人",
            "男性",
            "女性",
            "男孩",
            "女孩",
            "多人",
            "年轻",
            "成年",
            "长发",
            "短发",
            "眼镜",
            "帽子",
            "外套",
            "上衣",
            "连衣裙",
            "黑色",
            "白色",
            "红色",
            "蓝色",
            "绿色",
            "穿着",
            "woman",
            "man",
            "girl",
            "boy",
            "person",
            "people",
            "glasses",
            "hat",
            "wearing",
        )
        ocr_keywords = ("文字", "字幕", "标识", "海报", "标题", "logo", "text", "subtitle", "sign", "poster", "title")
        for segment in self._split_semantic_segments(visual_text):
            lowered = segment.lower()
            if any(keyword in lowered for keyword in ocr_keywords):
                self._append_unique(ocr_segments, segment)
                residual_segment = segment
                for keyword in ocr_keywords:
                    residual_segment = residual_segment.replace(keyword, "")
                segment = self.clean_text(residual_segment.strip("，,；; "))
                if not segment:
                    continue
            if any(keyword in segment or keyword in lowered for keyword in person_keywords):
                self._append_unique(person_segments, segment)
            else:
                self._append_unique(action_segments, segment)
        if not person_segments and not action_segments and visual_text:
            action_segments.append(visual_text)
        return person_segments, action_segments, ocr_segments

    def _build_semantic_document(
        self,
        visual_text="",
        subtitle_text="",
        asr_text="",
        post_text="",
    ):
        visual_text = self._normalize_visual_caption(visual_text)
        subtitle_text = self.clean_text(subtitle_text)
        asr_text = self.clean_text(asr_text)
        post_text = self.clean_text(post_text)
        sections = []
        sources = []
        person_segments, action_segments, ocr_segments = self._classify_visual_segments(
            visual_text
        )
        if person_segments:
            sections.append(f"人物特征：{self._trim_text('；'.join(person_segments), 120)}")
            sources.append("visual")
        if action_segments:
            sections.append(f"动作场景：{self._trim_text('；'.join(action_segments), 140)}")
            if "visual" not in sources:
                sources.append("visual")
        if app_config.caption_include_ocr_hint and ocr_segments:
            sections.append(f"画面文字：{self._trim_text('；'.join(ocr_segments), 100)}")
            if "visual" not in sources:
                sources.append("visual")

        speech_text = subtitle_text or asr_text
        if speech_text:
            sections.append(f"语音文本：{self._trim_text(speech_text, 180)}")
            sources.append("subtitle" if subtitle_text else "asr")
        if post_text and post_text != speech_text:
            sections.append(f"正文线索：{self._trim_text(post_text, 180)}")
            sources.append("post")
        return "\n".join(sections).strip(), sources

    def _build_embedding_document(self, text):
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        if "[Visual]" in cleaned or "[Speech]" in cleaned or "[Post]" in cleaned:
            cleaned = (
                cleaned.replace("[Visual]", "人物特征:")
                .replace("[Speech]", "语音文本:")
                .replace("[Post]", "正文线索:")
            )
        if app_config.semantic_corpus_style == "structured_zh":
            return self._trim_text(cleaned, 360)
        return cleaned

    def _build_asr_prompt_options(self):
        options = {}
        initial_prompt = self.clean_text(app_config.transcription_initial_prompt)
        hotwords = [self.clean_text(item) for item in app_config.transcription_hotwords]
        hotwords = [item for item in hotwords if item]
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
        if hotwords:
            options["hotwords"] = "，".join(hotwords)
        return options

    def _resolve_face_quality_config(self, face_quality_config=None):
        resolved = app_config.face_quality_config()
        for key, value in (face_quality_config or {}).items():
            if key in resolved and value is not None:
                resolved[key] = value
        resolved["enabled"] = bool(resolved.get("enabled", True))
        resolved["min_face_size"] = int(resolved.get("min_face_size", 56))
        resolved["min_face_ratio"] = float(resolved.get("min_face_ratio", 0.035))
        resolved["min_laplacian_var"] = float(resolved.get("min_laplacian_var", 80.0))
        resolved["max_pose_deviation"] = float(resolved.get("max_pose_deviation", 0.35))
        resolved["blur_eval_size"] = int(resolved.get("blur_eval_size", 96))
        return resolved

    def _extract_face_region(self, frame_bgr, face):
        height, width = frame_bgr.shape[:2]
        frame_min_side = max(int(min(width, height)), 1)
        bbox = np.asarray(face.bbox, dtype=int)
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[2]))
        y2 = min(height, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        face_img = frame_bgr[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        return {
            "bbox": np.array([x1, y1, x2, y2], dtype=int),
            "face_img": face_img,
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "min_side": int(min(x2 - x1, y2 - y1)),
            "frame_width": int(width),
            "frame_height": int(height),
            "frame_min_side": frame_min_side,
            "face_ratio": float(min(x2 - x1, y2 - y1) / frame_min_side),
            "area": int(max(x2 - x1, 0) * max(y2 - y1, 0)),
        }

    def _compute_laplacian_variance(self, face_img, blur_eval_size=96):
        if face_img is None or face_img.size == 0:
            return 0.0
        eval_size = max(int(blur_eval_size or 96), 1)
        if face_img.shape[0] != eval_size or face_img.shape[1] != eval_size:
            face_img = cv2.resize(
                face_img,
                (eval_size, eval_size),
                interpolation=cv2.INTER_AREA
                if face_img.shape[0] > eval_size or face_img.shape[1] > eval_size
                else cv2.INTER_LINEAR,
            )
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_pose_deviation(self, face, bbox):
        kps = getattr(face, "kps", None)
        if kps is None:
            return 1.0

        points = np.asarray(kps, dtype=np.float32)
        if points.shape[0] < 5:
            return 1.0

        left_eye, right_eye, nose, left_mouth, right_mouth = points[:5]
        bbox_width = max(float(bbox[2] - bbox[0]), 1.0)
        bbox_height = max(float(bbox[3] - bbox[1]), 1.0)

        eye_dx = float(right_eye[0] - left_eye[0])
        eye_dy = float(right_eye[1] - left_eye[1])
        eye_distance = max(float(np.hypot(eye_dx, eye_dy)), 1.0)
        mouth_width = max(float(np.hypot(*(right_mouth - left_mouth))), 1.0)

        eye_roll = min(abs(eye_dy) / eye_distance, 1.0)
        mid_eye = (left_eye + right_eye) / 2.0
        nose_offset = min(abs(float(nose[0] - mid_eye[0])) / bbox_width, 1.0)

        mouth_center = (left_mouth + right_mouth) / 2.0
        mouth_offset = min(abs(float(mouth_center[0] - mid_eye[0])) / bbox_width, 1.0)
        mouth_tilt = min(abs(float(right_mouth[1] - left_mouth[1])) / mouth_width, 1.0)
        nose_vertical = min(abs(float(nose[1] - mid_eye[1])) / bbox_height, 1.0)

        return float(
            np.mean(
                [
                    eye_roll,
                    nose_offset,
                    mouth_offset,
                    mouth_tilt,
                    nose_vertical * 0.5,
                ]
            )
        )

    def _evaluate_face_candidate(self, frame_bgr, face, face_quality_config=None):
        quality_config = self._resolve_face_quality_config(face_quality_config)
        region = self._extract_face_region(frame_bgr, face)
        if region is None:
            return {
                "accepted": False,
                "reason": "filtered_invalid_bbox",
                "embedding": getattr(face, "embedding", None),
                "metrics": {
                    "min_face_size": 0,
                    "laplacian_var": 0.0,
                    "pose_deviation": 1.0,
                },
            }

        laplacian_var = self._compute_laplacian_variance(
            region["face_img"],
            blur_eval_size=quality_config["blur_eval_size"],
        )
        pose_deviation = self._compute_pose_deviation(face, region["bbox"])
        metrics = {
            "min_face_size": region["min_side"],
            "face_ratio": region["face_ratio"],
            "frame_min_side": region["frame_min_side"],
            "blur_eval_size": quality_config["blur_eval_size"],
            "laplacian_var": laplacian_var,
            "pose_deviation": pose_deviation,
            "area": region["area"],
        }

        if quality_config["enabled"]:
            if region["min_side"] < quality_config["min_face_size"]:
                return {
                    **region,
                    "accepted": False,
                    "reason": "filtered_min_face_size",
                    "embedding": getattr(face, "embedding", None),
                    "metrics": metrics,
                }
            if region["face_ratio"] < quality_config["min_face_ratio"]:
                return {
                    **region,
                    "accepted": False,
                    "reason": "filtered_min_face_ratio",
                    "embedding": getattr(face, "embedding", None),
                    "metrics": metrics,
                }
            if laplacian_var < quality_config["min_laplacian_var"]:
                return {
                    **region,
                    "accepted": False,
                    "reason": "filtered_blur",
                    "embedding": getattr(face, "embedding", None),
                    "metrics": metrics,
                }
            if pose_deviation > quality_config["max_pose_deviation"]:
                return {
                    **region,
                    "accepted": False,
                    "reason": "filtered_pose",
                    "embedding": getattr(face, "embedding", None),
                    "metrics": metrics,
                }

        return {
            **region,
            "accepted": True,
            "reason": None,
            "embedding": getattr(face, "embedding", None),
            "metrics": metrics,
        }

    def _summarize_face_failure(self, candidates):
        if not candidates:
            return "no_face_detected"

        reasons = [candidate.get("reason") for candidate in candidates if candidate.get("reason")]
        if not reasons:
            return "embedding_failed"
        unique_reasons = list(dict.fromkeys(reasons))
        if len(unique_reasons) == 1:
            return unique_reasons[0]
        return "all_faces_filtered"

    def filter_face_candidates(self, frame_bgr, faces=None, face_quality_config=None):
        detected_faces = faces if faces is not None else self._ensure_face_app().get(frame_bgr)
        if not detected_faces:
            return []
        return [
            self._evaluate_face_candidate(
                frame_bgr,
                face,
                face_quality_config=face_quality_config,
            )
            for face in detected_faces
        ]

    def _select_best_face_candidate(self, frame_bgr, faces=None, face_quality_config=None):
        candidates = self.filter_face_candidates(
            frame_bgr,
            faces=faces,
            face_quality_config=face_quality_config,
        )
        accepted = [candidate for candidate in candidates if candidate.get("accepted")]
        if not accepted:
            return None, self._summarize_face_failure(candidates)

        best = max(
            accepted,
            key=lambda candidate: (
                candidate["metrics"]["area"],
                candidate["metrics"]["laplacian_var"],
                -candidate["metrics"]["pose_deviation"],
            ),
        )
        return best, None

    def get_face_embedding_result(self, img_bgr, face_quality_config=None):
        best_face, failure_reason = self._select_best_face_candidate(
            img_bgr,
            face_quality_config=face_quality_config,
        )
        if best_face is None:
            return {
                "embedding": None,
                "bbox": None,
                "failure_reason": failure_reason,
                "metrics": None,
            }

        return {
            "embedding": self._normalize(best_face["embedding"]),
            "bbox": best_face["bbox"],
            "failure_reason": None,
            "metrics": best_face["metrics"],
        }

    def get_face_embedding(self, img_bgr, face_quality_config=None):
        result = self.get_face_embedding_result(
            img_bgr,
            face_quality_config=face_quality_config,
        )
        return result["embedding"], result["bbox"]

    def get_face_embedding_from_path(self, image_path, face_quality_config=None):
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
        return self.get_face_embedding(image, face_quality_config=face_quality_config)

    def get_face_embedding_result_from_path(self, image_path, face_quality_config=None):
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                "embedding": None,
                "bbox": None,
                "failure_reason": "image_load_failed",
                "metrics": None,
            }
        return self.get_face_embedding_result(
            image,
            face_quality_config=face_quality_config,
        )

    def generate_description(self, pil_image):
        try:
            processor, model = self._ensure_vlm()
            import torch

            prompt = self._build_caption_prompt()
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            inputs = processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            ).to(self.device, dtype=dtype)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )

            text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            parsed_answer = processor.post_process_generation(
                text,
                task=prompt,
                image_size=(pil_image.width, pil_image.height),
            )
            return self._normalize_visual_caption(parsed_answer.get(prompt, ""))
        except Exception:
            self._reset_vlm()
            return ""

    def compose_semantic_text(
        self,
        visual_text="",
        subtitle_text="",
        asr_text="",
        post_text="",
    ):
        return self._build_semantic_document(
            visual_text=visual_text,
            subtitle_text=subtitle_text,
            asr_text=asr_text,
            post_text=post_text,
        )

    def compute_text_similarity(self, query, target_text):
        if not target_text or not isinstance(target_text, str):
            return 0.0

        try:
            from sentence_transformers import util

            encoder = self._ensure_text_encoder()
            normalized_query = self._normalize_search_query(query)
            normalized_text = self._build_embedding_document(target_text)
            embedding_1 = encoder.encode(
                normalized_query,
                convert_to_tensor=True,
                **self._query_encode_kwargs(),
            )
            embedding_2 = encoder.encode(normalized_text, convert_to_tensor=True)
            return util.pytorch_cos_sim(embedding_1, embedding_2).item()
        except Exception:
            query_tokens = set(self._normalize_search_query(query).lower().split())
            text_tokens = set(self._build_embedding_document(target_text).lower().split())
            if not query_tokens or not text_tokens:
                return 0.0
            overlap = len(query_tokens & text_tokens)
            return overlap / max(len(query_tokens), len(text_tokens))

    def rank_texts_by_similarity(self, query, texts):
        valid_items = [
            (idx, self._build_embedding_document(text))
            for idx, text in enumerate(texts)
            if text and isinstance(text, str) and text.strip()
        ]
        if not valid_items:
            return []

        try:
            from sentence_transformers import util

            encoder = self._ensure_text_encoder()
            text_indices = [idx for idx, _ in valid_items]
            text_values = [text for _, text in valid_items]
            normalized_query = self._normalize_search_query(query)
            query_embedding = encoder.encode(
                normalized_query,
                convert_to_tensor=True,
                normalize_embeddings=True,
                **self._query_encode_kwargs(),
            )
            text_embeddings = encoder.encode(
                text_values,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            scores = util.cos_sim(query_embedding, text_embeddings)[0]
            return [
                (text_indices[position], float(score))
                for position, score in enumerate(scores)
            ]
        except Exception:
            return [
                (idx, self.compute_text_similarity(query, text))
                for idx, text in valid_items
            ]

    def transcribe_video(self, video_path):
        try:
            model = self._ensure_asr_model()
            if model is None:
                return []
            asr_prompt_options = self._build_asr_prompt_options()

            def _transcribe_with_loaded_model():
                segments = []
                if self.asr_backend == "faster_whisper":
                    result, _ = model.transcribe(
                        video_path,
                        vad_filter=True,
                        beam_size=3,
                        **asr_prompt_options,
                    )
                    for segment in result:
                        text = self.clean_text(segment.text)
                        if not text:
                            continue
                        segments.append(
                            {
                                "start": float(segment.start),
                                "end": float(segment.end),
                                "text": text,
                            }
                        )
                    return segments

                if self.asr_backend == "whisper":
                    whisper_options = {"verbose": False}
                    if asr_prompt_options.get("initial_prompt"):
                        whisper_options["initial_prompt"] = asr_prompt_options["initial_prompt"]
                    result = model.transcribe(video_path, **whisper_options)
                    for segment in result.get("segments", []):
                        text = self.clean_text(segment.get("text", ""))
                        if not text:
                            continue
                        segments.append(
                            {
                                "start": float(segment.get("start", 0.0)),
                                "end": float(segment.get("end", 0.0)),
                                "text": text,
                            }
                        )
                return segments

            current_model_size = self.asr_model_size
            try:
                return _transcribe_with_loaded_model()
            except Exception:
                remaining_candidates = self._build_asr_model_candidates(current_model_size)[1:]
                self._reset_asr_model()
                with self._lock:
                    fallback_model = self._load_asr_with_fallback(remaining_candidates)
                if fallback_model is None:
                    return []

                model = fallback_model
                try:
                    return _transcribe_with_loaded_model()
                except Exception:
                    self._reset_asr_model()
                    return []
        finally:
            if self.asr_model is not None:
                self._reset_asr_model()

    def _process_media(
        self,
        frame_bgr,
        subtitle_text="",
        asr_text="",
        post_text="",
        face_quality_config=None,
    ):
        faces = self._ensure_face_app().get(frame_bgr)
        if not faces:
            return []

        try:
            from PIL import Image

            pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            visual_text = self.generate_description(pil_frame)

            semantic_text, semantic_source = self.compose_semantic_text(
                visual_text=visual_text,
                subtitle_text=subtitle_text,
                asr_text=asr_text,
                post_text=post_text,
            )

            results = []
            for candidate in self.filter_face_candidates(
                frame_bgr,
                faces=faces,
                face_quality_config=face_quality_config,
            ):
                if not candidate.get("accepted"):
                    continue

                results.append(
                    {
                        "embedding": self._normalize(candidate["embedding"]),
                        "face_img": candidate["face_img"],
                        "full_frame": frame_bgr,
                        "visual_text": visual_text,
                        "subtitle_text": self.clean_text(subtitle_text),
                        "asr_text": self.clean_text(asr_text),
                        "post_text": self.clean_text(post_text),
                        "semantic_text": semantic_text,
                        "semantic_source": semantic_source,
                        "face_metrics": candidate["metrics"],
                    }
                )

            return results
        finally:
            if app_config.release_vlm_after_task:
                self._reset_vlm()

    def process_frame(
        self,
        frame_bgr,
        subtitle_text="",
        asr_text="",
        post_text="",
        face_quality_config=None,
    ):
        return self._process_media(
            frame_bgr,
            subtitle_text=subtitle_text,
            asr_text=asr_text,
            post_text=post_text,
            face_quality_config=face_quality_config,
        )

    def process_image(self, image_bgr, post_text="", face_quality_config=None):
        return self._process_media(
            image_bgr,
            post_text=post_text,
            face_quality_config=face_quality_config,
        )
