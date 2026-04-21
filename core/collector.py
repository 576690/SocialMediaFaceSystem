import glob
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
import yt_dlp

from core.config import app_config
from core.weibo_adapter import WeiboUserCollector


class VideoCollector:
    def __init__(self, save_dir=None, weibo_adapter=None):
        self.save_dir = Path(save_dir or app_config.videos_dir)
        self.content_dir = Path(app_config.content_dir)
        self.weibo_adapter = weibo_adapter or WeiboUserCollector()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)

    def detect_platform(self, url):
        host = urlparse(url).netloc.lower()
        if "bilibili.com" in host or "b23.tv" in host:
            return "bilibili"
        if "weibo.com" in host or "weibo.cn" in host:
            return "weibo"
        if "youtube.com" in host or "youtu.be" in host:
            return "youtube"
        return "generic"

    def normalize_keywords(self, keywords):
        return self.weibo_adapter.normalize_keywords(keywords)

    def normalize_source_url(self, source_url, platform=None, source_type=None):
        platform = (platform or self.detect_platform(source_url)).lower()
        source_type = (source_type or "").lower()
        if platform == "weibo" and source_type == "weibo_user":
            return self.weibo_adapter.normalize_user_source(source_url)["canonical_url"]
        return source_url

    def _find_subtitle_file(self, media_path):
        base = os.path.splitext(media_path)[0]
        candidates = glob.glob(f"{base}*.srt")
        if candidates:
            return sorted(candidates)[0]
        return ""

    def _resolve_media_path(self, filename, info):
        if os.path.exists(filename):
            return filename

        ext = info.get("ext") or "mp4"
        merged_path = os.path.splitext(filename)[0] + f".{ext}"
        if os.path.exists(merged_path):
            return merged_path

        mp4_path = os.path.splitext(filename)[0] + ".mp4"
        if os.path.exists(mp4_path):
            return mp4_path
        return filename

    def _build_content_id(self, text):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return digest[:16]

    def _build_video_url(self, platform, entry):
        for key in ("webpage_url", "original_url", "url"):
            value = entry.get(key)
            if not value:
                continue
            if str(value).startswith("http"):
                return value

        entry_id = entry.get("id") or ""
        if platform == "bilibili" and entry_id.startswith("BV"):
            return f"https://www.bilibili.com/video/{entry_id}"
        return ""

    def _base_ydl_opts(self):
        cookie_path = app_config.storage_dir / "www.youtube.com_cookies.txt"
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(self.save_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": app_config.subtitle_languages,
            "subtitlesformat": app_config.subtitle_formats,
            "merge_output_format": "mp4",
        }
        if cookie_path.exists():
            ydl_opts["cookiefile"] = str(cookie_path)
        return ydl_opts

    def download(self, url):
        with yt_dlp.YoutubeDL(self._base_ydl_opts()) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = self._resolve_media_path(ydl.prepare_filename(info), info)

        subtitle_path = self._find_subtitle_file(filename)
        platform = self.detect_platform(url)
        external_id = info.get("id") or self._build_content_id(url)
        return {
            "platform": platform,
            "external_id": external_id,
            "id": external_id,
            "title": info.get("title") or external_id,
            "path": filename,
            "subtitle_path": subtitle_path,
            "url": info.get("webpage_url") or url,
            "metadata": info,
        }

    def fetch_source_entries(
        self,
        source_url,
        limit=None,
        platform=None,
        source_type=None,
        metadata=None,
    ):
        limit = limit or app_config.source_sync_limit
        platform = (platform or self.detect_platform(source_url)).lower()
        source_type = (source_type or "channel").lower()
        metadata = metadata or {}

        if platform == "weibo" and source_type == "weibo_user":
            keywords = self.normalize_keywords(metadata.get("keywords", []))
            return self.weibo_adapter.fetch_user_posts(
                source_url,
                limit=limit,
                keywords=keywords,
            )

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)

        raw_entries = info.get("entries") or []
        entries = []
        for entry in list(raw_entries)[:limit]:
            item_url = self._build_video_url(platform, entry)
            external_id = entry.get("id")
            if not item_url or not external_id:
                continue
            entries.append(
                {
                    "platform": platform,
                    "external_id": external_id,
                    "title": entry.get("title") or external_id,
                    "url": item_url,
                }
            )

        return {
            "platform": platform,
            "title": info.get("title") or source_url,
            "entries": entries,
        }

    def derive_external_id(self, url="", fallback_text=""):
        if url:
            path = urlparse(url).path.rstrip("/").split("/")
            if path and path[-1]:
                return path[-1]
        return self._build_content_id(fallback_text or url or "post")

    def extract_post_metadata(self, url):
        if not url:
            return {}

        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return {}

        image_urls = []
        if info.get("thumbnail"):
            image_urls.append(info["thumbnail"])
        for item in info.get("thumbnails", []):
            thumbnail = item.get("url")
            if thumbnail:
                image_urls.append(thumbnail)

        deduped = []
        seen = set()
        for image_url in image_urls:
            if image_url in seen:
                continue
            deduped.append(image_url)
            seen.add(image_url)

        return {
            "platform": self.detect_platform(url),
            "external_id": info.get("id") or self.derive_external_id(url, url),
            "title": info.get("title") or "",
            "post_text": info.get("description") or info.get("title") or "",
            "image_urls": deduped,
            "source_url": info.get("webpage_url") or url,
        }

    def download_post_images(self, platform, external_id, image_urls, request_headers=None):
        saved_files = []
        for index, image_url in enumerate(image_urls):
            parsed = urlparse(image_url)
            suffix = os.path.splitext(parsed.path)[1] or ".jpg"
            filename = f"{platform}_{external_id}_{index}{suffix}"
            target_path = self.content_dir / filename

            try:
                response = requests.get(
                    image_url,
                    headers=request_headers or {},
                    timeout=(
                        app_config.weibo_timeout_seconds
                        if platform == "weibo"
                        else 20
                    ),
                )
                response.raise_for_status()
                data = response.content
                with open(target_path, "wb") as f:
                    f.write(data)
            except Exception:
                continue

            saved_files.append(
                {
                    "local_path": str(target_path),
                    "web_path": f"/content/{filename}",
                    "source_url": image_url,
                }
            )

        return saved_files

    def get_platform_request_headers(self, platform, referer=None):
        if (platform or "").lower() == "weibo":
            return self.weibo_adapter.build_request_headers(referer=referer)
        return {}
