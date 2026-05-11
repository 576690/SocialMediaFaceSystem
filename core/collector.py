import glob
import hashlib
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
import yt_dlp
from yt_dlp.utils import DownloadError

from core.config import app_config
from core.weibo_adapter import WeiboUserCollector


class VideoCollector:
    def __init__(self, save_dir=None, weibo_adapter=None):
        self.save_dir = Path(save_dir or app_config.videos_dir)
        self.content_dir = Path(app_config.content_dir)
        self.weibo_adapter = weibo_adapter or WeiboUserCollector()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.last_bilibili_impersonate_status = None

    def detect_platform(self, url):
        host = urlparse(url).netloc.lower()
        if "bilibili.com" in host or "b23.tv" in host:
            return "bilibili"
        if "weibo.com" in host or "weibo.cn" in host:
            return "weibo"
        if host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
            return "x"
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

    def _base_ydl_opts(self, platform=None):
        platform = (platform or "").lower()
        cookie_path = app_config.storage_dir / "www.youtube.com_cookies.txt"
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(self.save_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": app_config.platform_timeout_seconds(platform or "generic"),
            "retries": app_config.platform_retry_count(platform or "generic"),
            "extractor_retries": app_config.platform_retry_count(platform or "generic"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": app_config.subtitle_languages,
            "subtitlesformat": app_config.subtitle_formats,
            "merge_output_format": "mp4",
        }
        if (
            platform == "youtube"
            and app_config.platform_auth_enabled("youtube")
            and cookie_path.exists()
        ):
            ydl_opts["cookiefile"] = str(cookie_path)
        return ydl_opts

    def _bilibili_headers(self):
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/135.0.0.0 Safari/537.36"
            ),
            "Referer": app_config.bilibili_referer,
        }

    def _supports_impersonate_target(self, target):
        requested = str(target or "").strip()
        if not requested:
            return False, "No impersonate target configured."
        if not importlib.util.find_spec("curl_cffi"):
            return False, (
                f'Impersonate target "{requested}" requested, but curl_cffi is not installed. '
                "Downgrading to standard requests mode."
            )
        return True, f'Impersonate target "{requested}" is available.'

    def _bilibili_ydl_opts(self, download=False):
        opts = {
            "quiet": True,
            "no_warnings": True,
            "http_headers": self._bilibili_headers(),
            "socket_timeout": app_config.platform_timeout_seconds("bilibili"),
            "retries": app_config.platform_retry_count("bilibili"),
            "extractor_retries": app_config.platform_retry_count("bilibili"),
            "retry_sleep": "extractor:1:2",
        }
        if download:
            opts.update(self._base_ydl_opts(platform="bilibili"))
            opts["http_headers"] = self._bilibili_headers()

        requested_target = app_config.bilibili_impersonate
        impersonate_enabled, impersonate_reason = self._supports_impersonate_target(
            requested_target
        )
        self.last_bilibili_impersonate_status = {
            "impersonate_requested": requested_target,
            "impersonate_enabled": impersonate_enabled,
            "impersonate_reason": impersonate_reason,
        }
        if impersonate_enabled:
            opts["impersonate"] = requested_target

        if (
            app_config.platform_auth_enabled("bilibili")
            and app_config.bilibili_cookie_path.exists()
        ):
            opts["cookiefile"] = str(app_config.bilibili_cookie_path)
        return opts

    def _build_ydl_opts(self, platform, download=False, extra_opts=None):
        platform = (platform or "").lower()
        if platform == "bilibili":
            opts = self._bilibili_ydl_opts(download=download)
        elif download:
            opts = self._base_ydl_opts(platform=platform)
        else:
            opts = {
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": app_config.platform_timeout_seconds(platform or "generic"),
                "retries": app_config.platform_retry_count(platform or "generic"),
                "extractor_retries": app_config.platform_retry_count(platform or "generic"),
            }
            youtube_cookie_path = app_config.storage_dir / "www.youtube.com_cookies.txt"
            if (
                platform == "youtube"
                and app_config.platform_auth_enabled("youtube")
                and youtube_cookie_path.exists()
            ):
                opts["cookiefile"] = str(youtube_cookie_path)

        if extra_opts:
            opts.update(extra_opts)
        return opts

    def _format_bilibili_download_error(self, exc):
        message = str(exc)
        impersonate_status = self.last_bilibili_impersonate_status or {}
        if "Impersonate target" in message and "not available" in message:
            detail = impersonate_status.get("impersonate_reason") or (
                "当前运行环境不支持请求的浏览器模拟目标。"
            )
            return (
                "Bilibili 下载需要使用浏览器模拟，但当前环境不支持。"
                f"{detail} 请安装可选的浏览器模拟依赖，以提升采集稳定性。"
            )
        if "HTTP Error 412" not in message and "Precondition Failed" not in message:
            return message
        if not app_config.bilibili_cookie_path.exists():
            return (
                "Bilibili 下载失败（HTTP 412）。"
                "缺少 Bilibili Cookie 文件：storage/bilibili_cookies.txt"
            )
        if not impersonate_status.get("impersonate_enabled", True):
            return (
                "Bilibili 下载失败（HTTP 412），当前处于未启用浏览器模拟的降级模式。"
                "现有 bilibili_cookies.txt 可能已过期或被拦截，请刷新 Cookie 文件。"
                "建议安装浏览器模拟所需的可选依赖，以提升采集稳定性。"
            )
        return (
            "Bilibili 下载失败（HTTP 412）。"
            "现有 bilibili_cookies.txt 可能已过期或被拦截，请刷新 Cookie 文件后重试。"
        )

    def _extract_info(self, url, platform, download=False, extra_opts=None):
        opts = self._build_ydl_opts(platform, download=download, extra_opts=extra_opts)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=download)
                filename = ydl.prepare_filename(info) if download else ""
            return info, filename
        except DownloadError as exc:
            if platform == "bilibili":
                raise RuntimeError(self._format_bilibili_download_error(exc)) from exc
            raise

    def download(self, url):
        platform = self.detect_platform(url)
        info, prepared_filename = self._extract_info(url, platform, download=True)
        filename = self._resolve_media_path(prepared_filename, info)

        subtitle_path = self._find_subtitle_file(filename)
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
        platform = (platform or self.detect_platform(source_url)).lower()
        if not app_config.platform_enabled(platform):
            raise RuntimeError(f"{platform} 采集当前未启用。")
        limit = limit or app_config.platform_sync_limit(platform)
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
        info, _ = self._extract_info(
            source_url,
            platform,
            download=False,
            extra_opts=ydl_opts,
        )

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

        platform = self.detect_platform(url)
        if platform == "weibo":
            return self.weibo_adapter.fetch_single_post(url)

        try:
            info, _ = self._extract_info(
                url,
                platform,
                download=False,
                extra_opts={"quiet": True, "no_warnings": True},
            )
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
            "platform": platform,
            "external_id": info.get("id") or self.derive_external_id(url, url),
            "title": info.get("title") or "",
            "post_text": info.get("description") or info.get("title") or "",
            "image_urls": deduped,
            "source_url": info.get("webpage_url") or url,
        }

    def download_post_images(self, platform, external_id, image_urls, request_headers=None):
        saved_files = []
        platform = (platform or "").lower()
        for index, image_url in enumerate(image_urls):
            parsed = urlparse(image_url)
            suffix = os.path.splitext(parsed.path)[1] or ".jpg"
            filename = f"{platform}_{external_id}_{index}{suffix}"
            target_path = self.content_dir / filename
            headers = request_headers
            if headers is None and platform == "weibo":
                try:
                    headers = self.get_platform_request_headers(platform, referer=image_url)
                except Exception:
                    headers = {}

            try:
                response = requests.get(
                    image_url,
                    headers=headers or {},
                    timeout=(
                        app_config.weibo_timeout_seconds
                        if platform == "weibo"
                        else app_config.platform_timeout_seconds(platform)
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
