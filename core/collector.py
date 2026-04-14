import yt_dlp
import os


class VideoCollector:
    def __init__(self, save_dir="storage/videos"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(self, url):
        cookie_path = "storage/www.youtube.com_cookies.txt"
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": f"{self.save_dir}/%(id)s.%(ext)s",
            "cookiefile": cookie_path,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return {
                "id": info["id"],
                "title": info["title"],
                "path": filename,
                "url": url,  # 【新增】保存原始输入的链接
            }
