# 社交媒体人脸数据采集与智能检索系统

## 项目概述

本项目是一个面向社交媒体内容的人脸数据采集、语义增强、检索和聚类管理系统。系统可以从视频链接、账号来源或图文内容中采集媒体数据，自动检测人脸、抽取人脸向量，并结合画面描述、字幕、语音识别文本和图文正文形成可检索的语义信息。

系统的核心目标不是只做人脸识别，而是把“人脸是谁”“出现在哪里”“对应内容语境是什么”统一管理起来。用户可以通过文本语义检索、上传图片检索相似人脸，也可以对人脸库进行聚类、命名、合并、回滚和质量阈值调优。

## 技术路线

整体流程可以概括为：

```text
视频/图文/账号来源采集
  -> 视频抽帧或图片下载
  -> 人脸检测与质量过滤
  -> InsightFace 生成人脸 embedding
  -> 视觉描述、字幕、ASR、正文语义融合
  -> SQLite 保存元数据，FAISS 保存向量索引
  -> 文本检索、以图搜图、人物聚类和人工维护
```

系统采用 FastAPI 提供后端接口，`static/index.html` 提供单页前端。数据层使用 SQLite 保存内容、人物、人脸记录和采集来源，使用 FAISS 维护人脸向量索引。AI 能力主要包括 InsightFace 人脸检测与特征抽取、Florence-2 画面描述、faster-whisper/Whisper 语音转写，以及 sentence-transformers 文本相似度计算。

## 核心模块

### `app.py`

`app.py` 是系统入口，负责创建 FastAPI 应用、初始化数据库、AI 处理器和采集器，并注册前端静态资源与 API 路由。

重要职责包括：

- 管理后台任务状态，避免采集、处理和系统重置互相冲突。
- 调度视频处理、图文处理和账号来源同步。
- 提供系统设置、人脸质量配置、采集、检索、聚类、人物维护等 API。

关键函数和接口：

- `process_video_task(video_info)`：处理已下载视频，解析字幕或 ASR，按时间间隔抽帧，检测人脸并入库。
- `process_post_task(post_info)`：处理图文图片，提取图片中的人脸并关联正文语义。
- `sync_source_task(source_record, limit)`：同步频道、账号或微博用户来源。
- `POST /api/collect`：采集单个视频链接。
- `POST /api/collect/source`：注册并同步采集来源。
- `GET /api/search/text`：根据语义文本检索人脸记录。
- `POST /api/search/image`：上传图片进行相似人脸检索。
- `POST /api/cluster/run`：执行人物聚类。
- `POST /api/cluster/rollback`：回滚上一次全量聚类结果。

### `core/collector.py`

`VideoCollector` 负责外部内容采集。它屏蔽不同平台的差异，把视频、账号来源和图文图片转换为系统统一的数据结构。

主要能力包括：

- `detect_platform(url)`：识别 bilibili、YouTube、微博或通用平台。
- `download(url)`：调用 `yt-dlp` 下载视频、字幕和元数据。
- `fetch_source_entries(...)`：从频道或账号来源提取待处理条目。
- `extract_post_metadata(url)`：从图文链接中提取标题、正文和图片地址。
- `download_post_images(...)`：下载图文中的图片到本地 `storage/content`。

### `core/analyzer.py`

`AIProcessor` 是项目的 AI 能力中心。它负责模型懒加载、设备选择、人脸质量过滤、向量抽取和语义文本生成。

主要能力包括：

- `get_face_embedding_result_from_path(...)` / `get_face_embedding(...)`：从图片中检测最佳人脸并返回归一化 embedding。
- `filter_face_candidates(...)`：按人脸尺寸、清晰度和姿态过滤候选人脸。
- `_compute_laplacian_variance(...)`：用拉普拉斯方差评估人脸区域清晰度。
- `_compute_pose_deviation(...)`：根据关键点估算姿态偏移。
- `generate_description(...)`：使用 Florence-2 为画面生成视觉描述。
- `transcribe_video(...)`：在没有字幕时使用 ASR 生成语音片段。
- `compose_semantic_text(...)`：将 Visual、Speech、Post 三类信息融合成统一语义文本。
- `rank_texts_by_similarity(...)`：对语义文本进行相似度排序。

人脸质量过滤是系统效果的重要环节。默认配置会过滤过小、模糊或姿态偏差过大的人脸，从源头减少低质量 embedding 对检索和聚类的影响。

### `core/database.py`

`DatabaseManager` 负责持久化和索引管理。它同时维护 SQLite 元数据表和 FAISS 向量索引。

核心数据对象包括：

- `contents`：视频、图文等内容条目。
- `faces`：检测到的人脸、embedding、语义文本、来源信息和人物编号。
- `people`：人物 ID 与人工命名。
- `collection_sources`：已注册的采集来源。
- `cluster_snapshots`：全量聚类前的人物分组快照，用于回滚。

关键函数包括：

- `upsert_content(...)`：新增或更新内容记录。
- `add_face(...)`：保存人脸记录，同时写入 FAISS 索引和全文检索表。
- `search_faces_by_embedding(...)`：使用 FAISS 按 cosine 相似度检索。
- `search_faces_by_metric(...)`：按 cosine 或 euclidean 遍历检索。
- `replace_all_person_ids(...)`：全量替换人物聚类结果。
- `save_cluster_snapshot(...)` / `restore_cluster_snapshot(...)`：保存和恢复聚类前状态。
- `delete_source_with_data(...)`：删除采集来源及其关联数据和文件。

### `core/clustering.py`

该模块负责人脸聚类和聚类效果评估。系统支持 DBSCAN、HDBSCAN 和 OPTICS，距离度量支持 cosine 和 euclidean。

关键函数包括：

- `cluster_embeddings(...)`：根据算法和参数对 embedding 聚类。
- `perform_clustering(...)`：从数据库读取人脸，执行增量或全量聚类，并写回人物 ID。
- `evaluate_embedding_clusters(...)`：计算 purity、NMI、ARI 等聚类指标。
- `evaluate_embedding_retrieval(...)`：计算以图搜图 top1、top5 命中率。
- `compare_image_search_metrics(...)`：在已标注人物数据上比较检索指标。

全量聚类会先保存快照，便于结果不满意时回滚。增量聚类只处理未分配人脸，适合日常采集后的快速更新。

### `core/benchmark.py` 与 `scripts/run_benchmark.py`

benchmark 模块用于在标准人脸数据集上评估 embedding、检索、聚类和质量过滤参数。

主要能力包括：

- `discover_identity_dataset(...)`：读取“每个身份一个目录”的标准数据集。
- `extract_dataset_embeddings(...)`：批量抽取 embedding，并保存缓存和失败样本。
- `run_benchmark_suite(...)`：运行聚类和检索评估。
- `evaluate_face_quality_grid(...)`：遍历人脸质量参数网格，比较不同阈值组合。
- `recommend_face_quality(...)`：根据综合分数推荐质量过滤配置。
- `export_benchmark_results(...)`：导出 CSV 结果。

`scripts/run_benchmark.py` 是命令行入口。需要注意的是，全量数据集加上 OPTICS 和多组质量参数会非常耗时，尤其在样本数上万时，聚类和检索阶段可能成为主要瓶颈。

Thesis-oriented ablation benchmarks can be run with a smaller, deterministic
stratified sample:

```bash
python scripts/run_benchmark.py --experiment-preset thesis-small --retrieval-backend auto
```

This preset writes `sample_manifest.csv`, `sample_manifest.json`,
`quality_ablation_results.csv`, and `cluster_ablation_results.csv`. The quality
ablation output keeps the legacy `balanced_score` and adds `balanced_score_v2`,
which combines retrieval accuracy, clustering stability, retention rate, and
accepted face quality for thesis tables.

### `core/weibo_adapter.py`

`WeiboUserCollector` 负责微博用户源适配。它对 `weibo-spider` 做兼容封装，处理 cookie、用户地址归一化、分页抓取、图片提取和关键词过滤。

关键函数包括：

- `normalize_user_source(...)`：把用户 ID、短路径或完整 URL 归一化为微博用户来源。
- `load_cookie()`：读取微博 cookie 文件。
- `build_request_headers(...)`：构造微博请求头。
- `fetch_user_posts(...)`：抓取微博用户图文，按关键词和图片存在性筛选。
- `normalize_picture_urls(...)`：解析微博图片字段。

该模块是微博采集稳定性的关键位置，因为微博页面结构、cookie 状态和访问限制都会直接影响采集结果。

### `core/alignment.py`

字幕对齐模块负责把字幕或 ASR 片段映射到视频帧时间点。

关键函数包括：

- `parse_srt_file(path)`：读取 SRT 字幕，转换为 `start/end/text` 片段。
- `write_srt_file(segments, path)`：把 ASR 片段写回 SRT 文件。
- `align_text_to_timestamp(segments, timestamp, tolerance)`：找到当前帧对应的字幕文本，或在容差范围内选择最近片段。

这部分让每一张人脸截图能够带上当时的说话内容，从而提升语义检索能力。

### `core/config.py`

`AppConfig` 管理系统默认配置和本地运行时配置。配置文件位于 `storage/system_config.json`，如果不存在会自动生成。

主要配置包括：

- 抽帧间隔和字幕容差。
- 文本检索阈值、图片检索阈值和 top-k。
- 采集来源同步数量、字幕语言、微博 cookie 设置。
- ASR 后端和模型大小。
- 人脸质量过滤阈值。
- 默认聚类算法和参数。

### `static/index.html`

前端是一个单页控制台，直接调用 FastAPI 接口完成操作。

主要界面包括：

- 单链接视频采集。
- 账号或频道来源同步。
- 图文导入。
- 文本检索和以图搜图。
- 已登记采集源管理。
- 人脸质量参数配置。
- 聚类配置、人物库、人物详情、重命名、合并和回滚。
- 系统重置等维护操作。

## 数据与存储

项目运行时数据默认放在 `storage/` 下：

- `storage/videos/`：下载的视频文件。
- `storage/content/`：图文图片等原始内容文件。
- `storage/faces/`：裁剪后的人脸图和对应完整帧图。
- `storage/artifacts/asr/`：ASR 生成的字幕文件。
- `storage/metadata.db`：SQLite 元数据数据库。
- `storage/face_index.faiss`：FAISS 人脸向量索引。
- `storage/system_config.json`：系统运行配置。
- `storage/benchmarks/`：benchmark 输出、embedding 缓存和 CSV 结果。

这些文件属于本地运行数据，通常不应提交到版本库。

## 运行方式

建议使用项目虚拟环境运行：

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```

应用入口会在启动早期默认设置：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

因此直接使用普通启动命令时，`transformers`、`sentence-transformers` 和 `huggingface_hub` 的下载默认会走国内镜像。

如果需要显式使用项目自带脚本，也可以运行：

```powershell
.\scripts\start_with_hf_mirror.ps1
```

说明：

- 该配置会影响 `transformers`、`sentence-transformers` 和 `huggingface_hub` 的模型下载地址。
- 如果你在启动前已经显式设置了 `HF_ENDPOINT`，应用会保留外部值，不会强制覆盖。
- `HF_ENDPOINT` 需要在相关模块导入前确定；不要指望应用运行后再动态切换 endpoint。
- `https://hf-mirror.com` 是第三方国内镜像，不是 Hugging Face 官方站点。

启动后访问：

```text
http://127.0.0.1:8000
```

依赖集中在 `requirements.txt`，主要包括：

- Web 服务：FastAPI、Uvicorn。
- 媒体采集：yt-dlp、requests、weibo-spider、Tweepy。
- 图像与向量：OpenCV、InsightFace、ONNX Runtime、FAISS。
- 深度学习与语义：PyTorch、Transformers、sentence-transformers、faster-whisper。
- 聚类与评估：scikit-learn、hdbscan、numpy。

### B站 Cookie

如果需要稳定下载 B 站视频，请将手动导出的 cookie 文件放到：

```text
storage/bilibili_cookies.txt
```

说明：

- `storage/www.youtube.com_cookies.txt` 仅用于 YouTube。
- `storage/bilibili_cookies.txt` 仅用于 B 站。
- 若 B 站返回 `HTTP 412`，优先检查或更新 `bilibili_cookies.txt`。
- 项目会优先尝试 B 站 impersonate 增强模式；若环境缺少相关依赖，会自动降级为普通请求模式。
- `impersonate` 是增强项，不是使用 B 站 cookie 下载的前置必需项。

### 来源适配器

系统通过 `core/source_adapters.py` 统一账号或频道来源同步。内置适配器包括：

- `weibo_user`：包装 `core/weibo_adapter.py`，同步微博用户图文。
- `x_user`：包装 `core/x_adapter.py`，使用 Tweepy 和 X API v2 同步 X/Twitter 用户图片推文。
- `yt_dlp`：包装 `yt-dlp`，同步 B 站、YouTube 和通用视频频道来源。

适配器统一返回 `platform`、`title`、`source_url`、`entries`、`stats` 和 `cursor`。单条 `entry` 至少包含：

- `content_type`：`video` 或 `post`。
- `platform`、`external_id`、`title`、`url`。
- 图文条目额外提供 `post_text`、`image_urls` 和可选 `metadata`。

后台“来源适配器”区域只允许上传 JSON/YAML 配置，不允许上传 Python 代码。复杂网站需要技术人员先把 Python 适配器模块放到：

```text
storage/source_adapter_modules/
```

再上传配置文件启用。配置文件保存到：

```text
storage/source_adapters/
```

配置示例：

```json
{
  "adapter_id": "custom_site",
  "display_name": "Custom Site",
  "platform": "custom_site",
  "enabled": true,
  "module": "custom_site_adapter:CustomSiteAdapter",
  "source_types": ["user", "channel"],
  "url_patterns": ["https://example.com/*"],
  "default_limit": 10,
  "settings": {}
}
```

如果省略 `module`，该配置会作为 `yt-dlp` 风格来源使用，适合 `yt-dlp` 已支持的网站。若指定 `module`，模块名必须是 `storage/source_adapter_modules/` 下的本地 Python 文件，不从后台上传代码，避免把管理员配置入口变成远程代码执行入口。

自定义 Python 适配器建议实现以下方法：

- `match(source_url, platform=None, source_type=None)`：判断是否支持该来源。
- `normalize_source(source_url, platform=None, source_type=None, metadata=None)`：归一化来源 URL。
- `fetch_entries(source_record, limit)`：返回统一来源结果。
- `get_request_headers(entry)`：图文图片下载时可返回平台请求头。

### X/Twitter 同步

X/Twitter 用户同步使用 Tweepy 调用 X API v2，需要将 Bearer Token 放到：

```text
storage/x_bearer_token.txt
```

支持的来源格式包括：

```text
OpenAI
@OpenAI
https://x.com/OpenAI
https://twitter.com/OpenAI
```

第一版只同步含图片的推文，图片会进入现有图文人脸处理流水线；视频、GIF 和无图文本推文会被跳过。该功能受 X API 权限、额度、账号可见性和平台限流影响。X API v2 通过 fields 和 expansions 返回媒体等关联对象，Tweepy 提供 `get_user`、`get_users_tweets` 等封装。

## 项目亮点

- 多来源采集：支持视频链接、频道/账号来源、微博/X 图文和手动图文导入。
- 多模态语义融合：把视觉描述、字幕、ASR 和正文融合为统一语义文本。
- 双检索路径：既支持文本语义检索，也支持上传图片做人脸相似检索。
- 可调质量过滤：通过人脸尺寸、清晰度和姿态阈值控制入库质量。
- 人物库维护：支持聚类、命名、重分配、合并和全量聚类回滚。
- 可评估可调参：benchmark 模块可以在标准数据集上比较聚类、检索和质量过滤配置。

## 局限与改进方向

- 全量 benchmark 和 OPTICS 聚类在大样本上耗时较长，需要根据数据规模选择评估范围。
- 前端目前是单文件页面，适合原型和演示；如果功能继续扩大，可以拆分为组件化前端。
- 微博采集依赖 cookie 和页面结构，实际可用性会受平台风控和页面变化影响。
- AI 模型加载和 GPU/CPU provider 配置会显著影响处理速度，需要在部署环境中单独验证。

## 测试

项目已有单元测试覆盖核心行为：

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

其中重点测试包括：

- `tests/test_benchmark.py`：benchmark 数据发现、缓存、导出和质量网格。
- `tests/test_clustering.py`：聚类、人物继承和回滚相关逻辑。
- `tests/test_database.py`：数据库、索引、来源删除和快照。
- `tests/test_face_quality.py`：人脸质量过滤。
- `tests/test_alignment.py`：字幕解析与时间戳对齐。
- `tests/test_weibo_adapter.py`：微博来源归一化、抓取和错误处理。
