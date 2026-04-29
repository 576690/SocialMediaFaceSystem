# 本科毕业论文定稿规划与 Composer Handoff

## 1. 输入与约束

- 仓库根目录：`D:\Personal\Documents\code\Python\SocialMediaFaceSystem`
- 当前初稿：`D:\Personal\Documents\毕业设计\毕业论文初稿.md`
- 论文要求：`D:\Personal\Documents\毕业设计\论文要求.md`
- 任务阶段：规划阶段，不直接生成定稿，不覆盖初稿。
- 用户要求：暂时保留现有截图占位；优先规范图表样式，并补充必要的工程图表。
- 学校/教师结构要求：绪论、系统分析、系统设计、系统实现、系统测试、结论；正文从绪论到结论超过 45 页，各章篇幅相对均衡；第四章不能直接粘贴源码，应使用流程图、代码逻辑描述和系统截图。

## 2. 证据边界

### 2.1 可作为论文事实依据的材料

| 证据类型 | 证据文件或目录 | 可支撑内容 |
|---|---|---|
| 项目说明 | `README.md` | 系统目标、技术路线、模块职责、运行方式、测试说明、局限性 |
| 依赖配置 | `requirements.txt` | FastAPI、SQLite/FAISS、InsightFace、Florence-2、faster-whisper、sentence-transformers、scikit-learn、hdbscan 等技术选型 |
| 后端入口与接口 | `app.py` | 系统启动、后台任务、管理员会话、采集、检索、聚类、人物维护、系统配置 API |
| 配置管理 | `core/config.py` | 抽帧间隔、字幕容差、检索阈值、ASR、视觉描述、人脸质量过滤、聚类参数、管理员口令哈希配置 |
| 数据库与索引 | `core/database.py` | `faces`、`face_fts`、`people`、`contents`、`collection_sources`、`cluster_snapshots` 表，SQLite 与 FAISS 索引同步、聚类快照与回滚 |
| AI 处理 | `core/analyzer.py` | InsightFace 人脸检测/embedding、质量过滤、Florence-2 视觉描述、ASR、语义文本融合、文本相似度排序 |
| 字幕对齐 | `core/alignment.py` | SRT 解析、ASR 片段写回、按时间戳匹配字幕 |
| 采集与适配器 | `core/collector.py`、`core/source_adapters.py`、`core/weibo_adapter.py`、`core/x_adapter.py` | yt-dlp 视频/频道采集、微博图文、X/Twitter 图文、适配器配置与归一化 |
| 聚类与评估 | `core/clustering.py`、`core/benchmark.py`、`scripts/run_benchmark.py` | DBSCAN/HDBSCAN/OPTICS、cosine/euclidean、purity/NMI/ARI、top1/top5、质量过滤网格 |
| 前端页面 | `static/index.html` | 以文搜图、以图搜图、人物库、管理员采集与配置、聚类、质量参数、来源适配器、系统重置界面 |
| 自动化测试 | `tests/*.py` | 数据库、聚类、benchmark、质量过滤、字幕对齐、语义融合、采集器、适配器、API 管理权限等单元测试 |
| benchmark 结果 | `storage\benchmarks\lfw_deepfunneled_full_20260429_121651\*` | LFW deepfunneled 数据集评估、检索 top1/top5、聚类指标、质量过滤参数推荐、失败样本原因 |
| 现有论文图资源 | `output\thesis_figures\*.drawio|*.png|*.svg` | 已生成的架构图、业务流程图、E-R 图、检索聚类流程图，可作为重绘基础 |

### 2.2 允许写入论文的核心结论

- 系统是一个本地化原型系统，围绕社交媒体视频、图文和账号/频道来源完成采集、人脸处理、语义增强、双模态检索、人物聚类与维护。
- 后端采用 FastAPI，前端为单页控制台，存储层使用 SQLite 保存元数据、FAISS 保存人脸向量索引。
- 人脸处理基于 InsightFace，质量过滤包括尺寸、相对占比、拉普拉斯清晰度和姿态偏移。
- 语义增强由标题、正文、字幕、ASR 文本和视觉描述组成；文本检索由 sentence-transformers 相似度排序支撑。
- 采集来源支持单视频链接、账号/频道同步、图文导入；微博和 X/Twitter 依赖适配器、cookie/token 与平台可访问性。
- 聚类支持 DBSCAN、HDBSCAN、OPTICS，并支持全量聚类快照、回滚、人工命名、合并、重分配。
- benchmark 可说明在 LFW deepfunneled 本地数据集上的辅助评估结果，但不能泛化为生产环境或所有社交媒体数据的性能结论。

### 2.3 禁止或必须弱化的结论

- 不得写“大规模生产部署”“线上用户规模”“高并发验证”“显著提升行业效率”等无证据结论。
- 不得声称提出全新人脸识别算法或多模态基础模型；应表述为成熟模型与工程流程的集成实现。
- 不得把 LFW benchmark 结果直接等同于真实社交媒体视频场景识别精度。
- 不得把所有测试用例都写成已真实人工执行，除非后续 composer/playwright 阶段补足运行截图或日志。
- “性能满足需求”只能限定为“本地原型、中小规模样本、辅助观察”，需要引用 benchmark 或可运行系统观察。
- 微博/X 采集必须说明受 cookie、API 权限、平台限流和页面结构变化影响。

## 3. 清理后的章节大纲

### 第1章 绪论

1.1 研究背景和意义  
1.2 国内外研究现状  
1.2.1 人脸识别与特征表示研究现状  
1.2.2 向量检索与多模态语义检索研究现状  
1.2.3 社交媒体数据采集与人物信息管理研究现状  
1.3 主要研究内容  
1.4 论文组织结构

写作边界：第一章只写背景、研究现状和本文工作定位，不提前展开数据库、接口、具体实现和测试表格。

### 第2章 系统分析

2.1 系统建设背景与问题分析  
2.2 用户角色与用例分析  
2.3 功能需求分析  
2.3.1 多来源数据采集需求  
2.3.2 人脸检测与质量过滤需求  
2.3.3 人脸特征提取与向量索引需求  
2.3.4 多模态语义融合需求  
2.3.5 文本语义检索需求  
2.3.6 图片人脸检索需求  
2.3.7 人物聚类与人工维护需求  
2.3.8 来源适配器与系统配置需求  
2.4 核心业务流程分析  
2.5 非功能需求分析  
2.6 本章小结

建议调整：当前 2.5 的用例描述可以保留，但压缩表格数量，避免第二章过长；2.2 中现有 use case 图可保留但需重绘成黑白工程风格。

### 第3章 系统设计

3.1 总体架构设计  
3.1.1 架构设计目标  
3.1.2 系统整体架构  
3.1.3 技术选型与分层说明  
3.2 功能模块设计  
3.3 数据流程设计  
3.4 数据库与文件存储设计  
3.4.1 数据库概念结构  
3.4.2 主要数据表结构  
3.4.3 文件存储结构  
3.4.4 向量索引与全文索引设计  
3.5 核心接口设计  
3.6 安全与配置设计  
3.7 本章小结

建议调整：初稿 3.7 “部署架构设计”如果没有真实部署环境，应改为“本地运行与部署结构”，不要写生产集群。

### 第4章 系统实现

4.1 开发环境与实现概述  
4.2 多来源数据采集实现  
4.3 视频帧与图文图片处理实现  
4.4 人脸检测、质量过滤与特征提取实现  
4.5 多模态语义融合实现  
4.6 数据存储与向量索引实现  
4.7 文本检索与图片人脸检索实现  
4.8 人物聚类、回滚与人工维护实现  
4.9 前端可视化管理界面实现  
4.10 本章小结

写作方式：禁止整段粘贴源码；每节采用“关键类/函数 + 流程说明 + 少量伪代码或逻辑描述 + 对应截图占位”的形式。

### 第5章 系统测试

5.1 测试目标与方法  
5.2 测试环境与测试数据  
5.3 功能测试  
5.3.1 采集功能测试  
5.3.2 人脸处理功能测试  
5.3.3 语义融合功能测试  
5.3.4 文本与图片检索功能测试  
5.3.5 聚类与人物维护功能测试  
5.3.6 系统配置与异常处理测试  
5.4 自动化测试  
5.5 benchmark 辅助评估  
5.6 测试结果分析  
5.7 本章小结

建议调整：现有 5.3 至 5.10 过细，可合并，避免第五章失衡；benchmark 应作为“辅助评估”，不要作为系统真实性能的唯一证明。

### 结论

总结系统完成内容、工程特点、局限与改进方向。篇幅控制在半页到一页，不再新增未验证功能。

## 4. 当前初稿 Keep / Rewrite / Delete 矩阵

| 初稿范围 | 处理 | 理由与操作 |
|---|---|---|
| 第1章 1.1 | Rewrite | 背景可保留方向，但应用价值和“挑战”表述偏泛，需要压缩并避免过度宏大叙述。 |
| 第1章 1.2 | Rewrite | 研究现状可保留文献方向，但需重新组织为人脸特征、向量检索、多模态语义、社交媒体采集四类；引用需后续核对格式与真实性。 |
| 第1章 1.3 | Rewrite | 与项目功能接近，但存在重复段落，需按代码证据重新写成 4 个研究内容。 |
| 第1章 1.4 | Keep with minor edits | 结构基本符合要求，按最终章节名同步调整。 |
| 第2章 2.1 | Keep with downgrade | 问题分析方向可保留，降低“平台级”“大规模”表述。 |
| 第2章 2.2 用例建模 | Keep but redraw | 角色和功能可由前端/API 支撑，但图表必须改为黑白工程风格，去除彩色 draw.io XML 内嵌内容。 |
| 第2章 2.3 | Keep with edits | 功能需求基本贴合代码；需要删除未实现或不能证明的过强约束。 |
| 第2章 2.4 | Keep but redraw | 业务流程与代码流程一致，需重绘为标准流程图。 |
| 第2章 2.5 | Rewrite/Compress | 当前用例表较多，建议保留 4-5 个核心用例即可，避免篇幅挤压设计与实现章节。 |
| 第2章 2.6 | Keep with downgrade | 非功能需求只能写“设计目标/本地原型约束”，不能写已生产验证。 |
| 第3章 3.1 | Rewrite with evidence | 架构内容可保留，但图 3.1 需要从现有图资源重绘成黑白分层架构。 |
| 第3章 3.2 | Keep but redraw | 数据流图应以 app.py 调度、analyzer、database、storage 为基础。 |
| 第3章 3.3 | Keep with correction | 数据表必须严格按 `core/database.py` 的实际表和字段写；保留 E-R 图但需更新字段和关系。 |
| 第3章 3.4 | Rewrite | 模块设计应按真实文件和函数组织，减少抽象空话。 |
| 第3章 3.5 | Keep with edits | 接口设计可由 `app.py` 支撑，应列核心 API，不必覆盖所有端点。 |
| 第3章 3.6 | Rewrite | 安全设计限于管理员密码哈希、受保护管理接口、适配器配置限制、运行数据本地化与隐私提醒。 |
| 第3章 3.7 | Delete/Rewrite | 若写部署，只能写本地运行部署；删除生产部署、集群、负载均衡等无证据内容。 |
| 第4章 4.1 | Keep with edits | 开发环境需与 benchmark device report、requirements 对齐。 |
| 第4章 4.2-4.3 | Keep with code-grounded rewrite | 采集、视频帧、图文处理与代码一致，补充函数名和流程图。 |
| 第4章 4.4 | Keep | 人脸质量过滤有代码和测试支撑，是重点章节。 |
| 第4章 4.5 | Keep | 字幕、ASR、视觉描述、统一语义文本有代码支撑，可加强为核心实现。 |
| 第4章 4.6 | Keep | SQLite、FAISS、FTS、文件存储证据充分。 |
| 第4章 4.7 | Keep | 文本检索与图片检索可由 API 和 analyzer/database 支撑。 |
| 第4章 4.8 | Keep | 聚类、快照、回滚、人工维护有代码和测试支撑。 |
| 第4章 4.9 | Keep but screenshot later | 前端功能可由 `static/index.html` 支撑；截图占位先保留，后续运行系统补图。 |
| 第5章开头“下面给出第五章正文初稿” | Delete | 这是过程性文字，定稿必须删除。 |
| 第5章测试用例表 | Rewrite/Verify | 可保留结构，但“通过”需要后续由实际运行截图、测试命令输出或已有 unittest 证明；不能无证据全量宣称。 |
| 第5章性能测试 | Rewrite | 当前“较短时间”“可接受”过泛；改用 benchmark 输出和本地观察，避免编造响应时间。 |
| 第5章 benchmark | Keep with exact numbers | 可引用 2026-04-29 LFW 结果：13233 张图、12185 个有效样本、1048 失败、top1=0.9883、top5=0.9914、elapsed=5924.951s、推荐质量参数。 |
| 参考文献 | Rewrite/Verify | 只保留真实可查文献，统一格式；删除乱码或不完整中文文献条目。 |

## 5. 章节证据映射

| 章节 | 主要论点 | 证据来源 |
|---|---|---|
| 第1章 | 课题聚焦社交媒体人脸采集、语义增强与双模态检索，不是提出新基础模型 | README 项目概述、requirements 模型依赖、app/API 能力 |
| 第2章 | 系统需要采集、处理、检索、聚类、配置与维护等功能 | `static/index.html` 页面区域、`app.py` API、README 核心模块 |
| 第2章 | 角色可分为普通检索用户、数据处理/管理员 | `static/index.html` 中普通用户可查看检索/人物，管理员可采集/配置/维护；`app.py` admin 保护接口 |
| 第3章 | 系统采用前端展示层、后端服务层、智能处理层、数据存储层 | README 技术路线，`app.py`、`core/analyzer.py`、`core/database.py`、`static/index.html` |
| 第3章 | 数据库由内容、人脸、人物、采集源、聚类快照构成 | `core/database.py` 的 `CREATE TABLE` |
| 第3章 | 索引包括 FAISS 人脸向量索引和 FTS 文本索引 | `core/database.py` 的 `face_fts`、`init_index`、`search_faces_by_embedding` |
| 第3章 | 系统配置可调整采集、检索、ASR、视觉描述、人脸质量、聚类参数 | `core/config.py`、`app.py` `/api/system/config`、`/api/system/face-quality` |
| 第4章 | 视频和图文采集通过 yt-dlp、微博/X 适配器统一输出 | `core/collector.py`、`core/source_adapters.py`、`core/weibo_adapter.py`、`core/x_adapter.py` |
| 第4章 | 人脸质量过滤采用尺寸、占比、清晰度和姿态 | `core/analyzer.py` `_evaluate_face_candidate`、`tests/test_face_quality.py` |
| 第4章 | 多模态语义融合来自视觉、字幕/ASR、图文正文 | `core/analyzer.py` `compose_semantic_text`、`core/alignment.py`、`tests/test_semantic.py` |
| 第4章 | 文本检索使用语义编码相似度，图片检索使用人脸 embedding 和 FAISS/距离度量 | `core/analyzer.py`、`core/database.py`、`app.py` `/api/search/text`、`/api/search/image` |
| 第4章 | 聚类支持 DBSCAN/HDBSCAN/OPTICS、快照回滚、人物维护 | `core/clustering.py`、`core/database.py`、`app.py` 聚类/人物接口、`tests/test_clustering.py` |
| 第5章 | 自动化测试覆盖数据库、聚类、benchmark、采集、适配器、质量过滤、语义、API | `tests/*.py` |
| 第5章 | benchmark 辅助说明检索与聚类评估能力 | `storage\benchmarks\lfw_deepfunneled_full_20260429_121651\benchmark_run.json` 与 CSV |

## 6. 图表与截图规划

### 6.1 图表样式统一规则

- 所有论文工程图使用白底、黑/灰线、无装饰图标、无彩色语义依赖。
- draw.io 图中禁止大面积蓝/绿/黄/红填充；已有图需要改为灰阶。
- 图题格式建议为“图 3.x xxx图”，表题格式建议为“表 3.x xxx表”。
- 图放在首次说明之后，不要把 draw.io XML 直接放进正文。
- 现有截图占位先保留；后续 composer/playwright 阶段再用真实运行系统截图替换。

### 6.2 需要保留并重绘的工程图

| 编号 | 放置章节 | 图名 | 当前资源/基础 | 处理 |
|---|---|---|---|---|
| 图2.1 | 2.2 | 系统用例图 | 初稿内 draw.io XML | 重绘为 UML 用例图，黑白风格；角色为管理员/数据处理人员/检索用户。 |
| 图2.2 | 2.4 | 核心业务流程图 | 初稿内流程图 XML | 重绘为标准流程图，保留视频/图文分支、入库、检索、聚类。 |
| 图3.1 | 3.1.2 | 系统整体架构图 | `output\thesis_figures\figure-3-1-architecture.drawio` | 改为四层架构灰阶图：前端展示、后端服务、智能处理、数据存储。 |
| 图3.2 | 3.3.1 | 数据库 E-R 图 | `output\thesis_figures\figure-3-3-er.drawio` | 保留但校正为真实表：contents、faces、people、collection_sources、cluster_snapshots。 |
| 图3.3 | 3.2/3.3 | 数据流图 | 初稿/现有图可扩展 | 建议新增，展示 content -> face -> semantic -> index -> search 的数据流。 |
| 图4.1 | 4.2 | 多来源采集流程图 | 需要新增 | 基于 collector/source_adapters/weibo/x 代码补充。 |
| 图4.2 | 4.4 | 人脸处理与质量过滤流程图 | 需要新增 | 展示检测、候选评估、尺寸/占比/清晰度/姿态过滤、embedding、入库。 |
| 图4.3 | 4.5 | 多模态语义融合流程图 | 需要新增 | 展示视觉描述、字幕对齐、ASR、图文正文、semantic_text。 |
| 图4.4 | 4.7-4.8 | 检索与聚类流程图 | `output\thesis_figures\figure-4-1-search-cluster.drawio` | 改为黑白，覆盖文本检索、图片检索、聚类、人工维护和回滚。 |
| 图5.1 | 5.1.3 | 系统测试流程图 | 初稿内 draw.io XML | 重绘为黑白流程图，避免彩色块。 |

### 6.3 表格规划

| 编号 | 放置章节 | 表名 | 依据 | 处理 |
|---|---|---|---|---|
| 表3.1 | 3.1.3 | 技术选型表 | `requirements.txt`、README | 保留，按前端/后端/AI/存储/测试分类。 |
| 表3.2 | 3.4.2 | contents 表结构 | `core/database.py` | 补全真实字段，不写不存在字段。 |
| 表3.3 | 3.4.2 | faces 表结构 | `core/database.py` | 包括后续迁移字段：content_id、content_type、semantic_text 等。 |
| 表3.4 | 3.4.2 | people、collection_sources、cluster_snapshots 表结构 | `core/database.py` | 可合并成一张“辅助表结构表”。 |
| 表3.5 | 3.5 | 核心接口表 | `app.py` | 只列采集、检索、聚类、人物维护、配置接口。 |
| 表4.1 | 4.4 | 人脸质量过滤参数表 | `core/config.py` | 使用默认值：min_face_size=56、min_face_ratio=0.035、min_laplacian_var=80.0、max_pose_deviation=0.35。 |
| 表5.1 | 5.2 | 测试环境表 | benchmark device report、requirements | 可写 RTX 3060 Laptop GPU、CUDA 12.8、torch 2.10.0+cu128、onnxruntime 1.23.2。 |
| 表5.2 | 5.4 | 自动化测试覆盖表 | `tests/*.py` | 按测试文件列验证目标。 |
| 表5.3 | 5.5 | benchmark 数据集与运行信息表 | `benchmark_run.json` | LFW deepfunneled、5749 identities、13233 images、12185 kept、1048 failed、5924.951s。 |
| 表5.4 | 5.5 | benchmark 检索结果表 | `retrieval_results.csv` | cosine/euclidean top1=0.9883、top5=0.9914、queries=8442。 |
| 表5.5 | 5.5 | 质量过滤参数对比表 | `quality_filter_results.csv` | 三组参数与 balanced_score，说明推荐配置来自辅助评估。 |

### 6.4 截图占位保留与后续补图清单

先保留初稿中 `【截图位置：...】` 的占位。后续替换时按以下最小截图集执行：

| 放置章节 | 截图内容 | 角色/前置数据 | 必要性 |
|---|---|---|---|
| 4.9 / 5.3 | 单链接视频采集或图文导入界面 | 管理员登录，准备一个可采集链接或手动图文 | 必需 |
| 4.9 / 5.6 | 文本检索结果界面 | 已有人脸记录和 semantic_text | 必需 |
| 4.9 / 5.7 | 图片人脸检索结果界面 | 已有人脸库，准备查询图片 | 必需 |
| 4.9 / 5.8 | 人物库界面 | 已执行聚类或有人工人物 | 必需 |
| 4.9 / 5.8 | 人物详情/时间线界面 | 人物下有多张人脸 | 建议 |
| 4.9 / 5.9 | 系统配置与人脸质量参数界面 | 管理员登录 | 必需 |
| 5.10 | 自动化测试命令输出 | 运行 `python -m unittest discover -s tests` | 建议 |
| 5.10 | benchmark CSV 或终端摘要 | 使用已有 `storage\benchmarks\...` 结果 | 必需 |

## 7. Composer Handoff

### 7.1 文件安全

- 原始初稿：`D:\Personal\Documents\毕业设计\毕业论文初稿.md`
- composer 必须先复制初稿到新的工作文件，再改正文；不得覆盖原文件。
- 本轮没有指定最终 DOCX 路径，composer 启动前需确定输出路径。
- 选取正文锚点时只能匹配正文标题，不得用目录段落作为锚点。

### 7.2 Preserve List

- 保留截图占位文本，后续运行系统后替换。
- 保留与实际代码一致的模块主题：采集、人脸处理、语义融合、检索、聚类、人物维护、配置、benchmark。
- 保留可核验的 benchmark 结果，但正文中必须注明其为 LFW deepfunneled 本地辅助评估。
- 保留数据库分表说明结构，但字段必须按 `core/database.py` 修正。

### 7.3 Replace List

- 替换所有内嵌 draw.io XML 为正式图片引用或 Word 中插图。
- 替换彩色/信息图风格图表为黑白工程论文图。
- 重写所有“下面给出初稿”“后续你可以...”等过程性文字。
- 重写缺乏证据的性能、部署、应用规模、创新性表述。
- 重写参考文献并核验格式。

### 7.4 重点重写边界

- 第1章：整体重写，但保留研究方向。
- 第2章：保留需求结构，压缩用例表，重绘用例图和业务流程图。
- 第3章：以代码证据重写架构、数据表、接口和配置设计。
- 第4章：以核心类/函数和流程图重写，避免源码粘贴。
- 第5章：测试用例只保留可验证内容；增加自动化测试与 benchmark 的证据表。

### 7.5 图表生成交给 drawio/composer 的任务

- 使用 `output\thesis_figures` 现有 drawio 作为基础，统一转为灰阶样式。
- 新增至少 3 张图：多来源采集流程图、人脸质量过滤流程图、多模态语义融合流程图。
- 修正 E-R 图，使字段与 `core/database.py` 一致。
- 最终 DOCX 中插图必须为图片，不得出现 XML 源码。

### 7.6 风险与人工复核点

- 参考文献需逐条核验，当前初稿中部分中文文献存在编码或格式问题。
- 若最终需要真实运行截图，应启动本地服务并准备演示数据；不要用占位图冒充实际截图。
- `storage` 下 benchmark 结果是本地运行数据，引用时需写清数据集、运行时间、硬件环境与局限。
- 微博/X 采集截图可能受 cookie/token 和平台限制影响，可优先使用手动图文导入或本地可访问来源完成截图。
- 若学校要求 DOCX 模板，composer 阶段必须按模板处理目录、页码、题注和参考文献格式。
