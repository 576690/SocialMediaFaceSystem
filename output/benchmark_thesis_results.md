# thesis-small benchmark 参数对照实验结果

## 实验概况

- 实验目录：`d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843`
- 数据集：LFW deepfunneled，本地路径 `D:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\datasets\lfw-deepfunneled`
- 抽样设置：500 个身份，1612 张图片；每个身份 2-5 张；随机种子 20260511。
- 运行时间：2026-05-11 17:58:48 至 2026-05-11 18:11:14，总耗时 750.435 秒。
- 运行设备：NVIDIA GeForce RTX 3060 Laptop GPU；Torch 2.10.0+cu128，CUDA 12.8，ONNX Runtime 1.23.2。
- 主检索结果：cosine / torch-cuda，Top-1=0.9906，Top-5=0.9906，有效查询数=1493。

## 表 1 质量过滤参数对照实验

| 配置 | min_face_size | min_face_ratio | min_laplacian_var | max_pose_deviation | 保留样本 | 失败率 | Top-1 | Top-5 | NMI | ARI | balanced_score_v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 默认均衡组 | 56 | 0.035 | 80.0 | 0.35 | 1517 | 0.0589 | 0.9906 | 0.9906 | 0.9382 | 0.1450 | 0.8910 |
| 整体宽松组 | 40 | 0.020 | 50.0 | 0.45 | 1591 | 0.0130 | 0.9943 | 0.9943 | 0.9450 | 0.1728 | 0.9027 |
| 整体严格组 | 72 | 0.050 | 120.0 | 0.25 | 1208 | 0.2506 | 0.9928 | 0.9928 | 0.9080 | 0.0808 | 0.8545 |
| 降低人脸尺寸阈值 | 40 | 0.035 | 80.0 | 0.35 | 1517 | 0.0589 | 0.9906 | 0.9906 | 0.9382 | 0.1450 | 0.8910 |
| 提高人脸尺寸阈值 | 72 | 0.035 | 80.0 | 0.35 | 1512 | 0.0620 | 0.9926 | 0.9926 | 0.9390 | 0.1482 | 0.8920 |
| 降低占比阈值 | 56 | 0.020 | 80.0 | 0.35 | 1517 | 0.0589 | 0.9906 | 0.9906 | 0.9382 | 0.1450 | 0.8910 |
| 提高占比阈值 | 56 | 0.050 | 80.0 | 0.35 | 1517 | 0.0589 | 0.9906 | 0.9906 | 0.9382 | 0.1450 | 0.8910 |
| 降低清晰度阈值 | 56 | 0.035 | 50.0 | 0.35 | 1591 | 0.0130 | 0.9943 | 0.9943 | 0.9450 | 0.1728 | 0.9027 |
| 提高清晰度阈值 | 56 | 0.035 | 120.0 | 0.35 | 1221 | 0.2426 | 0.9778 | 0.9778 | 0.9036 | 0.0750 | 0.8471 |
| 放宽姿态阈值 | 56 | 0.035 | 80.0 | 0.45 | 1518 | 0.0583 | 0.9893 | 0.9893 | 0.9379 | 0.1438 | 0.8903 |
| 收紧姿态阈值 | 56 | 0.035 | 80.0 | 0.25 | 1517 | 0.0589 | 0.9906 | 0.9906 | 0.9382 | 0.1450 | 0.8910 |

## 表 2 聚类参数对照实验

| 配置 | 算法 | 距离 | eps | min_samples | 簇数量 | 噪声率 | Purity | NMI | ARI | cluster_score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 增大 eps 至 0.5 | dbscan | cosine | 0.50 | 2 | 455 | 0.0475 | 1.0000 | 0.9838 | 0.6005 | 0.8626 |
| 默认聚类组 | dbscan | cosine | 0.40 | 2 | 423 | 0.1332 | 1.0000 | 0.9382 | 0.1450 | 0.6860 |
| OPTICS | optics | cosine | 0.40 | 2 | 451 | 0.1503 | 1.0000 | 0.9229 | 0.1037 | 0.6625 |
| HDBSCAN | hdbscan | cosine | 0.40 | 2 | 282 | 0.2044 | 0.9039 | 0.8831 | 0.0692 | 0.6214 |
| min_samples=3 | dbscan | cosine | 0.40 | 3 | 231 | 0.3863 | 1.0000 | 0.7803 | 0.0178 | 0.5182 |
| 减小 eps 至 0.3 | dbscan | cosine | 0.30 | 2 | 322 | 0.4146 | 1.0000 | 0.7489 | 0.0111 | 0.4949 |
| 欧氏距离 | dbscan | euclidean | 0.40 | 2 | 1 | 0.9987 | 1.0000 | 0.0033 | 0.0000 | 0.0019 |

## 表 3 默认配置与最优配置对比

| 对比项 | 默认配置 | 默认得分 | 最优配置 | 最优得分 | 次优配置 | 次优得分 | 结论 |
|---|---|---:|---|---:|---|---:|---|
| 质量过滤 | 默认均衡组 | 0.8910 | 整体宽松组 | 0.9027 | 降低清晰度阈值 | 0.9027 | 默认组不是扩大抽样后的最高分，但识别指标和失败率保持稳定；若按综合得分优化，宽松质量阈值更优。 |
| 聚类参数 | 默认聚类组 | 0.6860 | 增大 eps 至 0.5 | 0.8626 | 默认聚类组 | 0.6860 | 默认 eps=0.4 仍是稳定基线，但 eps=0.5 在本次抽样中显著降低噪声率并提升 ARI。 |

## 论文说明文字

本节采用 LFW deepfunneled 数据集进行辅助 benchmark 评估。为降低单次实验规模并便于开展对照实验，实验从完整数据集中按身份进行确定性分层抽样，共抽取 500 个身份、1612 张图像，随机种子为 20260511。实验环境中使用 NVIDIA GeForce RTX 3060 Laptop GPU，检索阶段采用 torch-cuda 后端。该实验不直接等同于真实社交媒体视频场景中的最终识别精度，而是用于比较不同人脸质量过滤参数和聚类参数对检索、聚类及样本保留情况的影响。

在人脸质量过滤参数对照实验中，默认配置为 min_face_size=56、min_face_ratio=0.035、min_laplacian_var=80.0、max_pose_deviation=0.35。该配置保留 1517 个有效样本，失败率为 0.0589，Top-1 为 0.9906，Top-5 为 0.9906，综合均衡得分 balanced_score_v2 为 0.8910。从扩大抽样后的结果看，整体宽松组和“降低清晰度阈值”组的 balanced_score_v2 最高，均为 0.9027，主要原因是其样本保留率更高且检索指标略有提升。因此，默认配置不能表述为所有指标上的绝对最优，但可以作为兼顾质量控制和稳定性的保守折中；若目标是提高 benchmark 综合得分，可考虑进一步评估宽松配置在真实社交媒体数据中的误入库风险。

在聚类参数对照实验中，默认配置为 DBSCAN、cosine 距离、eps=0.4、min_samples=2，对应 cluster_score 为 0.6860。对照结果显示，将 eps 增大到 0.5 后，噪声率由 0.1332 降至 0.0475，ARI 由 0.1450 提升至 0.6005，cluster_score 提升至 0.8626。这说明在本次抽样数据上，较大的聚类半径更有利于减少噪声样本并形成更完整的身份簇；但 eps 增大也可能在更复杂数据中增加不同身份合并的风险，因此论文中应将其表述为 benchmark 条件下的参数候选，而不是直接替代所有场景下的默认配置。

综合来看，扩大抽样后的结果与前一轮小样本实验趋势一致：宽松质量阈值在综合得分上更高，DBSCAN 的 eps=0.5 在聚类对照中表现更好。该结果验证了 benchmark 模块能够从检索准确率、聚类稳定性、样本保留率和入库质量等多个角度评价参数组合。论文中可将现有默认参数描述为稳定基线，并说明后续可依据 balanced_score_v2 和 cluster_score 对质量过滤阈值、DBSCAN 半径等参数进行进一步调优。

## 数据来源

- `d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843\benchmark_run.json`
- `d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843\sample_manifest.csv`
- `d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843\retrieval_results.csv`
- `d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843\quality_ablation_results.csv`
- `d:\Personal\Documents\code\Python\SocialMediaFaceSystem\storage\benchmarks\lfw_deepfunneled_thesis_small_20260511_175843\cluster_ablation_results.csv`
