# 本地运行数据

目录由程序按需创建，内容不提交到 Git：

- `v1.0/cache/`：CLAP 缓存。
- `v1.0/outputs/`：1.0 生成的视频。
- `v2.0/projects/`：Studio 的音频副本、原始预测、修订、计划、检查点及生成视频。
- `v2.0/outputs/`：批量导出结果。
- `v2.0/validation-output/`：验证日志、报告和测试产物。
- `annotations/continuous/`：下一版的人工连续 V/A 标注。
- `datasets/deam/raw/`：DEAM 原始音频、标注、特征及官方元数据。
- `datasets/deam/prepared/v1/`：下一版模型使用的连续标签、歌曲索引、数据划分与校验报告；这里的 `v1` 是数据准备版本。
- `datasets/pmemo/`、`datasets/memo2496/`：补充数据的原始包和连续标签候选版本。
- `datasets/merp/`：已下载的 MERP；时间原点未解决，派生数据保留在隔离目录。
- `datasets/combined/prepared/v1/`：经过跨库重叠筛查的固定混合划分。
- `datasets/.tools/`、`.dependencies/`、`.fingerprints/`：本地校验工具、独立依赖与音频指纹缓存。
- `training/next_va/`：独立 GPU 环境、特征缓存、训练记录和自动接续队列。

DEAM 的读取、复现和已知问题见 [数据准备说明](../tools/datasets/README.md)，其他可用训练资源见 [资源清单](../tools/datasets/RESOURCES.md)。原始音频不拆分，现有模型不因数据准备而改变。

备份或移动 Studio 项目时应复制完整项目目录，并先停止使用该目录的服务和任务。
