# 下一版模型的数据准备

这里管理外部数据集及其读取工具，独立于 `versions/v1.0`、`versions/v2.0` 和人工标注工具。准备脚本不训练模型，也不写入 `models/emotion_model.pth`。

## DEAM 本地目录

工作区根目录原来的 `MEMD_audio`、`annotations`、`features` 已整体移动到下列 `raw` 目录，文件内容保持原样。音乐、原始标签、派生数据和审计记录均由现有 `.gitignore` 排除，不进入 GitHub。

```text
data/datasets/deam/
  raw/
    audio/                    1,802 个原始 MP3
    annotations/              3,610 个原始标注文件，包括逐标注者数据
    features/                 1,802 个原始 openSMILE 特征 CSV
    metadata.zip              官方歌手、曲名、年份等元数据
  prepared/v1/                默认训练数据版本，与项目 1.0 无关
    manifest.jsonl            每首录音的路径、时长、来源、划分和 SHA256
    dynamic/<song_id>.npz      完整连续标签序列和逐维有效掩码
    splits/                   train.txt、validation.txt、test.txt
    config.json               标签顺序、量纲和读取约定
    report.json               全量校验结果及数据统计
    excluded.json             未进入训练清单的音频及原因
    applied-exclusions.json   本次采用的人工核查排除规则
    annotation-provenance.json 原始标注与官方元数据的校验值
    verification.json         派生标签回读、与原 CSV 逐值比较及音频读取结果
  layout-migration.json       迁移前后的文件大小、时间与路径记录
  audit-history/              历次审计产物，不用于训练
```

第一次导入同样结构的 DEAM 文件时可运行 `powershell -File tools/datasets/organize_deam.ps1`。本机已经完成迁移，**不要重复运行**。脚本遇到已有目标目录会拒绝合并。官方元数据来自 [DEAM 发布页](https://cvml.unige.ch/databases/DEAM/)，下载其中的 `metadata.zip` 并放入 `raw`。

## 标签如何使用

每个 NPZ 对应一首原始录音，不拆成 5 秒文件。DEAM 自身包含 45 秒节选和完整歌曲；本工具不能还原节选之外的音频。

| 字段 | 类型与形状 | 含义 |
| --- | --- | --- |
| `times_seconds` | float64 `[T]` | 相对于本地 MP3 开始的时间，来自原 CSV 列名 |
| `targets` | float32 `[T, 2]` | 列顺序固定为 **效价 V、唤醒 A**，范围 `[-1, 1]` |
| `mask` | bool `[T, 2]` | 每个时间、每个维度是否有有效监督 |

标签按原始 2 Hz 时间轴保存。首个标注时间为 15 秒，不给前 15 秒补值，不平移标签，不人为生成转折点。7 首歌曲的 V/A 长度相差一个采样点，使用时间戳并集和独立掩码保留数据。无效位置存储的 `0` 只是占位符，必须用 `mask` 从损失和指标中排除；有效位置的零值仍是正常标签。

整曲静态均值和标准差单独保存在 manifest 的 `static_labels` 中。原量纲 `1–9` 的均值同时提供 `(x - 5) / 4` 转换值；不能把静态均值铺满时间轴作为动态真值。上述标注约定参见 [DEAM 官方手册](https://cvml.unige.ch/databases/DEAM/manual.pdf)。

## 当前发现与处理

- 原始 ID 在音频、两个动态维度、静态标签、特征、官方元数据之间一致，共 1,802 首。默认清单排除 `1952.mp3`，其余 1,801 首用于后续训练准备。
- `1952.mp3` 在 FFmpeg 严格解码中通过，但本机 libsndfile/mpg123 报告比特流错误。原件保留，SHA256 和原因记在 [deam_exclusions.json](deam_exclusions.json)。更换该文件后必须先重新核查排除规则，脚本不会静默接纳变化后的文件。
- 743 份附带特征的末尾时间超过本地音频时长；例如 `2.mp3` 约 45 秒，而特征延伸到 224 秒。不能按行与本地标签拼接。
- `1640`、`1667`、`1952` 的特征步长约 0.501134 秒，不是精确的 0.5 秒。审计保留原时间戳并记录偏差。
- 部分 MP3 文件头的帧数大于实际可解码帧数。manifest 使用实际解码长度，原始头部帧数单独记录，不复制音频缓冲区或补零来填长。
- 所有附带特征均标记 `training_enabled=false`。独立训练入口从本地音频重新提取统一特征，保留每帧时间戳，再与标签对齐。CLAP 缓存、检查点与评估结果属于 [训练流程](../train_next/README.md)，不写回原始数据。

完整结果以本地 `prepared/v1/report.json` 为准。首次检查的失败目录 `*.pending-*` 仅用于诊断，不是可训练的数据版本。

## 数据划分

按规范化歌手名分组，字节内容相同的音频也强制归入同组，再按约 `70% / 15% / 15%` 分配到训练、验证、测试集，固定种子 `20261002`。一首歌的整条时间序列始终属于同一集合。先划分歌曲，再在各集合内部构造训练序列或批次。

这是本项目自定义划分，**不是 DEAM 官方评测划分**，不能直接与论文中的官方测试分数比较。歌手别名、翻唱及近似重复录音尚未做音频指纹识别；未来合并其他数据集时必须跨数据集去重。同一录音的片段与整曲也要绑定到同一集合。

本机 2026-10-02 的可用版本共 1,801 首、约 25.57 小时，947 个歌手/音频分组：训练 1,261 首，验证 270 首，测试 270 首。被排除的 `1952` 不属于任何训练划分。

标准化器只在训练集拟合。验证集用于选模型、调参和选择时延；测试集最后评估，不用于这些选择。建议分别报告 V/A 的 MAE、RMSE、CCC，并按歌曲汇总，避免长歌占据所有权重。

## 读取与复现

当前 `venv` 已有 NumPy、SoundFile；准备阶段还使用项目已有 FFmpeg，也可用 `--ffmpeg` 指定路径。无须加载现有情绪模型或下载 CLAP。

新环境可用 `python -m pip install -r tools/datasets/requirements.txt` 安装本工具的两项依赖。锁定版本在本机 Python 3.13.15 下验证；FFmpeg 单独准备。

在工作区根目录快速读取训练集并解码第一首：

```powershell
.\venv\Scripts\python.exe tools/datasets/deam_dataset.py --split train --audio
.\venv\Scripts\python.exe tools/datasets/verify_deam.py
.\venv\Scripts\python.exe -m unittest discover -s tools/datasets/tests -v
```

本次全量回读已核对 1,801 首的标签与原 CSV：效价 129,938 个有效值、唤醒 129,939 个有效值，7 个单维缺失位置；三个集合各选短音频和最长音频测试读取。10 项回归测试通过。

训练代码可以直接读取连续序列：

```python
from tools.datasets.deam_dataset import DeamSequences

dataset = DeamSequences(split="train", include_audio=True)
song = dataset[0]
audio = song["audio"]                 # [音频采样数, 声道数]，原始采样率
sample_rate = song["sample_rate"]
times = song["times_seconds"]         # 标签时间轴，单位秒
targets = song["targets"]             # [T, 2]，V 在前、A 在后
mask = song["mask"]                   # 只在 True 的位置计算损失
# 假设 predictions 已在同一 times 上输出 [T, 2]：
# mse = ((predictions[mask] - targets[mask]) ** 2).mean()
```

默认 `include_audio=False`，只读取标签和元数据。音频不自动混声道或重采样；未来模型应明确这些处理参数。不同歌曲长度不同，批训练时自行做 padding，并将 padding 掩码与标签 `mask` 合并。

需要重新准备时写入一个**尚不存在**的新目录，旧结果不会被覆盖：

```powershell
.\venv\Scripts\python.exe tools/datasets/prepare_deam.py --output data/datasets/deam/prepared/v2
```

`--root` 可指定其他数据根目录，`--output` 必须位于对应根目录的 `prepared/` 内。处理成功后才发布目标目录，中途失败会留下 `*.pending-*` 审计目录。逐文件生成 SHA256、严格解码音频、解析所有特征行；现有版本代码与模型不参与此流程。

DEAM 手册声明非商业用途的 CC BY-NC 条款，并要求引用其论文；附带特征另有 openSMILE 引用要求。数据保留在本地，后续发布或使用时应查阅 [原始发布说明](https://cvml.unige.ch/databases/DEAM/manual.pdf)。

更多资源与接入顺序见 [RESOURCES.md](RESOURCES.md)。

PMEmo、Memo2496 和 MERP 的实际下载、校验与接续训练见 [SUPPLEMENTAL.md](SUPPLEMENTAL.md)。其中 MERP 暂时隔离，动态训练使用通过校验和跨库重复筛查的 PMEmo 与 Memo2496 样本。
