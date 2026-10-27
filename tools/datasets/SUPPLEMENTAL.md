# 补充数据与自动接续训练

所有下载、缓存、工具依赖、解压音频和训练结果均保存在 `D:\2_datas\Vm` 内。原始包和派生文件由现有 Git 忽略规则排除。

## 本次接入

| 数据 | 实际发布文件 | 动态标签处理 | 用途 |
| --- | --- | --- | --- |
| [PMEmo](https://github.com/HuiZhangDB/PMEmo) | 作者 Google Drive 的 PMEmo2019.zip，794 首副歌音频 | 767 首有动态标签；CSV 的 `frameTime` 原样保留，通常从 15.5 秒开始；V/A 从 `[0,1]` 转为 `[-1,1]` | 通过去重和划分后加入连续监督训练 |
| [Memo2496](https://figshare.com/articles/dataset/Memo2496/25827034) | 官方 Figshare 第 8 版，2,496 首原始音频及均值、标准差和原始评分 | 原生 **1 Hz**，使用 CSV 的 `sample_0ms` 等时间戳；缺失值和音频之外的标签保留独立掩码 | 通过去重和划分后加入连续监督训练 |
| [MERP](https://huggingface.co/datasets/amaai-lab/MERP) | 固定发布修订的 54 首 WAV 和 Parquet 标注 | 保留原始文件，并按采样索引聚合；不人为指定绝对时间原点 | 暂存隔离，当前训练不读取 |

PMEmo 的静态均值单独存储，不铺满时间轴。Memo2496 的原始人工评分逐值重建了 784,582 个均值和同等数量的总体标准差，与发布 CSV 一致。模型仍在 0.5 秒特征网格上运行，Memo2496 的半秒位置没有人工监督，不插值生成真值。

本机完整解码后，PMEmo 794 首均通过；Memo2496 的 827、920、1095、1569、1773 五个 ID 解码失败，原始音频保留，但不进入候选清单。剩余 2,491 首为后续筛查候选，不等同于最终新增训练数量。原始标签中超出实际音频长度的值被置为无效掩码并记录，未平移或拉伸时间轴。

MERP 的公开代码从前部裁剪多余标注，而论文描述从末尾裁剪；数据未保存播放时钟时间戳。其 `prepared/quarantined-v1` 只有采样索引、均值和有效评分者计数，`audio_time_origin_seconds=null`、`training_enabled=false`。已知 DEAM 锚点为 115、343、745、1334。依据：[处理代码](https://github.com/dorienh/MERP/blob/master/processing/pruning.py)、[采集页面](https://github.com/dorienh/MERP/blob/master/amazon_Merged.html)。

## 下载与校验

```powershell
.\venv\Scripts\python.exe -u tools/datasets/fetch_supplemental.py pmemo
.\venv\Scripts\python.exe -u tools/datasets/fetch_supplemental.py merp
.\venv\Scripts\python.exe -u tools/datasets/fetch_supplemental.py memo2496
```

文件放在 `data/datasets/<数据集>/raw`。下载器固定发布修订、记录 URL、大小和本地 SHA256；MERP 核对发布方 SHA256，Memo2496 核对发布方 MD5。PMEmo 发布页没有独立校验值，因此以下载文件大小、ZIP 成员 CRC 和本地 SHA256 留存完整性证据。已有文件先复核，不静默覆盖；断点文件后缀为 `.part`。

PyArrow 仅安装在 `data/datasets/.dependencies`，用于读取 MERP；Chromaprint 位于 `data/datasets/.tools`，用于本地音频指纹计算，不向识曲服务上传音频。工具的发布地址、版本与哈希保存在同目录。下载的研究脚本作为资料保留，不执行。

## 数据准备与划分

以下默认目标已发布后不能再次覆盖，重复准备需指定新的输出目录：

```powershell
.\venv\Scripts\python.exe -u tools/datasets/prepare_pmemo.py --workers 3
.\venv\Scripts\python.exe -u tools/datasets/prepare_memo2496.py --workers 3
.\venv\Scripts\python.exe tools/datasets/prepare_merp_quarantine.py
.\venv\Scripts\python.exe -u -m tools.datasets.index_supplemental
```

前两项保留完整音频序列，严格解码、检查有限采样值、统计实际可解码帧数，并使用原始标签时间轴。单库清单先标为 `candidate`，混合训练只读取后续发布的 `data/datasets/combined/prepared/v1`。

合并阶段使用文件 SHA256、规范化歌手/曲名与允许时间偏移的 Chromaprint 音频指纹。与任意 DEAM 样本存在录音或元数据重叠的补充连通组整体排除；补充数据内部的重复录音及相同歌手/曲名绑定到同一集合。DEAM 原有划分不变，补充分组使用固定种子按约 80% / 10% / 10% 划分。最终数量见合并目录 `report.json`，排除原因和指纹证据分别在 `exclusions.json`、`duplicates.json`。

2026-10-03 发布的混合清单逐首回读验证通过，共 5,042 首、1,111,339 个有效维度标签值。跨库筛查排除 17 首补充候选，并记录 113 条指纹匹配关系；保留的匹配录音均绑定在同一划分。首次指纹提取超时的样本重试通过，本次最终没有指纹缺失样本。

| 数据集 | 训练 | 验证 | 最终测试 |
| --- | ---: | ---: | ---: |
| DEAM | 1,261 | 270 | 270 |
| PMEmo | 592 | 68 | 101 |
| Memo2496 | 1,983 | 246 | 251 |
| 合计 | 3,836 | 584 | 622 |

这是保守筛查，不能保证识别所有翻唱、重度剪辑或很短的重复片段。Memo2496 未提供完整歌手信息，不能宣称其划分实现了完整的歌手隔离。测试音频的指纹仅用于防止重复录音泄漏，不作为模型输入或选模依据。

## 自动接续

当前全量 DEAM 运行编号为 `teacher-20261002`。完成后，队列从它的 `best.pth` 初始化 `mixed-20261003`，使用 DEAM、PMEmo、Memo2496 的真实标签继续训练：

```powershell
.\data\training\next_va\env\Scripts\python.exe -u -m tools.train_next.continue_training --run-id mixed-20261003
```

队列在当前会话前台运行，记录位于 `data/training/next_va/queues/mixed-20261003`。它检查上游完成状态再启动接续训练；上游失败、中断或等待超过 12 小时会停止，不伪装成功。此方式不保证关闭应用或系统重启后仍继续，电脑休眠期间也不会训练。

接续阶段冻结 DEAM 训练集标准化器，保留预训练参数，学习率 `1e-4`；每轮按数据集等量抽样，小数据集轮换采样。最多 30 轮，8 轮未改善提前结束。各库单独报告 V/A 的 MAE、RMSE、CCC，并按数据集等权选模，同时要求 DEAM 的平均 CCC 相对初始化模型下降不超过 0.02；无合格改善时保留初始化模型为最佳检查点。

中断恢复可直接运行以下命令；不要与原队列或同一编号任务并发：

```powershell
.\data\training\next_va\env\Scripts\python.exe -u -m tools.train_next.mixed --run-id mixed-20261003 --resume
```

模型写入 `models/next_va/mixed-20261003`，指标和配置写入 `data/training/next_va/runs/mixed-20261003`。数据、源代码、标签、特征约定和初始化检查点均记录校验值。本轮不生成伪标签；只有真实监督教师完成评估后才开展伪标签校准。新模型不会自动替换现有应用的五维模型。

数据原始说明与许可随文件保留。PMEmo 的仓库代码许可不代表流行歌曲的再分发授权；MERP 音频采用逐曲 CC 条款；Memo2496 的 Figshare 发布标注为 CC BY 4.0，具体音源信息在 `songs_info_all.csv` 中。
