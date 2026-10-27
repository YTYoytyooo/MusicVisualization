# 下一版 V/A 教师模型训练

独立于现有 1.0/2.0 运行入口。先训练 DEAM 真实标注基线，再用通过校验的补充真实标签接续训练；当前不生成或混入伪标签，不写入 `models/emotion_model.pth`。

## 训练内容

- 固定使用 `data/datasets/deam/prepared/v1` 的歌手分组划分。
- 每首歌保留一条完整时间序列，0.5 秒采样一次。CLAP 使用以当前时间为中心的 2 秒上下文，边缘补零；音频平均为单声道并按编码器需要重采样。这是离线双向模型，不是实时因果模型。
- 输入为冻结 CLAP 的 512 维嵌入及 19 维声学特征；不读取 DEAM 附带的时间对应关系未确认的特征 CSV。
- 标签通过时间戳精确对齐，首 15 秒无真实监督，缺失维度与批次 padding 分别保留掩码。
- 按训练集拟合标准化器。一层双向 LSTM（单向隐藏维度 128）输出 `[valence, arousal]`，范围 `[-1, 1]`。
- 使用按歌曲、维度平均的 Huber + 0.2 × CCC 损失；常量目标的 CCC 项跳过。验证指标为逐歌 MAE、RMSE、CCC 及歌曲等权平均，常量目标不计入 CCC 均值。
- AdamW，学习率 `3e-4`、权重衰减 `1e-4`，梯度裁剪 1。验证集平均 V/A CCC 用于选择检查点，耐心值 8 轮。测试集不参加训练、特征提取或模型选择。

## 独立 GPU 环境

本机检查到 RTX 5080 Laptop GPU；原 `venv` 使用 CPU 版 PyTorch。下面的脚本在 `data/training/next_va/env` 新建环境，安装 PyTorch 2.11.0 CUDA 12.8 和锁定依赖，不修改原环境。CUDA 安装命令依据 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v2110)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/train_next/setup_gpu.ps1
```

`ExecutionPolicy Bypass` 仅影响此进程，用于本机默认禁止脚本执行的情况，不设置系统策略。依赖下载会占用数 GB 空间。中断后继续安装使用 `-Resume`；脚本最后执行 `pip check` 和实际 GPU 张量运算，失败会中止。

## 首轮试训

从工作区根目录运行：

```powershell
.\data\training\next_va\env\Scripts\python.exe -m tools.train_next.run --stage pilot --run-id pilot-20261002 --device cuda
```

选择 20 首训练歌和 8 首验证歌，固定随机种子并包含各集合的最长歌曲，训练 5 轮。用于验证真实音频特征提取、长序列、显存、梯度及保存恢复；试训分数不能代表正式精度。

```text
data/training/next_va/
  env/                         独立 Python 环境
  pip-cache/                   依赖下载缓存
  features/<配置指纹>/         按源音频 SHA256 保存的特征
  runs/<run-id>/
    config.json                固定配置、数据与代码校验值、选中歌曲 ID
    environment.json           Python/PyTorch/GPU 和旧模型 SHA256
    status.json                当前阶段或错误信息
    metrics.json               每轮训练损失和验证指标
    mean-baseline.json         训练标签均值预测的验证基线
    report.json                成功结束后才生成的报告
models/next_va/<run-id>/
  best.pth                     验证集选出的模型、标准化器和特征约定
  last.pth                     最近完成一轮的状态，包含优化器及随机数状态
```

缓存按源音频、编码器权重/配置、依赖版本和特征代码区分。已完成歌曲可以复用；未完成的当前歌曲会重新提取。所有生成文件均在已有 Git 忽略目录内。

## 已完成的首轮试训

`pilot-20261002` 已在 RTX 5080 Laptop GPU 上完成：20 首训练、8 首验证、5 轮，包含特征提取共 168.5 秒（不含环境安装）。原模型 SHA256 保持不变。重新加载 `best.pth` 后，对全部 8 首验证歌曲重新计算的指标与训练报告完全一致，记录位于运行目录的 `reload-verification.json`。

| 维度 | 验证 MAE（越低越好） | 验证 CCC（越接近 1 越好） | 训练标签均值基线 MAE |
| --- | ---: | ---: | ---: |
| 效价 | 0.3017 | 0.0044 | 0.2500 |
| 唤醒 | 0.1428 | 0.0959 | 0.1670 |

当前仅证明训练流程可运行：效价误差仍高于均值基线，两个维度的时间曲线一致性都较弱，不能作为可用教师生成伪标签。下一步是全量 DEAM 监督训练和验证。此运行编号已存在，重新试训请换用新编号；不要把检查点直接接入现有应用。

## 中断恢复与正式训练

保留相同参数并增加 `--resume`。只从完整检查点恢复，不覆盖原有独立训练记录；数据、参数或代码变化时拒绝恢复，改用新的 `run-id`。

```powershell
.\data\training\next_va\env\Scripts\python.exe -m tools.train_next.run --stage pilot --run-id pilot-20261002 --device cuda --resume
```

确认试训流程与耗时后，可单独开始全量监督训练，使用 1,261 首训练歌、270 首验证歌，最多 50 轮：

```powershell
.\data\training\next_va\env\Scripts\python.exe -m tools.train_next.run --stage full --run-id teacher-20261002 --device cuda
```

不要并发运行同一 `run-id`。`--device cpu` 可用于小型诊断，不会自动替代请求的 GPU 模式。训练程序只加载已有本地 CLAP，不自动下载模型。以上命令为前台有限任务，可通过 Ctrl+C 中断。

## 验证与当前边界

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tools/train_next/tests -v
```

单元测试包含真实优化器训练及保存/恢复，但使用小型合成数据，不是 DEAM 精度验证。正式结果以各次运行的 `report.json` 为准。

当前检查点是下一版两维序列模型，不能直接通过旧五维模型加载器读取。下一阶段再做与旧模型的公平对照、教师可信度校准、额外音乐去重及伪标签学生训练，并实现可视化接口的兼容加载。

## 补充数据接续训练

全量 DEAM 运行 `teacher-20261002` 与补充数据准备同步进行。已新增从教师检查点接续训练的 `mixed.py` 和等待上游完成的 `continue_training.py`，接续编号为 `mixed-20261003`。下载、原生时间轴、跨库重复筛查、采样策略、选模限制与恢复命令见 [补充数据与自动接续训练](../datasets/SUPPLEMENTAL.md)。

## 2026-10-03 恢复与定时检查

原全量运行在完成 1,018 / 1,531 首特征后因 Windows 状态文件替换失败而退出，尚未开始参数训练。修复后使用 `teacher-20261003-recovery` → `mixed-20261003-recovery` 两个新编号，保留旧日志并复用已有特征缓存。状态写入使用独立临时文件和有限重试；最终仍失败时只记录错误，不终止计算。配置、报告和模型检查点写入失败仍会明确报错。

每个训练进程持有操作系统文件锁，退出后自动释放；接续队列据此区分正在训练与已经退出但状态陈旧的任务。当前运行编号、日志和恢复命令记录在 `data/training/next_va/active-training.json`。手动读取状态：

```powershell
.\data\training\next_va\env\Scripts\python.exe -m tools.train_next.status
```

定时检查通过本对话的 Codex 自动任务每 15 分钟进行，确认两阶段完成并验证检查点后暂停。检查期间需要电脑保持开机、联网并运行 Codex，避免睡眠。恢复任务前必须确认没有活跃的同编号进程；已有配置时保留相同参数并加 `--resume`，不得跳过代码和数据一致性校验。新增程序采用文件锁，旧版未加锁进程必须通过进程命令行和启动时间另行核实。

## 代表歌曲曲线诊断

`diagnostics.py` 只读分析已完成的 `mixed-20261003-recovery/best.pth`，校验检查点、训练代码、标签、原音频和特征版本，要求已有缓存齐全。复算全部 584 首验证歌曲并与检查点逐曲指标比较，再每库选取 6 首代表；不读取测试标签，不提取新特征，不训练或修改模型。

```powershell
$env:PYTHONUTF8 = '1'
.\data\training\next_va\env\Scripts\python.exe -m tools.train_next.diagnostics --output data/training/next_va/diagnostics/mixed-20261003-curves --device cuda
.\venv\Scripts\python.exe -m tools.train_next.render_diagnostics data/training/next_va/diagnostics/mixed-20261003-curves
```

采集输出目录必须不存在；重复诊断使用新目录，渲染可复用已保存的 `diagnostics.json`。采集使用独立 GPU 环境；渲染使用已有 Matplotlib 的原环境，不为画图修改训练环境依赖。输出为离线 `index.html`、19 页 PDF、统一／局部纵轴 PNG、全库指标 CSV、代表歌曲原始时间序列 CSV。网页支持数据集／类型筛选和纵轴切换，图像内嵌，无外部网络资源。

抽样规则为双维真实变化与 CCC 的库内分位：典型、较平稳、变化明显、拟合较好、拟合较差和最长未选歌曲。均值与振幅仅在真实标注位置计算；P90 振幅是 P95−P5，标准差比反映相对波动。时延扫描比较 `y(t)` 和 `p(t+lag)`，正值表示预测相对标注较晚；每种移位使用相同的中间真实标注点，不插值、不校正正式指标。低波动、低相关、边缘峰等启发式标记不是统计置信区间。全部结果只支持诊断，不用于确定统一标签平移量。

```powershell
.\data\training\next_va\env\Scripts\python.exe -m unittest tools.train_next.tests.test_diagnostics -v
```
