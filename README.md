# Music Visualization

音乐情绪驱动的粒子可视化项目：音频特征与 CLAP 嵌入 → BiLSTM 情绪预测 → MCTS 视觉规划 → 粒子渲染 → FFmpeg 音视频合成。1.0 与 2.0 分别维护，共用工作区的模型和素材目录。

| 版本 | 用途 | 代码 |
| --- | --- | --- |
| 1.0 | 命令行分析音乐并生成视频；可叠加 V/A 情绪轨迹 | [versions/v1.0](versions/v1.0/README.md) |
| 2.0 | Studio：分析、时间线编辑、修订、预览、多模式粒子与导出 | [versions/v2.0](versions/v2.0/README.md) |

## 目录

```text
versions/v1.0/              1.0 代码与测试
versions/v2.0/              2.0 代码、网页、测试与文档
models/emotion_model.pth    共享模型权重
assets/audio/demo/         原 demo 音乐
assets/audio/test/         原 testdemo 音乐
assets/audio/showdemo/     原 show/showdemo 音乐
data/v1.0/outputs/          1.0 视频，保留原素材子目录层级
data/v1.0/cache/            1.0 CLAP 缓存，保留原素材子目录层级
data/v2.0/projects/        Studio 项目、音频副本、预测、修订、缓存、视频、任务记录
data/v2.0/outputs/         Studio 历史批量导出结果
data/v2.0/validation-output/ 验证报告与产物
scripts/                   启动与整理验证工具
venv/                      本地 Python 环境
```

Git 仓库位于工作区根目录，并保留整理前的提交历史。两版算法文件分别维护，不跨版本导入代码。音乐、模型权重、视频、缓存、项目数据、运行工具及迁移备份不包含在仓库中；所需目录由程序按需创建或由使用者准备。
不要将 Studio 的 projects 目录拆散，也不要手工修改其中带哈希的修订文件。

## 环境准备

两版依赖记录来自 Python 3.13 环境。安装 Python 后，在仓库根目录创建环境并安装所需版本的依赖；依赖与目标平台的兼容性仍需自行验证：

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r versions/v2.0/requirements.txt
```

仅使用 1.0 时可改用 `versions/v1.0/requirements.txt`。将情绪模型权重放到 `models/emotion_model.pth`，音频放到 `assets/audio`；参见 [模型说明](models/README.md) 和 [素材说明](assets/README.md)。CLAP 首次运行可能需要从 Hugging Face 下载预训练模型。

正式 MP4 需要支持 H.264/AAC 的 FFmpeg。将其加入 PATH，或为 Studio 配置 `STUDIO_FFMPEG` / `versions/v2.0/.runtime/bin/ffmpeg.exe`。

启动器会检查现有环境，无法使用时显示明确提示，不自动安装依赖。也可通过 `MUSIC_PYTHON` 指定已准备好的 Python，例如在 PowerShell 中：

```powershell
$env:MUSIC_PYTHON = 'D:\你的环境\Scripts\python.exe'
& '.\Start V2.cmd'
```

启动器本身使用 PATH 上的 `python`。也可以直接用准备好的 Python 执行下面的版本入口。Windows 启动脚本之外，Python 入口支持直接从命令行调用。

## 1.0 使用

从工作区根目录运行，或将音频文件拖到 `Start V1.cmd`：

```powershell
& '.\Start V1.cmd' 'assets\audio\demo\song.mp3' --test-output
# 已激活可用环境时：
python versions/v1.0/main.py 'assets/audio/demo/song.mp3' --test-output
```

默认读取 `models/emotion_model.pth`；视频写入 `data/v1.0/outputs`，CLAP 缓存写入 `data/v1.0/cache`。
工作区素材沿用相对目录，因此迁移后的旧缓存可复用；外部音频按文件内容指纹区分同名文件。
可通过 `--model`、`--output-dir`、`--cache-dir` 指定路径。已有视频及分离输出不会被覆盖，会添加编号。
1.0 保留原有“指定模型不存在时训练”的行为；不希望训练时，请确认模型路径存在。

## 2.0 使用

准备好环境后双击 `Start V2.cmd`，浏览器打开 http://127.0.0.1:8765 。版本目录内的 `Start Studio.cmd` 也会调用同一入口。

```powershell
python versions/v2.0/main.py doctor
python versions/v2.0/main.py serve
python versions/v2.0/main.py analyze 'assets/audio/demo/song.mp3' --model 'models/emotion_model.pth'
python versions/v2.0/main.py render 'data/v2.0/projects/p-实际ID' --revision r000001
```

`serve`、`analyze` 和 `demo` 的默认项目位置为 `data/v2.0/projects`，可用 `--projects` 指定。
网页新建分析时填写音频和模型的完整路径。已有项目的音频副本、修订和视频随项目整体迁移。
已有本地迁移记录时，历史元数据中的原始文件路径仍表示当时的来源；分析任务重试和运动验收脚本可依据本地迁移清单解析实际搬迁过的文件。全新克隆不依赖此清单。
另一台机器上的旧来源路径若本就不存在，不会被虚构映射。

## 验证与维护

人工数据标注：双击根目录 **Start Annotation.cmd**，访问 http://127.0.0.1:8766 。工具只收集下一版的整首连续 V/A 标注与独立转折点，保存到 `data/annotations/continuous`；不会训练或覆盖现有模型。操作与标注标准见 [标注工具说明](tools/annotate/README.md)。

在有完整依赖的环境中，从相应版本目录运行原有测试。

```powershell
Set-Location versions/v2.0
python -m unittest discover -s tests -v
node scripts/check_math.cjs
node scripts/check_motion_math.cjs
```

浏览器验证脚本使用标准 `playwright` 模块，或由 `STUDIO_PLAYWRIGHT` 指定模块路径；`STUDIO_PYTHON` / `MUSIC_PYTHON` 可指定 Python。
浏览器检查仍需已启动的服务及可用 Edge。验证产物统一写入 `data/v2.0/validation-output`。
历史 outputs 内的批处理脚本、日志和 docs 内的验收记录保留原文，属于历史证据，不应直接当作当前启动命令。

## 本次整理的验证范围

本地数据迁移校验、旧项目与修订读取、历史媒体接口、临时副本修订保存、Python/JavaScript 语法及前端数学检查通过。选取的 31 项 Python 测试中 29 项通过，2 项因验证环境缺少 soundfile 失败；存储测试也被该依赖阻塞。
以上是目录整理当时的结果。后续本地环境修复（2026-09-30）已将原 venv 连接到已安装的 Python 3.13.15，并复用原依赖；主要依赖导入、pip check 和 19 项相关测试通过。
CLAP 模型已准备到本地缓存；使用真实音乐的 2 秒片段，1.0 完成 1280×720/30 FPS 视频生成，2.0 完成 CLAP 分析、五维预测和 flow-v1 的 320×180/10 FPS 视频生成，两份输出均通过音视频解码检查。Studio 启动入口及 HTTP 接口也通过检查。长歌曲与全部编辑功能不在此次短流程验证范围。
启动器自动为两个版本配置已有本地 FFmpeg 和 CLAP 缓存，不改变系统环境变量。缓存、venv 和模型仍不包含在 GitHub 仓库中。

本地迁移备份与记录保留在 `archive/`，不上传到 GitHub。
