> 目录整理（2026-09-30）：代码现位于 `versions/v2.0`；项目与输出位于工作区 `data/v2.0`，模型位于 `models`。下文旧绝对路径及历史验收记录保留作为来源记录，当前启动与路径说明见工作区根目录 `README.md`。

# MusicVisualization Studio 2.0

本地音乐可视化编辑工作台：分析一次 → 保存数值 → 修改时间线 → 保存修订 → 再生成视频。

本目录是独立升级版。旧目录 `../MusicVisualization`、外部模型和音乐不会被覆盖。
当前为开发验收版；详见 [实施计划](docs/IMPLEMENTATION.md) 和 [验收记录](docs/VERIFICATION.md)。
不足、已完成优化和后续优先级见 [优化清单](docs/OPTIMIZATION_PLAN.md)。
最新任务控制、时间轴、局部预览、同步对比、检查点与可选平滑见 [优化版工作台说明](docs/UPGRADE_WORKBENCH.md)。
新增的情绪驱动多模式粒子、人工运动区间和运动数值导出见 [多模式运动工作台说明](docs/MOTION_WORKBENCH.md)。

## 开始使用（Windows）

双击 **Start Studio.cmd**，然后在浏览器打开 <http://127.0.0.1:8765>。
启动器复用 `D:\2_datas\Vm\venv`，不会重建虚拟环境。换电脑时修改启动器中的 Python 路径。
如果已经启动服务，直接打开网页，不要重复启动。

1. 新建分析项目：填写音频、模型的完整路径，例如 `D:\2_datas\Vm\happy_adveture.mp3`、`D:\2_datas\Vm\emotion_model.pth`。
2. 等待分析任务完成，刷新/选择左侧项目。模型不存在会报错，不自动训练。
3. 先生成原始修订的视频，或直接编辑曲线。
4. 在右侧选择情绪/视觉字段、时间、目标值和过渡时间：输入时曲线立即预览，离开输入框或按 Enter 确认这一步。也可用“添加修改 / 确认调整”按钮。
5. 可以先生成局部预览并与已有参考视频对比；满意后点击“保存为新修订”，再点击“生成当前已保存版本”。
6. 在输出选择框查看修订视频、下载逐帧数值；原始版本始终保留。

界面会明确显示“草稿已修改”和当前视频所属版本。修改曲线不会自动修改已经生成的视频。
可编辑已有操作、启用/禁用、删除，也可以把历史版本恢复为一个新修订。
通过已选情绪关键帧拖动调整时间/目标值；同字段冲突、越界时间、非法数值会拒绝。
草稿只缓存在本机浏览器；正式修订保存于项目目录。
草稿可撤销/重做，最多50步；支持 Ctrl/Cmd+Z、Ctrl/Cmd+Shift+Z 和 Ctrl+Y。
输入框内保留浏览器原生撤销；切换项目、修订或保存后清空草稿操作历史。
新修改默认使用更平滑的五次过渡，可切换线性或三次过渡；旧修改仍按原线性方式解释。
Esc 或“取消本次调整”放弃尚未确认的输入。首次查看视觉参数需等待真实规划完成；
已有视觉修改后再修改情绪，也会自动准备新规划并继续校验，不需要先删除视觉修改。
完整操作与平滑边界见 [即时编辑说明](docs/LIVE_EDITING.md)。

## 多模式粒子运动

旧项目及未配置运动层的修订默认使用 `legacy`，不会自动改变原有效果。
选择项目后，在右侧“运动形态与区间覆盖”中，将“运动引擎”切换为“多模式 · 情绪驱动”（`flow-v1`）。
中间下方的“粒子运动轨道”显示自动轨道、生效轨道，以及当前时间的模式混合权重和参数。

- 支持上升、下沉、环流、螺旋、扩散、汇聚、流星、波浪和湍流九种形态。
- 自动选择使用当前生效 V/A 与真实音频的强弱、频谱变化；这是美术规则映射，不是模型识别出的运动类别，也不是分类置信度。
- 点击“＋ 添加运动区间”，指定模式、时间和进入/退出过渡；输入即时更新轨道，离开输入框、Enter 或“确认运动调整”确认一步。Esc 取消未确认调整，已确认草稿可撤销/重做。
- 点击“生成局部预览快照”会冻结当前草稿，不创建正式修订；满意后“保存为新修订”，再生成已保存版本。即时轨道反馈不是实时视频生成。
- 单独调整运动层不重新运行 CLAP，也不重新计算视觉 MCTS。实际生成视频仍使用视觉规划的颜色、亮度、粒子数量等；没有可用视觉缓存时，生成流程会先准备它。

速度按画面短边/秒、半径按短边比例表达，拖尾最多 2 秒。过渡时的真实控制可能含多个模式，不能把导出的主模式参数当成完整混合结果。
参数范围、人工覆盖边界、数值文件和复现限制请见 [完整操作说明](docs/MOTION_WORKBENCH.md)。

## FFmpeg / 正式 MP4

正式输出需要支持 `libx264` 和 `aac` 的 FFmpeg。运行：

```bat
"D:\2_datas\Vm\venv\Scripts\python.exe" "D:\2_datas\Vm\show\MusicVisualizationStudio\main.py" doctor
```

支持系统 PATH、`STUDIO_FFMPEG` 指定完整路径、本目录 `.runtime/bin/ffmpeg.exe`、
`.runtime/imageio_ffmpeg/binaries`，以及本机可访问的WinGet安装位置。
本机已将现有FFmpeg复制到 `.runtime/bin`，避免不同启动环境的访问权限差异；未改系统安装。
依赖准备脚本仅会在显式执行 `scripts/validate_release.py --prepare-ffmpeg` 时安装隔离的
`imageio-ffmpeg` 到 `.runtime`，不会修改已安装的项目依赖。没有编码器时提前报错，不冒充生成成功。

## 不使用界面的命令

在本目录并激活原来的虚拟环境后：

```bat
python main.py doctor
python main.py analyze "D:\2_datas\Vm\happy_adveture.mp3" "D:\2_datas\Vm\battleThemeA.mp3" --model "D:\2_datas\Vm\emotion_model.pth"
python main.py edit "projects\p-实际项目ID" --base r000000 --edits examples\edits.json
python main.py edit "projects\p-实际项目ID" --base r000001 --edits examples\edits.json --motion examples\motion.json
python main.py render "projects\p-实际项目ID" --revision r000001 --mode analysis
python main.py render "projects\p-实际项目ID" --revision r000001 --mode presentation --start 10 --end 20
```

普通示例编辑仅适合至少4秒的歌曲；`examples/motion.json` 的环流覆盖含过渡，需要至少18秒。
`--edits` 应提供想保留的完整修改列表；`--motion` 省略时继承基准修订的运动配置。示例中的修订号需替换为实际当前修订。
项目 ID 由分析命令返回。`--start/--end` 仍会从歌曲开头推进粒子状态，
首次没有匹配检查点时，精确片段预览并非立即完成；已有同配置检查点可从较早状态恢复。非整帧的请求按帧边界扩展，实际起止时间记录在 `render.json`。

纯显示演示（合成音调和人为数值，不是真实模型预测）：

```bat
python main.py demo --duration 8
```

## 文件与数值

每首歌独立放在 `projects/p-*/`。音频复制到新项目，原文件不改。

| 文件 | 意义 |
|---|---|
| `project.json` | 音频/模型哈希、采样与时间对齐信息、种子、当前修订 |
| `analysis/predictions_raw.npz` | 不可覆盖的五维原始预测，读取时校验哈希 |
| `analysis/predictions_raw.csv` | 每0.1秒的原始预测查看表 |
| `analysis/waveform.json` | 界面波形缩略数据 |
| `revisions/rXXXXXX/revision.json` | 不可变修改快照，含父版本和校验值 |
| `plans/*/plan.json` | 自动视觉规划；视觉单独修改复用缓存 |
| `motion_features/*.npz` / `*.json` | 从真实音频提取的运动驱动特征缓存，不替代原始模型预测 |
| `previews/s-*/snapshot.json` | 不覆盖正式修订的冻结草稿预览 |
| `checkpoints/*/*.json` / `*.npz` | 同配置恢复所需的完整模拟状态与哈希 |
| `renders/v-*/predictions_effective.csv` | 生成前保存的整首生效情绪时间线 |
| `renders/v-*/visual_plan.csv` | 生成前保存的整首视觉参数计划 |
| `renders/v-*/motion_plan.json` / `motion_plan.csv` | flow-v1 的自动/生效运动轨道和整首逐帧运动控制计划 |
| `renders/v-*/frame_values.csv` | 真正提交给视频渲染器的每帧值，仅包含输出片段 |
| `renders/v-*/render.json` | 修订、环境、实际范围、完成状态、输出哈希 |
| `renders/v-*/output.mp4` | H.264/AAC 带音轨输出 |
| `.jobs/` | 本机队列、日志、取消信号 |

数值表区分 `*_raw`、`*_effective` 和 `visual_*`，不要混用。
flow-v1 另有 `motion_*`、真实音频特征和节拍控制列；`motion_components_json` 才是完整混合控制，运动标量列只是当前权重最大的模式参数。
V/A 编辑影响自动规划；Energy、Tension、Brightness 当前只读参考，不参与新增控制。
视觉 `brightness` 与模型预测 `brightness` 不是同一个参数。

## 时间、修改和复现边界

- 编辑时间内部使用整数微秒；视频时间由帧号直接计算，避免累积漂移。
- 原始预测保持采样值；人工影响权重支持线性、三次 smoothstep、五次 smootherstep。单点需要前后非零过渡，且完整影响范围必须在歌曲内。
- 平滑的是人工修改的进入/退出权重，不会改写原始预测，也不保证消除原始每0.1秒采样跳变。
- 另有默认关闭的渲染基线平滑，开启后仅处理V/A并保留原始值；人工编辑在平滑之后应用，设置随修订保存。
- 区间固定/偏移包含两端目标时刻，过渡在区间外；零过渡意味着突变。
- 同字段重叠编辑拒绝，不能悄悄覆盖。偏移造成越界会报错，不静默裁剪。
- Hue 走色环短弧；粒子数量/拖尾在最终送入渲染器时取整。
- 修改某段后，粒子位置和MCTS历史可能影响后续画面；“数值作用区间不变”不等于“后续像素不变”。
- MCTS 与粒子使用独立种子。相同环境可回归，跨库版本/硬件不承诺逐字节视频一致。
- 新版推理窗口包含当前 embedding，修复旧版一帧偏移；旧项目原始结果不迁移覆盖。
- CLAP 使用从当前时间起的2秒音频，属于离线分析，不能标为因果实时识别。
- 图表保留共同缩放、全范围小图和最近2秒高亮；人工修订明确标注，不代表模型准确率提高。
- 默认1280×720、30FPS。低分辨率/不同FPS输出标记为近似草稿，不承诺与正式视频逐像素相同。
- 尾部视频按整帧向上取整，音频补静音至片段时长，不丢弃最后数值行。

## 恢复与安全

- 只监听 `127.0.0.1`，POST需要会话令牌并检查来源；音乐不上传。
- 同一项目采用写锁和基准版本校验。新修订不会改正在运行任务的快照。
- 取消在阶段/帧检查点响应；CLAP某次调用期间可能需要等待当前调用结束。
- 关闭网页不停止工作进程。服务重启会核对进程身份，失效任务标记 interrupted。
- 写锁由操作系统持有，进程异常退出后自动释放；`.write.lock` 文件保留仅作诊断，不需要删除。
- 新项目在隐藏 `.pending-p-*` 目录中准备，全部保存成功后才出现在列表；失败暂存文件保留，不自动删除。
- 有声MP4失败时保留无声AVI、数据表和编码日志；失败结果不当作成功输出。
- 不向远程Git推送，不删除旧项目，不自动训练或上传个人音频。

## 验证

```bat
python -m unittest discover -s tests -v
node --check web/app.js
node scripts/check_math.cjs
node scripts/check_motion_math.cjs
node scripts/browser_live_check.cjs
node scripts/benchmark_browser_preview.cjs
python scripts/verify_transitions.py
python scripts/verify_motion_release.py --source projects\p-已有真实分析项目ID
python scripts/validate_release.py --prepare-ffmpeg --model "D:\2_datas\Vm\emotion_model.pth" --audio "D:\2_datas\Vm\happy_adveture.mp3" "D:\2_datas\Vm\battleThemeA.mp3"
```

浏览器检查需已启动本机工作台；使用本机 Edge/Playwright 路径。实时编辑回归会创建独立合成测试项目，性能检查使用合成接口且不修改项目。
过渡验收会在 validation-output/transitions 下生成合成音频视频，不重新进行模型推理。
运动专项创建独立验收项目、启动独立测试服务，复用指定项目的预测而不重复运行模型；它不会改动源项目。浏览器检查使用本机 Edge/Playwright 路径。
运动专项及最后一项需要较长时间；最后一项还可能下载隔离编码器。适合后台任务，Windows 当前不支持 Codex Process Jobs 时需明确采用前台执行。
