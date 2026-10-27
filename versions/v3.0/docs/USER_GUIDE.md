# V3 研究预览使用指南

V3 使用现有 B01 与 B00 权重分析整首音乐，并可选择计算旧 V2 对照。它不会训练模型、接入新标注或使用封存的最终测试集。当前交付身份是研究预览，正式模型替换门槛仍未通过。

## 启动与日常使用

1. 在 `D:\2_datas\Vm` 双击 `Start V3.cmd`，保留启动窗口。
2. 浏览器打开 `http://127.0.0.1:8768`。服务只监听本机。
3. 从音频库选择已登记的已有音频。选择 GPU 或 CPU；GPU 不可用时明确改选 CPU，不会偷偷切换设备。
4. 如需对照旧版，勾选“同时计算旧 V2 对照”。这会调用旧版原始 CPU 特征与推理，可能明显增加等待时间。
5. 点击“创建分析项目”，观察解码、特征、推理和保存阶段。完成后播放音频，点击曲线定位，使用 A/B 控件设置局部循环。
6. 查看 B01、B00、旧 V2 曲线。选择“显示曲线”才能启用显示平滑；原始文件保持独立不变。
7. 可下载原始 CSV、显示 CSV、项目 JSON。网页短视频范围最长 30 秒；命令行/API 的底层上限为 60 秒。每次导出到新的目录，使用原音频与固定粒子艺术映射。
8. 重新启动后在项目列表中打开已有项目。完整结果会先检查内容与来源凭证，损坏或不完整结果不会显示为可用。

没有勾选旧 V2 的项目会明确显示其未计算。改变对照选项应创建新项目，不能将已有结果伪装成另一组模型输出。

## 如何读这些数值

| 字段 | 含义 | 边界 |
| --- | --- | --- |
| 效价 V | 模型预测的负面到正面情绪轴，范围 −1 至 +1 | 不是客观情绪真值 |
| 唤醒 A | 模型预测的平静到激动情绪轴，范围 −1 至 +1 | 不等同于声音响度 |
| 能量 RMS/dBFS | 音频局部幅度的声学测量 | 不是 LUFS，也不是心理唤醒分数 |
| 明亮度／频谱重心 Hz | 频谱能量重心的声学测量 | 与画面亮度控制分开 |
| 三种子分歧 | 三个固定种子模型的样本标准差 | 不代表准确概率或可靠性区间 |
| 校准与紧张度 | 当前不可用 | 不会用 0 或猜测值冒充结果 |

B01/B00 显示三种子等权预测集成，保留各成员原始输出。它与“分别评估三个种子，再平均指标”是不同口径。V/A 的原始时间步长为 0.5 秒；旧 V2 保留其 0.1 秒时间轴。播放时的插值、首尾端点保持和显示平滑仅服务于预览，不会改写原始采样点。

当前模型使用双向整曲上下文，是离线分析。输出点间隔不代表实时识别延迟。视觉粒子变化是一种明确的艺术映射，不能用画面是否好看来判断情绪预测精度。

## 文件位置与恢复

| 内容 | 位置 |
| --- | --- |
| 启动器 | `D:\2_datas\Vm\Start V3.cmd` |
| 固定模型清单 | `D:\2_datas\Vm\models\next_va_v2\v3-preview-v1\manifest.json` |
| 已有音频清单 | `D:\2_datas\Vm\data\v3.0\library.json` |
| 项目、原始结果、导出 | `D:\2_datas\Vm\data\v3.0\projects` |
| 任务原始命令、环境与日志 | `D:\2_datas\Vm\data\v3.0\jobs` |
| 特征与旧 V2 结果缓存 | `D:\2_datas\Vm\data\v3.0\cache` |

每个项目的 `analysis/result.json` 是原始结果，`analysis/receipt.json` 是最后写入的完整性凭证。不要手工修改这两份文件或挪走项目内的元数据。项目引用清单中的原音频，移动或替换音频后会明确拒绝复用旧结果。

取消是协作式的，会在安全检查点生效。等待任务显示已取消或中断，再使用“恢复任务”。恢复保留原配置、命令与匹配缓存；代码、模型、音频发生变化时会拒绝恢复。校验失败或孤立的半成品应保留并检查，不能靠删除凭证或放宽校验继续。

## 命令行复现

以下命令都在 `D:\2_datas\Vm` 的 PowerShell 中执行。仅使用已经存在的独立环境，不需要安装依赖。`AUDIO_ID`、`PROJECT_ID`、`JOB_ID` 是清单、项目列表或任务输出中的实际标识，需替换后运行。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py doctor
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py status
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py serve --port 8768
```

在独立终端中按需运行单个有限任务：

```powershell
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py analyze AUDIO_ID --device cpu
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py analyze AUDIO_ID --device cuda --include-v2
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py export PROJECT_ID --start 0 --end 15 --smoothing 1
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py resume JOB_ID
```

只读验证已有结果：

```powershell
& '.\data\training\next_va\env\Scripts\python.exe' -m tools.train_next_v2.v3_inference verify --project 'data/v3.0/projects/PROJECT_ID'
```

模型和音频清单只在首次准备时由明确的 `prepare` 命令创建。已有清单需要通过校验，不会被静默刷新；日常启动不需要重新准备。

```powershell
& '.\data\training\next_va\env\Scripts\python.exe' versions/v3.0/main.py prepare
```

## 如何理解验收结论

独立合同测试覆盖路径、HTTP、身份凭证、缓存、时间轴、显示平滑及失败恢复边界；运行命令见 `versions/v3.0/tests/README.md`。真实推理一致性、浏览器使用、实际视频解码、速度及内存/显存需要对应的运行报告，不能仅凭单元测试通过判断。

已有验证数据表明 B01 整体优于旧 V2，但 PMEmo 唤醒维度仍存在退化，动态预测也并非普遍优于自身均值输出。V3 工程预览不修改这一研究结论。622 首最终测试继续封存；没有自动生成的“人工试听通过”结论。
