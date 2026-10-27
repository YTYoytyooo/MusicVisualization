# V3 研究预览工程验收结果

验收日期：2026-10-15。交付状态：`engineering_preview_complete`。

独立 V3 应用的自动工程验收已完成，未发现尚未解决的实质交付阻塞。模型仍为研究预览，未提升为正式模型。本次仅使用现有权重和已登记音频，没有新数据训练、标签拟合或最终 622 首测试集评估；没有把 A—F 研究计划标记为全部完成。

本报告依据 [delivery.json](D:/2_datas/Vm/data/v3.0/reports/delivery.json) 及其登记的原始证据。报告中的 Pass 表示指定范围内已核验；Partial 表示已有证据但覆盖不足。人工试听和情绪主观正确性不属于自动通过结论。

## 交付与模型身份

入口为 [Start V3.cmd](<D:/2_datas/Vm/Start V3.cmd>)，页面地址为 [http://127.0.0.1:8768](http://127.0.0.1:8768)。具体操作和可复现命令见 [使用指南](D:/2_datas/Vm/versions/v3.0/docs/USER_GUIDE.md)。

[冻结清单](D:/2_datas/Vm/models/next_va_v2/v3-preview-v1/manifest.json) 的 SHA256 为：

```text
293452709ed3caa8c9ac596bf68e1c66e0bb7308cbab6437254bea6801c6180b
```

B01/B00 使用固定三种子的等权预测集成，保留成员原始输出；三种子分歧为样本标准差，不能解释为准确概率。旧 V2 为可选的原始 CPU 对照。V/A 保留 0.5 秒时间轴，V2 保留 0.1 秒时间轴；显示插值与平滑不改写原始值。RMS/dBFS 和频谱重心 Hz 属于声学测量，校准与紧张度仍不可用。

原有逐库逐维模型门槛仍未全部通过，PMEmo 唤醒退化和动态预测不足仍是研究限制。推理一致性证明工程实现保留冻结模型的输出，不证明模型新增精度，也不构成正式替换旧版的依据。

## 实测结果

| 项目 | 已核验结果 | 证据 |
| --- | --- | --- |
| 独立合同测试 | 56/56 通过，用时 7.499 秒；包含最终待处理项目与 HTTP 变更 | [测试日志](D:/2_datas/Vm/data/v3.0/reports/unit-tests.log) |
| 已有音频推理 | 24/24 完成；20 MP3、4 FLAC；全部双声道；22 首 44.1 kHz、2 首 48 kHz；时长 28.029–401.032 秒 | [engineering-v2](D:/2_datas/Vm/data/v3.0/reports/engineering-v2/report.json) |
| 原始推理一致性 | 70/70 检查通过；18 首历史 B01、3 首历史 B00、1 首旧 V2、3 首 CPU/GPU 对比；最大绝对差 5.0664×10⁻⁷，沿用 rtol=10⁻⁵、atol=10⁻⁶ | [一致性报告](D:/2_datas/Vm/data/v3.0/reports/runtime-equivalence.json) |
| 真实任务恢复 | 观察到真实进程身份；竞争任务被拒绝；取消后进程退出；按原配置恢复完成；恢复前后 launch SHA256 相同 | [任务恢复报告](D:/2_datas/Vm/data/v3.0/reports/job-recovery.json) |
| 浏览器工作流 | 网页创建项目并观察 pending→completed，模型切换、原始/显示切换、2 秒平滑、播放/暂停、定位、A/B 循环、导出及已有项目重开 | [最终浏览器记录](D:/2_datas/Vm/data/v3.0/reports/browser-final.json)、[播放观察](D:/2_datas/Vm/data/v3.0/reports/browser-observation.json) |
| 页面数值 | 与独立平滑及插值参考相比，最大差 0.000449，处于三位小数显示误差 0.0005 以内 | [交付记录](D:/2_datas/Vm/data/v3.0/reports/delivery.json) |
| 原资产保护 | 97/97 项受保护文件 SHA256 前后相同，覆盖已登记的旧版本文件、入口及模型等保护对象 | [之前](D:/2_datas/Vm/data/v3.0/reports/protected-before.json)、[之后](D:/2_datas/Vm/data/v3.0/reports/protected-after.json) |

实际网页新项目为 `p-22cc95f60fa8-9eeb91f0`，任务为 `j-ffa71c31dda84b5e`；已重新打开含 V2 和导出记录的项目 `p-22cc95f60fa8-d1d8b430`。页面截图保存在 [studio-desktop.jpg](D:/2_datas/Vm/data/v3.0/reports/studio-desktop.jpg)。浏览器观察为自动操作证据，不等于有人听过音频。

24 首工程推理保留其执行时的代码绑定；后续 UI 项目登记、HTTP 拒绝响应和 CSV 换行修复由最终 56 项测试及实际浏览器/媒体检查覆盖。冻结模型与推理定义未因这些界面修复改变。各证据的时间、命令和校验值分别保留，没有将历史失败覆盖为成功。

## 速度和资源边界

设备记录为 NVIDIA GeForce RTX 5080 Laptop GPU，使用既有独立 Python 3.13 环境。以下为进程耗时，包含相应启动和保存开销；模型加载、解码、特征及推理阶段的细项见逐曲报告。

| 场景 | 实测 |
| --- | --- |
| engineering-v2 中 21 首已验证缓存复用 | 中位数 4.50395 秒，范围 4.11353–5.19364 秒 |
| engineering-v2 中 3 首新特征计算 | 22.7286、31.9709、39.2200 秒 |
| 首次冷特征尝试，保留于 engineering-v1 | `happy_adveture.mp3` 进程耗时 97.5506 秒，其中特征阶段约 92.3381 秒 |
| PyTorch GPU 峰值 | allocated 820.9067 MiB；reserved 934.0 MiB |

这些是有限样本的一次实测，未做重复计时稳定性实验，也没有扣除系统睡眠/挂起时间，不能当作延迟或吞吐保证。PyTorch allocated/reserved 不是整机或整个 GPU 的总显存占用。首次冷启动明显慢于之后的新特征样本，不能只展示较快结果。真实兼容性范围限于本次 MP3/FLAC 双声道样本；单声道、WAV、OGG 尚未做真实音频验收。

## 导出和时间同步

[媒体报告](D:/2_datas/Vm/data/v3.0/reports/media-verification.json) 对源音频 2–12 秒片段进行了实际编码、解码、逐帧数值与下载检查：

- 视频为 640×360、24 fps、240 帧、10 秒，成功解码。最后一帧对应源音频 11.958333 秒，按帧定义不要求在 12 秒再增加一帧。
- 导出音轨解码后为 160,000 个 16 kHz 单声道比较样本，即 10 秒。AAC 为有损编码；与源片段的零延迟相关为 0.998102649，搜索得到最佳延迟 0 个样本。
- 240 条逐帧值与独立参考的最大绝对差为 5.5511×10⁻¹⁷；原始结果 SHA256 保持不变。
- 视频、原始 CSV、逐帧 CSV 的下载内容与保存文件逐字节一致。显示 CSV 的平滑窗口序列化为 `2` 或 `2.0`，文本字节不同，数值完全一致；未把它计为字节一致。
- 原始 CSV 保留 B01/B00 各 57 个原始点、旧 V2 的 280 个原始点，未统一拉伸时间轴。

可直接检查 [导出视频](D:/2_datas/Vm/data/v3.0/projects/p-22cc95f60fa8-d1d8b430/exports/e-222983ade629/preview.mp4) 和 [抽取帧](D:/2_datas/Vm/data/v3.0/reports/export-frame.png)。以上同步与解码是数值和自动播放检查；**人工试听未执行**，不能宣称听感、情绪曲线主观准确性或艺术效果已获人工认可。

## 验收矩阵结论

对应 [验收约定](D:/2_datas/Vm/versions/v3.0/docs/ACCEPTANCE.md) 的编号。分组中的每个编号均采用该行结论。

| 编号 | 结论 | 证据与适用范围 |
| --- | --- | --- |
| M01 | Pass | 冻结身份和真实权重加载；原始输出一致性、保护校验均通过 |
| M02 | Pass | 缺失/不匹配身份及不合法输入的合同检查；不自动训练或静默更换模型 |
| M03 | Pass | 三成员等权均值及样本标准差合同；历史成员/均值/分歧一致性 |
| T01 | Pass | 原生时钟合同及 24 首真实完整解码输出 |
| T02 | Pass | 旧 V2 原生时钟实测与独立比较；播放、定位及模型切换观察 |
| T03 | Pass | 原生解码帧数、时长及实际片段媒体边界检查 |
| D01 | Pass | 独立平滑合同、浏览器切换、逐帧参考及原始 SHA256 不变 |
| D02 | Pass | 不可用字段及非置信分歧的负例测试、清单和页面标识 |
| D03 | Pass | 声学单位、数值关系检查及真实页面/JSON 字段；CSV 曲线文件不单独承载所有单位元数据，项目 JSON 保留完整说明 |
| C01、C02 | Pass | 内容身份、缓存 key/shape/dtype/clock、部分写入及篡改拒绝；实际已验证缓存复用 |
| P01 | Pass | 路径逃逸、ID 和导出目录负例测试 |
| P02 | Pass | 独立写入位置及旧版保护快照一致 |
| H01、H02、H03 | Pass | 真实本机 HTTP 合同：Host/Origin/token、Range/HEAD、非法请求拒绝；最终浏览器正常读取 |
| J01、J02 | Pass | PID/创建身份与竞争拒绝合同；真实任务和独占锁观察 |
| J03、J04 | Pass | 真实取消、退出、原配置恢复；失败与过期身份合同 |
| E01 | Pass | JSON/CSV 下载解析、原始时钟、数值及来源凭证；显示 CSV 文本差异明确记录 |
| E02 | Pass | 上述 10 秒真实视频、音轨、帧值及同步检查；限定自动工程检查，不包含人工听感 |
| E03 | Pass | 编码失败与目录逃逸合同；成功导出发布可校验的完整凭证 |
| U01、U02 | Pass | 实际网页创建、待处理显示、完成、播放循环、切换、导出和重开；研究状态与任务状态可见 |
| R01 | Partial | 已有阶段时序和 PyTorch 显存实测；仅一次有限样本，未测稳定重复分布或总进程/整机资源，不承诺性能 |
| R02 | Pass | 冻结 24 首已有音频及完整完成记录；无新标签/最终测试输入；格式/声道覆盖限制另列 |
| R03 | Pass | 97/97 保护对象的最终快照与起始快照相同 |
| 人工试听与主观接受度 | Partial | 自动播放与同步已有证据，人工试听尚未执行，主观结论保持空缺 |

原模型 [emotion_model.pth](D:/2_datas/Vm/models/emotion_model.pth) 的 SHA256 未变：

```text
f0c304aec8120ba09e21b2540cc5ae0de23a1536c449673edd0388b896e31667
```

## 保留的失败及修复

| 真实问题 | 修复与验收 | 保留记录 |
| --- | --- | --- |
| Windows 并发读取短时占用导致原子替换 WinError 5，首轮在第 16 首中断 | 仅对 Windows 5/32/33 做有限退避重试；持续失败仍抛错并保留旧文件与待提交内容；相关回归通过，新的 engineering-v2 24/24 完成 | [engineering-v1](D:/2_datas/Vm/data/v3.0/reports/engineering-v1/report.json) 及其原绑定未改写 |
| 先拒绝 POST 而未读取请求体，Windows socket reset 使浏览器收不到 403 | 先读取最多 64 KiB 原始字节，设置 5 秒读取超时，然后执行来源/令牌校验；授权前不解析或分派任务；最终 HTTP 测试通过 | [修复前测试日志](D:/2_datas/Vm/data/v3.0/reports/unit-tests-before-http-drain.log) |
| Windows 自动换行翻译使 CSV 出现双 CR | 写文件关闭额外换行翻译，保留旧导出，另建新导出；实际下载/文件比对完成 | [首轮媒体检查](D:/2_datas/Vm/data/v3.0/reports/media-verification-attempt1.json) |
| 网页创建分析后，解码前尚无正式 project.json | 启动前单独写 pending.json，绑定原任务身份；正式原生时长元数据随后优先使用；不伪造完成结果；新增两个回归及真实网页创建通过 | [最终浏览器记录](D:/2_datas/Vm/data/v3.0/reports/browser-final.json)、[最终测试日志](D:/2_datas/Vm/data/v3.0/reports/unit-tests.log) |

当前没有需继续执行的重型工程验收任务。下一步可由用户用现有项目进行试听和使用反馈；这不需要新增训练数据。若未来开展模型推广，应另行满足原研究计划的逐库逐维门槛及冻结流程，本交付不会自动触发额外训练或最终测试。
