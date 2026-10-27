# 独立下一版连续 V/A 实验

本目录仅用于 A–F 优化计划中的新版研究，不替换旧应用或旧模型。当前实现阶段 B，使用既有 CLAP512 + 声学19 特征；每次模型输入为一首完整歌曲，输出间隔为 0.5 秒。

## 冻结实验协议

`python -m tools.train_next_v2.protocol` 创建 `data/training/next_va/optimization/protocols/b-v1.json`，已有协议必须完全一致。

- 仅从原始 train 歌曲按 `group_id` 与相同音频 SHA256 的连通组划分 fit/development/calibration，目标 80/10/10，按数据集保持比例。Memo2496 艺人信息不足，不能声称已完全隔离艺人或所有重编码录音。
- 模型随机初始化，归一化只拟合内部 fit。旧教师、混合模型已经见过这些歌曲，禁止用它们的权重做此内部对照初始化。
- 仅读取 fit 和 development 的缓存、音频校验与标签。内部 calibration、原 584 首验证、622 首测试的标签不参与 B 筛选。
- 校验协议、源清单、源配置、音频、缓存、标签、依赖特征版本与代码。特征缺失时立即停止，不重新提取。
- 8 组分别比较原结构、分支投影、独立 V/A 头、CCC 权重 0.5/1.0、变化率损失，以及按样本量平方根/样本量采样。每个变体仅改变一个因素。
- 每组 20,000 次更新，每次 4 首，共 80,000 次歌曲曝光；每 1,000 步内部开发评估、每 250 步完整检查点。按步固定随机抽样，有放回；不以旧训练轮数比较，也不按分数无限续训。
- 使用歌曲等权、数据集与维度等权的平均 CCC 选最佳检查点，同时记录每库每维误差、均值偏差、振幅、动态与常量对照。变化率按原生 0.5/1 秒时间间隔计算，缺失区间不连接，两维掩码独立。
- 入围两组以及原结构对照均用固定三个种子复验。晋级要求平均 CCC 增益 ≥ 0.02、任何库/维点估计 CCC 下降 ≤ 0.01、MAE 增加 ≤ 0.01，且歌曲组配对 bootstrap 的各库/维 CCC 差异 95% 区间下界 > -0.02。目标不代表保证；未通过可报告无收益，保留基线。

## 登记、执行与恢复

使用 `plan.json` 登记的独立 GPU Python，以下 `python` 指该解释器。先设置环境中的 TEMP/TMP 和所有新缓存到 Vm 内，设置 `PYTHONUTF8=1`、`HF_HUB_OFFLINE=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。实际 argv、环境、代码和数据校验会写入 `launch.json` 与计划。

```powershell
python -m unittest discover -s tools/train_next_v2/tests -v
python -m tools.train_next_v2.protocol
python -u -m tools.train_next_v2.run --run-id b00-reference-s20261003 --experiment b00-reference --device cuda --register-only
python -u -m tools.train_next_v2.run --run-id b00-reference-s20261003 --experiment b00-reference --device cuda --resume
python -m tools.train_next_v2.status
```

后续执行与恢复使用登记的 `original_argv` 和 `environment`。`--resume` 必须保留原配置；完整 `last.pth` 包括当前与最佳权重、优化器、随机状态、采样步、归一化、历史指标。代码或数据变动时拒绝恢复，不能删除旧配置绕过。若没有完整检查点，严格相同随机初始化重做未保存的步骤。完整任务拒绝重启。

每个任务有 OS 文件锁与 Windows PID 创建时间身份记录，全局 GPU 锁防止不同编号重叠。状态命令按真实锁、进程身份和有限日志报告，不把旧 JSON 当作活动证据。完成前重新加载 `best.pth`、复算内部开发指标并检查原模型与历史混合模型 SHA256。

运行记录在 `data/training/next_va/optimization/runs/<run_id>`，检查点在 `models/next_va_v2/<run_id>`。每次阶段推进和结果记录到 `plan.json`/`events.jsonl`。当前模块不会自动消费校准、原验证或测试集，也没有自动切换应用模型的功能。
