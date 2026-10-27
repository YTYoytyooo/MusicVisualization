# 独立研究版推理

此适配器加载 B00 的三个固定种子检查点，使用训练时的 CLAP512＋声学19 特征及训练集归一化器。它是研究对照，不代表已超过 V2 或已通过发布门槛，不改变 1.0／2.0 的模型选择。

`ResearchPredictor(manifest_path, device)` 从 `models/next_va_v2/research-reference-v1/manifest.json` 严格核对检查点、代码、特征、依赖和归一化来源。`predict_features(times, features, feature_key, duration_seconds)` 接收原始 float32 特征，要求原生文件零点开始的 0.5 秒采样轴和完整序列，不允许平移、拉伸或缺帧。底层为双向整曲模型，只适用于离线推理。

输出分开保存：

- `raw_va`：三个种子预测的算术平均，列顺序为效价、唤醒，范围 [-1, 1]。
- `member_raw_va` 和 `seed_disagreement_std`：各成员原始预测和样本标准差。分歧不代表已校准的置信度。
- `calibration`：现有校准未通过，区间保持不可用。
- `acoustic`：由未归一化声学特征还原的局部 RMS、dBFS 和频谱重心 Hz。RMS 不等于心理唤醒或 LUFS，频谱重心不等于视觉亮度。为保持训练特征口径，数值来自对数特征的时间插值。
- `display`：未应用平滑或视觉映射。
- `tension`：没有真实监督，保持不可用。

完整音频基准通过独立 GPU Python 运行 `-m tools.train_next_v2.research_inference.benchmark`。先 `--tests`，再 `--prepare` 固定三个样本和完整启动参数。之后使用生成的 `launch.json` 中 `original_argv`／`verification_argv` 及对应环境。`--status` 查询 OS 锁、进程创建身份和退出状态、有限日志、缓存数和完成凭证；不能根据旧状态文件认定进程仍在运行。

缓存位于 `data/training/next_va/optimization/inference_features/research-reference-v1`，预测、测量和恢复凭证位于 `data/training/next_va/optimization/inference_jobs/f-research-reference-benchmark-v1`。中断后确认同编号进程和 GPU 锁都已释放，检查原因后才使用原命令加 `--resume`。只接受哈希匹配的完整单曲凭证；发现未附凭证的缓存或预测应先保留并检查，不能将其默认为完整结果。

本基准使用没有读取人工标签的已审核目标音频，仅检验加载、音频时间、训练／推理一致性和速度显存。三个样本不构成精度评估，最终 622 首测试标签保持封存，实际人工试听仍待完成。
