# ============================================================
# emotion_model.py
# ------------------------------------------------------------
# 实现两阶段情感推理流水线：
#
#   阶段 1 — CLAP 特征提取（extract_clap_embeddings）
#     使用冻结的预训练 CLAP 模型将 2s 音频窗口编码为 512 维向量。
#     这一步计算量大（~1-2 分钟/3分钟音乐），结果缓存到 .npy 文件。
#
#   阶段 2 — EmotionModel 推理（EmotionInterface）
#     小型 BiLSTM 适配器，接收 10 帧连续 CLAP embeddings，
#     输出 5 维情感向量 [valence, arousal, energy, tension, brightness]。
#     首次运行时用规则伪标签（generate_pseudo_labels）自监督训练。
#
# 为什么用 CLAP 而非直接用 Librosa 特征？
#   CLAP 经过大规模音频-文本对比学习，embedding 空间中
#   语义相近的音乐段落距离更近，远优于手工特征的泛化性。
# ============================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Optional, Tuple

from feature_extraction import FrameFeatures, GlobalInfo, generate_pseudo_labels

# ── 模型超参数 ─────────────────────────────────────────────────
WINDOW_SIZE = 10    # BiLSTM 输入的连续帧数：10 帧 × 0.1s = 1s 上下文
CLAP_DIM = 512      # CLAP audio projection 输出维度（laion/clap-htsat-fused 固定为 512）
HIDDEN_DIM = 128    # BiLSTM 单向隐层维度；双向拼接后 → 256
N_EMOTIONS = 5      # 输出维度：[valence, arousal, energy, tension, brightness]


# ── EmotionModel：小型 BiLSTM 适配器 ──────────────────────────

class EmotionModel(nn.Module):
    """
    以 10 帧 CLAP embeddings 为输入，输出 5 维情感向量的轻量神经网络。

    架构（参数量约 67 万）：
      Input  (B, 10, 512)
        → LayerNorm(512)          # 归一化 CLAP 嵌入，稳定训练
        → BiLSTM(512→128, 1层)    # 捕捉时间序列上的情感动态
        → output[:, -1, :]        # 取最后一步的双向隐状态 → (B, 256)
        → Linear(256→64) + GELU   # 非线性压缩
        → Linear(64→5) + Tanh     # 输出范围 (-1, 1)，对应情感强度

    选择 BiLSTM 而非单向 LSTM 的原因：
      双向模型同时看到"前文"和"后文"的音频上下文，
      对判断当前帧的情感更稳定（例如：一个渐强段落，
      单向模型在开头看不到后续高潮，而双向可以）。
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(CLAP_DIM)
        self.lstm = nn.LSTM(
            input_size=CLAP_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,    # 输入形状 (Batch, Seq, Feature) 更直观
            bidirectional=True,  # 前向 + 反向，拼接后 hidden 维度 ×2
            dropout=0.0,         # 单层 LSTM 无需 dropout（多层才有效）
        )
        self.fc1 = nn.Linear(HIDDEN_DIM * 2, 64)  # 256 → 64
        self.fc2 = nn.Linear(64, N_EMOTIONS)        # 64 → 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, CLAP_DIM) = (B, 10, 512)
        x = self.norm(x)           # 归一化 → (B, 10, 512)
        out, _ = self.lstm(x)      # LSTM 输出 → (B, 10, 256)（128×2 双向）
        last = out[:, -1, :]       # 取序列最后一步 → (B, 256)
        h = F.gelu(self.fc1(last)) # 全连接 + GELU 激活 → (B, 64)
        return torch.tanh(self.fc2(h))  # 输出 → (B, 5)，值域 (-1, 1)


# ── CLAP 可用性检测与加载 ──────────────────────────────────────

def _check_transformers() -> bool:
    """检查 transformers 库是否已安装（CLAP 依赖它）。"""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load_clap():
    """
    加载 CLAP 预训练模型（laion/clap-htsat-fused）。
    首次调用会从 Hugging Face 下载约 600MB 权重到本地缓存。
    model.eval() 确保 BatchNorm/Dropout 等层处于推理模式。
    """
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model.eval()
    return model, processor


# ── CLAP Embedding 提取 ────────────────────────────────────────

def extract_clap_embeddings(
    y: np.ndarray,
    sr: int,
    frame_duration: float = 0.1,    # 每个情感帧的时长（秒）
    window_duration: float = 2.0,   # 送入 CLAP 的音频窗口时长（秒）
    clap_sr: int = 48000,           # CLAP 要求的采样率（laion 模型固定 48kHz）
    cache_path: Optional[str] = None,
    batch_size: int = 16,           # 每批处理的窗口数，影响内存占用
) -> np.ndarray:
    """
    为每个 0.1s 情感帧提取 CLAP 音频嵌入。

    每帧对应一个以该帧为起点、长 2s 的音频窗口，
    2s 窗口能包含足够的音频上下文，让 CLAP 感知旋律/和声/节奏。
    返回 (N_frames, 512) float32 数组。

    流程：
      1. 若 cache_path 存在，直接加载缓存（避免重复计算）
      2. 将 22050Hz 音频重采样到 48kHz（CLAP 要求）
      3. 切分为 N_frames 个 2s 窗口（允许重叠）
      4. 分批送入 CLAP，提取 pooler_output → audio_projection → 512 维
      5. 结果保存到 cache_path（.npy 格式）
    """
    # 缓存命中：直接返回上次结果，跳过耗时的 CLAP 推理
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading CLAP cache from {cache_path}")
        return np.load(cache_path)

    if not _check_transformers():
        print("ERROR: 'transformers' package not found. Install with:")
        print("  d:\\2_datas\\Vm\\venv\\Scripts\\pip.exe install transformers")
        sys.exit(1)

    import librosa as _librosa

    print("  Loading CLAP model (laion/clap-htsat-fused)...")
    clap_model, processor = _load_clap()

    # 重采样到 CLAP 要求的 48kHz（使用高质量 soxr 重采样算法）
    y_48k = _librosa.resample(y, orig_sr=sr, target_sr=clap_sr, res_type='soxr_hq')
    y_48k = y_48k.astype(np.float32)

    segment_len_orig = int(frame_duration * sr)         # 22050Hz 下每帧样本数
    window_len_48k   = int(window_duration * clap_sr)   # 48kHz 下 2s 的样本数 = 96000
    n_frames = max(1, len(y) // segment_len_orig)

    # 为每个情感帧构建对应的 2s 音频窗口（允许末尾零填充）
    windows = []
    for i in range(n_frames):
        start_48k = int(i * frame_duration * clap_sr)  # 该帧在 48kHz 音频中的起始位置
        end_48k   = start_48k + window_len_48k
        win = y_48k[start_48k:end_48k]
        if len(win) < window_len_48k:
            # 末尾帧不足 2s，用零补齐（静音）
            win = np.pad(win, (0, window_len_48k - len(win)))
        windows.append(win)

    # 分批提取 CLAP 嵌入
    embeddings = []
    print(f"  Extracting CLAP embeddings ({n_frames} frames, batch={batch_size})...")
    for batch_start in range(0, n_frames, batch_size):
        batch_wins = windows[batch_start : batch_start + batch_size]
        # ClapProcessor 将原始波形转为模型输入（梅尔频谱 + is_longer 标志）
        inputs = processor(
            audio=batch_wins,
            return_tensors="pt",
            sampling_rate=clap_sr,
            padding=True,  # 批内对齐到最长样本
        )
        with torch.no_grad():
            # 调用 CLAP 内部的音频编码器（HTS-AT 变换器）
            # 注意：不能直接调用 clap_model.get_audio_features()，
            # 该方法返回 BaseModelOutputWithPooling 而非 Tensor，
            # 需要手动取 pooler_output 再经 audio_projection 投影到 512 维
            audio_out = clap_model.audio_model(
                input_features=inputs.get('input_features'),
                is_longer=inputs.get('is_longer'),
            )
            feats = audio_out.pooler_output          # (B, encoder_hidden_dim)
            if hasattr(clap_model, 'audio_projection'):
                feats = clap_model.audio_projection(feats)  # 投影到统一的 512 维空间
        embeddings.append(feats.cpu().numpy().astype(np.float32))
        # 每 10 批打印一次进度（避免刷屏）
        if (batch_start // batch_size) % 10 == 0:
            pct = min(100, int(100 * batch_start / n_frames))
            print(f"    {pct}%", end='\r')

    result = np.concatenate(embeddings, axis=0)  # 拼接所有批次 → (N, 512)
    print(f"\n  Done. Embeddings shape: {result.shape}")

    # 保存缓存，下次跳过 CLAP 推理
    if cache_path:
        np.save(cache_path, result)
        print(f"  Cached to {cache_path}")

    return result


# ── 滑动窗口数据集构建 ─────────────────────────────────────────

def _build_windows(
    embeddings: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CLAP embeddings 序列构建滑动窗口训练集。

    每个样本 X[i] 是连续 window_size 帧的 embeddings，
    对应标签 y[i] 是窗口最后一帧的情感值（因果预测）。

    X: (N-W, W, 512)  — 输入序列
    y: (N-W, 5)       — 对应的情感标签

    注意：前 window_size-1 帧没有足够的历史窗口，因此数据集
    从第 window_size 帧开始（共 N-W 个样本）。
    推理时用零向量左侧填充解决这个问题（见 predict_sequence）。
    """
    N = len(embeddings)
    xs, ys = [], []
    for i in range(N - window_size):
        xs.append(embeddings[i : i + window_size])  # 连续 W 帧 → (W, 512)
        ys.append(labels[i + window_size - 1])        # 窗口最后一帧的标签
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


# ── EmotionInterface：训练 / 加载 / 推理的统一接口 ────────────

class EmotionInterface:
    """
    EmotionModel 的高层封装，提供三种使用方式：
      - train()：用规则伪标签训练新模型并保存权重
      - load()：从 .pth 文件加载已训练的模型
      - predict_sequence()：对整首歌的所有帧批量推理
    """
    def __init__(self, model: EmotionModel):
        self.model = model
        self.model.eval()
        self._embed_buffer: List[np.ndarray] = []  # 保留接口，当前未使用

    @staticmethod
    def train(
        clap_embeddings: np.ndarray,    # (N, 512) 全曲 CLAP 嵌入
        features: List[FrameFeatures],  # Librosa 逐帧特征，用于生成规则标签
        global_info: GlobalInfo,        # 全曲信息（BPM 等），同上
        epochs: int = 100,
        lr: float = 3e-4,              # Adam 学习率
        weight_decay: float = 1e-4,    # L2 正则化，防止过拟合
        save_path: str = 'emotion_model.pth',
    ) -> 'EmotionInterface':
        """
        自监督训练流程：
          1. generate_pseudo_labels() 基于 Librosa 特征生成规则标签
          2. 构建滑动窗口数据集（CLAP embeddings + 规则标签对齐）
          3. 用 MSE 损失训练 EmotionModel 100 个 epoch
          4. CosineAnnealingLR 调度器：学习率从 lr 余弦退火到 0，
             避免后期振荡，使最终收敛更平滑
          5. 保存权重（含 window_size 以便 load 时恢复）
        """
        print("  Generating pseudo-labels...")
        rule_labels = generate_pseudo_labels(features, global_info)

        # CLAP 和 features 的帧数可能因边界处理差 1 帧，取较小值对齐
        n = min(len(clap_embeddings), len(rule_labels))
        clap_embeddings = clap_embeddings[:n]
        rule_labels = rule_labels[:n]

        X, y = _build_windows(clap_embeddings, rule_labels)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        # shuffle=True 使每 epoch 批次顺序随机，提升泛化
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

        model = EmotionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # CosineAnnealingLR：学习率按余弦曲线从 lr 降至接近 0
        # T_max=epochs 表示整个训练周期内完成一个余弦周期
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"  Training EmotionModel ({len(X)} samples, {epochs} epochs)...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                pred = model(X_batch)
                loss = F.mse_loss(pred, y_batch)  # MSE：情感值是连续回归目标
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()  # 每 epoch 末更新学习率
            if (epoch + 1) % 20 == 0:
                avg = total_loss / len(loader)
                print(f"    Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

        # 保存权重 + window_size（load 时需要匹配）
        torch.save({'state_dict': model.state_dict(), 'window_size': WINDOW_SIZE}, save_path)
        print(f"  Model saved to {save_path}")
        model.eval()
        return EmotionInterface(model)

    @staticmethod
    def load(path: str) -> 'EmotionInterface':
        """
        从 .pth 文件加载已训练的模型。
        weights_only=True 防止加载不受信任的 pickle 对象（PyTorch 安全建议）。
        map_location='cpu' 确保在无 GPU 环境下也能正常加载。
        """
        data = torch.load(path, map_location='cpu', weights_only=True)
        model = EmotionModel()
        model.load_state_dict(data['state_dict'])
        model.eval()
        return EmotionInterface(model)

    def predict_sequence(
        self,
        clap_embeddings: np.ndarray,  # (N, 512) 全曲 CLAP 嵌入
    ) -> List[Dict]:
        """
        对全曲所有帧批量推理，返回每帧的情感字典列表。
        返回格式：List[{'valence': float, 'arousal': float, 'energy': float,
                        'tension': float, 'brightness': float, 'emotion': np.ndarray}]

        早期帧处理（冷启动问题）：
          前 WINDOW_SIZE-1 帧没有足够的历史 embeddings。
          解决方法：在序列开头拼接 WINDOW_SIZE 个零向量作为"静音填充"，
          使每帧都能取到完整的 WINDOW_SIZE 长度窗口，
          而不是从第 WINDOW_SIZE 帧才开始预测。

        推理分批（batch_size=256）：
          全曲可能有数千帧，一次性放入 GPU/内存可能不够，
          分批处理后拼接结果。
        """
        N = len(clap_embeddings)
        results = []

        # 在序列前填充 WINDOW_SIZE 个零向量（代表"无历史"的静音状态）
        pad = np.zeros((WINDOW_SIZE, CLAP_DIM), dtype=np.float32)
        padded = np.concatenate([pad, clap_embeddings], axis=0)  # → (W+N, 512)

        # 滑动窗口：每帧 i 取 padded[i : i+W] 作为输入
        windows = np.stack([padded[i : i + WINDOW_SIZE] for i in range(N)])  # → (N, W, 512)
        X = torch.tensor(windows, dtype=torch.float32)

        batch_size = 256
        all_preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                pred = self.model(X[i : i + batch_size])  # → (B, 5)
                all_preds.append(pred.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)  # → (N, 5)

        # 将 (N, 5) 数组转换为字典列表，方便 main.py 按键名访问
        emotion_keys = ['valence', 'arousal', 'energy', 'tension', 'brightness']
        for i in range(N):
            d = {k: float(preds[i, j]) for j, k in enumerate(emotion_keys)}
            d['emotion'] = preds[i]  # 保留原始 (5,) 数组，供可能的后续分析使用
            results.append(d)

        return results
