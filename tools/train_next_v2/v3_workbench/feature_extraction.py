"""Acoustic measurements for local rendering; no emotion labels or training."""
import numpy as np
import librosa
from typing import TypedDict, List, Tuple, Optional


# ── 类型定义 ──────────────────────────────────────────────────

class FrameFeatures(TypedDict):
    """
    单帧（0.1 秒）音频特征的类型定义。
    所有字段由 extract_frame_features() 填充。
    """
    mfcc: np.ndarray       # (13,) float32 — 13 个 Mel 倒谱系数，时间轴取均值
    mfcc_mean: np.ndarray  # (13,) float32 — 同 mfcc，保留别名方便后续代码引用
    centroid: float        # 频谱质心（Hz）：能量重心频率，高值=音色明亮
    rolloff: float         # 85% 能量滚降频率（Hz）：反映高频能量占比
    zcr: float             # 过零率：波形每秒穿越零轴次数，打击乐/噪声段值高
    rms: float             # 均方根能量：反映瞬时响度
    chroma: np.ndarray     # (12,) float32 — 12 个半音强度（C, C#, D, ..., B）
    mel_mean: np.ndarray   # (64,) float32 — Mel 频谱图归一化后沿时间轴的均值
    onset: float           # Onset strength 均值：节奏冲击强度
    frame_time: float      # 该帧起始时间（秒）


class GlobalInfo(TypedDict):
    """全曲级别的信息，贯穿 main.py 整个流程传递。"""
    tempo: float           # 全曲估算 BPM（librosa 自动节拍追踪）
    beat_times: np.ndarray # (N_beats,) float32 — 每个 beat 的时间点（秒）
    duration: float        # 总时长（秒）
    sr: int                # 采样率（Hz），通常 22050
    y: np.ndarray          # (N_samples,) float32 — 原始波形（备用）
    n_frames: int          # 以 0.1s 分帧后的总帧数


# ── 音频加载 ──────────────────────────────────────────────────

def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并统一重采样到目标采样率。

    librosa.load 自动处理 MP3/FLAC/WAV/OGG 等格式；
    mono=True 将多声道混合为单声道，保证后续特征维度一致。
    返回 float32 数组避免后续频繁类型转换。
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr_out


def extract_global_info(y: np.ndarray, sr: int) -> GlobalInfo:
    """
    提取全曲级别特征（节奏 / 节拍 / 时长）。

    注意：librosa 0.11 中 beat_track() 返回的 tempo 是 shape=(1,) 的 ndarray，
    而非 Python float。若直接使用会导致后续打印出现 array([xxx]) 包装，
    或传入 NumPy 运算时触发广播异常。
    用 np.squeeze() 确保提取为标量再转 float。
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # librosa 0.11 返回 shape=(1,) 的 tempo 数组，squeeze 后转标量
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(np.float32)
    duration = len(y) / sr
    n_frames = int(duration / 0.1)
    return GlobalInfo(
        tempo=float(np.squeeze(tempo)),
        beat_times=beat_times,
        duration=duration,
        sr=sr,
        y=y,
        n_frames=n_frames,
    )


# ── 辅助函数 ──────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    """
    将数组线性归一化到 [0, 1]。
    若数组全为同一值（分母趋近 0），返回全零数组，避免除零异常。
    用于 Mel 频谱 dB 值的归一化，使不同响度的片段可比较音色形状。
    """
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── 逐帧特征提取 ──────────────────────────────────────────────

def extract_frame_features(
    y: np.ndarray,
    sr: int,
    frame_duration: float = 0.1,  # 每帧 0.1 秒 → 22050Hz 下为 2205 个样本
    n_mfcc: int = 13,
    n_mels: int = 64,
    hop_length: int = 512,        # STFT 跳步约 23ms，每 0.1s 帧内约有 4 个 STFT 帧
) -> List[FrameFeatures]:
    """
    将全曲切成等长片段，逐帧提取多维音频特征。

    每帧 0.1s 内包含约 4 个 STFT 帧（hop=512），
    各特征均沿时间轴取均值，压缩为单一标量或向量，
    仅作为可解释的声学测量，不产生情绪标签。

    mel_mean 的计算步骤：
      1. 计算 Mel 频谱图（线性幅度平方）→ (64, T)
      2. power_to_db 转换为 dB 刻度，压缩动态范围 → (64, T)
      3. 按帧内最大值归一化到 [0,1] → (64, T)
      4. 沿时间轴取均值 → (64,)
    归一化后不同响度片段的音色"形状"可直接比较。
    """
    segment_len = int(frame_duration * sr)  # 每帧样本数，默认 2205
    n_frames = max(1, len(y) // segment_len)
    features: List[FrameFeatures] = []

    for i in range(n_frames):
        start = i * segment_len
        seg = y[start : start + segment_len]
        # 最后一帧可能不足整帧长度，用零填充保持维度一致
        if len(seg) < segment_len:
            seg = np.pad(seg, (0, segment_len - len(seg)))

        # MFCC：将对数 Mel 频谱压缩为 13 个系数，捕捉音色/音素信息
        # mean(axis=1) 沿时间轴折叠，得到帧内平均 MFCC
        mfcc_mat = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc_val = mfcc_mat.mean(axis=1).astype(np.float32)  # → (13,)

        # 频谱质心：能量重心所在频率（Hz），音色越亮则值越高
        centroid = float(librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=hop_length).mean())
        # 滚降频率：累积 85% 频谱能量时对应的频率，反映高频丰富程度
        rolloff = float(librosa.feature.spectral_rolloff(y=seg, sr=sr, hop_length=hop_length).mean())
        # 过零率：波形每采样点穿越零轴的概率，打击乐/噪声段值高
        zcr = float(librosa.feature.zero_crossing_rate(seg, hop_length=hop_length).mean())
        # RMS 能量：帧内响度的均方根
        rms_val = float(librosa.feature.rms(y=seg, hop_length=hop_length).mean())

        # Chroma：将频谱映射到 12 个音高类别（C~B），反映和声调性特征
        # mean(axis=1) 折叠时间轴 → (12,) 向量
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, hop_length=hop_length).mean(axis=1).astype(np.float32)

        # Mel 频谱均值：归一化后的音色"频谱形状"
        mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)   # dB 刻度，压缩动态范围
        mel_mean = _normalize(mel_db).mean(axis=1).astype(np.float32)  # → (64,)

        # Onset Strength：检测音符/打击乐开始的冲击强度均值
        onset = float(librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop_length).mean())

        features.append(FrameFeatures(
            mfcc=mfcc_val,
            mfcc_mean=mfcc_val,    # 别名，两者指向同一数组
            centroid=centroid,
            rolloff=rolloff,
            zcr=zcr,
            rms=rms_val,
            chroma=chroma,
            mel_mean=mel_mean,
            onset=onset,
            frame_time=start / sr, # 帧起始时间（秒）
        ))

    return features
