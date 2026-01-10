"""
Music Visualization Pipeline
Usage: python main.py input1.mp3 input2.wav input3.flac ...
       python main.py input1.wav input2.mp3 input3.flac ... --model emotion_model.pth
"""

# ============================================================
# main.py
# ------------------------------------------------------------
# 整个音乐可视化流水线的入口和编排脚本。
#
# 6 步流程：
#   [1/6] 加载音频 → 提取全曲信息（BPM、节拍、时长）
#   [2/6] 逐帧特征提取 → 每 0.1s 一帧的 Librosa 特征
#   [3/6] CLAP 嵌入提取 → 每帧 512 维语义向量（可缓存）
#   [4/6] 情感推理 → 加载或训练 EmotionModel，预测全曲情感序列
#   [5/6] MCTS 视觉参数搜索 → 每 0.5s 搜索一次最优 VisualState
#   [6/6] 视频渲染 → 30fps 逐帧合成，ffmpeg 合并音视频
#
# 各步骤的时间成本（3分钟歌曲参考）：
#   1-2：<10s（Librosa 特征提取，CPU 较快）
#   3：  1-5min（首次 CLAP 推理，缓存后 <1s）
#   4：  30-60s（训练 100 epoch，取决于 CPU）
#   5：  <5s（MCTS 纯 numpy 运算）
#   6：  3-30min（OpenCV 渲染，取决于 CPU 性能）
# ============================================================

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import numpy as np
import soundfile as sf

from feature_extraction import (
    load_audio,
    extract_global_info,
    extract_frame_features,
    FrameFeatures,
)
from emotion_model import extract_clap_embeddings, EmotionInterface
from mcts import MCTS, VisualState
from renderer import VideoRenderer

# ── 全局常量 ──────────────────────────────────────────────────
FRAME_DUR = 0.1        # 情感帧时长（秒）：每 0.1s 提取一次特征/CLAP 嵌入
MCTS_INTERVAL = 0.5    # MCTS 搜索间隔（秒）：每 5 个情感帧运行一次 MCTS
FPS = 30               # 输出视频帧率（每秒 30 帧）


# ── VisualState 插值 ──────────────────────────────────────────

def _interpolate_visual_state(
    states: List[VisualState],   # MCTS 计算出的关键帧状态列表
    times: List[float],          # 各关键帧对应的时间点（秒）
    t: float,                    # 当前视频帧的时间点（秒）
) -> VisualState:
    """
    在 MCTS 关键帧（每 0.5s 一个）之间插值，生成当前视频帧（1/30s 精度）的 VisualState。

    为什么需要插值：
      MCTS 每 0.5s 运行一次，但视频是 30fps（每 1/30s 一帧）。
      若直接用最近的 MCTS 结果，视觉参数会每隔 15 帧跳变一次。
      线性插值使参数在两个 MCTS 关键帧之间平滑过渡。

    特殊处理 hue_base（色相短弧插值）：
      色相是 0-360° 的环形空间，直接线性插值会走"长弧"。
      例如从 350° → 10°，直接插值经过 350→180→10（走 340°），
      短弧插值只走 20°（350→360/0→10）。
      方法：diff = (s1-s0+180)%360 - 180，得到 (-180,180] 内的有向差值，
      然后 hue = s0 + alpha×diff（再取模 360）。

    整型字段（particle_count, trail_length）用 round() 而非截断，
    避免插值偏向低端（如 int(0.9) = 0 而非期望的 1）。
    """
    if len(states) == 1:
        return states[0]  # 只有一个关键帧时直接返回

    # searchsorted 找到 t 应插入 times 的位置，即 t 之前最近的关键帧索引
    idx = int(np.searchsorted(times, t, side='right')) - 1
    idx = int(np.clip(idx, 0, len(states) - 2))  # 确保 idx+1 不越界

    s0, s1 = states[idx], states[idx + 1]
    dt = times[idx + 1] - times[idx]
    # alpha 是 t 在区间 [times[idx], times[idx+1]] 内的归一化位置 [0, 1]
    alpha = float(np.clip((t - times[idx]) / dt, 0.0, 1.0)) if dt > 0 else 0.0

    # 色相短弧插值：diff 被压缩到 (-180, 180]，保证走最短圆弧
    diff = ((s1.hue_base - s0.hue_base + 180.0) % 360.0) - 180.0
    hue_interp = (s0.hue_base + alpha * diff) % 360.0

    return VisualState(
        hue_base=hue_interp,
        hue_range=s0.hue_range + alpha * (s1.hue_range - s0.hue_range),
        saturation=s0.saturation + alpha * (s1.saturation - s0.saturation),
        brightness=s0.brightness + alpha * (s1.brightness - s0.brightness),
        particle_count=int(round(s0.particle_count + alpha *
                           (s1.particle_count - s0.particle_count))),
        particle_speed=s0.particle_speed + alpha *
        (s1.particle_speed - s0.particle_speed),
        field_turbulence=s0.field_turbulence + alpha *
        (s1.field_turbulence - s0.field_turbulence),
        trail_length=int(round(s0.trail_length + alpha *
                         (s1.trail_length - s0.trail_length))),
    )


# ── ffmpeg 工具函数 ────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    """
    在系统中搜索 ffmpeg 可执行文件。

    搜索顺序：
      1. shutil.which()：检查 PATH 环境变量（最常见情况）
      2. Windows 常见安装路径（winget install Gyan.FFmpeg 的默认位置）
    返回第一个找到的路径，未找到则返回 None。
    """
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    candidates = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        os.path.expanduser(r'~\AppData\Local\Programs\ffmpeg\bin\ffmpeg.exe'),
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def _merge_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    """
    将无声视频（.avi）和音频（.wav）合并为有声视频（.mp4）。

    ffmpeg 命令说明：
      -y：覆盖已存在的输出文件，不询问
      -c:v copy：视频流直接复制（不重新编码），保留画质且速度快
      -c:a aac：音频编码为 AAC（MP4 容器标准音频格式）
      -shortest：以较短的流为准截断（防止无声或无画面的尾部）

    成功后删除临时文件（.avi 和 .wav）。
    若 ffmpeg 不可用，回退方案：保留分离的 _video.avi 和 _audio.wav，
    并打印手动合并命令（用户可稍后安装 ffmpeg 再执行）。
    """
    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'   # 或 ignore
        )
        if result.returncode == 0:
            # 合并成功，删除临时文件
            for f in [video_path, audio_path]:
                if os.path.exists(f):
                    os.remove(f)
            return
        print(f"[!] ffmpeg error:\n{result.stderr[-500:]}")

    # 回退方案：ffmpeg 未找到或执行失败，保留分离文件
    vid_out = output_path.replace('.mp4', '_video.avi')
    aud_out = output_path.replace('.mp4', '_audio.wav')
    shutil.move(video_path, vid_out)
    shutil.move(audio_path, aud_out)
    print("\n[!] ffmpeg not found. Output saved as separate files:")
    print(f"    Video: {vid_out}")
    print(f"    Audio: {aud_out}")
    print(f"\nTo merge, install ffmpeg (winget install Gyan.FFmpeg) then run:")
    print(
        f"  ffmpeg -i \"{vid_out}\" -i \"{aud_out}\" -c:v copy -c:a aac \"{output_path}\"")


# ── 主流程 ─────────────────────────────────────────────────────

def main(input_path: str, output_path: str, model_path: str = 'emotion_model.pth') -> None:
    # ── 1. 加载音频 ───────────────────────────────────────────────
    print(f"[1/6] Loading audio: {input_path}")
    y, sr = load_audio(input_path)          # y: (N_samples,) float32，sr: 22050
    global_info = extract_global_info(y, sr)  # BPM、beat_times、duration 等全曲信息
    duration = global_info['duration']
    print(
        f"      Duration: {duration:.1f}s  Tempo: {global_info['tempo']:.1f} BPM")

    # ── 2. 逐帧特征提取 ───────────────────────────────────────────
    print(f"[2/6] Extracting features ({FRAME_DUR*1000:.0f}ms frames)...")
    features = extract_frame_features(y, sr, frame_duration=FRAME_DUR)
    n_frames = len(features)
    print(f"      {n_frames} frames extracted")

    # ── 3. CLAP 嵌入提取 ──────────────────────────────────────────
    # 缓存路径与输入音频同目录，命名为 <音频文件名>_clap_cache.npy
    # 首次运行需要 1-5 分钟（CLAP 模型下载 + 推理），之后 <1s 直接加载缓存
    audio_stem = os.path.splitext(os.path.basename(input_path))[0]
    cache_path = os.path.join(os.path.dirname(
        input_path), f"{audio_stem}_clap_cache.npy")

    print("[3/6] Extracting CLAP embeddings...")
    clap_embeddings = extract_clap_embeddings(
        y, sr,
        frame_duration=FRAME_DUR,
        cache_path=cache_path,
    )  # 返回 (N_frames, 512) float32

    # ── 4. 情感推理 ───────────────────────────────────────────────
    print("[4/6] Emotion inference...")
    if os.path.exists(model_path):
        # 已有训练好的模型权重，直接加载（跳过训练）
        print(f"      Loading model from {model_path}")
        iface = EmotionInterface.load(model_path)
    else:
        # 首次运行：基于规则伪标签自监督训练，保存权重供下次使用
        print(
            f"      No model found at {model_path}. Training from pseudo-labels...")
        iface = EmotionInterface.train(
            clap_embeddings=clap_embeddings,
            features=features,
            global_info=global_info,
            save_path=model_path,
        )

    # 批量推理全曲所有帧，返回 List[{'valence', 'arousal', ...}]
    emotion_states = iface.predict_sequence(clap_embeddings)
    print(f"      Predicted {len(emotion_states)} emotion states")

    # ── 5. MCTS 视觉参数搜索 ──────────────────────────────────────
    print("[5/6] MCTS visual parameter search...")
    mcts = MCTS(n_iter=200, branching=5)
    mcts_states: List[VisualState] = []
    mcts_times: List[float] = []
    # 每 frames_per_mcts 个情感帧运行一次 MCTS（默认每 5 帧 = 0.5s）
    frames_per_mcts = max(1, int(MCTS_INTERVAL / FRAME_DUR))

    prev_state: Optional[VisualState] = None
    for i in range(0, n_frames, frames_per_mcts):
        es = emotion_states[i]
        valence = float(es['valence'])
        arousal = float(es['arousal'])
        # 传入上次 MCTS 结果作为起点，保证相邻时间点的视觉连贯性
        vs = mcts.search(valence, arousal, init_state=prev_state)
        mcts_states.append(vs)
        mcts_times.append(i * FRAME_DUR)  # 该关键帧对应的时间（秒）
        prev_state = vs

    print(f"      {len(mcts_states)} visual states computed")

    # ── 6. 渲染视频 ───────────────────────────────────────────────
    print("[6/6] Rendering video...")
    # 临时文件：无声视频（OpenCV 写出）和音频（soundfile 写出）
    tmp_video = output_path.replace('.mp4', '_noaudio.avi')
    tmp_audio = output_path.replace('.mp4', '_audio.wav')

    # 将音频导出为 PCM_16 WAV（ffmpeg 合并时使用，也可单独播放）
    sf.write(tmp_audio, y, sr, subtype='PCM_16')

    total_video_frames = int(duration * FPS)         # 总视频帧数
    samples_per_video_frame = max(1, sr // FPS)      # 每视频帧对应的音频样本数（约 735）

    renderer = VideoRenderer(tmp_video, fps=FPS)

    for frame_idx in range(total_video_frames):
        t = frame_idx / FPS  # 当前帧对应的时间（秒）

        # 在 MCTS 关键帧之间线性插值，得到当前帧的 VisualState
        vs = _interpolate_visual_state(mcts_states, mcts_times, t)

        # 提取当前帧对应的音频样本（用于波形叠加渲染）
        chunk_start = int(t * sr)
        audio_chunk = y[chunk_start: chunk_start + samples_per_video_frame]
        if len(audio_chunk) < samples_per_video_frame:
            # 最后几帧音频可能不足，用零填充
            audio_chunk = np.pad(
                audio_chunk, (0, samples_per_video_frame - len(audio_chunk)))

        # 节拍冲击检测：当前帧时间与任意 beat_time 差值在 1 帧内则视为节拍帧
        # 节拍帧时粒子系统会施加随机方向冲击（"跳舞"效果）
        beat_impulse = 1.0 if any(abs(t - bt) < (1.0 / FPS)
                                  for bt in global_info['beat_times']) else 0.0

        # 全局时间相位：缓慢增长（×0.05），驱动向量场的整体漂移动画
        # 不使用整数帧索引，避免 30fps 的离散跳变
        phase = t * 0.05

        # 节拍内相位：在每个 beat 周期内从 0→1 循环
        # beat_period = 60s / BPM；用于驱动向量场的高频节律扰动
        beat_phase = (t % (
            60.0 / max(global_info['tempo'], 1.0))) / (60.0 / max(global_info['tempo'], 1.0))

        frame = renderer.render_frame(
            vs, audio_chunk, beat_impulse, phase, beat_phase)
        renderer.write_frame(frame)

        # 每 300 帧打印一次进度（\r 覆盖同行，不刷屏）
        if frame_idx % 300 == 0:
            pct = int(100 * frame_idx / total_video_frames)
            print(
                f"      Frame {frame_idx}/{total_video_frames}  ({pct}%)", end='\r')

    renderer.release()  # 必须释放，否则 AVI 文件不完整
    print(f"\n      Rendered {total_video_frames} frames")

    # 合并音视频（有 ffmpeg 则生成 .mp4，否则保留分离文件）
    _merge_audio_video(tmp_video, tmp_audio, output_path)

    if os.path.exists(output_path):
        print(f"\nDone! Output: {output_path}")


# ── 命令行入口 ─────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Visualization')

    parser.add_argument('inputs', nargs='+',
                        help='Input audio files (WAV/MP3/FLAC)')

    parser.add_argument('--model', default='emotion_model.pth',
                        help='Path to emotion model weights')

    args = parser.parse_args()

    for input_file in args.inputs:
        if not os.path.isfile(input_file):
            print(f"ERROR: Input file not found: {input_file}")
            continue

        # 原始输出路径（同目录，同名.mp4）
        base = os.path.splitext(input_file)[0]
        output_file = base + '.mp4'

        # === 自动重命名避免覆盖 ===
        i = 1
        while os.path.exists(output_file):
            output_file = base + f'_{i}.mp4'
            i += 1

        print(f"Processing: {input_file} -> {output_file}")
        main(input_file, output_file, args.model)
