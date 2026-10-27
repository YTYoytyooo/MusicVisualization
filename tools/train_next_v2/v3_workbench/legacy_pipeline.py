"""Versioned visual interpolation only; no legacy analysis or training entry."""
from typing import List
import numpy as np
from .mcts import VisualState

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
