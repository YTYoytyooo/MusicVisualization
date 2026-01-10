# ============================================================
# renderer.py
# ------------------------------------------------------------
# 负责将 MCTS 输出的 VisualState 渲染为视频帧，写入 AVI 文件。
#
# 渲染层次（每帧按顺序叠加）：
#   1. 背景：径向渐变色彩场（中心亮、边缘暗，色相随情感变化）
#   2. 粒子轨迹：历史位置半透明叠加（旧→暗，新→亮，fade-in 效果）
#   3. 粒子本体：活跃粒子的实心圆点
#   4. 波形：底部 80px 区域显示当前帧音频波形折线
#
# 性能策略：
#   - 粒子物理更新向量化（numpy 批量运算，不用 Python 循环）
#   - 向量场每 3 帧重算一次（节省 2/3 的 meshgrid 计算）
#   - 颜色仅在色相变化 >3° 时重建（避免每帧的色相微抖导致颜色闪烁）
# ============================================================

from mcts import VisualState
import numpy as np
import cv2
from typing import Optional
from collections import deque
import colorsys

TEST = False  # test


# 输出视频分辨率和帧率（与 main.py 中的 FPS 保持一致）
WIDTH = 1280
HEIGHT = 720
FPS = 30


# ── 颜色空间转换工具 ──────────────────────────────────────────

def _hsv_to_bgr(h: float, s: float, v: float) -> tuple:
    """
    将 HSV 颜色转换为 OpenCV 使用的 BGR uint8 格式。
    h 范围 [0, 360]，s/v 范围 [0, 1]。
    colorsys 库要求 h 归一化到 [0, 1]，故除以 360。
    """
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV 是 BGR 顺序


# ── ParticleSystem：粒子物理系统 ──────────────────────────────

class ParticleSystem:
    """
    管理最多 max_particles 个粒子的生命周期和物理运动。

    数据结构（预分配 numpy 数组，避免动态内存分配）：
      pos   (max_p, 2) float32 — 粒子位置 [x, y]（像素坐标）
      vel   (max_p, 2) float32 — 粒子速度向量
      color (max_p, 3) uint8  — 粒子 BGR 颜色
      age   (max_p,)   int32  — 粒子当前帧龄
      life  (max_p,)   int32  — 粒子最大寿命（60-180帧随机）
      active (max_p,)  bool   — 是否激活（控制可见粒子数量）

    _hue_offsets (max_p,) float32 — 每粒子固定的色相偏移量 [-0.5, 0.5]。
      这是防止颜色闪烁的关键：每个粒子的色相相对 hue_base 的偏移
      在粒子生成时随机分配，之后固定不变，
      只在 hue_base 整体改变时才按比例缩放（set_colors）。
      若每帧重新随机分配颜色，粒子颜色会因微小 hue_base 变动而剧烈跳变。
    """

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        max_particles: int = 500,
        rng_seed: Optional[int] = None,
    ):
        self.W = width
        self.H = height
        self.max_p = max_particles
        self._rng = np.random.default_rng(rng_seed)

        # 初始化所有粒子到随机位置（均匀分布）
        self.pos = self._rng.random((max_particles, 2)).astype(np.float32)
        self.pos[:, 0] *= width
        self.pos[:, 1] *= height

        # 初始速度：小随机向量（[-1, 1] 范围），后续由向量场累积
        self.vel = (self._rng.random((max_particles, 2)
                                     ).astype(np.float32) - 0.5) * 2.0
        self.color = np.zeros((max_particles, 3), dtype=np.uint8)

        # 固定的每粒子色相偏移（生命周期内不变），用于制造色彩多样性
        # 范围 [-0.5, 0.5]，乘以 hue_range 后得到实际色相偏移角度
        self._hue_offsets = (self._rng.random(
            max_particles).astype(np.float32) - 0.5)

        # 粒子寿命随机化：错开各粒子的死亡时间，避免大批同时重生导致闪烁
        self.age = self._rng.integers(
            0,   60,  size=max_particles).astype(np.int32)
        self.life = self._rng.integers(
            60, 180,  size=max_particles).astype(np.int32)
        self.active = np.ones(max_particles, dtype=bool)

        # 初始激活 200 个粒子（其余处于休眠状态，按需激活）
        self._n_active = 200
        self.active[self._n_active:] = False

        # 轨迹历史：最多保存 60 帧的粒子位置快照（deque 自动丢弃旧帧）
        self.trail_history: deque = deque(maxlen=60)

        # 向量场缓存：每 3 帧重建一次，减少 meshgrid 计算开销
        self._cached_field: Optional[np.ndarray] = None
        self._field_age = 0  # 当前缓存已使用的帧数

        # 色相防抖：记录上次构建颜色时的 hue_base，变化 <3° 时跳过重建
        self._last_hue_base = -999.0

    def set_count(self, n: int) -> None:
        """
        动态调整活跃粒子数量，对应 VisualState.particle_count。

        增加粒子：激活 [_n_active, n) 范围内的粒子，重置其位置/速度/寿命
        减少粒子：将 [n, _n_active) 标记为非激活（pos/vel 数据保留，不清零）
        """
        n = int(np.clip(n, 1, self.max_p))
        if n > self._n_active:
            for i in range(self._n_active, n):
                self.pos[i, 0] = self._rng.random() * self.W
                self.pos[i, 1] = self._rng.random() * self.H
                self.vel[i] = (self._rng.random(2) - 0.5) * 2.0
                self.age[i] = 0
                self.life[i] = int(self._rng.integers(60, 180))
            self.active[self._n_active:n] = True
        elif n < self._n_active:
            self.active[n:self._n_active] = False
        self._n_active = n

    def set_colors(self, hue_base: float, hue_range: float,
                   saturation: float, brightness: float) -> None:
        """
        根据当前 VisualState 重建所有粒子的颜色。

        防抖机制：仅当 hue_base 变化超过 3° 时才重建颜色。
        原理：相邻帧之间 hue_base 的微小浮动（<3°）在视觉上不可见，
        但若每帧都重建颜色数组，ColorsHSV→BGR 的循环会在帧间造成随机抖动感。
        3° 阈值既过滤掉插值噪声，又不会延误真实色相变化的响应。

        角度差的圆弧计算：(delta + 180) % 360 - 180，确保结果在 (-180, 180]，
        正确处理跨 0°/360° 的色相变化（如从 350° 变到 10°，实际差值是 20°）。

        brightness 乘以 1.5：粒子比背景需要更亮才能可见，
        clip 到 [0.3, 1.0] 防止过曝。
        """
        if abs(((hue_base - self._last_hue_base + 180) % 360) - 180) < 3.0:
            return  # 色相变化不足 3°，跳过重建，避免微抖引起的视觉闪烁
        self._last_hue_base = hue_base

        # 每粒子色相 = hue_base + 固定偏移（偏移量按 hue_range 缩放）
        hue_offsets = self._hue_offsets * \
            hue_range  # [-hue_range/2, +hue_range/2]
        hues = (hue_base + hue_offsets) % 360.0
        for i in range(self.max_p):
            r, g, b = colorsys.hsv_to_rgb(hues[i] / 360.0,
                                          float(np.clip(saturation, 0.3, 1.0)),
                                          float(np.clip(brightness * 1.5, 0.3, 1.0)))
            self.color[i] = (int(b * 255), int(g * 255), int(r * 255))

    def build_vector_field(
        self,
        turbulence: float,
        phase: float,       # 全局时间相位，随时间缓慢增长（t×0.05），驱动场的整体漂移
        beat_phase: float,  # 节拍相位，在每个 beat 周期内从 0→1 循环，驱动高频扰动
    ) -> np.ndarray:
        """
        构建 (H, W, 2) float32 向量场，每个像素存储 [fx, fy] 方向向量。

        由三层正弦波叠加组成，各层有不同的空间频率和时间频率：

        层 1（低频大波浪，幅度随 turbulence 衰减）：
          空间频率 3.0×，时间频率 0.5×phase
          turbulence 高时幅度从 1.0 降至 0.7（被湍流层盖过）

        层 2（中频旋转涡旋，幅度固定 0.5）：
          fx 和 fy 使用交叉的空间坐标（X-2Y, 5X+7Y），产生旋转效果
          相同的时间相位驱动使整体旋转

        层 3（高频随机扰动，幅度正比于 turbulence）：
          空间频率 13×，时间快速变化（phase×2.7），增加混沌感
          beat_phase 加入其中，使每个节拍产生可见的场变动
        """
        xx = np.linspace(0.0, 1.0, self.W, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, self.H, dtype=np.float32)
        X, Y = np.meshgrid(xx, yy)  # 广播生成完整坐标网格，各为 (H, W)

        # 层 1：大尺度流动，turbulence 高时被削弱（流动感退位于混沌感）
        A1 = 1.0 - turbulence * 0.3
        fx1 = A1 * np.sin(2 * np.pi * (3.0 * X + 0.5 * phase))
        # +0.25 相位偏移使 x/y 方向不同步
        fy1 = A1 * np.cos(2 * np.pi * (3.0 * Y + 0.5 * phase + 0.25))

        # 层 2：中频旋转涡旋，贡献固定强度的旋流感
        A2 = 0.5
        fx2 = A2 * np.sin(2 * np.pi * (7.0 * X - 2.0 * Y + phase * 1.3))
        fy2 = A2 * np.cos(2 * np.pi * (5.0 * X + 7.0 * Y + phase * 1.3))

        # 层 3：高频湍流，turbulence=0 时完全消失，turbulence=1 时最强
        A3 = turbulence * 0.8
        fx3 = A3 * np.sin(2 * np.pi * (13.0 * X + 11.0 *
                          Y + phase * 2.7 + beat_phase))
        fy3 = A3 * np.cos(2 * np.pi * (11.0 * X - 13.0 * Y + phase * 2.7))

        field = np.stack([fx1 + fx2 + fx3, fy1 + fy2 + fy3], axis=-1)
        return field.astype(np.float32)  # (H, W, 2)

    def update(
        self,
        visual_state: VisualState,
        phase: float,
        beat_phase: float,
        beat_impulse: float = 0.0,  # 1.0 表示当前帧是节拍，0.0 表示非节拍
    ) -> None:
        """
        推进粒子物理状态一帧（物理更新 + 寿命管理 + 轨迹记录）。

        物理更新步骤（向量化，无 Python 循环）：
          1. 从缓存向量场中采样每粒子所在位置的场向量
          2. 将场向量累加到粒子速度（缩放系数 = speed × turbulence × 0.1）
          3. 若当前帧是节拍，对所有粒子施加随机方向的冲击（"跳舞"效果）
          4. 速度乘以阻尼系数 0.95（模拟空气阻力，防止速度无限累积）
          5. 位置 += 速度 × speed × 0.1（速度倍率保证不同 speed 下的比例感）
          6. 边界处理：位置取模 [W, H]（环形边界，粒子从对边出现）

        寿命管理：
          超过最大寿命的粒子随机重生（新位置/速度/寿命），
          这使画面保持动态流动感，不会出现"死寂"的空白区域。
        """
        # 向量场每 3 帧重建一次（节省 2/3 的 meshgrid 计算）
        self._field_age += 1
        if self._cached_field is None or self._field_age >= 3:
            self._cached_field = self.build_vector_field(
                visual_state.field_turbulence, phase, beat_phase
            )
            self._field_age = 0

        field = self._cached_field
        speed = visual_state.particle_speed

        # 只操作活跃粒子
        idx = np.where(self.active)[0]
        if len(idx) == 0:
            return

        # 采样向量场：将粒子浮点位置转换为整数像素索引，查找对应场向量
        px = np.clip(self.pos[idx, 0].astype(np.int32), 0, self.W - 1)
        py = np.clip(self.pos[idx, 1].astype(np.int32), 0, self.H - 1)
        fv = field[py, px]  # (N_active, 2)：每粒子对应的场向量

        self._last_sampled_fv = fv  # test

        # 速度累积：场向量 × 速度倍率 × 湍流强度（turbulence=0 时场向量不影响速度）
        self.vel[idx] += fv * speed * visual_state.field_turbulence * 0.1

        # 节拍冲击：在节拍时刻向随机方向推动粒子（创造"节奏跳动"的视觉感）
        if beat_impulse > 0.05:
            rand_dirs = self._rng.standard_normal(
                (len(idx), 2)).astype(np.float32)
            norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-8
            rand_dirs /= norms  # 归一化为单位方向向量
            self.vel[idx] += beat_impulse * rand_dirs * speed * 0.5

        # 阻尼：每帧速度衰减 5%，防止粒子无限加速
        self.vel[idx] *= 0.95

        # 位置更新：位置增量 = 速度 × speed × 0.1（0.1 是全局速度缩放系数）
        self.pos[idx] += self.vel[idx] * speed * 0.1

        # 环形边界：粒子超出边界时从对边出现（避免粒子累积在边缘）
        self.pos[idx, 0] %= self.W
        self.pos[idx, 1] %= self.H

        # 寿命管理：超龄粒子随机重生
        self.age[idx] += 1
        dead = idx[self.age[idx] >= self.life[idx]]
        for i in dead:
            self.pos[i, 0] = self._rng.random() * self.W
            self.pos[i, 1] = self._rng.random() * self.H
            self.vel[i] = (self._rng.random(2) - 0.5) * 2.0
            self.age[i] = 0
            self.life[i] = int(self._rng.integers(60, 180))

        # 记录本帧活跃粒子位置快照（用于轨迹渲染）
        self.trail_history.append(self.pos[idx].copy())

    def render_trails(self, frame: np.ndarray, trail_length: int) -> None:
        """
        将粒子历史轨迹半透明叠加到帧上，制造"拖尾"效果。

        实现原理：
          1. 取 trail_history 最近 trail_length 帧的位置快照
          2. 在 overlay（帧的副本）上绘制每帧的粒子点（半径 1px）
          3. 用 addWeighted(overlay×0.4 + frame×0.6) 混合，使轨迹半透明

        alpha 渐变：从旧→新，alpha 从 0.05 线性增长到 0.4，
        视觉效果：近期位置较亮，远期位置渐隐（fade-out 效果）。

        注意：每帧都完整重绘整个轨迹历史，计算量随 trail_length 线性增加。
        trail_length 越长，渲染越慢，但视觉上流畅感越强。
        """
        history = list(
            self.trail_history)[-trail_length:]  # 只取最近 trail_length 帧
        n_hist = len(history)
        if n_hist < 2:
            return  # 历史不足 2 帧时无需绘制

        overlay = frame.copy()  # 在副本上绘制，最后与原帧 alpha 混合
        idx = np.where(self.active)[0]
        n_active = len(idx)

        for hi, positions in enumerate(history):
            # alpha 从 0.05（最老帧）线性增长到 0.4（最新帧）
            alpha = 0.05 + 0.35 * (hi / max(1, n_hist - 1))
            n_pts = min(len(positions), n_active)
            for pi in range(n_pts):
                x, y = int(positions[pi, 0]), int(positions[pi, 1])
                if 0 <= x < self.W and 0 <= y < self.H:
                    ci = idx[pi] if pi < len(idx) else pi % len(idx)
                    bgr = tuple(int(c) for c in self.color[ci])
                    cv2.circle(overlay, (x, y), 1, bgr, -1)  # 1px 实心圆（轨迹点）

        # 将轨迹层以 40% 权重叠加到帧上（40% overlay + 60% 原帧）
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    def render_particles(self, frame: np.ndarray) -> None:
        """
        在帧上绘制所有活跃粒子（2px 实心圆点）。
        在轨迹之后绘制，确保粒子本体始终位于轨迹上方，视觉层次清晰。
        """
        idx = np.where(self.active)[0]
        for i in idx:
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            if 0 <= x < self.W and 0 <= y < self.H:
                bgr = tuple(int(c) for c in self.color[i])
                cv2.circle(frame, (x, y), 2, bgr, -1)  # 2px 实心圆（粒子本体）


# ── VideoRenderer：帧合成与视频写入 ──────────────────────────

class VideoRenderer:
    """
    完整帧的合成器，将多个渲染层按顺序叠加，输出到 VideoWriter。

    渲染顺序（从底层到顶层）：
      背景（全帧径向渐变）→ 轨迹（半透明叠加）→ 粒子（实心点）→ 波形（折线）
    """

    def __init__(self, output_video_path: str, fps: int = FPS,
                 width: int = WIDTH, height: int = HEIGHT):
        self.W = width
        self.H = height
        self.fps = fps
        self.frame_idx = 0

        # mp4v 编码器：生成 .avi 临时文件，后续由 ffmpeg 合并音频并转为 .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (width, height))
        self.particles = ParticleSystem(width, height)

    def _render_background(self, visual_state: VisualState, phase: float) -> np.ndarray:
        """
        渲染径向渐变背景：从画面中心向边缘渐变。

        颜色计算：
          center_bgr：中心颜色，使用主色相，饱和度 ×0.6（略去饱和），亮度 ×0.4（偏暗）
          edge_bgr：边缘颜色，色相偏移 30°（轻微互补色），饱和度/亮度更低，营造深邃感

        dist 归一化到 [0, 1]（0=中心，1=角落），每个通道独立线性插值。
        背景刻意偏暗，为粒子和轨迹提供高对比度的展示空间。
        """
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        cx, cy = self.W / 2.0, self.H / 2.0
        Y_grid, X_grid = np.mgrid[0:self.H, 0:self.W]
        dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)
        dist /= (dist.max() + 1e-8)  # 归一化，防止除零

        h = visual_state.hue_base
        s = visual_state.saturation
        b = visual_state.brightness

        center_bgr = _hsv_to_bgr(h, s * 0.6, b * 0.4)          # 中心：低饱和/低亮
        edge_bgr = _hsv_to_bgr((h + 30) % 360.0, s * 0.4, b * 0.15)  # 边缘：更暗更淡

        # 每通道独立插值（numpy 广播，一次操作完成全帧）
        for c in range(3):
            frame[:, :, c] = (
                center_bgr[c] * (1.0 - dist) + edge_bgr[c] * dist
            ).astype(np.uint8)

        return frame

    def _render_waveform(self, frame: np.ndarray, audio_chunk: np.ndarray,
                         hue_base: float) -> None:
        """
        在画面底部 80px 区域绘制当前帧的音频波形折线。

        实现步骤：
          1. 将 audio_chunk（sr/fps 个样本，约 735 个）插值重采样到 W=1280 个点
          2. 映射到 [底部-80px, 底部] 的像素坐标（以中线为零点，振幅 36px）
          3. 用 cv2.polylines 绘制抗锯齿折线（颜色与当前主色相一致）

        波形颜色：使用 hue_base，全饱和全亮（S=1, V=1），
        在深色背景上呈现鲜艳的示波器效果。
        """
        if len(audio_chunk) == 0:
            return

        # 将 audio_chunk 重采样到 W 个像素点（线性插值）
        waveform = np.interp(
            np.linspace(0, len(audio_chunk) - 1, self.W),
            np.arange(len(audio_chunk)),
            audio_chunk.astype(np.float32),
        )

        wave_region_h = 80                        # 波形区域高度（像素）
        baseline_y = self.H - wave_region_h // 2  # 波形中心线的 y 坐标
        amplitude = wave_region_h // 2 - 4        # 波形最大振幅（像素）

        # 将归一化音频值 [-1, 1] 映射到像素 y 坐标（向上为负）
        ys = np.clip(
            (baseline_y - waveform * amplitude).astype(np.int32),
            self.H - wave_region_h,  # 上边界
            self.H - 1,               # 下边界（不超出画面）
        )
        xs = np.arange(self.W, dtype=np.int32)
        # cv2.polylines 要求 (N,1,2) 格式
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)

        bgr = _hsv_to_bgr(hue_base, 1.0, 1.0)  # 全饱和全亮，与主色相一致
        cv2.polylines(frame, [pts], False, bgr, 1, cv2.LINE_AA)  # 抗锯齿折线

    # test
    def _render_debug_text(self, frame, visual_state, particles):  # test
        idx = np.where(particles.active)[0]  # test
        if len(idx) == 0:  # test
            return  # test
# test
        # 平均速度#test
        vel = particles.vel[idx]  # test
        speed_mean = float(np.mean(np.linalg.norm(vel, axis=1)))  # test
# test
        # 全局向量场强度#test
        if particles._cached_field is not None:  # test
            field_mag = np.linalg.norm(particles._cached_field, axis=2)  # test
            fv_mean = float(np.mean(field_mag))  # test
        else:  # test
            fv_mean = 0.0  # test
# test
        # 粒子采样到的局部向量#test
        if hasattr(particles, "_last_sampled_fv"):  # test
            fv_local = np.linalg.norm(
                particles._last_sampled_fv, axis=1)  # test
            fv_local_mean = float(np.mean(fv_local))  # test
        else:  # test
            fv_local_mean = 0.0  # test
# test
        text = (  # test
            f"speed={speed_mean:.2f}  "  # test
            f"fv={fv_mean:.3f}  "  # test
            f"fv_local={fv_local_mean:.3f}  "  # test
            f"turb={visual_state.field_turbulence:.2f}"  # test
        )  # test
# test
        cv2.putText(  # test
            frame,  # test
            text,  # test
            (20, 40),  # test
            cv2.FONT_HERSHEY_SIMPLEX,  # test
            0.7,  # test
            (255, 255, 255),  # test
            2,  # test
            cv2.LINE_AA,  # test
        )  # test

    # test
    def _render_vector_field(self, frame, particles, step=80):  # test
        if particles._cached_field is None:  # test
            return  # test
# test
        field = particles._cached_field  # test
        H, W = field.shape[:2]  # test
# test
        for y in range(0, H, step):  # test
            for x in range(0, W, step):  # test
                fx, fy = field[y, x]  # test
# test
                end_x = int(x + fx * 20)  # test
                end_y = int(y + fy * 20)  # test
# test
                cv2.arrowedLine(  # test
                    frame,  # test
                    (x, y),  # test
                    (end_x, end_y),  # test
                    (0, 255, 0),  # test
                    1,  # test
                    tipLength=0.3,  # test
                )  # test

    # test
    def _render_velocity(self, frame, particles, max_draw=100):  # test
        idx = np.where(particles.active)[0][:max_draw]  # test
# test
        for i in idx:  # test
            x = int(particles.pos[i, 0])  # test
            y = int(particles.pos[i, 1])  # test
            vx, vy = particles.vel[i]  # test
# test
            end = (int(x + vx * 5), int(y + vy * 5))  # test
# test
            cv2.arrowedLine(  # test
                frame,  # test
                (x, y),  # test
                end,  # test
                (255, 0, 0),  # test
                1,  # test
                tipLength=0.3,  # test
            )  # test

    def render_frame(
        self,
        visual_state: VisualState,
        audio_chunk: np.ndarray,    # 当前视频帧对应的音频样本（约 sr/fps 个）
        beat_impulse: float,        # 节拍冲击信号（1.0=本帧是节拍，0.0=非节拍）
        phase: float,               # 全局时间相位（t × 0.05），驱动场的整体漂移
        beat_phase: float,          # 节拍内相位（0→1 循环），驱动高频场扰动
    ) -> np.ndarray:
        """
        合成完整一帧（(H, W, 3) uint8 BGR 图像），按以下顺序叠加：
          1. 背景：径向渐变，确立色调氛围
          2. 粒子更新：同步更新物理状态（在渲染前更新，保证位置正确）
          3. 轨迹：半透明历史路径，增加流动感
          4. 粒子：实心点，视觉焦点
          5. 波形：音频可视化，增加音乐感
        """
        # 1. 生成背景帧（全黑 canvas 上绘制径向渐变）
        frame = self._render_background(visual_state, phase)

        # 2. 更新粒子状态（数量、颜色、物理位置）
        self.particles.set_count(visual_state.particle_count)
        self.particles.set_colors(
            visual_state.hue_base,
            visual_state.hue_range,
            visual_state.saturation,
            visual_state.brightness,
        )
        self.particles.update(visual_state, phase, beat_phase, beat_impulse)

        # 3. 绘制轨迹（在粒子之前绘制，轨迹在下层）
        self.particles.render_trails(frame, visual_state.trail_length)

        # 4. 绘制粒子本体（在轨迹上方）
        self.particles.render_particles(frame)

        # 5. 绘制底部波形（最顶层，始终可见）
        self._render_waveform(frame, audio_chunk, visual_state.hue_base)

        # 6.test
        if TEST:  # test
            self._render_debug_text(
                frame, visual_state, self.particles)  # test
            self._render_vector_field(frame, self.particles)  # test
            self._render_velocity(frame, self.particles)  # test

        self.frame_idx += 1
        return frame

    def write_frame(self, frame: np.ndarray) -> None:
        """将帧写入 VideoWriter（按 fps 速率写入，不自动添加时间戳）。"""
        self.writer.write(frame)

    def release(self) -> None:
        """释放 VideoWriter 资源，完成视频文件写入。必须调用，否则输出文件可能损坏。"""
        self.writer.release()
