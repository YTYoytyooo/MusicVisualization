"""Optional test-video HUD. Displays model outputs, never pseudo-labels.

Coordinates use a 1280x720 design canvas and scale with the output frame.
Prediction i belongs to the pipeline timestamp i * frame_duration. Values are
held until the next prediction; no smoothing or future trajectory is displayed.
Zoom bounds are calculated once from the full offline sequence, with real-value
ticks and a full-range inset. Only display coordinates change, not predictions.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def format_playback_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    minutes, remainder = divmod(milliseconds, 60000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f'{minutes:02d}:{whole_seconds:02d}.{fraction:03d}'


def _title_font(title: str, font_path=None):
    # OpenCV's built-in fonts cannot draw Chinese song names.
    if font_path:
        return ImageFont.truetype(str(font_path), 24)
    candidates = [
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc',
        Path('/System/Library/Fonts/PingFang.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ]
    for path in candidates:
        if path.is_file():
            if not title.isascii() and path.name == 'DejaVuSans.ttf':
                continue
            return ImageFont.truetype(str(path), 24)
    if not title.isascii():
        raise ValueError('No Unicode title font found. Use --overlay-font with a font supporting the song name.')
    return ImageFont.load_default(size=24)


class PredictionOverlay:
    BACKGROUND = (32, 22, 12)  # BGR
    WHITE = (245, 238, 225)
    CYAN = (235, 213, 96)
    ORANGE = (108, 180, 255)
    MIN_ZOOM_SPAN = 0.1
    HIGHLIGHT_SECONDS = 2.0
    MAIN_RECT = (68, 82, 308, 322)
    OVERVIEW_RECT = (228, 397, 316, 485)
    FULL_BOUNDS = ((-1., 1.), (-1., 1.))

    def __init__(self, song_name, duration, emotion_states, frame_duration=0.1,
                 font_path=None):
        if not np.isfinite(frame_duration) or frame_duration <= 0:
            raise ValueError('frame_duration must be positive and finite')
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('duration must be positive and finite')
        self.values = np.asarray([(s['valence'], s['arousal'])
                                  for s in emotion_states], dtype=np.float64)
        if self.values.size == 0 or not np.isfinite(self.values).all():
            raise ValueError('The prediction overlay requires nonempty, finite V-A predictions')
        if (np.abs(self.values) > 1).any():
            raise ValueError('V-A predictions must be in [-1, 1]')
        self.duration = float(duration)
        self.frame_duration = float(frame_duration)
        self.times = np.arange(len(self.values)) * frame_duration
        self.zoom_bounds = self._zoom_bounds(self.values)
        self.points = np.array([self.point_for(v, a) for v, a in self.values],
                               dtype=np.int32)
        self.overview_points = np.array([
            self._map_point(v, a, self.FULL_BOUNDS, self.OVERVIEW_RECT)
            for v, a in self.values], dtype=np.int32)
        self.song_name = ' '.join(str(song_name).split()) or 'Untitled'
        self.title_max_width = 850
        font = _title_font(self.song_name, font_path)
        title = self.song_name
        if font.getlength(title) > self.title_max_width:
            while title and font.getlength(title + '...') > self.title_max_width:
                title = title[:-1]
            title += '...'
        self.title_width = font.getlength(title)
        header_image = Image.new('RGB', (1248, 80), self.BACKGROUND[::-1])
        ImageDraw.Draw(header_image).text((18, 11), title, font=font,
                                         fill=self.WHITE[::-1])
        self.header = np.array(header_image)[:, :, ::-1].copy()
        self._text(self.header, 'TEST OUTPUT / MODEL PREDICTION', (18, 65), 0.48)
        self.panel = np.full((510, 340, 3), self.BACKGROUND, dtype=np.uint8)
        self._text(self.panel, 'VALENCE - AROUSAL', (18, 28), 0.65)
        self._text(self.panel, 'ZOOMED VIEW / fixed per song', (18, 49), 0.43)
        self._text(self.panel, 'Arousal', (68, 71), 0.43)
        left, top, right, bottom = self.MAIN_RECT
        (v_low, v_high), (a_low, a_high) = self.zoom_bounds
        for value in np.linspace(v_low, v_high, 3):
            x, _ = self.point_for(value, a_low)
            cv2.line(self.panel, (x, top), (x, bottom), (76, 62, 46), 1)
            label = f'{value:+.3f}'
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .36, 1)[0][0]
            self._text(self.panel, label, (min(x - width // 2, 335 - width), 342), .36)
        for value in np.linspace(a_low, a_high, 3):
            _, y = self.point_for(v_low, value)
            cv2.line(self.panel, (left, y), (right, y), (76, 62, 46), 1)
            self._text(self.panel, f'{value:+.3f}', (8, y + 4), .36)
        self._text(self.panel, 'Valence', (152, 363), .43)
        self._text(self.panel, 'FULL [-1, +1]', (222, 384), .36)
        left, top, right, bottom = self.OVERVIEW_RECT
        for value in (-1, 0, 1):
            x, y = self._map_point(value, value, self.FULL_BOUNDS, self.OVERVIEW_RECT)
            cv2.line(self.panel, (x, top), (x, bottom), (76, 62, 46), 1)
            cv2.line(self.panel, (left, y), (right, y), (76, 62, 46), 1)
        for value in (-1, 1):
            x, y = self._map_point(value, value, self.FULL_BOUNDS, self.OVERVIEW_RECT)
            self._text(self.panel, f'{value:+d}', (x - 8, 500), .32)
            self._text(self.panel, f'{value:+d}', (206, y + 4), .32)
        zoom_top_left = self._map_point(v_low, a_high, self.FULL_BOUNDS, self.OVERVIEW_RECT)
        zoom_bottom_right = self._map_point(v_high, a_low, self.FULL_BOUNDS, self.OVERVIEW_RECT)
        cv2.rectangle(self.panel, tuple(zoom_top_left), tuple(zoom_bottom_right), self.ORANGE, 1)
        self._text(self.panel, 'Last 2s highlighted', (18, 465), .40)
        self._text(self.panel, 'Earlier history: dim', (18, 482), .40)
        self._text(self.panel, 'Prediction, not ground truth', (18, 503), .34)

    @staticmethod
    def _text(image, text, position, scale=0.55, color=None):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color or PredictionOverlay.WHITE, 1, cv2.LINE_AA)

    @classmethod
    def _zoom_bounds(cls, values):
        lows, highs = values.min(axis=0), values.max(axis=0)
        # Equal units per pixel preserve V-A geometry. Leave 10% padding on
        # each side, limit magnification to 20x, and never hide any samples.
        span = min(2., max(cls.MIN_ZOOM_SPAN, float(max(highs - lows)) * 1.2))
        bounds = []
        for low, high in zip(lows, highs):
            start = float(np.clip((low + high) / 2 - span / 2, -1., 1. - span))
            bounds.append((start, min(1., start + span)))
        return tuple(bounds)

    @staticmethod
    def _map_point(valence, arousal, bounds, rect):
        (v_low, v_high), (a_low, a_high) = bounds
        left, top, right, bottom = rect
        return np.rint([left + (valence - v_low) / (v_high - v_low) * (right - left),
                        top + (a_high - arousal) / (a_high - a_low) * (bottom - top)]).astype(np.int32)

    def point_for(self, valence, arousal):
        return self._map_point(valence, arousal, self.zoom_bounds, self.MAIN_RECT)

    def recent_start_at(self, time_seconds):
        elapsed = float(np.clip(time_seconds, 0, self.duration))
        cutoff = max(0., elapsed - self.HIGHLIGHT_SECONDS)
        return int(np.searchsorted(self.times, cutoff - 1e-9, side='left'))

    def _draw_trajectory(self, panel, points, index, recent_start, radius):
        history = points[:index + 1]
        if len(history) > 1:
            cv2.polylines(panel, [history], False, (120, 105, 85), 1, cv2.LINE_AA)
        recent = points[recent_start:index + 1]
        if len(recent) > 1:
            cv2.polylines(panel, [recent], False, self.CYAN, 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(points[index]), radius, self.ORANGE, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(points[index]), radius + 2, self.WHITE, 1, cv2.LINE_AA)

    def sample_at(self, time_seconds):
        if not np.isfinite(time_seconds):
            raise ValueError('Playback time must be finite')
        # Small tolerance avoids selecting the preceding sample at e.g. 0.3s.
        index = int(np.searchsorted(self.times, time_seconds + 1e-9, side='right') - 1)
        index = int(np.clip(index, 0, len(self.values) - 1))
        return index, float(self.values[index, 0]), float(self.values[index, 1])

    @staticmethod
    def _paste(frame, block, x, y):
        sx, sy = frame.shape[1] / 1280, frame.shape[0] / 720
        left, top = round(x * sx), round(y * sy)
        width, height = round(block.shape[1] * sx), round(block.shape[0] * sy)
        if width > 0 and height > 0:
            frame[top:top + height, left:left + width] = cv2.resize(block, (width, height))

    def draw(self, frame, time_seconds):
        index, valence, arousal = self.sample_at(time_seconds)
        elapsed = float(np.clip(time_seconds, 0, self.duration))
        header = self.header.copy()
        self._text(header, f'{format_playback_time(elapsed)} / {format_playback_time(self.duration)}',
                   (900, 32), 0.57, self.CYAN)
        self._text(header, 'AUDIO PLAYBACK TIME', (900, 62), 0.43)
        panel = self.panel.copy()
        recent_start = self.recent_start_at(elapsed)
        self._draw_trajectory(panel, self.points, index, recent_start, radius=5)
        self._draw_trajectory(panel, self.overview_points, index, recent_start, radius=2)
        self._text(panel, f'V {valence:+.3f}', (18, 400), .62)
        self._text(panel, f'A {arousal:+.3f}', (18, 425), .62)
        self._text(panel, f'Sample @ {self.times[index]:.3f}s', (18, 447), .40)
        self._paste(frame, header, 16, 16)
        self._paste(frame, panel, 924, 118)
        return frame
