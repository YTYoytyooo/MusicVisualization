"""Optional test-video HUD. Distinguishes raw predictions and manual revisions.

Coordinates use a 1280x720 design canvas and scale with the output frame.
Prediction i belongs to the pipeline timestamp i * frame_duration. Values are
held until the next prediction; no smoothing or future trajectory is displayed.
Zoom bounds are calculated once from the full offline sequence, with real-value
ticks and a full-range inset. Only display coordinates change, not predictions.
"""

import os
import hashlib
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
                 font_path=None, raw_emotion_states=None,
                 source_label='MODEL PREDICTION', times=None):
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
        self.times = (np.arange(len(self.values)) * frame_duration if times is None
                      else np.asarray(times, dtype=np.float64))
        if (self.times.shape != (len(self.values),)
                or not np.isfinite(self.times).all()
                or (np.diff(self.times) <= 0).any() or self.times[0] != 0
                or self.times[-1] > duration + 1e-9):
            raise ValueError('times must match predictions, start at zero, increase, and fit duration')
        self.raw_values = None
        if raw_emotion_states is not None:
            self.raw_values = np.asarray([(s['valence'], s['arousal'])
                                         for s in raw_emotion_states], dtype=np.float64)
            if (self.raw_values.shape != self.values.shape
                    or not np.isfinite(self.raw_values).all()
                    or (np.abs(self.raw_values) > 1).any()):
                raise ValueError('Raw predictions must match effective predictions and lie in [-1, 1]')
        self.source_label = ' '.join(str(source_label).split())
        if not self.source_label or not self.source_label.isascii() or len(self.source_label) > 90:
            raise ValueError('source_label must be 1-90 ASCII characters')
        combined = (self.values if self.raw_values is None else
                    np.concatenate([self.values, self.raw_values]))
        self.zoom_bounds = self._zoom_bounds(combined)
        self.points = np.array([self.point_for(v, a) for v, a in self.values],
                               dtype=np.int32)
        self.overview_points = np.array([
            self._map_point(v, a, self.FULL_BOUNDS, self.OVERVIEW_RECT)
            for v, a in self.values], dtype=np.int32)
        self.raw_points = None
        self.raw_overview_points = None
        if self.raw_values is not None:
            self.raw_points = np.array([self.point_for(v, a) for v, a in self.raw_values], dtype=np.int32)
            self.raw_overview_points = np.array([
                self._map_point(v, a, self.FULL_BOUNDS, self.OVERVIEW_RECT)
                for v, a in self.raw_values], dtype=np.int32)
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
        self._text(self.header, 'TEST OUTPUT / ' + self.source_label, (18, 65), 0.43)
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
        self._text(self.panel, 'Raw: gray / Used: cyan' if self.raw_values is not None
                   else 'Earlier history: dim', (18, 482), .36)
        self._text(self.panel, 'Estimates / edits, not ground truth', (18, 503), .31)
        fingerprint = hashlib.sha256()
        for data in (self.header, self.panel, self.times, self.values, self.raw_values):
            if data is not None:
                fingerprint.update(np.ascontiguousarray(data).tobytes())
        self._fingerprint = fingerprint.hexdigest()
        self.reset_history()

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

    def _draw_trajectory(self, panel, points, index, recent_start, radius, current_point=None):
        current = points[index] if current_point is None else current_point
        # Historical samples already live in the incremental mask. Only the
        # exact current-frame connector and bounded 2s highlight are transient.
        if not np.array_equal(points[index], current):
            cv2.line(panel, tuple(points[index]), tuple(current), (120, 105, 85), 1, cv2.LINE_AA)
        recent = points[recent_start:index + 1]
        if not len(recent) or not np.array_equal(recent[-1], current):
            recent = np.vstack([recent, current])
        if len(recent) > 1:
            cv2.polylines(panel, [recent], False, self.CYAN, 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(current), radius, self.ORANGE, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(current), radius + 2, self.WHITE, 1, cv2.LINE_AA)

    def reset_history(self):
        self._history_index = -1
        self._raw_history_mask = np.zeros(self.panel.shape[:2], dtype=np.uint8)
        self._effective_history_mask = np.zeros(self.panel.shape[:2], dtype=np.uint8)

    def _advance_history(self, index):
        if index < self._history_index:
            self.reset_history()
        # A forward seek and sequential playback use exactly the same ordered
        # segments, including AA endpoint accumulation, so pixels are identical.
        for current in range(max(1, self._history_index + 1), index + 1):
            for points in (self.points, self.overview_points):
                cv2.line(self._effective_history_mask, tuple(points[current - 1]),
                         tuple(points[current]), 255, 1, cv2.LINE_AA)
            if self.raw_points is not None:
                for points in (self.raw_points, self.raw_overview_points):
                    cv2.line(self._raw_history_mask, tuple(points[current - 1]),
                             tuple(points[current]), 255, 1, cv2.LINE_AA)
        self._history_index = index

    @staticmethod
    def _composite_history(panel, mask, color):
        alpha = mask.astype(np.uint16)[:, :, None]
        panel[:] = ((panel.astype(np.uint16) * (255 - alpha)
                     + np.asarray(color, dtype=np.uint16) * alpha + 127) // 255).astype(np.uint8)

    def export_state(self):
        return {'metadata': {'schema_version': 1, 'fingerprint': self._fingerprint,
                             'history_index': self._history_index},
                'arrays': {'raw_history_mask': self._raw_history_mask.copy(),
                           'effective_history_mask': self._effective_history_mask.copy()}}

    def restore_state(self, snapshot):
        metadata, arrays = snapshot['metadata'], snapshot['arrays']
        index = metadata.get('history_index')
        if (metadata.get('schema_version') != 1 or metadata.get('fingerprint') != self._fingerprint
                or isinstance(index, bool) or not isinstance(index, int)
                or not -1 <= index < len(self.values)):
            raise ValueError('HUD checkpoint does not match this prediction timeline')
        if set(arrays) != {'raw_history_mask', 'effective_history_mask'}:
            raise ValueError('HUD checkpoint arrays are incomplete')
        for array in arrays.values():
            if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape != self.panel.shape[:2]:
                raise ValueError('Invalid HUD checkpoint canvas')
        self._raw_history_mask = arrays['raw_history_mask'].copy()
        self._effective_history_mask = arrays['effective_history_mask'].copy()
        self._history_index = index

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

    def draw(self, frame, time_seconds, current_values=None):
        """Draw past samples plus optional exact (valence, arousal) at this frame."""
        index, valence, arousal = self.sample_at(time_seconds)
        if current_values is not None:
            current_values = np.asarray(current_values, dtype=np.float64)
            if (current_values.shape != (2,) or not np.isfinite(current_values).all()
                    or (np.abs(current_values) > 1).any()):
                raise ValueError('current_values must contain finite V/A in [-1, 1]')
            valence, arousal = map(float, current_values)
        elapsed = float(np.clip(time_seconds, 0, self.duration))
        header = self.header.copy()
        self._text(header, f'{format_playback_time(elapsed)} / {format_playback_time(self.duration)}',
                   (900, 32), 0.57, self.CYAN)
        self._text(header, 'AUDIO PLAYBACK TIME', (900, 62), 0.43)
        panel = self.panel.copy()
        self._advance_history(index)
        if self.raw_points is not None:
            self._composite_history(panel, self._raw_history_mask, (150, 150, 150))
        self._composite_history(panel, self._effective_history_mask, (120, 105, 85))
        recent_start = self.recent_start_at(elapsed)
        if self.raw_points is not None:
            for points in (self.raw_points, self.raw_overview_points):
                cv2.circle(panel, tuple(points[index]), 2, (150, 150, 150), -1, cv2.LINE_AA)
        self._draw_trajectory(panel, self.points, index, recent_start, radius=5,
                              current_point=self.point_for(valence, arousal))
        self._draw_trajectory(panel, self.overview_points, index, recent_start, radius=2,
                              current_point=self._map_point(valence, arousal, self.FULL_BOUNDS,
                                                            self.OVERVIEW_RECT))
        self._text(panel, f'V {valence:+.3f}', (18, 400), .62)
        self._text(panel, f'A {arousal:+.3f}', (18, 425), .62)
        self._text(panel, f'Frame @ {elapsed:.3f}s' if current_values is not None
                   else f'Sample @ {self.times[index]:.3f}s', (18, 447), .40)
        self._paste(frame, header, 16, 16)
        self._paste(frame, panel, 924, 118)
        return frame

    def draw_values(self, frame, time_seconds, valence, arousal):
        return self.draw(frame, time_seconds, current_values=(valence, arousal))
