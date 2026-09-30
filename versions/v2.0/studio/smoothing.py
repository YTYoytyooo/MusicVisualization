"""Optional offline V/A presentation baseline; never modifies model samples."""
import math
import numpy as np


def validate_smoothing(config=None):
    if config is None:
        return {'enabled': False, 'window_seconds': .5}
    if not isinstance(config, dict) or set(config) - {'enabled', 'window_seconds'}:
        raise ValueError('非法平滑配置')
    enabled, window = config.get('enabled', False), config.get('window_seconds', .5)
    if not isinstance(enabled, bool):
        raise ValueError('平滑开关必须为布尔值')
    if isinstance(window, bool) or not isinstance(window, (int, float)) or not math.isfinite(window) or not .1 <= window <= 5:
        raise ValueError('平滑窗口必须在0.1至5秒之间')
    return {'enabled': enabled, 'window_seconds': float(window)}


def smooth_source(raw, step_us, config=None):
    config = validate_smoothing(config)
    result = np.array(raw, dtype=np.float64, copy=True)
    if not config['enabled']:
        return result
    radius = int(math.floor(config['window_seconds'] * 1e6 / step_us / 2))
    cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(result[:, :2], axis=0)))
    indexes = np.arange(len(result))
    lo, hi = np.maximum(0, indexes - radius), np.minimum(len(result), indexes + radius + 1)
    result[:, :2] = (cumulative[hi] - cumulative[lo]) / (hi - lo)[:, None]
    return result


def baseline_at(raw, times_us, step_us=100000, config=None):
    from .timeline import raw_at
    config = validate_smoothing(config)
    result = raw_at(raw, times_us, step_us)
    if config['enabled']:
        source = smooth_source(raw, step_us, config)
        clock = np.arange(len(source)) * step_us
        for column in (0, 1):
            result[:, column] = np.interp(times_us, clock, source[:, column])
    return result


def emotion_values(raw, times_us, edits, step_us=100000, config=None):
    from .timeline import apply_edits, EMOTIONS
    return apply_edits(baseline_at(raw, times_us, step_us, config), times_us, edits, 'emotion', EMOTIONS)


def check_times(raw, duration_us, edits, step_us=100000, config=None):
    """All baseline knots and offset stationary points on integer microseconds."""
    from .timeline import validation_times, support
    points = set(validation_times(len(raw), duration_us, edits, step_us).tolist())
    if not validate_smoothing(config)['enabled']:
        return np.array(sorted(points), dtype=np.int64)
    source = smooth_source(raw, step_us, config)
    for edit in edits:
        if not edit['enabled'] or edit['layer'] != 'emotion' or edit['operation'] != 'interval_offset' or not edit['value']:
            continue
        column = ('valence', 'arousal').index(edit['field'])
        left, right = support(edit)
        for index in range(max(0, left // step_us), min(len(source) - 1, right // step_us + 1)):
            a, b = index * step_us, (index + 1) * step_us
            slope = (source[index + 1, column] - source[index, column]) / step_us
            for start, end, sign in ((left, edit['start_us'], 1), (edit['end_us'], right, -1)):
                if end <= start or end < a or start > b:
                    continue
                target = -slope * (end - start) / (edit['value'] * sign)
                roots = []
                if edit['interpolation'] == 'smoothstep' and 0 <= target <= 1.5:
                    d = math.sqrt(max(0, 1 - 2 * target / 3))
                    roots = [(1 - d) / 2, (1 + d) / 2]
                elif edit['interpolation'] == 'smootherstep' and 0 <= target <= 1.875:
                    d = math.sqrt(max(0, 1 - 4 * math.sqrt(target / 30)))
                    roots = [(1 - d) / 2, (1 + d) / 2]
                for root in roots:
                    t = start + root * (end - start) if sign == 1 else end - root * (end - start)
                    if a <= t <= b:
                        points.update(x for x in (math.floor(t)-1, math.floor(t), math.ceil(t), math.ceil(t)+1) if 0 <= x <= duration_us)
    return np.array(sorted(points), dtype=np.int64)
