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
    from .timeline import TimedValues, source_clock, source_mask, connected
    config = validate_smoothing(config)
    result = np.array(raw, dtype=np.float64, copy=True)
    clock, mask = source_clock(raw, step_us), source_mask(raw)
    if config['enabled']:
        radius = config['window_seconds'] * 1e6 / 2
        joins = connected(raw, step_us)
        for column in (0, 1):
            # Separate connected runs per dimension. A gap in V does not erase A,
            # and smoothing never borrows observations across a missing span.
            starts = np.flatnonzero(mask[:, column] & np.r_[True, ~(mask[:-1, column] & joins)])
            for start in starts:
                end = start + 1
                while end < len(raw) and mask[end, column] and joins[end-1]:
                    end += 1
                times = clock[start:end]
                series = np.asarray(raw)[start:end, column]
                cumulative = np.r_[0., np.cumsum(series)]
                lo = np.searchsorted(times, times-radius, side='left')
                hi = np.searchsorted(times, times+radius, side='right')
                result[start:end, column] = (cumulative[hi]-cumulative[lo])/(hi-lo)
    return TimedValues(result, clock, mask, getattr(raw, 'step_us', None) or step_us,
                       getattr(raw, 'duration_us', None), getattr(raw, 'hold_last_to_end', False))


def baseline_at(raw, times_us, step_us=100000, config=None):
    from .timeline import raw_at, source_clock, source_mask, connected
    config = validate_smoothing(config)
    result = raw_at(raw, times_us, step_us)
    if config['enabled']:
        source = smooth_source(raw, step_us, config)
        clock = source_clock(source, step_us)
        t = np.asarray(times_us, dtype=np.int64)
        index = np.clip(np.searchsorted(clock, t, side='right')-1, 0, len(source)-1)
        right = np.minimum(index+1, len(source)-1)
        joins = np.r_[connected(source, step_us), False]
        mask = source_mask(source)
        for column in (0, 1):
            use = ((t >= clock[index]) & (t < clock[right]) & joins[index]
                   & mask[index, column] & mask[right, column])
            ratio = (t[use]-clock[index[use]])/(clock[right[use]]-clock[index[use]])
            result[use, column] = source[index[use], column]*(1-ratio)+source[right[use], column]*ratio
            # Singleton runs and the final sample keep native held values.
            hold = np.isfinite(result[:, column]) & ~use
            result[hold, column] = source[index[hold], column]
    return result


def emotion_values(raw, times_us, edits, step_us=100000, config=None):
    from .timeline import apply_edits, EMOTIONS
    return apply_edits(baseline_at(raw, times_us, step_us, config), times_us, edits, 'emotion', EMOTIONS)


def check_times(raw, duration_us, edits, step_us=100000, config=None):
    """All baseline knots and offset stationary points on integer microseconds."""
    from .timeline import validation_times, support, source_clock, source_mask, connected
    points = set(validation_times(len(raw), duration_us, edits, step_us).tolist())
    clock = source_clock(raw, step_us)
    points.update(int(t) for t in clock)
    points.update(max(0, int(t)-1) for t in clock)
    points.update(min(duration_us, int(t)+(getattr(raw, 'step_us', None) or step_us)) for t in clock)
    if not validate_smoothing(config)['enabled']:
        return np.array(sorted(points), dtype=np.int64)
    source = smooth_source(raw, step_us, config)
    mask, joins = source_mask(source), connected(source, step_us)
    for edit in edits:
        if not edit['enabled'] or edit['layer'] != 'emotion' or edit['operation'] != 'interval_offset' or not edit['value']:
            continue
        column = ('valence', 'arousal').index(edit['field'])
        left, right = support(edit)
        for index in range(len(source)-1):
            a, b = int(clock[index]), int(clock[index+1])
            if b < left or a > right or not joins[index] or not mask[index:index+2, column].all():
                continue
            slope = (source[index + 1, column] - source[index, column]) / (b-a)
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
