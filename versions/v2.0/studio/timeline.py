"""Pure, deterministic evaluation of immutable source data and bounded edits."""
from copy import deepcopy
import math
import numpy as np

EMOTIONS = ('valence', 'arousal', 'energy', 'tension', 'brightness')
VISUAL_BOUNDS = {
    'hue_base': (0., 360.), 'hue_range': (20., 120.),
    'saturation': (.3, 1.), 'brightness': (.1, .9),
    'particle_count': (50, 500), 'particle_speed': (.5, 8.),
    'field_turbulence': (0., 1.), 'trail_length': (5, 60),
}
INTEGER_FIELDS = {'particle_count', 'trail_length'}
INTERPOLATIONS = ('linear', 'smoothstep', 'smootherstep')


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} 必须是大于等于 {minimum} 的整数')
    return value


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 必须是有限数值')
    return float(value)


def support(edit):
    if edit['operation'] == 'point_target':
        start = end = edit['time_us']
    else:
        start, end = edit['start_us'], edit['end_us']
    return start - edit['transition_in_us'], end + edit['transition_out_us']


def validate_edits(edits, duration_us):
    if not isinstance(edits, list) or len(edits) > 1000:
        raise ValueError('编辑必须是列表，最多 1000 条')
    checked, ids, occupied = [], set(), {}
    for original in edits:
        if not isinstance(original, dict):
            raise ValueError('编辑项必须是对象')
        e = deepcopy(original)
        eid = e.get('id')
        if not isinstance(eid, str) or not eid or len(eid) > 100 or eid in ids:
            raise ValueError('编辑 ID 必须唯一且非空')
        ids.add(eid)
        layer, field = e.get('layer'), e.get('field')
        if layer == 'emotion':
            if field not in ('valence', 'arousal'):
                raise ValueError('情绪编辑第一版仅支持 valence / arousal')
            low, high = -1., 1.
        elif layer == 'visual' and field in VISUAL_BOUNDS:
            low, high = VISUAL_BOUNDS[field]
        else:
            raise ValueError('未知编辑层或字段')
        operation = e.get('operation')
        if operation not in ('point_target', 'interval_set', 'interval_offset'):
            raise ValueError('未知编辑操作')
        e['value'] = number(e.get('value'), 'value')
        if operation != 'interval_offset' and not low <= e['value'] <= high:
            raise ValueError(f'{field} 目标值必须在 [{low}, {high}]')
        if field in INTEGER_FIELDS and operation != 'interval_offset' and not e['value'].is_integer():
            raise ValueError(f'{field} 目标必须是整数')
        for key in ('transition_in_us', 'transition_out_us'):
            e[key] = integer(e.get(key, 0), key)
        if operation == 'point_target':
            e['time_us'] = integer(e.get('time_us'), 'time_us')
            if min(e['transition_in_us'], e['transition_out_us']) <= 0:
                raise ValueError('单点目标必须有非零的前后过渡时间')
        else:
            e['start_us'] = integer(e.get('start_us'), 'start_us')
            e['end_us'] = integer(e.get('end_us'), 'end_us')
            if e['end_us'] <= e['start_us']:
                raise ValueError('区间结束时间必须晚于开始时间')
        left, right = support(e)
        if left < 0 or right > duration_us:
            raise ValueError('编辑及其过渡范围必须位于歌曲时间范围内')
        e.setdefault('enabled', True)
        if not isinstance(e['enabled'], bool):
            raise ValueError('enabled 必须为布尔值')
        e.setdefault('interpolation', 'linear')
        if e['interpolation'] not in INTERPOLATIONS:
            raise ValueError('过渡必须为 linear / smoothstep / smootherstep')
        e['note'] = str(e.get('note', ''))[:2000]
        if e['enabled']:
            key = (layer, field)
            for a, b, previous in occupied.setdefault(key, []):
                overlaps = left < b and a < right
                # Intervals include their endpoints. Touching support is safe
                # only if at least one transition has zero weight there.
                boundary = left if left == b else right if right == a else None
                shared_endpoint = (boundary is not None
                                   and weight_at(e, [boundary])[0] > 0
                                   and weight_at(previous, [boundary])[0] > 0)
                if overlaps or shared_endpoint:
                    raise ValueError(f'{field} 编辑区间重叠，请先禁用或删除已有编辑')
            occupied[key].append((left, right, e))
        checked.append(e)
    return checked


def weight_at(e, times_us):
    t = np.asarray(times_us, dtype=np.float64)
    if e['operation'] == 'point_target':
        start = end = e['time_us']
    else:
        start, end = e['start_us'], e['end_us']
    before, after = e['transition_in_us'], e['transition_out_us']
    w = np.zeros_like(t)
    w[(t >= start) & (t <= end)] = 1.
    if before:
        mask = (t >= start - before) & (t < start)
        w[mask] = (t[mask] - (start - before)) / before
    if after:
        mask = (t > end) & (t <= end + after)
        w[mask] = (end + after - t[mask]) / after
    curve = e.get('interpolation', 'linear')
    if curve == 'smoothstep':
        return w * w * (3. - 2. * w)
    if curve == 'smootherstep':
        return w * w * w * (w * (w * 6. - 15.) + 10.)
    if curve != 'linear':
        raise ValueError('未知过渡曲线')
    return w


def apply_edits(base, times_us, edits, layer, fields):
    values = np.array(base, dtype=np.float64, copy=True)
    if values.shape != (len(times_us), len(fields)) or not np.isfinite(values).all():
        raise ValueError('时间线数据形状或数值非法')
    for e in edits:
        if not e['enabled'] or e['layer'] != layer:
            continue
        column = fields.index(e['field'])
        weights = weight_at(e, times_us)
        current = values[:, column]
        if e['operation'] == 'interval_offset':
            result = current + weights * e['value']
        elif e['field'] == 'hue_base':
            delta = (e['value'] - current + 180) % 360 - 180
            result = current + weights * delta
        else:
            result = current * (1 - weights) + e['value'] * weights
        if e['field'] == 'hue_base':
            result %= 360
        else:
            low, high = (-1., 1.) if layer == 'emotion' else VISUAL_BOUNDS[e['field']]
            if np.any(result < low - 1e-9) or np.any(result > high + 1e-9):
                raise ValueError(f"编辑 {e['id']} 使 {e['field']} 越界；请减小偏移量")
            result = np.clip(result, low, high)
        values[:, column] = result
    return values


def raw_at(raw, times_us, step_us=100000):
    indexes = np.clip(np.asarray(times_us, dtype=np.int64) // step_us, 0, len(raw) - 1)
    return np.asarray(raw, dtype=np.float64)[indexes].copy()


def emotion_at(raw, times_us, edits, step_us=100000):
    return apply_edits(raw_at(raw, times_us, step_us), times_us, edits, 'emotion', EMOTIONS)


def event_times(edits, layer=None):
    points = set()
    for e in edits:
        if not e['enabled'] or (layer is not None and e['layer'] != layer):
            continue
        points.update(support(e))
        if e['operation'] == 'point_target':
            points.add(e['time_us'])
        else:
            points.update((e['start_us'], e['end_us']))
    return points


def validation_times(raw_count, duration_us, edits, step_us=100000):
    # Check both sides of zero-order-hold discontinuities, plus edit knots.
    points = set(range(0, duration_us + 1, step_us)) | event_times(edits)
    points.update(max(0, i * step_us - 1) for i in range(1, raw_count))
    points.add(duration_us)
    return np.array(sorted(points), dtype=np.int64)


def preview_times(raw_count, duration_us, edits, step_us=100000):
    """Display-only grid: bounded regular density, all source jumps and knots.

    This grid must never replace validation_times for validity checks. Curves
    add 32 subdivisions per transition; mandatory source/event points can make
    the total exceed the 6000 regular-point budget on long audio.
    """
    integer(raw_count, 'raw_count', 1)
    integer(duration_us, 'duration_us', 1)
    integer(step_us, 'step_us', 1)
    spacing = max(33334, (duration_us + 5999) // 6000)
    points = set(range(0, duration_us, spacing)) | {duration_us}
    for index in range(raw_count):
        point = index * step_us
        if point > duration_us:
            break
        points.add(point)
        points.add(max(0, point - 1))
    for e in edits:
        if not e['enabled']:
            continue
        if e['operation'] == 'point_target':
            start = end = e['time_us']
        else:
            start, end = e['start_us'], e['end_us']
        left, right = support(e)
        for point in (left, start, end, right):
            points.update((point - 1, point, point + 1))
        for a, b in ((left, start), (end, right)):
            for index in range(33):
                points.add(a + ((b - a) * index + 16) // 32)
    return np.array(sorted(p for p in points if 0 <= p <= duration_us), dtype=np.int64)


def visual_validation_times(plan, duration_us, edits):
    """Exact integer-time probes for visual offset bounds, not a display grid.

    Baseline float fields interpolate linearly. Offsets with eased weights can
    have extrema inside a transition, so knots alone are insufficient. Probe
    stationary points and both sides of baseline integer-rounding changes.
    Set/target blends stay between their valid baseline and target values.
    """
    integer(duration_us, 'duration_us', 1)
    times, states = plan['times_us'], plan['states']
    if not len(times) or len(times) != len(states) or times[0] != 0:
        raise ValueError('视觉规划时间与状态数量不匹配或不是从零开始')
    if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
           or t < 0 or t > duration_us for t in times):
        raise ValueError('视觉规划时间必须为歌曲范围内的整数微秒')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('视觉规划时间必须严格递增')
    for state in states:
        for field, (low, high) in VISUAL_BOUNDS.items():
            value = state[field]
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or not low <= value <= high
                    or (field in INTEGER_FIELDS and int(value) != value)):
                raise ValueError(f'视觉规划字段 {field} 非法')
    points = {0, duration_us}

    def near(value):
        for point in (math.floor(value) - 1, math.floor(value), math.ceil(value), math.ceil(value) + 1):
            if 0 <= point <= duration_us:
                points.add(point)

    for t in times:
        near(t)
    for t in event_times(edits):
        near(t)

    def roots(curve, target):
        if curve == 'smoothstep' and 0 <= target <= 1.5:
            delta = math.sqrt(max(0., 1. - 2. * target / 3.))
            return ((1. - delta) / 2., (1. + delta) / 2.)
        if curve == 'smootherstep' and 0 <= target <= 1.875:
            delta = math.sqrt(max(0., 1. - 4. * math.sqrt(target / 30.)))
            return ((1. - delta) / 2., (1. + delta) / 2.)
        return ()

    for i in range(len(states) - 1):
        a, b, left, right = states[i], states[i + 1], times[i], times[i + 1]
        for field in INTEGER_FIELDS:
            if a[field] == b[field]:
                continue
            for value in np.arange(min(a[field], b[field]) + .5, max(a[field], b[field])):
                near(left + (value - a[field]) / (b[field] - a[field]) * (right - left))
        for e in edits:
            if (not e['enabled'] or e['layer'] != 'visual' or e['operation'] != 'interval_offset'
                    or not e['value'] or e['field'] == 'hue_base' or e['field'] in INTEGER_FIELDS):
                continue
            start, end = e['start_us'], e['end_us']
            support_start, support_end = support(e)
            slope = (b[e['field']] - a[e['field']]) / (right - left)
            for begin, finish, sign in ((support_start, start, 1), (end, support_end, -1)):
                if finish <= begin or finish < left or begin > right:
                    continue
                target = -slope * (finish - begin) / (e['value'] * sign)
                for root in roots(e.get('interpolation', 'linear'), target):
                    t = begin + root * (finish - begin) if sign == 1 else finish - root * (finish - begin)
                    if left <= t <= right:
                        near(t)
    return np.array(sorted(points), dtype=np.int64)


def frame_times(duration_us, fps):
    fps = integer(fps, 'fps', 1)
    count = (duration_us * fps + 999999) // 1000000
    return np.array([(i * 1000000 + fps // 2) // fps for i in range(count)], dtype=np.int64)
