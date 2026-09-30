"""Versioned motion data contract. No rendering/ML imports or implicit IO."""
from copy import deepcopy
import math

MODE_LABELS = {'rise':'轻盈上升', 'fall':'缓慢下沉', 'orbit':'环流绕圈',
               'spiral':'螺旋运动', 'expand':'向外扩散', 'gather':'向内汇聚',
               'meteor':'流星掠过', 'wave':'波浪平移', 'turbulent':'湍流运动'}
PARAM_BOUNDS = {'speed':(0.,.5), 'coherence':(0.,1.), 'turbulence':(0.,1.),
                'direction_deg':(0.,360.), 'center_x':(0.,1.), 'center_y':(0.,1.),
                'radius':(.05,.48), 'rotation':(-1.,1.), 'radial':(-1.,1.),
                'pulse':(0.,1.), 'trail_seconds':(0.,2.)}
_COMMON = dict(speed=.12, coherence=.92, turbulence=.08, direction_deg=0.,
               center_x=.5, center_y=.5, radius=.28, rotation=1., radial=0.,
               pulse=.2, trail_seconds=1.)
_PRESETS = {
    'rise':dict(speed=.10,direction_deg=270.,trail_seconds=.7),
    'fall':dict(speed=.07,direction_deg=90.,turbulence=.05,trail_seconds=1.2),
    'orbit':dict(speed=.13,coherence=.98,turbulence=.03,radius=.30,pulse=.12,trail_seconds=1.4),
    'spiral':dict(speed=.18,coherence=.94,turbulence=.07,radial=.25,pulse=.25,trail_seconds=1.5),
    'expand':dict(speed=.19,radial=1.,pulse=.45,trail_seconds=1.),
    'gather':dict(speed=.09,radial=-1.,pulse=.15,trail_seconds=1.2),
    'meteor':dict(speed=.30,direction_deg=45.,coherence=.98,turbulence=.02,pulse=.35,trail_seconds=1.),
    'wave':dict(speed=.10,turbulence=.03,pulse=.15,trail_seconds=1.),
    'turbulent':dict(speed=.13,coherence=.25,turbulence=.75,pulse=.35,trail_seconds=.8),
}
MOTION_DEFAULTS = {mode:{**_COMMON,**values} for mode,values in _PRESETS.items()}
DEFAULT_CONFIG = dict(engine='legacy', sensitivity=1., min_hold_seconds=6.,
                      transition_seconds=2., overrides=[])


def finite(value, name, low, high):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} 必须为有限数值')
    if not low <= value <= high:
        raise ValueError(f'{name} 必须在 {low}–{high} 之间')
    return float(value)


def microseconds(value, name):
    if isinstance(value,bool) or not isinstance(value,int) or value < 0:
        raise ValueError(f'{name} 必须为非负整数微秒')
    return value


def mode_params(mode, params=None):
    if not isinstance(mode,str) or mode not in MODE_LABELS:
        raise ValueError('未知运动模式')
    params = {} if params is None else params
    if not isinstance(params,dict) or set(params)-set(PARAM_BOUNDS):
        raise ValueError('运动参数含未知字段')
    result = deepcopy(MOTION_DEFAULTS[mode])
    for key,value in params.items():
        result[key] = finite(value,key,*PARAM_BOUNDS[key])
    if result['rotation'] not in (-1.,1.):
        raise ValueError('旋转方向只能为 -1 或 1')
    result['direction_deg'] %= 360.
    return result


def validate_config(config=None, duration_us=None):
    if config is None:
        return deepcopy(DEFAULT_CONFIG)
    if not isinstance(config,dict) or set(config)-set(DEFAULT_CONFIG):
        raise ValueError('运动配置含未知字段')
    result = {**deepcopy(DEFAULT_CONFIG),**deepcopy(config)}
    if result['engine'] not in ('legacy','flow-v1'):
        raise ValueError('运动引擎必须为 legacy 或 flow-v1')
    result['sensitivity'] = finite(result['sensitivity'],'运动敏感度',.5,2.)
    result['min_hold_seconds'] = finite(result['min_hold_seconds'],'效果最短保持',4.,12.)
    result['transition_seconds'] = finite(result['transition_seconds'],'自动过渡',1.,4.)
    if result['transition_seconds'] > result['min_hold_seconds']:
        raise ValueError('过渡时间不可超过最短保持时间')
    overrides = result['overrides']
    if not isinstance(overrides,list) or len(overrides)>1000:
        raise ValueError('运动覆盖必须为列表，最多 1000 项')
    checked, ids, occupied = [],set(),[]
    allowed = {'id','enabled','mode','start_us','end_us','transition_in_us',
               'transition_out_us','interpolation','note','params'}
    for original in overrides:
        if not isinstance(original,dict) or set(original)-allowed:
            raise ValueError('运动覆盖含未知字段')
        item = deepcopy(original)
        eid = item.get('id')
        if not isinstance(eid,str) or not eid or len(eid)>100 or eid in ids:
            raise ValueError('运动覆盖 ID 必须唯一且非空')
        ids.add(eid)
        item.setdefault('enabled',True)
        if not isinstance(item['enabled'],bool):
            raise ValueError('运动覆盖 enabled 必须为布尔值')
        for key in ('start_us','end_us'):
            item[key] = microseconds(item.get(key),key)
        for key in ('transition_in_us','transition_out_us'):
            item[key] = microseconds(item.get(key,2000000),key)
        if item['end_us'] <= item['start_us']:
            raise ValueError('运动覆盖结束时间必须晚于开始时间')
        left = item['start_us']-item['transition_in_us']
        right = item['end_us']+item['transition_out_us']
        if left<0 or duration_us is not None and right>duration_us:
            raise ValueError('运动覆盖及过渡范围必须位于歌曲内')
        item.setdefault('interpolation','smootherstep')
        if item['interpolation'] not in ('linear','smoothstep','smootherstep'):
            raise ValueError('未知运动过渡方式')
        item['params'] = mode_params(item.get('mode'),item.get('params'))
        item['note'] = str(item.get('note',''))[:2000]
        if item['enabled']:
            for a,b,previous in occupied:
                touches_active = (left==b and item['transition_in_us']==0 and previous['transition_out_us']==0
                                  or right==a and item['transition_out_us']==0 and previous['transition_in_us']==0)
                if left<b and a<right or touches_active:
                    raise ValueError('运动覆盖及过渡区间重叠，请先调整或禁用已有覆盖')
            occupied.append((left,right,item))
        checked.append(item)
    result['overrides'] = checked
    return result


def catalog():
    return {mode:{'label':MODE_LABELS[mode],'params':deepcopy(params)} for mode,params in MOTION_DEFAULTS.items()}
