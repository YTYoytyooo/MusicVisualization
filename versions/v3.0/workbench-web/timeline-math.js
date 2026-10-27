/* Pure preview math shared by the local UI and Node contract tests.
 * No fetch, storage writes, randomness, or mutation of caller-owned objects.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.StudioTimeline = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const EMOTIONS = ['valence', 'arousal'];
  const BOUNDS = {hue_base: [0, 360], hue_range: [20, 120], saturation: [.3, 1],
    brightness: [.1, .9], particle_count: [50, 500], particle_speed: [.5, 8],
    field_turbulence: [0, 1], trail_length: [5, 60]};
  const FIELDS = Object.keys(BOUNDS);
  const INTEGERS = new Set(['particle_count', 'trail_length']);
  const CURVES = ['linear', 'smoothstep', 'smootherstep'];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const mod = (x, divisor) => ((x % divisor) + divisor) % divisor;
  function number(value, name) {
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${name} 必须是有限数值`);
    return value;
  }
  function integer(value, name, min = 0) {
    if (!Number.isSafeInteger(value) || value < min) throw new Error(`${name} 必须是大于等于 ${min} 的整数`);
    return value;
  }
  function endpoints(e) {
    return e.operation === 'point_target' ? [e.time_us, e.time_us] : [e.start_us, e.end_us];
  }
  function support(e) {
    const [start, end] = endpoints(e);
    return [start - e.transition_in_us, end + e.transition_out_us];
  }
  function ease(value, curve = 'linear') {
    if (curve === 'linear') return value;
    if (curve === 'smoothstep') return value * value * (3 - 2 * value);
    if (curve === 'smootherstep') return value * value * value * (value * (value * 6 - 15) + 10);
    throw new Error('未知过渡曲线');
  }
  function weightAt(e, time) {
    const [start, end] = endpoints(e);
    let weight = time >= start && time <= end ? 1 : 0;
    if (e.transition_in_us && time >= start - e.transition_in_us && time < start)
      weight = (time - (start - e.transition_in_us)) / e.transition_in_us;
    if (e.transition_out_us && time > end && time <= end + e.transition_out_us)
      weight = (end + e.transition_out_us - time) / e.transition_out_us;
    return ease(weight, e.interpolation || 'linear');
  }
  function validateEdits(edits, durationUs) {
    integer(durationUs, 'duration_us', 1);
    if (!Array.isArray(edits) || edits.length > 1000) throw new Error('编辑必须是列表，最多 1000 条');
    const ids = new Set(), occupied = new Map();
    return edits.map(original => {
      if (!original || typeof original !== 'object' || Array.isArray(original)) throw new Error('编辑项必须是对象');
      const e = {...original};
      if (typeof e.id !== 'string' || !e.id.length || e.id.length > 100 || ids.has(e.id))
        throw new Error('编辑 ID 必须唯一且非空');
      ids.add(e.id);
      let range;
      if (e.layer === 'emotion' && ['valence', 'arousal'].includes(e.field)) range = [-1, 1];
      else if (e.layer === 'visual' && Object.hasOwn(BOUNDS, e.field)) range = BOUNDS[e.field];
      else throw new Error('未知编辑层或字段');
      if (!['point_target', 'interval_set', 'interval_offset'].includes(e.operation)) throw new Error('未知编辑操作');
      e.value = number(e.value, 'value');
      if (e.operation !== 'interval_offset') {
        if (e.value < range[0] || e.value > range[1]) throw new Error(`${e.field} 目标值越界`);
        if (INTEGERS.has(e.field) && !Number.isInteger(e.value)) throw new Error(`${e.field} 目标必须是整数`);
      }
      for (const key of ['transition_in_us', 'transition_out_us'])
        e[key] = integer(e[key] === undefined ? 0 : e[key], key);
      if (e.operation === 'point_target') {
        integer(e.time_us, 'time_us');
        if (Math.min(e.transition_in_us, e.transition_out_us) <= 0) throw new Error('单点目标必须有非零的前后过渡时间');
      } else {
        integer(e.start_us, 'start_us'); integer(e.end_us, 'end_us');
        if (e.end_us <= e.start_us) throw new Error('区间结束时间必须晚于开始时间');
      }
      const [left, right] = support(e);
      if (left < 0 || right > durationUs) throw new Error('编辑及其过渡范围必须位于歌曲时间范围内');
      e.enabled = e.enabled === undefined ? true : e.enabled;
      if (typeof e.enabled !== 'boolean') throw new Error('enabled 必须为布尔值');
      e.interpolation = e.interpolation === undefined ? 'linear' : e.interpolation;
      if (!CURVES.includes(e.interpolation)) throw new Error('过渡必须为 linear / smoothstep / smootherstep');
      e.note = String(e.note === undefined ? '' : e.note).slice(0, 2000);
      if (e.enabled) {
        const key = `${e.layer}:${e.field}`, previous = occupied.get(key) || [];
        for (const p of previous) {
          const [a, b] = support(p);
          const boundary = left === b ? left : right === a ? right : null;
          if ((left < b && a < right) || (boundary !== null && weightAt(e, boundary) > 0 && weightAt(p, boundary) > 0))
            throw new Error(`${e.field} 编辑区间重叠，请先禁用或删除已有编辑`);
        }
        previous.push(e); occupied.set(key, previous);
      }
      return e;
    });
  }
  function applyEdits(base, times, edits, layer, fields) {
    if (base.length !== times.length) throw new Error('时间线数据形状非法');
    const values = base.map(row => {
      if (!Array.isArray(row) || row.length !== fields.length || row.some(x => x !== null && !Number.isFinite(x)))
        throw new Error('时间线数据形状或数值非法');
      return row.slice();
    });
    for (const e of edits) {
      if (!e.enabled || e.layer !== layer) continue;
      const column = fields.indexOf(e.field);
      if (column < 0) throw new Error('未知时间线字段');
      for (let i = 0; i < times.length; ++i) {
        const weight = weightAt(e, times[i]), value = values[i][column];
        // A fully specified human target may supply a production value; offsets
        // and partial blends need a real baseline. Raw labels remain untouched.
        if (value === null) {
          if(weight===1 && e.operation!=='interval_offset')values[i][column]=e.value;
          continue;
        }
        let result;
        if (e.operation === 'interval_offset') result = value + weight * e.value;
        else if (e.field === 'hue_base') result = value + weight * (mod(e.value - value + 180, 360) - 180);
        else result = value * (1 - weight) + e.value * weight;
        if (e.field === 'hue_base') result = mod(result, 360);
        else {
          const [low, high] = layer === 'emotion' ? [-1, 1] : BOUNDS[e.field];
          if (result < low - 1e-9 || result > high + 1e-9) throw new Error(`编辑 ${e.id} 使 ${e.field} 越界；请减小偏移量`);
          result = clamp(result, low, high);
        }
        values[i][column] = result;
      }
    }
    return values;
  }
  function addEvents(points, edits, durationUs, subdivisions = false) {
    const add = point => { if (point >= 0 && point <= durationUs) points.add(point); };
    for (const e of edits) {
      if (!e.enabled) continue;
      const [start, end] = endpoints(e), [left, right] = support(e);
      for (const point of [left, start, end, right]) { add(point - 1); add(point); add(point + 1); }
      if (subdivisions) for (const [a, b] of [[left, start], [end, right]])
        for (let i = 0; i <= 32; ++i) add(a + Math.floor(((b - a) * i + 16) / 32));
    }
  }
  function sourcePoints(count, step, duration, edits, preview, nominal=null) {
    const points = new Set([0, duration]);
    if (preview) {
      const spacing = Math.max(33334, Math.ceil(duration / 6000));
      for (let t = 0; t < duration; t += spacing) points.add(t);
    }
    const native = Array.isArray(step) ? step : Array.from({length:count}, (_, i) => i * step);
    for (const t of native) {
      if (t > duration) break;
      points.add(t); points.add(Math.max(0, t - 1));
      if(nominal && t+nominal<duration){points.add(t+nominal);points.add(t+nominal+1);}
    }
    addEvents(points, edits, duration, preview);
    return [...points].sort((a, b) => a - b);
  }
  function rawAt(raw, times, step, nominal=null, holdLastToEnd=false) {
    if (Array.isArray(step)) return timedAt(raw, times, step, false,nominal,holdLastToEnd);
    return times.map(t => raw[clamp(Math.floor(t / step), 0, raw.length - 1)].slice());
  }
  function sourceClock(raw, stepOrTimes, duration) {
    if (!Array.isArray(stepOrTimes)) {
      integer(stepOrTimes, 'step_us', 1);
      return Array.from({length:raw.length}, (_, i) => i * stepOrTimes);
    }
    if (stepOrTimes.length !== raw.length) throw new Error('原始时间戳数量必须与 V/A 数量一致');
    stepOrTimes.forEach((t, i) => {
      integer(t, 'source time');
      if (t > duration || (i && t <= stepOrTimes[i-1])) throw new Error('原始时间戳须严格递增且位于音频内');
    });
    return stepOrTimes;
  }
  function upperBound(times, t) {
    let lo=0, hi=times.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<=t)lo=mid+1;else hi=mid;}
    return lo;
  }
  function timedAt(values, targets, times, linear, nominal=null,holdLastToEnd=false) {
    // Native clocks and preview targets are ordered. Walk the source once;
    // retain binary lookup for any backwards query without changing its value.
    let index=-1, previous=-Infinity;
    return targets.map(t => {
      if(t<previous)index=upperBound(times,t)-1;
      else while(index+1<times.length && times[index+1]<=t)index++;
      previous=t;
      if(index<0)return [null,null];
      const held=(holdLastToEnd&&index===times.length-1)||!(nominal && t-times[index]>nominal);
      const row=held?values[index].slice():[null,null];
      if(!linear || index+1>=times.length || (nominal&&times[index+1]-times[index]>1.5*nominal))return row;
      const alpha=clamp((t-times[index])/(times[index+1]-times[index]),0,1);
      // row is already a copy; interpolate there instead of allocating another
      // array per query. Never blend through a missing dimension.
      for(let c=0;c<row.length;c++) {
        if(values[index][c]!==null && values[index+1][c]!==null)
          row[c]=values[index][c]*(1-alpha)+values[index+1][c]*alpha;
      }
      return row;
    });
  }
  function validateSmoothing(config) {
    if (config == null) return {enabled:false,window_seconds:.5};
    if (typeof config !== 'object' || Array.isArray(config) || Object.keys(config).some(k=>!['enabled','window_seconds'].includes(k)))
      throw new Error('非法平滑配置');
    const enabled=config.enabled === undefined ? false : config.enabled;
    const window=config.window_seconds === undefined ? .5 : config.window_seconds;
    if(typeof enabled !== 'boolean' || typeof window !== 'number' || !Number.isFinite(window) || window<.1 || window>5)
      throw new Error('平滑窗口必须在0.1至5秒之间');
    return {enabled,window_seconds:window};
  }
  function smoothSource(raw, stepUs, config, nominal=null) {
    config=validateSmoothing(config);
    const result=raw.map(row=>row.slice());
    if(!config.enabled)return result;
    if(Array.isArray(stepUs)){
      const half=config.window_seconds*5e5;
      for(const c of [0,1]){
        let start=0;
        while(start<raw.length){
          if(raw[start][c]===null){start++;continue;}
          let end=start+1;while(end<raw.length&&raw[end][c]!==null&&(!nominal||stepUs[end]-stepUs[end-1]<=1.5*nominal))end++;
          const sums=[0];for(let i=start;i<end;i++)sums.push(sums[sums.length-1]+raw[i][c]);
          let lo=start,hi=start;
          for(let i=start;i<end;i++){
            while(lo<i&&stepUs[lo]<stepUs[i]-half)lo++;
            while(hi<end&&stepUs[hi]<=stepUs[i]+half)hi++;
            result[i][c]=(sums[hi-start]-sums[lo-start])/(hi-lo);
          }
          start=end;
        }
      }
      return result;
    }
    const radius=Math.floor(config.window_seconds*1e6/stepUs/2);
    for(const c of [0,1]) {
      const sums=[0];
      for(const row of raw)sums.push(sums[sums.length-1]+row[c]);
      for(let i=0;i<raw.length;i++){
        const lo=Math.max(0,i-radius),hi=Math.min(raw.length,i+radius+1);
        result[i][c]=(sums[hi]-sums[lo])/(hi-lo);
      }
    }
    return result;
  }
  function smoothAt(raw, source, times, step, enabled,nominal=null,holdLastToEnd=false) {
    if(Array.isArray(step))return timedAt(enabled?source:raw,times,step,enabled,nominal,holdLastToEnd);
    const result=rawAt(raw,times,step);
    if(enabled)times.forEach((t,i)=>{
      const left=clamp(Math.floor(t/step),0,source.length-1),right=Math.min(source.length-1,left+1);
      const alpha=clamp((t-left*step)/step,0,1);
      for(const c of [0,1])result[i][c]=source[left][c]*(1-alpha)+source[right][c]*alpha;
    });
    return result;
  }
  function smoothingCheckTimes(source, step, duration, edits, points) {
    const keep=new Set(points);
    const native=Array.isArray(step)?step:Array.from({length:source.length},(_,i)=>i*step);
    for(const e of edits){
      if(!e.enabled || e.layer!=='emotion' || e.operation!=='interval_offset' || !e.value)continue;
      const c=EMOTIONS.indexOf(e.field),[left,right]=support(e);
      for(let i=Math.max(0,upperBound(native,left)-1);i<source.length-1&&native[i]<=right;i++){
        if(source[i][c]===null||source[i+1][c]===null)continue;
        const a=native[i],b=native[i+1],slope=(source[i+1][c]-source[i][c])/(b-a);
        for(const [from,to,sign] of [[left,e.start_us,1],[e.end_us,right,-1]]){
          if(to<=from || to<a || from>b)continue;
          for(const root of derivativeRoots(e.interpolation,-slope*(to-from)/(e.value*sign))){
            const t=sign===1?from+root*(to-from):to-root*(to-from);
            if(t<a || t>b)continue;
            for(const p of [Math.floor(t)-1,Math.floor(t),Math.ceil(t),Math.ceil(t)+1])
              if(p>=0 && p<=duration)keep.add(p);
          }
        }
      }
    }
    return [...keep].sort((a,b)=>a-b);
  }
  function emotionPreview(raw, stepUs, durationUs, originalEdits, smoothing,nominalStepUs=null,holdLastToEnd=false) {
    integer(durationUs, 'duration_us', 1);
    if (!Array.isArray(raw) || !raw.length || raw.some(row => !Array.isArray(row) || row.length !== 2 ||
      row.some(x => x !== null && (typeof x !== 'number' || !Number.isFinite(x) || Math.abs(x) > 1))))
      throw new Error('原始预测必须为非空 N×2，范围 [-1,1]；缺失维度使用 null');
    const clock=sourceClock(raw,stepUs,durationUs);
    if(nominalStepUs!==null)integer(nominalStepUs,'nominal step_us',1);
    if(typeof holdLastToEnd!=='boolean')throw new Error('hold_last_to_end 必须为布尔值');
    const edits = validateEdits(originalEdits, durationUs);
    const config=validateSmoothing(smoothing), source=smoothSource(raw,clock,config,nominalStepUs);
    // Validation is never decimated: every zero-order-hold boundary and edit
    // knot is tested even if a consumer later reduces the displayed points.
    let checks = sourcePoints(raw.length, stepUs, durationUs, edits, false,nominalStepUs);
    if(config.enabled)checks=smoothingCheckTimes(source,stepUs,durationUs,edits,checks);
    applyEdits(smoothAt(raw,source,checks,stepUs,config.enabled,nominalStepUs,holdLastToEnd), checks, edits, 'emotion', EMOTIONS);
    const timesUs = sourcePoints(raw.length, stepUs, durationUs, edits, true,nominalStepUs);
    const baseline = rawAt(raw, timesUs, stepUs,nominalStepUs,holdLastToEnd);
    const processed = smoothAt(raw,source,timesUs,stepUs,config.enabled,nominalStepUs,holdLastToEnd);
    return {times: timesUs.map(t => t / 1e6), times_us: timesUs, raw: baseline,
      baseline:processed, smoothing:config,
      effective: applyEdits(processed, timesUs, edits, 'emotion', EMOTIONS), fields: EMOTIONS.slice()};
  }
  function sourcePreview(source, durationUs, edits = [], smoothing) {
    if(!source || typeof source!=='object')throw new Error('所选模型的原始对照数据不可用，请重新加载作品。');
    const raw=source.values||source.raw_va,clock=source.times_us||source.times_seconds?.map(t=>Math.round(t*1e6));
    if(!raw || !clock)throw new Error('所选模型缺少完整原始时间戳与 V/A 数据，请重新加载作品。');
    // This seam reads the selected model's immutable source only. In particular,
    // a reset switch must not first validate the old revision against that source.
    return emotionPreview(raw,clock,durationUs,edits,smoothing,source.step_us||null,Boolean(source.hold_last_to_end));
  }
  function roundEven(value) {
    const lower = Math.floor(value), fraction = value - lower;
    return fraction < .5 ? lower : fraction > .5 ? lower + 1 : lower % 2 === 0 ? lower : lower + 1;
  }
  function validatePlan(plan, durationUs) {
    if (!plan || !Array.isArray(plan.times_us) || !Array.isArray(plan.states) ||
        !plan.states.length || plan.times_us.length !== plan.states.length || plan.times_us[0] !== 0)
      throw new Error('视觉规划必须包含从零开始的时间点和匹配的状态');
    plan.times_us.forEach((t, i) => {
      integer(t, 'plan time');
      if (t > durationUs || (i && t <= plan.times_us[i - 1])) throw new Error('视觉规划时间点必须严格递增且位于歌曲内');
      for (const field of FIELDS) {
        const value = number(plan.states[i][field], field), [lo, hi] = BOUNDS[field];
        if (value < lo || value > hi || (INTEGERS.has(field) && !Number.isInteger(value)))
          throw new Error(`视觉规划字段 ${field} 非法`);
      }
    });
  }
  function visualAt(plan, times) {
    return times.map(t => {
      if (plan.states.length === 1) return FIELDS.map(k => plan.states[0][k]);
      let low = 0, high = plan.times_us.length;
      while (low < high) { const middle = (low + high) >> 1; if (plan.times_us[middle] <= t) low = middle + 1; else high = middle; }
      const i = clamp(low - 1, 0, plan.states.length - 2), a = plan.states[i], b = plan.states[i + 1];
      const alpha = clamp((t - plan.times_us[i]) / (plan.times_us[i + 1] - plan.times_us[i]), 0, 1);
      return FIELDS.map(field => {
        if (field === 'hue_base') return mod(a[field] + alpha * (mod(b[field] - a[field] + 180, 360) - 180), 360);
        const value = a[field] + alpha * (b[field] - a[field]);
        return INTEGERS.has(field) ? roundEven(value) : value;
      });
    });
  }
  function derivativeRoots(curve, target) {
    // Roots of ease'(u)=target on [0,1]. Used to catch offset extrema between
    // display samples, including narrow transitions on a linear base curve.
    if (curve === 'smoothstep' && target >= 0 && target <= 1.5) {
      const d = Math.sqrt(Math.max(0, 1 - 2 * target / 3)); return [(1 - d) / 2, (1 + d) / 2];
    }
    if (curve === 'smootherstep' && target >= 0 && target <= 1.875) {
      const d = Math.sqrt(Math.max(0, 1 - 4 * Math.sqrt(target / 30))); return [(1 - d) / 2, (1 + d) / 2];
    }
    return [];
  }
  function visualCheckTimes(plan, edits, durationUs) {
    const points = new Set([0, durationUs]);
    const near = value => {
      for (const t of [Math.floor(value) - 1, Math.floor(value), Math.ceil(value), Math.ceil(value) + 1])
        if (t >= 0 && t <= durationUs) points.add(t);
    };
    plan.times_us.forEach(near); addEvents(points, edits, durationUs, false);
    for (let i = 0; i + 1 < plan.states.length; ++i) {
      const a = plan.states[i], b = plan.states[i + 1], left = plan.times_us[i], right = plan.times_us[i + 1];
      // Python rounds these two baseline fields BEFORE applying an edit.
      for (const field of INTEGERS) if (a[field] !== b[field])
        for (let value = Math.min(a[field], b[field]) + .5; value < Math.max(a[field], b[field]); value += 1)
          near(left + (value - a[field]) / (b[field] - a[field]) * (right - left));
      for (const e of edits) {
        if (!e.enabled || e.layer !== 'visual' || e.operation !== 'interval_offset' || !e.value ||
            e.field === 'hue_base' || INTEGERS.has(e.field)) continue;
        const [start, end] = endpoints(e), [supportStart, supportEnd] = support(e);
        const slope = (b[e.field] - a[e.field]) / (right - left);
        for (const [from, to, sign] of [[supportStart, start, 1], [end, supportEnd, -1]]) {
          if (to <= from || to < left || from > right) continue;
          const target = -slope * (to - from) / (e.value * sign);
          for (const root of derivativeRoots(e.interpolation, target)) {
            const t = sign === 1 ? from + root * (to - from) : to - root * (to - from);
            if (t >= left && t <= right) near(t);
          }
        }
      }
    }
    return [...points].sort((a, b) => a - b);
  }
  function visualPreview(plan, durationUs, originalEdits) {
    integer(durationUs, 'duration_us', 1); validatePlan(plan, durationUs);
    const edits = validateEdits(originalEdits, durationUs);
    const checks = visualCheckTimes(plan, edits, durationUs);
    applyEdits(visualAt(plan, checks), checks, edits, 'visual', FIELDS);
    const points = new Set(sourcePoints(0, 1, durationUs, edits, true));
    for (const t of plan.times_us) { points.add(t); points.add(Math.max(0, t - 1)); }
    const timesUs = [...points].sort((a, b) => a - b), baseline = visualAt(plan, timesUs);
    const effective = applyEdits(baseline, timesUs, edits, 'visual', FIELDS);
    effective.forEach(row => FIELDS.forEach((field, i) => { if (INTEGERS.has(field)) row[i] = roundEven(row[i]); }));
    return {times: timesUs.map(t => t / 1e6), times_us: timesUs, raw: baseline, effective, fields: FIELDS.slice()};
  }
  function plotIndices(times, series, pixelWidth, mandatoryTimes = []) {
    // Drawing only: never use the retained subset for evaluating or validating
    // edits. Every column preserves the extrema of EVERY supplied series.
    const isArray = value => Array.isArray(value) ||
      (ArrayBuffer.isView(value) && !(value instanceof DataView));
    if (!isArray(times)) throw new Error('绘图时间必须是数组');
    const length = times.length;
    if (!length) return [];
    if (!isArray(series)) throw new Error('绘图序列必须是数组');
    const lines = series.length && isArray(series[0]) ? series : series.length ? [series] : [];
    for (const line of lines) {
      if (!isArray(line) || line.length !== length) throw new Error('绘图序列必须与时间长度匹配');
      for (const value of line) if(value!==null)number(value, 'plot value');
    }
    for (let i = 0; i < length; ++i) {
      number(times[i], 'plot time');
      if (i && times[i] < times[i - 1]) throw new Error('绘图时间必须按升序排列');
    }
    number(pixelWidth, 'pixel width');
    // Zero/negative/subpixel canvases are treated as a single column; retain
    // its extrema instead of returning an empty or misleading flat curve.
    const columns = Math.max(1, Math.floor(pixelWidth));
    const span = times[length - 1] - times[0];
    const kept = new Set([0, length - 1]);
    let bucket = -1, first = 0, last = 0, minima = [], maxima = [];
    const flush = () => {
      if (bucket < 0) return;
      kept.add(first); kept.add(last);
      minima.forEach(index => kept.add(index));
      maxima.forEach(index => kept.add(index));
    };
    for (let i = 0; i < length; ++i) {
      const column = span > 0 ? Math.min(columns - 1,
        Math.floor((times[i] - times[0]) / span * columns)) : 0;
      if (column !== bucket) {
        flush(); bucket = column; first = i;
        minima = lines.map(() => i); maxima = lines.map(() => i);
      } else {
        lines.forEach((line, j) => {
          if((line[i]===null)!==(line[i-1]===null)){kept.add(i);if(i)kept.add(i-1);}
          if (line[i]!==null && (line[minima[j]]===null || line[i] < line[minima[j]])) minima[j] = i;
          if (line[i]!==null && (line[maxima[j]]===null || line[i] > line[maxima[j]])) maxima[j] = i;
        });
      }
      last = i;
    }
    flush();
    if (!isArray(mandatoryTimes)) throw new Error('关键绘图时间必须是数组');
    const retain = index => { if (index >= 0 && index < length) kept.add(index); };
    for (const target of mandatoryTimes) {
      number(target, 'mandatory time');
      let lo = 0, hi = length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (times[mid] < target) lo = mid + 1; else hi = mid;
      }
      // A target between samples retains both sides. Exact targets additionally
      // retain the next sample so one-microsecond discontinuities stay visible.
      retain(lo - 1); retain(lo);
      if (lo < length && times[lo] === target) retain(lo + 1);
    }
    return [...kept].sort((a, b) => a - b);
  }
  // Plot series are immutable snapshots: edits must replace the row array.
  // Keep only per-channel extrema; weak keys do not retain superseded previews.
  function createExtentCache() {
    const cache = new WeakMap();
    return (rows, channel) => {
      let channels = cache.get(rows);
      if (!channels) { channels = new Map(); cache.set(rows, channels); }
      if (!channels.has(channel)) {
        let low = Infinity, high = -Infinity;
        for (const row of rows) {
          const value = row[channel];
          if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); }
        }
        channels.set(channel, Object.freeze({low, high}));
      }
      return channels.get(channel);
    };
  }
  return {validateEdits, emotionPreview, sourcePreview, visualPreview, ease, weightAt, applyEdits, visualAt, roundEven, plotIndices,validateSmoothing,smoothSource,sourceClock,createExtentCache,
    visualValidationTimes: (plan, durationUs, edits) => {
      integer(durationUs, 'duration_us', 1);
      validatePlan(plan, durationUs);
      return visualCheckTimes(plan, validateEdits(edits, durationUs), durationUs);
    },
    fields: FIELDS.slice(), emotions: EMOTIONS.slice()};
});
