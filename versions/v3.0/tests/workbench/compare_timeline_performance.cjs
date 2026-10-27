'use strict';
// Frozen implementation comparison, not a model-quality evaluation. All inputs
// are deterministic synthetic timing fixtures; no song labels are consumed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const [baselinePath, outputPath, mode = 'compare'] = process.argv.slice(2);
if (!baselinePath || !outputPath) throw new Error('Pass baseline module and report path');
const before = require(path.resolve(baselinePath));
const after = require('../../workbench-web/timeline-math.js');
function fixture(duration, step, editCount) {
  const times = Array.from({length: duration / step}, (_, i) => i * step);
  const raw = times.map((t, i) => [i % 173 === 0 ? null : .4 * Math.sin(t / 2e7),
    i % 251 === 0 ? null : .3 * Math.cos(t / 3e7)]);
  const edits = Array.from({length: editCount}, (_, i) => ({id: 'p' + i, layer: 'emotion',
    field: i % 2 ? 'arousal' : 'valence', operation: 'point_target',
    time_us: Math.floor((i + 1) * duration / (editCount + 1)), value: .5,
    transition_in_us: 1e6, transition_out_us: 1e6}));
  return [raw, times, duration, edits, {enabled: true, window_seconds: .5}, step, true];
}
function run(api, args) {
  const p = api.emotionPreview(...args);
  const indices = api.plotIndices(p.times, [p.effective.map(r => r[0]), p.effective.map(r => r[1])],
    1200, args[3].map(e => e.time_us / 1e6));
  return {p, indices};
}
let parityCases = 0;
if (mode === 'compare') {
  for (let n = 0; n < 96; n++) {
    const step = n % 2 ? 100000 : 500000;
    const times = Array.from({length: 13}, (_, i) => 37 + i * step + (i > 6 ? step * 3 : 0));
    const raw = times.map((t, i) => [i % 4 ? .7 * Math.sin(i + n) : null,
      i % 3 ? .6 * Math.cos(i + n) : null]);
    const duration = times.at(-1) + step * 3;
    const operation = ['interval_set', 'interval_offset', 'point_target'][n % 3];
    const e = {id: 'test', layer: 'emotion', field: n % 2 ? 'arousal' : 'valence', operation,
      value: operation === 'interval_offset' ? (n % 5 ? .1 : 1.2) : -.4,
      time_us: times[4], start_us: times[4], end_us: times[5],
      transition_in_us: step, transition_out_us: step,
      interpolation: ['linear', 'smoothstep', 'smootherstep'][Math.floor(n / 3) % 3]};
    const args = [raw, times, duration, n % 7 ? [e] : [],
      {enabled: Boolean(n % 2), window_seconds: .5 + n % 5}, step, Boolean(n % 4)];
    const untouched = structuredClone(args);
    const capture = api => {
      try { return {value: api.emotionPreview(...args)}; }
      catch (error) { return {error: error.message}; }
    };
    assert.deepStrictEqual(capture(after), capture(before), 'native-clock parity case ' + n);
    assert.deepStrictEqual(args, untouched, 'caller data mutated');
    parityCases++;
  }
}
const quantiles = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return {median_ms: sorted[Math.floor(sorted.length / 2)], p95_ms: sorted[Math.ceil(sorted.length * .95) - 1],
    min_ms: sorted[0], max_ms: sorted.at(-1)};
};
const cases = [];
for (const [seconds, step, editCount] of [[600, 500000, 20], [600, 100000, 20], [3600, 100000, 20]]) {
  const args = fixture(seconds * 1e6, step, editCount);
  if (mode === 'compare') {
    assert.deepStrictEqual(run(after, args), run(before, args)); parityCases++;
  }
  const elapsed = {before: [], after: []};
  for (let iteration = 0; iteration < 48; iteration++) {
    // Alternate order to reduce systematic warmup and thermal bias.
    const order = mode === 'baseline' ? ['before'] : iteration % 2 ? ['before', 'after'] : ['after', 'before'];
    for (const name of order) {
      const start = performance.now(); run(name === 'before' ? before : after, args);
      if (iteration >= 8) elapsed[name].push(performance.now() - start);
    }
  }
  const item = {seconds, step_us: step, native_samples: args[0].length, edits: editCount,
    measured_runs_per_version: 40, before: quantiles(elapsed.before)};
  if (mode !== 'baseline') {
    item.after = quantiles(elapsed.after);
    item.median_reduction_percent = 100 * (1 - item.after.median_ms / item.before.median_ms);
    item.p95_reduction_percent = 100 * (1 - item.after.p95_ms / item.before.p95_ms);
  }
  cases.push(item);
}
const report = {recorded_at: new Date().toISOString(), node: process.version, mode, parity_cases: parityCases,
  scope: 'Pure native-clock preview, edit validation, smoothing, extrema-preserving draw selection; excludes DOM, canvas, HTTP and model accuracy.', cases};
fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
