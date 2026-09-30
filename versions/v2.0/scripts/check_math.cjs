/* Run: node scripts/check_math.cjs. Known answers are shared with Python tests. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const math = require('../web/timeline-math.js');
const vectors = JSON.parse(fs.readFileSync(path.join(__dirname, '../tests/transition_vectors.json'), 'utf8'));
const close = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-10,
    `index ${i}: expected ${expected[i]}, got ${value}`));
};
const edit = patch => math.validateEdits([{...vectors.base_edit, ...patch}], vectors.duration_us)[0];
for (const c of vectors.weight_cases) close(c.times_us.map(t => math.weightAt(edit(c.patch), t)), c.expected);
for (const c of vectors.apply_cases) {
  const base = c.base_arousal.map(a => [.123, a, 0, 0, 0]), before = JSON.stringify(base);
  const result = math.applyEdits(base, c.times_us, [edit(c.patch)], 'emotion', math.emotions);
  close(result.map(row => row[1]), c.expected);
  assert.equal(JSON.stringify(base), before);
  close(result.map(row => row[0]), c.times_us.map(() => .123));
}
for (const patch of vectors.invalid_patches) assert.throws(() => edit(patch));
assert.equal(edit({}).interpolation, 'linear');
assert.equal(vectors.base_edit.interpolation, undefined);
assert.throws(() => math.validateEdits([vectors.base_edit, {...vectors.base_edit, id: 'overlap'}], 1000000));
assert.equal(math.validateEdits([vectors.base_edit, {...vectors.base_edit, id: 'off', enabled: false}], 1000000).length, 2);
const touchA = {...vectors.base_edit, start_us: 0, end_us: 500000, transition_in_us: 0, transition_out_us: 0};
assert.throws(() => math.validateEdits([touchA, {...touchA, id: 'touch', start_us: 500000, end_us: 1000000}], 1000000));
for (const [value, expected] of vectors.round_cases) assert.ok(math.roundEven(value) === expected);
const visualRows = math.visualAt(vectors.visual_plan, vectors.visual_samples.times_us);
for (const field of ['hue_base', 'particle_count', 'trail_length'])
  close(visualRows.map(row => row[math.fields.indexOf(field)]), vectors.visual_samples[field]);
const visualEdit = math.validateEdits([vectors.visual_edit], 1000000);
const edited = math.applyEdits(visualRows, vectors.visual_samples.times_us, visualEdit, 'visual', math.fields);
close(edited.map(row => math.roundEven(row[math.fields.indexOf('particle_count')])), vectors.visual_edited_count);
const raw = Array.from({length: 10}, () => [0, 0, 0, 0, 0]);
raw[1][0] = .9;
const rawBefore = JSON.stringify(raw), edits = [edit(vectors.weight_cases.at(-1).patch)];
const preview = math.emotionPreview(raw, 100000, 1000000, edits);
for (const time of [0, 99999, 100000, 365123, 352779, 377467, 1000000]) assert.ok(preview.times_us.includes(time));
assert.equal(preview.raw[preview.times_us.indexOf(99999)][0], 0);
assert.equal(preview.raw[preview.times_us.indexOf(100000)][0], .9);
assert.equal(preview.effective[preview.times_us.indexOf(365123)][1], 1);
assert.equal(JSON.stringify(raw), rawBefore);
raw[3][1] = .95;
assert.throws(() => math.emotionPreview(raw, 100000, 1000000,
  [edit({operation: 'interval_offset', value: .4, interpolation: 'smootherstep'})]));
const planBefore = JSON.stringify(vectors.visual_plan);
const visual = math.visualPreview(vectors.visual_plan, 1000000, [vectors.visual_edit]);
assert.deepEqual(visual.fields, math.fields);
assert.equal(JSON.stringify(vectors.visual_plan), planBefore);
assert.equal(visual.effective[visual.times_us.indexOf(0)][math.fields.indexOf('particle_count')], 50);
// Offset endpoints can be valid while an interior transition extremum exceeds
// the bound. Checking edit knots alone is insufficient.
const extremaPlan = {times_us: [0, 1000000], states: vectors.visual_plan.states.map(s => ({...s}))};
extremaPlan.states[0].particle_speed = vectors.offset_extrema.start_speed;
extremaPlan.states[1].particle_speed = vectors.offset_extrema.end_speed;
assert.throws(() => math.visualPreview(extremaPlan, 1000000, [vectors.offset_extrema.edit]), /越界/);
const probeTimes = math.visualValidationTimes(vectors.visual_plan, 1000000, []);
for (const time of vectors.visual_validation_required_times) assert.ok(probeTimes.includes(time));
// Pixel-column thinning preserves both positive/negative isolated peaks and
// edit targets. It is intentionally separate from preview validation.
const longTimes = Array.from({length: 100000}, (_, i) => i / 1000);
const positive = longTimes.map(() => 0), negative = longTimes.map(() => .1);
positive[12345] = 1; negative[87654] = -1;
const timesSnapshot = JSON.stringify(longTimes), seriesSnapshot = JSON.stringify([positive, negative]);
const width = 120;
const thinned = math.plotIndices(longTimes, [positive, negative], width);
assert.ok(thinned.includes(12345)); assert.ok(thinned.includes(87654));
assert.equal(thinned[0], 0); assert.equal(thinned.at(-1), longTimes.length - 1);
assert.equal(new Set(thinned).size, thinned.length);
assert.ok(thinned.every((index, i) => i === 0 || index > thinned[i - 1]));
assert.ok(thinned.length <= width * (2 + 2 * 2), `Too many vertices: ${thinned.length}`);
const mandatoryIndex = 43210;
const mandatory = math.plotIndices(longTimes, [positive, negative], width,
  [longTimes[mandatoryIndex], (longTimes[55555] + longTimes[55556]) / 2]);
for (const index of [43209, 43210, 43211, 55555, 55556]) assert.ok(mandatory.includes(index));
assert.equal(JSON.stringify(longTimes), timesSnapshot);
assert.equal(JSON.stringify([positive, negative]), seriesSnapshot);
assert.deepEqual(math.plotIndices([], [], 0), []);
assert.deepEqual(math.plotIndices([0], [1], 0), [0]);
assert.deepEqual(math.plotIndices([0, 1], [0, 1], .25), [0, 1]);
assert.ok(math.plotIndices(longTimes, positive, 0).includes(12345));
assert.throws(() => math.plotIndices([1, 0], [0, 1], 10));
assert.throws(() => math.plotIndices([0, 1], [0], 10));
console.log('Transition math: shared vectors, previews, bounds, rounding, immutability and peak-preserving plot thinning passed.');
