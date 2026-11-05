"use strict";
// Native hold-support and independent-mask expectations shared conceptually
// with SmoothingGapTests; neither side derives expected values from the other.
const assert = require("node:assert/strict");
const T = require("../../workbench-web/timeline-math.js");
const at = (p, t, field="effective") => {
  const i = p.times_us.indexOf(t);
  assert.ok(i >= 0, `Missing exact time ${t}`);
  return p[field][i];
};
const near = (row, expected) => row.forEach((v, i) => {
  if(expected[i] === null) assert.equal(v, null);
  else assert.ok(Math.abs(v - expected[i]) < 1e-14, `${v} != ${expected[i]}`);
});

const raw = [[0,0],[0,0],[1,1]], clock = [0,500000,1100000];
const before = JSON.stringify({raw, clock});
const gap = T.emotionPreview(raw, clock, 1600000, [], {enabled:true,window_seconds:2}, 500000);
near(at(gap,0),[0,0]);
near(at(gap,500000),[0,0]);
near(at(gap,1000000),[0,0]);
assert.deepEqual(at(gap,1000001),[null,null]);
assert.deepEqual(at(gap,1099999),[null,null]);
near(at(gap,1100000),[1,1]);
near(at(gap,1600000),[1,1]);
assert.equal(JSON.stringify({raw, clock}),before);
console.log("PASS smoothing never borrows samples or fills values across expired native support");

const regular = T.emotionPreview([[0,0],[null,.5],[1,1]], [0,500000,1000000],
  1500000, [], {enabled:true,window_seconds:.1}, 500000);
near(at(regular,500000),[null,.5]);
near(at(regular,1000000),[1,1]);
near(at(regular,1500000),[1,1]);
// Dense preview samples exercise unaffected A interpolation while V is absent.
const interior = regular.times_us.findIndex(t => t>500000 && t<1000000);
assert.ok(interior>=0);
near(regular.effective[interior],[null,regular.times_us[interior]/1e6]);
console.log("PASS regular-clock interpolation and independent masks remain unchanged");

const edit = {id:"human-fill",layer:"emotion",field:"valence",operation:"interval_set",
  start_us:1000001,end_us:1099999,value:.7,transition_in_us:0,transition_out_us:0};
const manual = T.emotionPreview(raw,clock,1600000,[edit],{enabled:true,window_seconds:2},500000);
assert.deepEqual(at(manual,1000001,"raw"),[null,null]);
near(at(manual,1000001),[.7,null]);
console.log("PASS explicit manual target changes only effective values in a gap");
