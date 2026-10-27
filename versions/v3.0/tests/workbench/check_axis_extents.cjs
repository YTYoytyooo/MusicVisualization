"use strict";
// Read-only series are replaced on edits. Verify cached axes against the actual
// pre-change application range calculation, including dynamic point targets.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto");
const {performance} = require("node:perf_hooks");
const T = require("../../workbench-web/timeline-math.js");
const baselinePath = "data/v3.0/optimization/20261024/rounds/003/app.js.before";
const appPath = "versions/v3.0/workbench-web/app.js";
const checks = [];
function test(name, run) { run(); checks.push(name); }
test("independent finite extrema and empty dimensions", () => {
  const extent = T.createExtentCache();
  const rows = Object.freeze([Object.freeze([-.9, null]), Object.freeze([.2, .7]),
    Object.freeze([NaN, -.4]), Object.freeze([Infinity, undefined])]);
  assert.deepEqual(extent(rows, 0), {low: -.9, high: .2});
  assert.deepEqual(extent(rows, 1), {low: -.4, high: .7});
  assert.deepEqual(extent([[null, null]], 0), {low: Infinity, high: -Infinity});
  assert.deepEqual(extent([], 1), {low: Infinity, high: -Infinity});
});
let rowReads;
test("unchanged series is scanned once per dimension; replacement invalidates", () => {
  const extent = T.createExtentCache(); let reads = 0;
  const rows = Array.from({length: 1000}, (_, i) => ({get 0() { reads++; return i/1000; }}));
  extent(rows, 0); const coldReads = reads;
  for(let i=0;i<100;i++) extent(rows, 0);
  assert.equal(coldReads, 1000); assert.equal(reads, coldReads);
  assert.deepEqual(extent([[.8, -.6]], 0), {low: .8, high: .8});
  assert.deepEqual(extent([[.8, -.6]], 1), {low: -.6, high: -.6});
  rowReads = {cold: coldReads, additional_100_refreshes: reads-coldReads};
});
function appRange(path) {
  const source = fs.readFileSync(path, "utf8");
  const draw = source.slice(source.indexOf("  function drawTimeline()"));
  const body = draw.slice(draw.indexOf("      let low = Infinity"), draw.indexOf("      const y = v =>"));
  assert.ok(body.includes("visibleEdits") && body.includes("state.drag"));
  // Both versions and the cached helper use the same JS realm. A separate VM
  // would add global lookup overhead to the old inner loop and bias timing.
  return vm.runInThisContext("(function(t,comparisons,channel,state,edits){const visibleEdits=()=>edits;" + body + ";return {low,high};})");
}
const oldRange = appRange(baselinePath), newRange = appRange(appPath);
const makeState = () => ({smoothing:{enabled:false},axisLocked:false,axisCache:{},drag:null,extentFor:T.createExtentCache()});
const plain = x => JSON.parse(JSON.stringify(x));
let differentialCases = 0;
test("actual application axes match with masks, smoothing, comparisons, targets, locks and drag", () => {
  const oldState = makeState(), newState = makeState();
  for(const raw of [[], [[null,null]], [[-.8,null],[.2,.3],[null,-.5]], [[0,0],[0,0]]]) {
    for(const smooth of [false,true]) for(const compare of [false,true]) for(const target of [false,true]) {
      const t={raw,effective:raw.map(row=>row.map(v=>Number.isFinite(v)?v*.7:v)),baseline:[[-.9,.6],[.4,-.8]]};
      const comparisons=compare?[{preview:{raw:[[-.95,.99],[null,-.9]]}}]:[];
      const edits=target?[{layer:"emotion",field:"valence",operation:"point_target",value:.99},
        {layer:"emotion",field:"arousal",operation:"point_target",value:-.99},
        {layer:"emotion",field:"valence",operation:"point_target",value:-1,enabled:false}]:[];
      for(const locked of [false,true]) for(const dragging of [false,true]) {
        for(const s of [oldState,newState]) {
          s.smoothing.enabled=smooth;s.axisLocked=locked;
          s.drag=dragging?{channel:0,geometry:{low:-.42,high:.32}}:null;
        }
        for(let channel=0;channel<2;channel++) {
          assert.deepEqual(plain(newRange(t,comparisons,channel,newState,edits)),plain(oldRange(t,comparisons,channel,oldState,edits)));
          differentialCases++;
        }
      }
    }
  }
});
function median(values) { return values.slice().sort((a,b)=>a-b)[Math.floor(values.length/2)]; }
const benchmarks=[];
for(const rows of [6001,36001,72001]) {
  const raw=Array.from({length:rows},(_,i)=>[i%41===0?null:Math.sin(i*.01)*.8,Math.cos(i*.02)*.7]);
  const t={raw,effective:raw.map(r=>r.map(v=>v===null?null:v*.9)),baseline:raw};
  const comparisons=[{preview:{raw:raw.map(r=>r.slice())}}];
  const before=JSON.stringify([t,comparisons]);
  const a=makeState(),b=makeState();a.smoothing.enabled=b.smoothing.enabled=true;
  const measure=(fn,state,frames)=>{const start=performance.now();for(let i=0;i<frames;i++)for(let ch=0;ch<2;ch++)fn(t,comparisons,ch,state,[]);return performance.now()-start;};
  measure(oldRange,a,2);const cold=measure(newRange,b,1);
  const oldSamples=[],newSamples=[];
  for(let n=0;n<5;n++) {
    if(n%2) {newSamples.push(measure(newRange,b,25));oldSamples.push(measure(oldRange,a,25));}
    else {oldSamples.push(measure(oldRange,a,25));newSamples.push(measure(newRange,b,25));}
  }
  assert.equal(JSON.stringify([t,comparisons]),before,"plotting must not mutate data");
  benchmarks.push({rows,frames_per_batch:25,cold_new_ms:cold,baseline_median_ms:median(oldSamples),cached_median_ms:median(newSamples),baseline_samples_ms:oldSamples,cached_samples_ms:newSamples,input_sha256:crypto.createHash('sha256').update(before).digest('hex')});
}
console.log(JSON.stringify({passed:true,checks,differential_cases:differentialCases,row_reads:rowReads,benchmarks,
  scope:"Only actual application axis range calculation under Node; excludes browser paint, layout, inference and scheduling.",
  baseline_sha256:crypto.createHash('sha256').update(fs.readFileSync(baselinePath)).digest('hex'),
  app_sha256:crypto.createHash('sha256').update(fs.readFileSync(appPath)).digest('hex')},null,2));
