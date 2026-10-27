"use strict";
// Exercise production viewport/locate/seek functions with paint observations.
// No audio, model predictions, rendering jobs or revisions are created.
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const source = fs.readFileSync(process.argv[2] || path.resolve(__dirname, "../../workbench-web/app.js"), "utf8");
const checks = [];
function test(name, fn) {
  try { fn(); checks.push({name, passed:true}); }
  catch (error) { checks.push({name, passed:false, error:error.message}); }
}
function functionSource(name) {
  const start = source.indexOf("  function " + name + "(");
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf("\n  function ", start + 1));
}
function harness(edit) {
  const state = {data:{project:{duration:30}},time:1,viewport:{start:0,end:30},
    editingId:"selected",drag:null,selectedRender:null,comparing:false};
  const elements = {audio:{paused:true},video:{hidden:true,duration:NaN},
    playhead:{},clock:{},"view-start":{},"view-end":{},"viewport-state":{},timeline:{dataset:{}}};
  const draws = [];
  const context = {state, $:id=>elements[id], clock:v=>v.toFixed(3),
    visibleEdits:()=>edit ? [{id:"selected",...edit}] : [],
    drawTimeline:()=>draws.push({time:state.time,start:state.viewport.start,end:state.viewport.end,
      playhead:elements.playhead.value,viewStart:elements["view-start"].value,viewEnd:elements["view-end"].value})};
  vm.createContext(context);
  vm.runInContext(["viewRange","updateViewportLabel","setViewport","zoomViewport","locateEdit","seek"]
    .map(functionSource).join("\n"),context);
  return {state,elements,draws,...context};
}
for (const [name, edit, expected] of [
  ["point target",{time_us:5e6,transition_in_us:1e6,transition_out_us:1e6},[3.7,6.3,5]],
  ["zero-time target",{time_us:0,transition_in_us:0,transition_out_us:1e6},[0,1.4,0]],
  ["tail target",{time_us:29e6,transition_in_us:1e6,transition_out_us:1e6},[27.4,30,29]],
  ["interval",{start_us:10e6,end_us:14e6,transition_in_us:1e6,transition_out_us:2e6},[7.95,17.05,10]]
]) test(name+": one synchronous paint with the final viewport and playhead",()=>{
  const h=harness(edit);h.locateEdit();
  assert.equal(h.draws.length,1,"one final paint, no intermediate old playhead");
  const d=h.draws[0];
  for(const [actual,want] of [[d.start,expected[0]],[d.end,expected[1]],[d.time,expected[2]]])
    assert.ok(Math.abs(actual-want)<1e-9,actual+" != "+want);
  assert.equal(d.playhead,expected[2].toFixed(3));
  assert.equal(d.viewStart,expected[0].toFixed(3));assert.equal(d.viewEnd,expected[1].toFixed(3));
});
test("ordinary viewport and zoom remain synchronous and paint once each",()=>{
  const h=harness();h.setViewport(4,10);assert.equal(h.draws.length,1);
  h.zoomViewport(2);assert.equal(h.draws.length,2);
  assert.equal(h.state.viewport.start,5.5);assert.equal(h.state.viewport.end,8.5);
});
test("missing selection and invalid range fail without changing the view or painting",()=>{
  const h=harness();assert.throws(()=>h.locateEdit(),/选择/);
  assert.throws(()=>h.setViewport(4,3),/终点/);
  assert.equal(h.draws.length,0);assert.equal(h.state.viewport.start,0);assert.equal(h.state.time,1);
});
test("viewport change remains blocked during a keyframe drag",()=>{
  const h=harness();h.state.drag={};h.setViewport(4,10);
  assert.equal(h.draws.length,0);assert.equal(h.state.viewport.start,0);assert.equal(h.state.viewport.end,30);
});
console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length},null,2));
if(checks.some(x=>!x.passed))process.exitCode=1;
