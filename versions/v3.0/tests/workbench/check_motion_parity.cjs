"use strict";
// Reads offline Python reference evidence; never loads audio, models or a browser.
const assert=require("node:assert/strict"), fs=require("node:fs");
const M=require("../../workbench-web/motion-math.js");
const T=require("../../workbench-web/timeline-math.js");
const fixture=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));
assert.equal(fixture.schema,1);assert.equal(fixture.tolerance,1e-9);
let numeric=0,maxError=0,frames=0;
function same(a,b,path=""){
  if(typeof b==="number"){
    assert.ok(Number.isFinite(a),path+" nonfinite");
    const error=path.endsWith(".direction_deg")?Math.abs(((a-b+540)%360)-180):Math.abs(a-b);
    numeric++;maxError=Math.max(maxError,error);
    assert.ok(error<=fixture.tolerance,`${path}: ${a} != ${b} (error ${error})`);return;
  }
  if(Array.isArray(b)){assert.ok(Array.isArray(a),path);assert.equal(a.length,b.length,path);
    b.forEach((v,i)=>same(a[i],v,`${path}[${i}]`));return;}
  if(b&&typeof b==="object"){
    assert.ok(a&&typeof a==="object",path);
    assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path+" fields");
    Object.keys(b).forEach(k=>same(a[k],b[k],path+"."+k));return;
  }
  assert.equal(a,b,path);
}
// Production metadata may carry diagnostics; compare the complete rendering
// contract plus source identity separately from optional diagnostic fields.
function contract(f){return {mode:f.mode,source:f.source,reason:f.reason,
  override_id:f.override_id??null,components:f.components};}
for(const c of fixture.motion){
  const before=JSON.stringify(c.plan), actual=M.applyOverrides(c.plan,c.config,c.duration_us);
  same(actual.config,c.expected_plan.config,c.name+" config");
  same(actual.times_us,c.expected_plan.times_us,c.name+" native clock");
  same(actual.segments,c.expected_plan.segments,c.name+" segments");
  for(const kind of ["auto","effective"])
    same(actual[kind].map(contract),c.expected_plan[kind].map(contract),c.name+" "+kind);
  c.times.forEach((t,i)=>{same(contract(M.sampleFrame(actual,t)),contract(c.expected[i]),c.name+" @"+t);frames++;});
  assert.equal(JSON.stringify(c.plan),before,c.name+" mutated source");
  console.log("PASS "+c.name+" ("+c.times.length+" samples)");
}
for(const c of fixture.visual){
  const before=JSON.stringify(c.plan), rows=T.visualAt(c.plan,c.times);
  same(rows,c.expected,c.name);assert.equal(JSON.stringify(c.plan),before,c.name+" mutated source");
  console.log("PASS "+c.name+" ("+c.times.length+" samples)");
}
console.log(JSON.stringify({status:"passed",motion_cases:fixture.motion.length,
  motion_samples:frames,visual_cases:fixture.visual.length,numeric_comparisons:numeric,max_absolute_error:maxError}));
