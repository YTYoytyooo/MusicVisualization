"use strict";
// Compare production canvas commands against a frozen pre-change function.
// Long axes below are deterministic synthetic plotting data, never training labels.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm"),{performance}=require("node:perf_hooks");
const T=require("../../workbench-web/timeline-math.js");
const root=path.resolve(__dirname,"../../../.."),round=path.join(root,"data/v3.0/optimization/20261024/rounds/013");
const source=fs.readFileSync(process.argv[2]||path.join(root,"versions/v3.0/workbench-web/app.js"),"utf8"),reference=fs.readFileSync(path.join(round,"app.before.js"),"utf8");
const fixture=JSON.parse(fs.readFileSync(path.join(root,"data/v3.0/optimization/20261024/rounds/010/parity-fixture.json"),"utf8")).visual[0];
const plain=x=>JSON.parse(JSON.stringify(x)),checks=[];
function fn(src,name){const start=new RegExp("^  (?:async )?function "+name+"\\(","m").exec(src);assert.ok(start,name);
  const rest=src.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);
  return src.slice(start.index,start.index+start[0].length+(end?end.index:rest.length));}
const ranges={hue_base:[0,360],hue_range:[20,120],saturation:[.3,1],brightness:[.1,.9],particle_count:[50,500],particle_speed:[.5,8],field_turbulence:[0,1],trail_length:[5,60]};
function harness(src,preview,edits=[],countReads=false){
  const operations=[],counters={rowReads:0};let record=true;
  const context2d=new Proxy({}, {get:(o,k)=>o[k]??((...args)=>{if(record)operations.push([k,...args.map(v=>Array.isArray(v)?Array.from(v):v)]);}),set:(o,k,v)=>{o[k]=v;if(record)operations.push(["set",k,v]);return true;}});
  const canvas={clientWidth:640,dataset:{},getContext:()=>context2d};
  const elements={"visual-timeline":canvas,"visual-field":{value:"brightness"}};
  const copyPreview=value=>{
    const p=plain(value);if(countReads)for(const key of ["raw","effective"])p[key]=p[key].map(row=>new Proxy(row,{get:(o,k)=>{if(/^\d+$/.test(String(k)))counters.rowReads++;return o[k];}}));
    return p;
  };
  const state={data:{project:{duration:preview.times.at(-1)}},sourceSystem:"b01",edits:plain(edits),visualPlan:{key:"fixture"},planSignature:"emotion",
    visualPreview:null,viewport:{start:0,end:preview.times.at(-1)},seriesCache:new WeakMap(),drawCache:new WeakMap(),extentFor:T.createExtentCache(),axisCache:{},axisLocked:false,time:0,tx:null,editingId:null};
  const c={state,$:id=>elements[id],window:{devicePixelRatio:1},StudioTimeline:T,fieldRanges:ranges,
    visibleEdits:()=>state.edits,emotionSignature:()=>"emotion",durationUs:()=>Math.round(state.data.project.duration*1e6),clock:time=>time.toFixed(3)};
  vm.createContext(c);vm.runInContext(["sampleAt","channelValues","viewRange","shadeImpact","drawingIndices","drawVisual"].map(name=>fn(src,name)).join("\n"),c);
  const replace=p=>{state.visualPlan={key:state.visualPlan.key+"-new"};state.visualPreview=copyPreview(p);state.visualPreview.signature=JSON.stringify([state.visualPlan.key,state.sourceSystem,state.edits]);};
  replace(preview);
  return {state,canvas,elements,counters,replace,setRecord:v=>record=v,draw:()=>{operations.length=0;c.drawVisual();return operations.slice();}};
}
function test(name,run){try{run();checks.push({name,passed:true});}catch(error){checks.push({name,passed:false,error:error.message});}}
const duration=fixture.plan.times_us.at(-1),preview=T.visualPreview(fixture.plan,duration,[]);
for(const field of Object.keys(ranges))test(field+": exact canvas commands across playback, resize, zoom and locked axes",()=>{
  const before=harness(reference,preview),after=harness(source,preview);
  for(const h of [before,after])h.elements["visual-field"].value=field;
  for(const [time,start,end,width,locked] of [[0,0,duration/1e6,640,false],[5,0,duration/1e6,640,false],[5,3.7,6.3,640,false],[5.1,3.7,6.3,375,true],[20,15,23,900,false]]){
    for(const h of [before,after]){h.state.time=time;h.state.viewport={start,end};h.canvas.clientWidth=width;h.state.axisLocked=locked;}
    assert.deepEqual(after.draw(),before.draw());
  }
});
test("new immutable preview invalidates extrema and columns, preserving manual target shading",()=>{
  const edit={id:"visual",layer:"visual",field:"brightness",operation:"point_target",time_us:5e6,value:.7,transition_in_us:1e6,transition_out_us:1e6,interpolation:"smootherstep",enabled:true};
  const next=T.visualPreview(fixture.plan,duration,[edit]),before=harness(reference,preview),after=harness(source,preview);
  assert.deepEqual(after.draw(),before.draw());
  for(const h of [before,after]){h.state.edits=[edit];h.state.editingId=edit.id;h.replace(next);h.state.time=5;}
  assert.deepEqual(after.draw(),before.draw());
  assert.equal(after.state.visualPreview.effective[after.state.visualPreview.times_us.indexOf(5e6)][3],.7);
});
test("returning to a field never reuses another field's axis or samples",()=>{
  const before=harness(reference,preview),after=harness(source,preview);
  for(const field of ["hue_base","brightness","particle_count","brightness","hue_base"]){
    for(const h of [before,after])h.elements["visual-field"].value=field;
    assert.deepEqual(after.draw(),before.draw());
  }
});
test("unmatched plan and absent field keep the existing unavailable display",()=>{
  for(const unavailable of ["stale","absent"]){const before=harness(reference,preview),after=harness(source,preview);
    for(const h of [before,after]){if(unavailable==="stale")h.state.planSignature="old";else h.elements["visual-field"].value="absent";}
    assert.deepEqual(after.draw(),before.draw());assert.equal(after.canvas.dataset.ready,"false");
  }
});
function longPreview(n){const times=Array.from({length:n},(_,i)=>i/10),raw=times.map((t,i)=>[180+90*Math.sin(t),50+10*Math.cos(t),.6+.1*Math.sin(t),.5+.1*Math.cos(t),200+i%100,3+Math.sin(t),.5+.1*Math.cos(t),20+i%20]);
  return {times,raw,effective:raw.map(row=>row.slice()),fields:Object.keys(ranges)};}
const synthetic=longPreview(36001),beforeCounter=harness(reference,synthetic,[],true),afterCounter=harness(source,synthetic,[],true);
for(const h of [beforeCounter,afterCounter]){h.state.viewport={start:1790,end:1810};h.draw();h.counters.rowReads=0;h.state.time=1800;}
const afterCommands=afterCounter.draw(),beforeCommands=beforeCounter.draw();
test("one-hour synthetic axis has identical visible peaks, paths and cursor",()=>assert.deepEqual(afterCommands,beforeCommands));
test("warm same-snapshot draw avoids whole-array rereads",()=>assert.ok(afterCounter.counters.rowReads<beforeCounter.counters.rowReads/4,
  `row reads before=${beforeCounter.counters.rowReads}, after=${afterCounter.counters.rowReads}`));
function benchmark(src,p){const h=harness(src,p);h.state.viewport={start:p.times.at(-1)/2-10,end:p.times.at(-1)/2+10};h.setRecord(false);h.draw();
  for(let i=0;i<10;i++){h.state.time=i;h.draw();}
  const samples=[];for(let run=0;run<5;run++){const start=performance.now();for(let i=0;i<60;i++){h.state.time=i/10;h.draw();}samples.push((performance.now()-start)/60);}
  return {runs:5,draws_per_run:60,ms_per_draw:samples,median_ms:samples.slice().sort((a,b)=>a-b)[2]};}
const beforeHash=JSON.stringify(preview),performanceResult={scope:"Node production drawVisual with no-op canvas sink, warm cache; not browser frame rate",cached_28s:{points:preview.times.length,before:benchmark(reference,preview),after:benchmark(source,preview)},synthetic_1h:{points:synthetic.times.length,before:benchmark(reference,synthetic),after:benchmark(source,synthetic)},warm_row_reads:{before:beforeCounter.counters.rowReads,after:afterCounter.counters.rowReads}};
test("read-only drawing never mutates the source fixture",()=>assert.equal(JSON.stringify(preview),beforeHash));
console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length,performance:performanceResult},null,2));
if(checks.some(x=>!x.passed))process.exitCode=1;
