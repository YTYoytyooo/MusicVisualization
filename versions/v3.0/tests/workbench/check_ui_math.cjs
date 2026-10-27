"use strict";
// Independent expectations for V3 native clocks, masks, edits and draw-only reduction.
// Run with Node; this module never imports browser state or performs model inference.
const assert=require("node:assert/strict");
const T=require("../../workbench-web/timeline-math.js");
const S=require("../../workbench-web/workbench-state.js");
let checks=0;
function test(name,fn){fn();checks++;console.log("PASS "+name);}
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
function at(p,time,key="effective"){
  const i=p.times_us.indexOf(time);assert.ok(i>=0,"expected exact checkpoint "+time);return p[key][i];
}
const raw=[[-.8,null],[-.2,.3],[.8,.7]],clock=[0,500000,2000000];
test("explicit native clocks preserve independent missing dimensions",()=>{
  const p=T.emotionPreview(raw,clock,3000000,[],null,500000);
  assert.deepEqual(p.fields,["valence","arousal"]);
  assert.deepEqual(at(p,0),[-.8,null]);assert.deepEqual(at(p,500000),[-.2,.3]);
  assert.deepEqual(at(p,1000000),[-.2,.3]);assert.deepEqual(at(p,1000001),[null,null]);
  assert.deepEqual(at(p,2000000),[.8,.7]);assert.deepEqual(at(p,2500001),[null,null]);
});
test("time-window smoothing does not cross a mask or timestamp gap",()=>{
  const p=T.emotionPreview(raw,clock,3000000,[],{enabled:true,window_seconds:1},500000);
  near(at(p,0)[0],-.5);assert.equal(at(p,0)[1],null);
  near(at(p,500000)[0],-.5);near(at(p,500000)[1],.3);
  assert.deepEqual(at(p,2000000),[.8,.7]);assert.deepEqual(at(p,1000001),[null,null]);
});
test("real time windows differ from sample-index smoothing",()=>{
  const rows=[[.1,.1],[.3,.3],[.9,.9]],times=[0,400000,900000];
  const p=T.emotionPreview(rows,times,1000000,[],{enabled:true,window_seconds:.8},500000);
  near(at(p,0)[0],.2);near(at(p,400000)[0],.2);near(at(p,900000)[0],.9);
});
test("full manual target can supply a production value while raw remains missing",()=>{
  const edit={id:"manual",layer:"emotion",field:"valence",operation:"interval_set",value:.4,
    start_us:2600000,end_us:2800000,transition_in_us:100000,transition_out_us:100000};
  const p=T.emotionPreview(raw,clock,3000000,[edit],null,500000);
  assert.deepEqual(at(p,2600000,"raw"),[null,null]);assert.deepEqual(at(p,2600000),[.4,null]);
  assert.deepEqual(at(p,2500001),[null,null]);
});
test("offset does not invent missing baseline values",()=>{
  const edit={id:"offset",layer:"emotion",field:"arousal",operation:"interval_offset",value:.1,
    start_us:1100000,end_us:1200000,transition_in_us:0,transition_out_us:0};
  const p=T.emotionPreview(raw,clock,3000000,[edit],null,500000);
  assert.deepEqual(at(p,1100000),[null,null]);
});
test("native V2 tail policy is explicit and does not fill interior gaps",()=>{
  const p=T.emotionPreview(raw,clock,3000000,[],null,500000,true);
  assert.deepEqual(at(p,1000001),[null,null]);assert.deepEqual(at(p,3000000),[.8,.7]);
});
test("legacy scalar clock remains supported for genuine two-dimensional source",()=>{
  const p=T.emotionPreview([[.1,.2],[.3,.4]],100000,200000,[]);
  assert.deepEqual(at(p,0),[.1,.2]);assert.deepEqual(at(p,100000),[.3,.4]);
});
test("native timestamps and two-dimensional shape are strictly checked",()=>{
  assert.throws(()=>T.emotionPreview([[0,0,0,0,0]],[0],100000,[]));
  assert.throws(()=>T.emotionPreview([[0,0],[0,0]],[0,0],100000,[]));
  assert.throws(()=>T.emotionPreview([[0,0]],[200000],100000,[]));
  assert.throws(()=>T.emotionPreview([[NaN,0]],[0],100000,[]));
});
test("out-of-range offset is rejected rather than clipped silently",()=>{
  const edit={id:"overflow",layer:"emotion",field:"valence",operation:"interval_offset",value:.2,
    start_us:100000,end_us:300000,transition_in_us:0,transition_out_us:0};
  assert.throws(()=>T.emotionPreview([[.9,0],[.9,0]],[0,500000],1000000,[edit],null,500000),/越界/);
});
test("draw-only reduction preserves peaks and both sides of missing runs",()=>{
  const times=Array.from({length:10001},(_,i)=>i/10),values=times.map(()=>0);
  values[3999]=.99;values[4000]=null;values[4001]=null;values[4002]=-.87;
  const kept=T.plotIndices(times,values,1);
  for(const i of [3999,4000,4001,4002])assert.ok(kept.includes(i),"missing boundary/peak "+i);
  assert.ok(kept.length<30,"decimation should remain bounded for a finite isolated gap");
});
test("manual edits and smoothing never mutate model samples",()=>{
  const before=JSON.stringify(raw),beforeClock=JSON.stringify(clock);
  T.emotionPreview(raw,clock,3000000,[],{enabled:true,window_seconds:1},500000);
  assert.equal(JSON.stringify(raw),before);assert.equal(JSON.stringify(clock),beforeClock);
});
test("driver reset remains possible after retained offsets fail on the new source",()=>{
  const target={id:"new-model",times_us:[0,500000],step_us:500000,values:[[.9,.2],[.9,.2]]};
  const offset={id:"old-offset",layer:"emotion",field:"valence",operation:"interval_offset",value:.2,
    start_us:100000,end_us:300000,transition_in_us:0,transition_out_us:0};
  const before=JSON.stringify(target);
  assert.throws(()=>T.sourcePreview(target,1000000,[offset]),/越界/);
  const reset=T.sourcePreview(target,1000000,[]);
  assert.deepEqual(at(reset,0),[.9,.2]);assert.equal(JSON.stringify(target),before);
  assert.throws(()=>T.sourcePreview(null,1000000,[]),/原始对照数据不可用/);
});
test("source list refresh preserves a still-available selected driver",()=>{
  assert.equal(S.preserveSelection(["b01","b00","v2"],"b00"),"b00");
  assert.equal(S.preserveSelection(["b01"],"b00"),"b01");
  assert.equal(S.preserveSelection([],"b00"),"");
});
test("bootstrap selection never overrides a manual project selection",()=>{
  assert.equal(S.shouldAutoSelectProject(0,0,false,false),true);
  assert.equal(S.shouldAutoSelectProject(0,1,false,true),false);
  assert.equal(S.shouldAutoSelectProject(0,1,true,false),false);
  assert.equal(S.shouldAutoSelectProject(0,1,false,false),false);
  assert.equal(S.shouldAutoSelectProject(0,0,true,false),false);
});
test("raw downloads follow the current driver even when the selected video stays unchanged",()=>{
  const systems=[{id:"v2",raw_csv_url:"/api/projects/p/raw?source_system=v2"},
    {id:"b01",raw_csv_url:"/api/projects/p/raw?source_system=b01"}];
  const render={id:"full-1",source_system:"v2",video_url:"/media/p/renders/full-1/video.mp4"};
  const previous=S.outputDownloads(systems,"v2",render),current=S.outputDownloads(systems,"b01",render);
  assert.deepEqual(previous[0],["下载原始预测（v2）",systems[0].raw_csv_url]);
  assert.deepEqual(current[0],["下载原始预测（b01）",systems[1].raw_csv_url]);
  assert.deepEqual(current.slice(1),previous.slice(1));
  assert.equal(S.outputDownloads(systems,"b01",null)[0][1],systems[1].raw_csv_url);
  assert.equal(S.outputDownloads(systems,"missing",render)[0][1],undefined);
});
async function navigationChecks(){
  const payload={project_id:"p-1",base_revision:"r-1",source_system:"b00",edits:[{value:.2}]};
  const fullStorage={setItem(){throw new Error("QuotaExceededError");},removeItem(){throw new Error("SecurityError");}};
  const failed=S.writeDraft(fullStorage,"draft",payload);assert.equal(failed.ok,false);
  let navigated=false,finished=false;
  await assert.rejects(async()=>{
    if(await S.prepareSwitch("keep",{finishPending:async()=>{finished=true;},persistDraft:()=>failed.ok,
      saveRevision:async()=>{throw new Error("unexpected save");}}))navigated=true;
  },/已阻止切换/);
  assert.equal(finished,true);assert.equal(navigated,false);checks++;console.log("PASS failed local draft storage prevents navigation");
  const values=new Map(),storage={setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  assert.equal(await S.prepareSwitch("keep",{finishPending:async()=>{},persistDraft:()=>S.writeDraft(storage,"draft",payload).ok}),true);
  assert.deepEqual(JSON.parse(values.get("draft")),payload);
  let saved=false;assert.equal(await S.prepareSwitch("save",{finishPending:async()=>{},saveRevision:async()=>{saved=true;},persistDraft:()=>false}),true);
  assert.equal(saved,true);
  let touched=false;assert.equal(await S.prepareSwitch("cancel",{finishPending:async()=>{touched=true;}}),false);assert.equal(touched,false);
  checks++;console.log("PASS keep, save and cancel preserve their separate navigation semantics");
  console.log(JSON.stringify({status:"passed",checks}));
}
navigationChecks().catch(error=>{console.error(error);process.exitCode=1;});
