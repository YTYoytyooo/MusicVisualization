"use strict";
// Production pointer handlers and selection/transaction/cancel code, with a
// controllable canvas/media-free harness. Browser drag is checked separately.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,"../../workbench-web/app.js"),"utf8");
const results=[];
const plain=x=>JSON.parse(JSON.stringify(x));
function fn(name){const i=source.indexOf("  function "+name+"(");assert.ok(i>=0,name);return source.slice(i,source.indexOf("\n  function ",i+1));}
function harness(channel=0){
  const handlers={},captures=new Set(),calls=[],fields={},notices=[];
  const edit={id:"point",layer:"emotion",field:channel?"arousal":"valence",operation:"point_target",time_us:5e6,value:.4};
  const oldGeometry={left:0,right:100,top:0,bottom:100,low:-1,high:1,duration:30,view:{start:0,end:30,span:30}};
  const freshGeometry={left:52,right:582,top:22,bottom:127,low:-.18,high:.48,duration:30,view:{start:3.7,end:6.3,span:2.6}};
  const state={data:{project:{id:"p",duration:30},revision:{id:"r"}},edits:[edit],editingId:null,loading:false,busy:false,
    handles:[{editId:"point",x:317,y:34,channel}],plotGeometry:[plain(oldGeometry),plain(oldGeometry)],
    tx:null,txCounter:0,drag:null,confirmQueue:[],finalizing:null,localFrame:null,requestSeq:0,loadSerial:1,
    smoothing:{enabled:false},motion:{},confirmedPreview:"confirmed",preview:"confirmed",motionTx:null};
  const canvas={addEventListener:(event,cb)=>handlers[event]=cb,getBoundingClientRect:()=>({left:10,top:20}),
    setPointerCapture:id=>captures.add(id),hasPointerCapture:id=>captures.has(id),releasePointerCapture:id=>captures.delete(id)};
  const get=id=>id==="timeline"?canvas:(fields[id]??={focus:()=>calls.push("focus")});
  const context={state,$:get,clone:plain,projectId:()=>"p",revisionId:()=>"r",pending:()=>Boolean(state.tx?.changed),
    activateInspector:()=>calls.push("select-layer"),fillEditForm:e=>{get("edit-start").value=e.time_us/1e6;get("edit-value").value=e.value;},
    drawTimeline:()=>{calls.push("paint");state.plotGeometry[channel]=plain(freshGeometry);},updateDraft:()=>calls.push("draft"),
    queueCurrentTransaction:()=>{throw Error("unexpected queued transaction");},
    seekFromCanvas:()=>calls.push("seek"),scheduleInput:()=>calls.push("input"),flushInput:()=>calls.push("flush"),
    finishPending:async()=>{calls.push("finish");state.tx=null;},notice:message=>notices.push(message),
    clearTimeout:()=>{},cancelAnimationFrame:id=>calls.push("cancel-frame:"+id),releaseTemporary:()=>calls.push("release-temp"),
    cancelMotion:()=>{},localEmotion:edits=>plain(edits),scheduleMotionPlan:()=>{},};
  vm.createContext(context);
  const start=source.indexOf('  $("timeline").addEventListener("pointerdown"');
  const end=source.indexOf('  $("audio").addEventListener("timeupdate"',start);
  assert.ok(start>=0&&end>start);
  vm.runInContext(["run","editRecord","beginTransaction","cancelAdjustment"].map(fn).join("\n")+source.slice(start,end),context);
  function event(extra={}){return {button:0,pointerId:7,clientX:327,clientY:54,currentTarget:canvas,
    preventDefault(){calls.push("prevent");},...extra};}
  return {state,edit,handlers,captures,calls,fields,notices,context,freshGeometry,event,get,
    fire:(type,extra)=>handlers[type](event(extra))};
}
async function test(name,body){try{await body();results.push({name,passed:true});}catch(e){results.push({name,passed:false,error:e.message});}}
(async()=>{
  for(const channel of [0,1])await test(`dimension ${channel}: one start paint, refreshed geometry and pointer capture`,()=>{
    const h=harness(channel);h.fire("pointerdown");
    assert.equal(h.calls.filter(x=>x==="paint").length,1);
    assert.deepEqual(plain(h.state.drag.geometry),h.freshGeometry);
    assert.notEqual(h.state.drag.geometry,h.state.plotGeometry[channel]);
    assert.equal(h.state.drag.channel,channel);assert.equal(h.state.tx.changed,false);assert.ok(h.captures.has(7));
    assert.equal(h.calls.includes("focus"),false);
  });
  await test("drag uses the captured zoom and axis, independent of later plot geometry",()=>{
    const h=harness();h.fire("pointerdown");h.state.plotGeometry[0]={};
    h.fire("pointermove",{clientX:327,clientY:94.5});
    assert.equal(h.get("edit-start").value,"5.000");assert.equal(h.get("edit-value").value,"0.1500");
    assert.ok(h.calls.includes("input"));
    h.fire("pointermove",{clientX:-1e5,clientY:1e5});
    assert.equal(h.get("edit-start").value,"0.000");assert.equal(h.get("edit-value").value,"-1.0000");
  });
  await test("pointerup flushes input before confirmation and releases capture",async()=>{
    const h=harness();h.fire("pointerdown");await h.fire("pointerup");
    assert.ok(h.calls.indexOf("flush")<h.calls.indexOf("finish"));assert.equal(h.state.drag,null);
    assert.equal(h.captures.has(7),false);assert.equal(h.state.tx,null);assert.equal(h.state.edits[0].value,.4);
  });
  await test("pointercancel drops scheduled preview and restores the saved edit",()=>{
    const h=harness();h.fire("pointerdown");h.state.tx.changed=true;h.state.localFrame=42;
    h.state.preview="temporary";h.get("edit-start").value="6";h.get("edit-value").value=".7";
    h.fire("pointercancel");
    assert.equal(h.state.drag,null);assert.equal(h.state.tx,null);assert.equal(h.state.localFrame,null);
    assert.ok(h.calls.includes("cancel-frame:42"));assert.equal(h.get("edit-start").value,5);
    assert.equal(h.get("edit-value").value,.4);assert.deepEqual(plain(h.state.preview),[h.edit]);
  });
  await test("unrelated pointer cannot move, finish or cancel an active drag",async()=>{
    const h=harness();h.fire("pointerdown");const relevant=()=>h.calls.filter(x=>x!=="prevent");
    const before=relevant();
    h.fire("pointermove",{pointerId:8});await h.fire("pointerup",{pointerId:8});h.fire("pointercancel",{pointerId:8});
    // The generic run wrapper prevents the default pointerup action, even when
    // its handler ignores this pointer. It must not alter the edit or capture.
    assert.deepEqual(relevant(),before);assert.equal(h.state.drag.pointerId,7);assert.ok(h.captures.has(7));
  });
  await test("pending edit blocks a new keyframe drag without painting or capture",()=>{
    const h=harness();h.state.tx={changed:true};h.fire("pointerdown");
    assert.equal(h.state.drag,null);assert.equal(h.captures.size,0);assert.ok(!h.calls.includes("paint"));
    assert.match(h.notices[0],/确认或取消/);
  });
  await test("empty plot clicks seek; busy/loading/right-button input does not",()=>{
    const h=harness();h.fire("pointerdown",{clientX:900});assert.deepEqual(h.calls,["seek"]);
    h.calls.length=0;h.fire("pointerdown",{button:2});h.state.loading=true;h.fire("pointerdown");
    h.state.loading=false;h.state.busy=true;h.fire("pointerdown");assert.equal(h.calls.length,0);
  });
  await test("failed confirmation restores the original control values and reports rejection",async()=>{
    const h=harness();h.fire("pointerdown");h.state.tx.changed=true;h.get("edit-value").value=".9";
    h.context.finishPending=async()=>{throw Error("rejected");};await h.fire("pointerup");
    assert.equal(h.state.drag,null);assert.equal(h.state.tx,null);assert.equal(h.get("edit-value").value,.4);
    assert.equal(h.captures.size,0);assert.match(h.notices.at(-1),/未生效.*rejected/);
  });
  console.log(JSON.stringify({checks:results,passed:results.filter(r=>r.passed).length,total:results.length},null,2));
  if(results.some(r=>!r.passed))process.exitCode=1;
})();
