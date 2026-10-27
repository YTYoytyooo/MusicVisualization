"use strict";
// Exercise the actual recovery/persistence/acceptance functions with isolated
// storage, DOM and service boundaries. Never touch real browser storage or files.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,"../../workbench-web/app.js"),"utf8");
const S=require("../../workbench-web/workbench-state.js"),plain=x=>JSON.parse(JSON.stringify(x)),checks=[];
function fn(name){
  const start=new RegExp("^  (?:async )?function "+name+"\\(","m").exec(source);assert.ok(start,name);
  const rest=source.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);
  return source.slice(start.index,start.index+start[0].length+(end?end.index:rest.length));
}
function declaration(name){const m=new RegExp("^  const "+name+" = [\\s\\S]*?;","m").exec(source);assert.ok(m,name);return m[0];}
const saved=[{id:"point",layer:"emotion",field:"valence",operation:"point_target",time_us:5e6,value:.4,
  transition_in_us:1e6,transition_out_us:1e6,interpolation:"smootherstep",enabled:true}];
const smoothing={enabled:false,window_seconds:.5},motion={engine:"flow-v1",sensitivity:1,min_hold_seconds:6,transition_seconds:2,
  overrides:[{id:"manual-orbit",enabled:true,mode:"orbit",start_us:1e6,end_us:3e6,params:{speed:.2,rotation:-1}}]};
function harness(){
  const elements={},values=new Map(),notices=[],requests=[],events={};
  const node=(tag,text)=>({tag,textContent:text,hidden:false,children:[],append(...children){this.children.push(...children);},
    replaceChildren(...children){this.children=children;},setAttribute(){}});
  const get=id=>(elements[id]??=node("div",""));
  const state={data:{project:{id:"p",analysis_id:"a",current_revision:"r"},revision:{id:"r"}},sourceSystem:"b01",
    edits:plain(saved),saved:JSON.stringify(saved),smoothing:plain(smoothing),savedSmoothing:plain(smoothing),
    motion:plain(motion),savedMotion:plain(motion),busy:false,loading:false,loadSerial:1,
    undoStack:[],redoStack:[],editingId:null,localFrame:null,tx:null,confirmQueue:[],finalizing:null,motionTx:null,motionFinal:null,motionFrame:null};
  const storage={getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v),removeItem:k=>values.delete(k)};
  const c={state,StudioWorkbenchState:S,localStorage:storage,$:get,node,clone:plain,Date,JSON,HISTORY_LIMIT:50,
    crypto:{randomUUID:()=>"recovery-test"},defaultMotion:()=>({engine:"legacy",sensitivity:1,min_hold_seconds:6,transition_seconds:2,overrides:[]}),
    notice:(text,error=false)=>{notices.push({text,error});get("notice").replaceChildren(node("span",text));},
    candidatePlanFor:async()=>null,api:async(route,body)=>{requests.push({route,body:plain(body),motion:plain(state.motion),smoothing:plain(state.smoothing)});return {};},
    localEmotion:edits=>({edits:plain(edits)}),emotionSignature:edits=>JSON.stringify(edits),scheduleMotionPlan:()=>{},adoptMotionPlan:()=>{},
    adoptVisualPlan:()=>{},invalidateVisualPlan:()=>{},updateDraft:()=>{},renderEdits:()=>{},renderMotionEdits:()=>{},fillMotionForm:()=>{},drawTimeline:()=>{},
    cancelEditing:()=>{state.editingId=null;},window:{addEventListener:(name,handler)=>{events[name]=handler;}}};
  vm.createContext(c);
  vm.runInContext(["dirty","projectId","revisionId","urlId","projectRoute","draftKey","pending"].map(declaration).join("\n")+"\n"+
    ["transactionContext","pushHistory","run","persistDraft","offerSavedDraft","applyDraft"].map(fn).join("\n")+
    "\nthis.getDraftKey=()=>draftKey(projectId(),revisionId());",c);
  const unload=source.match(/  window\.addEventListener\("beforeunload", event => \{[\s\S]*?\n  \}\);/);assert.ok(unload);vm.runInContext(unload[0],c);
  const payload=()=>({project_id:"p",base_revision:"r",source_system:"b01",analysis_id:"a",edits:[{...plain(saved[0]),value:.47}],
    smoothing:{enabled:true,window_seconds:1},motion:{...plain(motion),sensitivity:1.25}});
  const key=c.getDraftKey();
  const offer=(draft=payload())=>{values.set(key,typeof draft==="string"?draft:JSON.stringify(draft));c.offerSavedDraft();};
  const button=text=>get("notice").children.find(x=>x.tag==="button"&&x.textContent===text);
  return {c,state,values,key,storage,notices,requests,events,payload,offer,button,get};
}
async function test(name,run){try{await run();checks.push({name,passed:true});}catch(error){checks.push({name,passed:false,error:error.stack});}}
(async()=>{
  await test("accepted draft restores edits, smoothing and independent motion after server validation",async()=>{
    const h=harness(),draft=h.payload();h.offer(draft);await h.button("恢复草稿").onclick();
    assert.deepEqual(plain(h.state.edits),draft.edits);assert.deepEqual(plain(h.state.smoothing),draft.smoothing);assert.deepEqual(plain(h.state.motion),draft.motion);
    assert.equal(h.requests.length,1);assert.equal(h.requests[0].body.base_revision,"r");
    assert.deepEqual(h.requests[0].motion,draft.motion);assert.deepEqual(h.requests[0].smoothing,draft.smoothing);
    assert.equal(h.state.sourceSystem,"b01");assert.equal(h.state.data.revision.id,"r");assert.equal(h.state.undoStack.length,1);
    assert.deepEqual(plain(h.state.undoStack[0]),{edits:saved,smoothing,motion});
    assert.deepEqual(JSON.parse(h.values.get(h.key)).motion,draft.motion);assert.match(h.notices.at(-1).text,/已恢复并校验/);
  });
  for(const [field,value] of [["project_id","other"],["base_revision","other"],["source_system","v2"],["analysis_id","other"],["edits",null]]){
    await test("reject mismatched or invalid "+field+" without altering storage or saved state",async()=>{
      const h=harness(),draft={...h.payload(),[field]:value};h.offer(draft);
      assert.equal(h.button("恢复草稿"),undefined);assert.ok(h.button("丢弃损坏草稿"));
      assert.equal(h.values.get(h.key),JSON.stringify(draft));assert.deepEqual(plain(h.state.edits),saved);assert.equal(h.requests.length,0);
    });
  }
  await test("malformed JSON is retained until explicit discard",()=>{
    const h=harness();h.offer("{broken");assert.equal(h.button("恢复草稿"),undefined);
    assert.equal(h.values.get(h.key),"{broken");h.button("丢弃损坏草稿").onclick();assert.equal(h.values.has(h.key),false);
    assert.deepEqual(plain(h.state.edits),saved);assert.equal(h.requests.length,0);
  });
  await test("identical draft is removed without a false recovery prompt",()=>{
    const h=harness();h.offer({...h.payload(),edits:saved,smoothing,motion});
    assert.equal(h.values.has(h.key),false);assert.equal(h.button("恢复草稿"),undefined);assert.equal(h.requests.length,0);
  });
  await test("server rejection keeps draft, saved values and history intact",async()=>{
    const h=harness();h.offer();const before=h.values.get(h.key);h.c.api=async()=>{throw Error("server validation rejected");};
    await h.button("恢复草稿").onclick();assert.equal(h.values.get(h.key),before);
    assert.deepEqual(plain(h.state.edits),saved);assert.deepEqual(plain(h.state.smoothing),smoothing);assert.deepEqual(plain(h.state.motion),motion);
    assert.equal(h.state.undoStack.length,0);assert.equal(h.state.busy,false);assert.equal(h.notices.at(-1).error,true);
    assert.equal(h.notices.some(x=>x.text.includes("已恢复并校验")),false);
  });
  await test("old recovery and discard actions cannot touch a newly loaded project",async()=>{
    const h=harness();h.offer();const restore=h.button("恢复草稿"),discard=h.button("丢弃本地草稿"),before=h.values.get(h.key);
    h.state.loadSerial++;h.state.data.project.id="q";await restore.onclick();discard.onclick();
    assert.equal(h.requests.length,0);assert.equal(h.values.get(h.key),before);assert.deepEqual(plain(h.state.edits),saved);
  });
  await test("busy, loading and pending adjustments block recovery and discard actions",async()=>{
    for(const flag of ["busy","loading","tx"]){
      const h=harness();h.offer();const restore=h.button("恢复草稿"),discard=h.button("丢弃本地草稿"),before=h.values.get(h.key);
      h.state[flag]=flag==="tx"?{changed:true}:true;await restore.onclick();discard.onclick();
      assert.equal(h.requests.length,0);assert.equal(h.values.get(h.key),before);
    }
  });
  await test("persistence keeps accepted values, excluding unconfirmed temporary edits",()=>{
    const h=harness();h.state.edits=h.payload().edits;h.state.tx={changed:true,edits:[{...saved[0],value:.9}]};
    assert.equal(h.c.persistDraft(),true);const draft=JSON.parse(h.values.get(h.key));assert.deepEqual(draft.edits,h.payload().edits);
    assert.deepEqual(draft.motion,motion);assert.equal(draft.source_system,"b01");assert.equal(draft.analysis_id,"a");
    h.state.edits=plain(saved);h.state.tx=null;h.c.persistDraft();assert.equal(h.values.has(h.key),false);
  });
  await test("storage failure prevents keep-and-switch; cancellation never touches pending or storage",async()=>{
    const h=harness();h.state.edits=h.payload().edits;h.storage.setItem=()=>{throw Error("quota exceeded");};
    let finished=0;const actions={finishPending:async()=>{finished++;},persistDraft:h.c.persistDraft};
    await assert.rejects(S.prepareSwitch("keep",actions),/已阻止切换/);assert.equal(finished,1);assert.match(h.notices.at(-1).text,/无法保存本地草稿/);
    assert.equal(await S.prepareSwitch("cancel",actions),false);assert.equal(finished,1);assert.deepEqual(plain(h.state.edits),h.payload().edits);
  });
  await test("storage namespace separates project, revision, driver and fallback analysis checksum",()=>{
    const h=harness(),keys=[h.c.getDraftKey()];h.state.sourceSystem="v2";keys.push(h.c.getDraftKey());
    h.state.data.revision.id="r2";keys.push(h.c.getDraftKey());h.state.data.project.id="q";keys.push(h.c.getDraftKey());
    delete h.state.data.project.analysis_id;h.state.data.project.analysis_sha256="hash1";keys.push(h.c.getDraftKey());
    h.state.data.project.analysis_sha256="hash2";keys.push(h.c.getDraftKey());assert.equal(new Set(keys).size,keys.length);
  });
  await test("beforeunload warns for accepted or pending work, but not clean saved state",()=>{
    const h=harness(),attempt=()=>{const event={prevented:false,preventDefault(){this.prevented=true;}};h.events.beforeunload(event);return event;};
    assert.equal(attempt().prevented,false);h.state.edits=h.payload().edits;assert.equal(attempt().prevented,true);
    h.state.edits=plain(saved);h.state.motionTx={changed:true};const event=attempt();assert.equal(event.prevented,true);assert.equal(event.returnValue,"");
  });
  console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length},null,2));
  if(checks.some(x=>!x.passed))process.exitCode=1;
})();
