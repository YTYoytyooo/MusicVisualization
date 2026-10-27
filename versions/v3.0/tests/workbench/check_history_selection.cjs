"use strict";
// Production history/apply/form functions with controllable async validation.
// No data import, audio processing, model inference or saved revisions.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,"../../workbench-web/app.js"),"utf8");
const plain=x=>JSON.parse(JSON.stringify(x));
const checks=[];
function fn(name){
  const start=new RegExp("^  (?:async )?function "+name+"\\(","m").exec(source);assert.ok(start,name);
  const rest=source.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);
  return source.slice(start.index,start.index+start[0].length+(end?end.index:rest.length));
}
const smoothing={enabled:false,window_seconds:.5},motion={engine:"legacy",overrides:[]};
function snapshot(edits){return {edits:plain(edits),smoothing:plain(smoothing),motion:plain(motion)};}
function harness(current, previous){
  const elements={},calls=[];
  const get=id=>(elements[id]??={value:"",hidden:false,focus:()=>calls.push("focus")});
  const state={data:{project:{id:"p"},revision:{id:"r"}},edits:[plain(current)],editingId:current.id,
    smoothing:plain(smoothing),motion:plain(motion),undoStack:[snapshot(previous)],redoStack:[],
    busy:false,loading:false,drag:null,tx:null,confirmQueue:[],finalizing:null,loadSerial:1};
  const context={state,$:get,clone:plain,HISTORY_LIMIT:50,crypto:{randomUUID:()=>"history-test"},
    projectId:()=>"p",revisionId:()=>"r",projectRoute:()=>"/p",defaultMotion:()=>plain(motion),
    populateFields:()=>{get("edit-field").value="";},updateFieldHint:()=>{},drawTimeline:()=>calls.push("draw"),
    updateDraft:()=>{},persistDraft:()=>calls.push("persist"),renderEdits:()=>calls.push("render"),
    renderMotionEdits:()=>{},fillMotionForm:()=>{},candidatePlanFor:async()=>null,api:async()=>({}),
    localEmotion:edits=>({edits:plain(edits)}),emotionSignature:edits=>JSON.stringify(edits),
    scheduleMotionPlan:()=>{},adoptMotionPlan:()=>{},adoptVisualPlan:()=>{},invalidateVisualPlan:()=>{},
    finishPending:async()=>{calls.push("finish");state.tx=null;},
    cancelAdjustment:()=>{calls.push("cancel");state.tx=null;},cancelMotion:()=>{},notice:()=>{}};
  vm.createContext(context);
  vm.runInContext(["transactionContext","pushHistory","updateOperation","fillEditForm","cancelEditing","applyDraft","stepHistory"].map(fn).join("\n"),context);
  context.fillEditForm(current);get("apply-edit").textContent="确认调整";get("cancel-edit").hidden=false;
  return {state,get,calls,context,undo:()=>context.stepHistory("undo"),redo:()=>context.stepHistory("redo")};
}
function assertForm(h,edit){
  const expected={"edit-layer":edit.layer,"edit-field":edit.field,"edit-operation":edit.operation,
    "edit-start":(edit.time_us??edit.start_us)/1e6,"edit-end":(edit.end_us??edit.time_us)/1e6,
    "edit-value":edit.value,"edit-in":(edit.transition_in_us||0)/1e6,"edit-out":(edit.transition_out_us||0)/1e6,
    "edit-note":edit.note||"","edit-interpolation":edit.interpolation||"linear"};
  for(const [id,value] of Object.entries(expected))assert.equal(String(h.get(id).value),String(value),id);
  assert.equal(h.state.editingId,edit.id);assert.equal(h.get("apply-edit").textContent,"确认调整");
  assert.equal(h.get("cancel-edit").hidden,false);assert.equal(h.get("end-label").hidden,edit.operation==="point_target");
  assert.equal(h.calls.includes("focus"),false,"history must not steal focus");
}
const point={id:"edit",layer:"emotion",field:"valence",operation:"point_target",time_us:5e6,value:.4,
  transition_in_us:1e6,transition_out_us:1e6,interpolation:"smootherstep",note:"saved"};
async function test(name,run){try{await run();checks.push({name,passed:true});}catch(error){checks.push({name,passed:false,error:error.message});}}
(async()=>{
  for(const [layer,field,value] of [["emotion","valence",.1],["emotion","arousal",-.2],["visual","hue_base",359]]){
    await test(field+": undo and redo synchronize every field and retain identity",async()=>{
      const previous={...point,layer,field,operation:"interval_set",start_us:2e6,end_us:4e6,time_us:undefined,
        value,transition_in_us:2e5,transition_out_us:3e5,interpolation:"linear",note:"previous"};
      const current={...point,layer,field,value:field==="hue_base"?3:.6};
      const h=harness(current,[previous]);await h.undo();assertForm(h,previous);
      assert.equal(h.state.edits.length,1);assert.deepEqual(plain(h.state.preview.edits),[plain(previous)]);
      assert.equal(h.state.undoStack.length,0);assert.equal(h.state.redoStack.length,1);
      await h.redo();assertForm(h,current);assert.equal(h.state.edits.length,1);
      assert.equal(h.state.undoStack.length,1);assert.equal(h.state.redoStack.length,0);
    });
  }
  await test("undo newly added selection clears identity; redo never binds a nonexistent record",async()=>{
    const h=harness(point,[]);await h.undo();assert.equal(h.state.editingId,null);
    assert.equal(h.get("apply-edit").textContent,"添加修改");assert.equal(h.get("cancel-edit").hidden,true);
    await h.redo();assert.equal(h.state.edits.length,1);assert.equal(h.state.editingId,null);
  });
  await test("undo deletion restores the record without selecting it or overwriting an unselected form",async()=>{
    const h=harness(point,[point]);h.state.edits=[];h.state.editingId=null;h.get("edit-value").value="draft-template";
    await h.undo();assert.equal(h.state.edits.length,1);assert.equal(h.state.editingId,null);
    assert.equal(h.get("edit-value").value,"draft-template");
  });
  await test("pending adjustment finishes before choosing the undo snapshot",async()=>{
    const pending={...point,time_us:6e6,value:.7};const h=harness(pending,[{...point,value:-.1}]);
    let release;h.context.finishPending=()=>new Promise(resolve=>{release=()=>{
      h.state.edits=[plain(pending)];h.state.undoStack.push(snapshot([point]));resolve();};});
    const promise=h.undo();assert.equal(h.get("edit-value").value,.7);assert.equal(h.calls.includes("persist"),false);
    release();await promise;assertForm(h,point);assert.equal(h.state.undoStack.length,1);
  });
  await test("rejected history validation leaves form, selection and stacks unchanged",async()=>{
    const h=harness(point,[{...point,value:.2}]);h.context.api=async()=>{throw Error("validation rejected");};
    await assert.rejects(h.undo(),/validation rejected/);assertForm(h,point);
    assert.equal(h.state.undoStack.length,1);assert.equal(h.state.redoStack.length,0);
    assert.equal(h.calls.includes("persist"),false);assert.equal(h.state.busy,false);
  });
  await test("changed project during validation cannot refill the selected form",async()=>{
    const h=harness(point,[{...point,value:.2}]);h.context.api=async()=>{h.state.loadSerial++;return {};};
    await assert.rejects(h.undo(),/项目已切换/);assertForm(h,point);assert.equal(h.calls.includes("persist"),false);
  });
  await test("empty history and busy/dragging states leave the editor untouched",async()=>{
    const h=harness(point,[]);h.state.undoStack=[];await h.undo();assertForm(h,point);
    for(const flag of ["busy","loading","drag"]){h.state[flag]=true;await h.undo();h.state[flag]=false;assertForm(h,point);}
  });
  console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length},null,2));
  if(checks.some(x=>!x.passed))process.exitCode=1;
})();
