"use strict";
// Execute production key handlers at their DOM boundary; no browser state injection.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,'../../workbench-web/app.js'),'utf8'),checks=[];
function section(start,end){const a=source.indexOf(start),b=source.indexOf(end,a+start.length);assert.ok(a>=0&&b>a,start);return source.slice(a,b);}
function harness(actualCancel=false){
  const handlers={},tasks=[],calls={motion:0,emotion:0,cancelMotion:0,cancelEmotion:0,seek:[],history:[],notices:[],cancelledFrames:[],restoredForm:0};
  class Element {
    constructor(tag='div',editable=false){this.tag=tag;this.isContentEditable=editable;}
    matches(selector){return selector.split(',').map(x=>x.trim()).includes(this.tag);}
    closest(selector){return this.matches(selector)||(this.isContentEditable&&selector.includes('contenteditable'))?this:null;}
  }
  const state={data:{project:{duration:30}},time:5,loading:false,busy:false,drag:null,motionTx:null,motionFrame:null,undoStack:[{}],redoStack:[{}]};
  const controls={pending:false,dialog:false};
  const c={Element,state,document:{querySelector:()=>controls.dialog?{}:null,addEventListener:(event,handler)=>handlers.document=handler},
    $:id=>({addEventListener:(event,handler)=>handlers[id]=handler}),pending:()=>controls.pending,
    seek:t=>calls.seek.push(t),cancelMotion:()=>{calls.cancelMotion++;},cancelAdjustment:()=>{calls.cancelEmotion++;},notice:text=>calls.notices.push(text),
    finishMotion:()=>{calls.motion++;},finishPending:()=>{calls.emotion++;},stepHistory:direction=>calls.history.push(direction),
    clearTimeout:()=>{},cancelAnimationFrame:id=>calls.cancelledFrames.push(id),fillMotionForm:()=>{calls.restoredForm++;},scheduleMotionPlan:()=>{},updateDraft:()=>{},drawMotion:()=>{},
    run:action=>()=>{const task=Promise.resolve().then(action);tasks.push(task);return task;}};
  vm.createContext(c);
  vm.runInContext(section('  $("motion-form").addEventListener("keydown"','  $("motion-form").addEventListener("change"')+
    section('  $("edit-form").addEventListener("keydown"','  $("visual-field").onchange')+
    section('  document.addEventListener("keydown"','  // A native anchor'),c);
  if(actualCancel)vm.runInContext(section('  function cancelMotion(', '  async function finishMotion('),c);
  return {state,controls,calls,async key(where,key,options={}){const event={key,target:new Element(),defaultPrevented:false,isComposing:false,ctrlKey:false,metaKey:false,altKey:false,shiftKey:false,...options};
    if(options.tag)event.target=new Element(options.tag,options.editable);event.preventDefault=()=>{event.defaultPrevented=true;};handlers[where](event);await Promise.all(tasks);return event;}};
}
async function test(name,action){try{await action();checks.push({name,passed:true});}catch(e){checks.push({name,passed:false,error:e.message});}}
(async()=>{
  for(const where of ['motion-form','edit-form']){
    for(const boundary of ['isComposing','defaultPrevented'])await test(where+': '+boundary+' Enter does not confirm',async()=>{const h=harness();await h.key(where,'Enter',{tag:'input',[boundary]:true});assert.equal(h.calls.motion+h.calls.emotion,0);});
    await test(where+': ordinary Enter confirms once and prevents form default',async()=>{const h=harness(),event=await h.key(where,'Enter',{tag:'input'});assert.equal(h.calls.motion+h.calls.emotion,1);assert.equal(event.defaultPrevented,true);});
    await test(where+': other keys and button Enter are left to their controls',async()=>{const h=harness();for(const [key,tag] of [['Escape','input'],['a','input'],['Enter','button']])await h.key(where,key,{tag});assert.equal(h.calls.motion+h.calls.emotion,0);});
  }
  for(const boundary of ['isComposing','defaultPrevented']){
    for(const type of ['emotion','motion'])await test(type+': '+boundary+' Escape preserves pending adjustment',async()=>{const h=harness();h.controls.pending=true;if(type==='motion')h.state.motionTx={changed:true};
      await h.key('document','Escape',{tag:'input',[boundary]:true});assert.equal(h.calls.cancelMotion+h.calls.cancelEmotion,0);assert.equal(h.calls.notices.length,0);});
    await test(boundary+': arrow keys cannot seek the song',async()=>{const h=harness();await h.key('document','ArrowRight',{[boundary]:true});assert.deepEqual(h.calls.seek,[]);});
  }
  await test('plain Escape cancels pending emotion and releases drag',async()=>{const h=harness();h.controls.pending=true;h.state.drag={pointerId:1};const e=await h.key('document','Escape',{tag:'input'});assert.equal(h.calls.cancelEmotion,1);assert.equal(h.state.drag,null);assert.equal(e.defaultPrevented,true);});
  await test('plain Escape cancels motion first without cancelling emotion twice',async()=>{const h=harness();h.controls.pending=true;h.state.motionTx={changed:true};await h.key('document','Escape');assert.equal(h.calls.cancelMotion,1);assert.equal(h.calls.cancelEmotion,0);});
  await test('Escape before motion animation frame cancels motion rather than emotion',async()=>{const h=harness();h.controls.pending=true;h.state.motionTx={changed:false};h.state.motionFrame=7;await h.key('document','Escape',{tag:'input'});assert.equal(h.calls.cancelMotion,1);assert.equal(h.calls.cancelEmotion,0);});
  await test('production motion cancel removes queued frame and restores form baseline',async()=>{const h=harness(true);h.controls.pending=true;h.state.motionTx={changed:false};h.state.motionFrame=7;await h.key('document','Escape',{tag:'input'});assert.deepEqual(h.calls.cancelledFrames,[7]);assert.equal(h.state.motionFrame,null);assert.equal(h.state.motionTx,null);assert.equal(h.calls.restoredForm,1);assert.equal(h.calls.cancelEmotion,0);});
  await test('composing or consumed Escape preserves queued motion frame',async()=>{for(const boundary of ['isComposing','defaultPrevented']){const h=harness();h.controls.pending=true;h.state.motionTx={changed:false};h.state.motionFrame=7;await h.key('document','Escape',{tag:'input',[boundary]:true});assert.equal(h.calls.cancelMotion+h.calls.cancelEmotion,0);assert.equal(h.state.motionFrame,7);}});
  await test('Escape without pending work leaves native default intact',async()=>{const h=harness(),e=await h.key('document','Escape');assert.equal(e.defaultPrevented,false);assert.equal(h.calls.notices.length,0);});
  await test('open dialog, loading and busy state preserve pending work on Escape',async()=>{for(const flag of ['dialog','loading','busy']){const h=harness();h.controls.pending=true;if(flag==='dialog')h.controls.dialog=true;else h.state[flag]=true;await h.key('document','Escape');assert.equal(h.calls.cancelEmotion,0);}});
  await test('outside input ordinary arrows seek exactly one second',async()=>{const h=harness();await h.key('document','ArrowLeft');await h.key('document','ArrowRight');assert.deepEqual(h.calls.seek,[4,6]);});
  await test('input and editable content retain their own arrows and undo',async()=>{for(const target of [{tag:'input'},{tag:'select'},{tag:'textarea'},{tag:'span',editable:true}]){const h=harness();await h.key('document','ArrowLeft',target);await h.key('document','z',{...target,ctrlKey:true});assert.deepEqual(h.calls.seek,[]);assert.deepEqual(h.calls.history,[]);}});
  await test('Ctrl/Cmd undo and redo mappings remain intact',async()=>{const h=harness();for(const opts of [{ctrlKey:true},{metaKey:true},{ctrlKey:true,shiftKey:true},{metaKey:true,shiftKey:true}])await h.key('document','z',opts);await h.key('document','y',{ctrlKey:true});assert.deepEqual(h.calls.history,['undo','undo','redo','redo','redo']);});
  await test('composition, consumed key, dialog, modifiers and busy states do not undo',async()=>{for(const flags of [{isComposing:true},{defaultPrevented:true},{altKey:true},{dialog:true},{busy:true},{loading:true}]){const h=harness();h.controls.dialog=!!flags.dialog;h.state.busy=!!flags.busy;h.state.loading=!!flags.loading;await h.key('document','z',{ctrlKey:true,...flags});assert.deepEqual(h.calls.history,[]);}});
  await test('empty history does not swallow undo unless pending work exists',async()=>{const h=harness();h.state.undoStack=[];assert.equal((await h.key('document','z',{ctrlKey:true})).defaultPrevented,false);h.controls.pending=true;await h.key('document','z',{ctrlKey:true});assert.deepEqual(h.calls.history,['undo']);});
  console.log(JSON.stringify({checks,passed:checks.filter(c=>c.passed).length,total:checks.length,scope:'Production keydown handlers with controlled DOM events. Native OS IME not exercised.'},null,2));if(checks.some(c=>!c.passed))process.exitCode=1;
})();
