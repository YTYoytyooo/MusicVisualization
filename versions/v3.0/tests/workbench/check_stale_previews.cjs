"use strict";
// Actual API and preview producer functions; transport/timers are controlled.
// No HTTP service, source data, inference, revision write or rendering is used.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const helpers=require('../../workbench-web/workbench-state.js');
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,'../../workbench-web/app.js'),'utf8');
const plain=x=>JSON.parse(JSON.stringify(x)),tick=()=>new Promise(resolve=>setImmediate(resolve));
const ok=value=>({ok:true,status:200,json:async()=>value});
const checks=[];
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function fn(name){
  const start=new RegExp('^  (?:async )?function '+name+'\\(','m').exec(source);assert.ok(start,name);
  const rest=source.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);
  return source.slice(start.index,start.index+start[0].length+(end?end.index:rest.length));
}
function harness(fetch){
  const state={token:'local-token',sourceSystem:'b01',smoothing:{enabled:false,window_seconds:.5},
    motion:{engine:'flow-v1'},data:{project:{id:'p'},revision:{id:'r'}},loadSerial:1,
    loading:false,requestSeq:0,motionSeq:0,motionTimer:null,edits:[],tx:null,motionPlan:null};
  const timers=new Map(),adopted=[],draws=[];let timerId=0;
  const c={state,clone:plain,urlId:encodeURIComponent,projectRequests:helpers.createProjectRequestQueue(),fetch,
    projectId:()=>state.data.project.id,revisionId:()=>state.data.revision.id,
    visibleEdits:()=>state.edits,shownMotion:()=>state.motion,
    motionAutoSignature:(edits,motion)=>JSON.stringify([state.sourceSystem,state.smoothing,edits,motion]),
    candidatePlanFor:async()=>null,updateDraft:()=>{},drawTimeline:()=>draws.push('timeline'),
    drawMotion:()=>draws.push('motion'),adoptVisualPlan:()=>{},
    adoptMotionPlan:plan=>adopted.push(plan),localEmotion:edits=>({edits:plain(edits)}),
    setTimeout:callback=>{const id=++timerId;timers.set(id,callback);return id;},clearTimeout:id=>timers.delete(id)};
  vm.createContext(c);vm.runInContext(['transactionContext','api','validateTemporary','scheduleMotionPlan'].map(fn).join('\n'),c);
  return {c,state,adopted,draws,fire:()=>{const [id,callback]=timers.entries().next().value;timers.delete(id);return callback();}};
}
async function test(name,run){try{await run();checks.push({name,passed:true});}catch(error){checks.push({name,passed:false,error:error.message});}}
(async()=>{
  await test('queued stale temporary preview never reaches transport; latest still does',async()=>{
    const gate=deferred(),calls=[];const h=harness((url,options)=>{calls.push({url,body:JSON.parse(options.body||'null')});return calls.length===1?gate.promise:Promise.resolve(ok({latest:true}));});
    const block=h.c.api('/api/projects/p/motion-plan',{});await tick();let version=1;
    const stale=h.c.api('/api/projects/p/preview',{value:1},()=>version===1);version=2;
    const latest=h.c.api('/api/projects/p/preview',{value:2},()=>version===2);
    gate.resolve(ok({}));await block;assert.equal(await stale,null);assert.equal((await latest).latest,true);
    assert.deepEqual(calls.map(x=>x.body.value),[undefined,2]);
  });
  await test('in-flight temporary request completes without cancellation',async()=>{
    const gate=deferred();let current=true,calls=0;
    const h=harness(()=>{calls++;return gate.promise;});
    const active=h.c.api('/api/projects/p/preview',{},()=>current);await tick();assert.equal(calls,1);
    current=false;gate.resolve(ok({completed:true}));assert.equal((await active).completed,true);
  });
  await test('confirmation preview, revision save, detail and other routes are never implicitly dropped',async()=>{
    const calls=[];const h=harness(async(url)=>{calls.push(url);return ok({});});
    await h.c.api('/api/projects/p/preview',{});
    await h.c.api('/api/projects/p/revisions',{},()=>false);
    await h.c.api('/api/projects/p',undefined,()=>false);
    await h.c.api('/api/projects/p/previews',{},()=>false);
    assert.equal(calls.length,4);
  });
  await test('skip releases same-project queue and does not delay another project',async()=>{
    const gate=deferred(),calls=[];const h=harness(url=>{calls.push(url);return url==='/api/projects/p'?gate.promise:Promise.resolve(ok({}));});
    const block=h.c.api('/api/projects/p');await tick();
    const stale=h.c.api('/api/projects/p/motion-plan',{},()=>false);
    const save=h.c.api('/api/projects/p/revisions',{});
    await h.c.api('/api/projects/q/preview',{});assert.equal(calls.length,2);
    gate.resolve(ok({}));await Promise.all([block,stale,save]);
    assert.deepEqual(calls,['/api/projects/p','/api/projects/q/preview','/api/projects/p/revisions']);
  });
  await test('actual temporary-validation producer skips old input snapshots',async()=>{
    const gate=deferred(),values=[];let count=0;
    const h=harness((url,options)=>{count++;if(count===1)return gate.promise;values.push(JSON.parse(options.body).edits[0].value);return Promise.resolve(ok({}));});
    const block=h.c.api('/api/projects/p/motion-plan',{});await tick();
    const tx={project:'p',revision:'r',loadSerial:1,valid:true,changed:true,smoothing:h.state.smoothing};h.state.tx=tx;
    const pending=[];
    for(const value of [.1,.2,.3]){tx.edits=[{id:'edit',value}];pending.push(h.c.validateTemporary(tx));await tick();}
    gate.resolve(ok({}));await Promise.all([block,...pending]);
    assert.deepEqual(values,[.3]);assert.equal(tx.remoteStatus,'valid');assert.equal(tx.lastAcceptedEdits[0].value,.3);
  });
  await test('actual motion producer skips obsolete queued plans and adopts latest only',async()=>{
    const gate=deferred(),values=[];let count=0;
    const h=harness((url,options)=>{count++;if(count===1)return gate.promise;const value=JSON.parse(options.body).edits[0].value;values.push(value);return Promise.resolve(ok({value}));});
    const block=h.c.api('/api/projects/p/preview',{});await tick();const pending=[];
    for(const value of [.1,.2,.3]){h.state.edits=[{id:'edit',value}];h.c.scheduleMotionPlan();pending.push(h.fire());await tick();}
    gate.resolve(ok({}));await Promise.all([block,...pending]);
    assert.deepEqual(values,[.3]);assert.deepEqual(h.adopted,[{value:.3}]);
  });
  await test('project change invalidates waiting temporary motion work without showing stale error',async()=>{
    const gate=deferred();let calls=0;const h=harness(()=>{calls++;return gate.promise;});
    const block=h.c.api('/api/projects/p/preview',{});await tick();h.state.edits=[{id:'edit',value:.3}];
    h.c.scheduleMotionPlan();const motion=h.fire();await tick();h.state.loadSerial++;h.state.loading=true;
    gate.resolve(ok({}));await Promise.all([block,motion]);assert.equal(calls,1);assert.equal(h.adopted.length,0);assert.equal(h.state.motionError,'');
  });
  await test('latest temporary failure reaches its owner and preserves queue recovery',async()=>{
    let calls=0;const h=harness(async()=>++calls===1?{ok:false,status:400,json:async()=>({error:'validation rejected'})}:ok({saved:true}));
    await assert.rejects(h.c.api('/api/projects/p/preview',{},()=>true),/validation rejected/);
    assert.equal((await h.c.api('/api/projects/p/revisions',{})).saved,true);assert.equal(calls,2);
  });
  // Fixed artificial service time exposes avoidable backlog, not real server speed.
  const timings=[];
  for(let repeat=0;repeat<3;repeat++){
    let calls=0,version=0;const values=[];
    const h=harness(async(url,options)=>{calls++;const body=JSON.parse(options.body||'{}');values.push(body.value??null);await new Promise(r=>setTimeout(r,30));return ok({value:body.value});});
    const start=performance.now(),pending=[h.c.api('/api/projects/p/motion-plan',{})];await tick();
    for(let v=1;v<=6;v++){version=v;pending.push(h.c.api('/api/projects/p/preview',{value:v},()=>version===v));}
    const confirmation=h.c.api('/api/projects/p/preview',{value:6});pending.push(confirmation);
    await Promise.all(pending);assert.equal((await confirmation).value,6);
    timings.push({elapsed_ms:performance.now()-start,requests:calls,sent_values:values});
  }
  console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length,
    controlled_delay:{delay_per_request_ms:30,samples:timings,note:'Synthetic transport delay only; not whole-page latency.'}},null,2));
  if(checks.some(x=>!x.passed))process.exitCode=1;
})();
