"use strict";
// Exercise the production API boundary with controlled asynchronous responses.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const helpers=require('../../workbench-web/workbench-state.js');
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,'../../workbench-web/app.js'),'utf8');
const start=source.indexOf('  async function api('),end=source.indexOf('  function run(',start);
assert.ok(start>=0&&end>start);
const checks=[],tick=()=>new Promise(resolve=>setImmediate(resolve));
const ok=(value={ok:true})=>({ok:true,status:200,json:async()=>value});
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function harness(fetch){
  const state={token:'local-token',sourceSystem:'b01',smoothing:{enabled:false,window_seconds:.5},motion:{engine:'flow-v1'}};
  const context={state,clone:x=>JSON.parse(JSON.stringify(x)),projectRequests:helpers.createProjectRequestQueue(),fetch};
  vm.createContext(context);vm.runInContext(source.slice(start,end),context);return context;
}
async function test(name,run){try{await run();checks.push({name,passed:true});}catch(error){checks.push({name,passed:false,error:error.message});}}
(async()=>{
  await test('one project detail/preview/motion/save runs in request order',async()=>{
    const gates=[deferred(),deferred(),deferred(),deferred()],calls=[];
    const h=harness((url)=>{calls.push(url);return gates[calls.length-1].promise;});
    const urls=['/api/projects/p?revision=r000005&source=b01','/api/projects/p/preview','/api/projects/p/motion-plan','/api/projects/p/revisions'];
    const promises=urls.map((url,i)=>h.api(url,i?{edits:[]}:undefined));
    await tick();assert.deepEqual(calls,[urls[0]]);
    for(let i=0;i<4;i++){gates[i].resolve(ok({index:i}));assert.equal((await promises[i]).index,i);await tick();assert.deepEqual(calls,urls.slice(0,Math.min(i+2,4)));}
  });
  await test('different projects can proceed independently',async()=>{
    const gate=deferred(),calls=[];const h=harness(url=>{calls.push(url);return url.includes('/a/')?gate.promise:Promise.resolve(ok());});
    const a=h.api('/api/projects/a/preview',{});await h.api('/api/projects/ab/motion-plan',{});
    assert.equal(calls.length,2);gate.resolve(ok());await a;
  });
  await test('job polling, assets and unrelated routes do not wait for a project preview',async()=>{
    const gate=deferred(),calls=[];const h=harness(url=>{calls.push(url);return url.endsWith('/preview')?gate.promise:Promise.resolve(ok());});
    const pending=h.api('/api/projects/p/preview',{});
    await Promise.all(['/api/jobs','/api/projects','/api/projects/p/audio','/api/projects/p/visual-plan'].map(url=>h.api(url)));
    assert.equal(calls.length,5);gate.resolve(ok());await pending;
  });
  await test('waiting requests preserve caller data, driver, smoothing, motion and token',async()=>{
    const gate=deferred(),calls=[];const h=harness((url,options)=>{calls.push({url,options});return calls.length===1?gate.promise:Promise.resolve(ok());});
    const first=h.api('/api/projects/p/motion-plan',{});await tick();
    const body={base_revision:'r000005',edits:[{id:'one',value:.4}]};const second=h.api('/api/projects/p/preview',body);
    body.edits[0].value=.9;h.state.sourceSystem='v2';h.state.smoothing.enabled=true;h.state.motion.engine='legacy';h.state.token='changed';
    gate.resolve(ok());await Promise.all([first,second]);
    const saved=JSON.parse(calls[1].options.body);assert.equal(saved.edits[0].value,.4);assert.equal(saved.source_system,'b01');
    assert.equal(saved.smoothing.enabled,false);assert.equal(saved.motion.engine,'flow-v1');assert.equal(calls[1].options.headers['X-Studio-Token'],'local-token');
    assert.equal(calls[1].options.method,'POST');assert.equal(body.source_system,undefined);
  });
  for(const kind of ['http','json','network'])await test(kind+' failure reaches its caller once and releases the next request',async()=>{
    let calls=0;const failure=deferred();const h=harness(()=>{calls++;return calls===1?failure.promise:Promise.resolve(ok({next:true}));});
    const one=h.api('/api/projects/p/preview',{});const rejected=assert.rejects(one,kind==='http'?/项目正在写入/:kind==='json'?/无法识别/:/disconnected/);
    const two=h.api('/api/projects/p/motion-plan',{});await tick();
    if(kind==='network')failure.reject(Error('disconnected'));
    else if(kind==='json')failure.resolve({ok:true,status:200,json:async()=>{throw Error('invalid');}});
    else failure.resolve({ok:false,status:400,json:async()=>({error:'项目正在写入'})});
    await rejected;assert.equal((await two).next,true);assert.equal(calls,2,'no hidden retry');
  });
  await test('exclusive cache simulation accepts both overlapping UI requests',async()=>{
    let active=false,overlaps=0;const gate=deferred();const h=harness(async()=>{
      if(active){overlaps++;return {ok:false,status:400,json:async()=>({error:'项目正在写入'})};}
      active=true;await gate.promise;active=false;return ok();
    });
    const promises=Promise.allSettled([h.api('/api/projects/p/motion-plan',{}),h.api('/api/projects/p/preview',{})]);
    await tick();gate.resolve();const results=await promises;
    assert.equal(overlaps,0);assert.equal(results.filter(x=>x.status==='fulfilled').length,2);
  });
  await test('response parsing completes before the next project request starts',async()=>{
    const gate=deferred(),calls=[];const h=harness(url=>{calls.push(url);return Promise.resolve(calls.length===1?{ok:true,status:200,json:()=>gate.promise}:ok());});
    const one=h.api('/api/projects/p/preview',{}),two=h.api('/api/projects/p/motion-plan',{});
    await tick();assert.equal(calls.length,1);gate.resolve({ok:true});await Promise.all([one,two]);assert.equal(calls.length,2);
  });
  await test('a settled project queue can be reused without dropping later requests',async()=>{
    let calls=0;const h=harness(async()=>ok({call:++calls}));
    assert.equal((await h.api('/api/projects/p/preview',{})).call,1);await tick();
    const values=await Promise.all([h.api('/api/projects/p/preview',{}),h.api('/api/projects/p/motion-plan',{})]);
    assert.deepEqual(values.map(x=>x.call),[2,3]);
  });
  console.log(JSON.stringify({checks,passed:checks.filter(x=>x.passed).length,total:checks.length},null,2));
  if(checks.some(x=>!x.passed))process.exitCode=1;
})();
