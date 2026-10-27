"use strict";
// Frozen production comparison. Synthetic long axes exercise drawing only, not training.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{performance}=require('node:perf_hooks');
const T=require('../../workbench-web/timeline-math.js');
const root=path.resolve(__dirname,'../../../..'),round=path.join(root,'data/v3.0/optimization/20261024/rounds/014');
const source=fs.readFileSync(process.argv[2]||path.join(root,'versions/v3.0/workbench-web/app.js'),'utf8'),reference=fs.readFileSync(path.join(round,'app.before.js'),'utf8');
const fixture=JSON.parse(fs.readFileSync(path.join(round,'acoustic-fixture.json'),'utf8')).acoustic;
const checks=[],copy=x=>structuredClone(x);
function fn(src,name,optional=false){const start=new RegExp('^  function '+name+'\\(','m').exec(src);if(!start){assert.ok(optional,name);return '';}
  const rest=src.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);return src.slice(start.index,start.index+start[0].length+(end?end.index:rest.length));}
function harness(src,input,count=false){
  const ops=[],counts={reads:0};let record=true;const details={open:true};
  const ctx=new Proxy({}, {get:(o,k)=>o[k]??((...args)=>{if(record)ops.push([k,...args]);}),set:(o,k,v)=>{o[k]=v;if(record)ops.push(['set',k,v]);return true;}});
  const canvas={clientWidth:640,getContext:()=>ctx,closest:()=>details},elements={'acoustic-timeline':canvas,'acoustic-readout':{textContent:''}};
  const state={data:{project:{duration:28.029},timeline:{}},time:5,viewport:{start:0,end:28.029},drawCache:new WeakMap(),seriesCache:new WeakMap()};
  const c={state,$:id=>elements[id],window:{devicePixelRatio:1},StudioTimeline:T,visibleEdits:()=>[]};vm.createContext(c);
  vm.runInContext(['sampleAt','viewRange','drawingIndices','fitPlot','drawAcoustic'].map(n=>fn(src,n)).join('\n')+['acousticTimes','acousticExtent'].map(n=>fn(src,n,true)).join('\n'),c);
  const replace=input=>{const a=copy(input);if(count&&a)for(const key of ['times_us','rms_dbfs','spectral_centroid_hz'])if(a[key])a[key]=new Proxy(a[key],{get:(o,k)=>{if(/^\d+$/.test(String(k)))counts.reads++;return o[k];}});state.data.timeline.acoustic=a;};
  replace(input);
  return {state,canvas,details,counts,replace,setRecord:v=>record=v,draw:()=>{ops.length=0;c.drawAcoustic();return {ops:ops.slice(),readout:elements['acoustic-readout'].textContent};}};
}
function test(name,run){try{run();checks.push({name,passed:true});}catch(e){checks.push({name,passed:false,error:e.message});}}
function compare(input){const before=harness(reference,input),after=harness(source,input);
  for(const [time,start,end,width,open] of [[0,0,28.029,640,true],[5,0,28.029,640,true],[12.3,10,15,375,true],[28.029,0,28.029,900,false],[20,15,25,900,true]]){
    for(const h of [before,after]){h.state.time=time;h.state.viewport={start,end};h.canvas.clientWidth=width;h.details.open=open;}
    assert.deepEqual(after.draw(),before.draw());
  }
}
test('authorized acoustic measurements: exact canvas and readouts across seek, resize, zoom and collapse',()=>compare(fixture));
const micros=copy(fixture);micros.times_us=micros.times_seconds.map(t=>Math.round(t*1e6));delete micros.times_seconds;
test('microsecond fallback preserves exact conversion and sample selection',()=>compare(micros));
test('seconds clock takes precedence even when a different microsecond clock exists',()=>compare({...fixture,times_us:[0,1]}));
for(const [name,data] of Object.entries({absent:null,empty:{times_seconds:[]},emptySeconds:{times_seconds:[],times_us:[0,1e6]},mismatched:{...fixture,rms_dbfs:[-10]},constant:{...fixture,rms_dbfs:fixture.times_seconds.map(()=>-20)},missing:{...fixture,rms_dbfs:fixture.times_seconds.map((_,i)=>i%3?null:-20),spectral_centroid_hz:fixture.times_seconds.map((_,i)=>i%5?1200:null)},allMissing:{...fixture,rms_dbfs:fixture.times_seconds.map(()=>NaN),spectral_centroid_hz:fixture.times_seconds.map(()=>null)}}))test(name+': preserves drawing and no fabricated values',()=>compare(data));
test('non-finite invalid input still rejects instead of weakening plot validation',()=>{for(const src of [reference,source]){const a=copy(fixture);a.rms_dbfs[5]=NaN;assert.throws(()=>harness(src,a).draw(),/plot value/);}});
test('new measurement arrays invalidate cached ranges and microsecond conversion',()=>{
  const before=harness(reference,micros),after=harness(source,micros);assert.deepEqual(after.draw(),before.draw());
  const next=copy(micros);next.rms_dbfs=next.rms_dbfs.map(x=>x-30);next.spectral_centroid_hz=next.spectral_centroid_hz.map(x=>x*2);next.times_us=next.times_us.map(x=>x+125000);
  for(const h of [before,after])h.replace(next);assert.deepEqual(after.draw(),before.draw());
});
test('collapsed curve does not draw and still updates acoustic readout',()=>{const h=harness(source,fixture);h.details.open=false;const a=h.draw();h.state.time=10;const b=h.draw();assert.equal(a.ops.length,0);assert.equal(b.ops.length,0);assert.notEqual(a.readout,b.readout);});
const synthetic={times_seconds:Array.from({length:36001},(_,i)=>i/10),rms_dbfs:Array.from({length:36001},(_,i)=>i%101===0?null:-25+15*Math.sin(i/50)),spectral_centroid_hz:Array.from({length:36001},(_,i)=>i%137===0?null:1500+500*Math.cos(i/91))};
const syntheticMicros={...synthetic,times_us:synthetic.times_seconds.map(t=>Math.round(t*1e6))};delete syntheticMicros.times_seconds;
const counts={};
for(const [name,data] of [['seconds',synthetic],['microseconds',syntheticMicros]]){
  const before=harness(reference,data,true),after=harness(source,data,true);
  for(const h of [before,after]){h.state.data.project.duration=3600;h.state.viewport={start:1790,end:1810};h.draw();h.counts.reads=0;h.state.time=1800;}
  const b=before.draw(),a=after.draw();counts[name]={before:before.counts.reads,after:after.counts.reads};
  test(name+': one-hour synthetic axis keeps exact gaps, peaks and readout',()=>assert.deepEqual(a,b));
  test(name+': warm draw avoids whole-array rereads',()=>assert.ok(after.counts.reads<before.counts.reads/4,JSON.stringify(counts[name])));
}
function benchmark(src,data){const h=harness(src,data);h.state.data.project.duration=3600;h.state.viewport={start:1790,end:1810};h.setRecord(false);h.draw();for(let i=0;i<10;i++)h.draw();const samples=[];
  for(let r=0;r<5;r++){const t=performance.now();for(let i=0;i<60;i++){h.state.time=1800+i/10;h.draw();}samples.push((performance.now()-t)/60);}return {ms_per_draw:samples,median_ms:samples.slice().sort((a,b)=>a-b)[2],runs:5,draws_per_run:60};}
test('drawing leaves its actual measurement arrays unchanged',()=>{const h=harness(source,fixture),data=h.state.data.timeline.acoustic,original=copy(data);
  for(const value of Object.values(data))if(Array.isArray(value))Object.freeze(value);Object.freeze(data);h.draw();h.state.time=12;h.draw();assert.deepEqual(data,original);});
const perf={scope:'Node production drawAcoustic with no-op canvas, warm cache; not whole-page browser timing',reads:counts,synthetic_1h_seconds:{before:benchmark(reference,synthetic),after:benchmark(source,synthetic)},synthetic_1h_microseconds:{before:benchmark(reference,syntheticMicros),after:benchmark(source,syntheticMicros)}};
console.log(JSON.stringify({checks,passed:checks.filter(c=>c.passed).length,total:checks.length,performance:perf},null,2));if(checks.some(c=>!c.passed))process.exitCode=1;
