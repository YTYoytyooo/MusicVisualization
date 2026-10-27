// Finite ten-minute synthetic UI fixture; no training data or model evaluation.
'use strict';
const {performance}=require('node:perf_hooks');
const fs=require('node:fs');
const T=require('../../workbench-web/timeline-math.js');
const cases=[];
for(const step of [500000,100000]){
  const duration=600000000;
  const times=Array.from({length:duration/step},(_,i)=>i*step);
  const values=times.map((t,i)=>[.4*Math.sin(t/2e7),i%251===0?null:.3*Math.cos(t/3e7)]);
  const edits=Array.from({length:20},(_,i)=>({id:'p'+i,layer:'emotion',field:'valence',
    operation:'point_target',time_us:(10+i*28)*1e6,value:.5,transition_in_us:1e6,transition_out_us:1e6}));
  const elapsed=[];
  for(let n=0;n<110;n++){
    const started=performance.now();
    const p=T.emotionPreview(values,times,duration,edits,{enabled:true,window_seconds:.5},step,true);
    T.plotIndices(p.times,[p.effective.map(r=>r[0]),p.effective.map(r=>r[1])],1200,edits.map(e=>e.time_us/1e6));
    const ms=performance.now()-started;
    if(n>=10)elapsed.push(ms);
  }
  elapsed.sort((a,b)=>a-b);
  cases.push({step_us:step,samples:times.length,edits:edits.length,runs:elapsed.length,
    median_ms:elapsed[49],p95_ms:elapsed[94],max_ms:elapsed[99]});
}
const report={recorded_at:new Date().toISOString(),node:process.version,cases,
  scope:'Pure edit validation, smoothing and extrema-preserving draw selection; excludes DOM, canvas paint and HTTP latency.'};
fs.writeFileSync('data/v3.0/reports/workbench/timeline-performance.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
