// Bounded pure-math benchmark; no model, network or project mutations.
const M = require('../web/timeline-math.js');
const {performance} = require('node:perf_hooks');
const results = [];
for (const seconds of [180, 600]) {
  const raw = Array.from({length: seconds * 10}, (_, i) => [.2*Math.sin(i/50), .3+.1*Math.cos(i/70), 0, 0, 0]);
  const edit = {id:'benchmark',layer:'emotion',field:'arousal',operation:'point_target',
    time_us: seconds*500000+12345,value:.8,transition_in_us:2000000,transition_out_us:2000000,
    interpolation:'smootherstep',enabled:true};
  M.emotionPreview(raw,100000,seconds*1000000,[edit]);
  const samples=[];
  let points=0;
  for(let n=0;n<10;n++) {
    const start=performance.now();
    points=M.emotionPreview(raw,100000,seconds*1000000,[edit]).times.length;
    samples.push(performance.now()-start);
  }
  samples.sort((a,b)=>a-b);
  results.push({seconds,points,median_ms:samples[5],max_ms:samples[9]});
}
console.log(JSON.stringify({scope:'pure preview calculation only; excludes browser canvas/network',results},null,2));
