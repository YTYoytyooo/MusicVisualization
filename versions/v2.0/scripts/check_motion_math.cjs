"use strict";
const assert=require("node:assert/strict");
const Motion=require("../web/motion-math.js");
const duration=10000000;
const baseline={times_us:[0,5000000,duration],auto:[0,1,2].map(()=>({
  mode:"rise",source:"auto",reason:"test",components:[{mode:"rise",weight:1,params:Motion.params("rise")}]}))};
const edit={id:"one",mode:"orbit",start_us:4000000,end_us:6000000,
  transition_in_us:2000000,transition_out_us:2000000,params:{speed:.2}};
const config={engine:"flow-v1",overrides:[edit]};
const original=JSON.stringify({baseline,config});
const result=Motion.applyOverrides(baseline,config,duration);
assert.equal(JSON.stringify({baseline,config}),original,"Pure preview must not mutate inputs");
assert.equal(Motion.validateConfig(null).engine,"legacy");
assert.equal(Motion.validateConfig(config,duration).overrides[0].params.rotation,1);
assert.equal(Motion.params("meteor",{direction_deg:360}).direction_deg,0);
assert.equal(Motion.params("spiral").radial,.25);
assert.equal(Motion.applyOverrides(null,{engine:"legacy"},duration).times_us.length,0);
assert.throws(()=>Motion.applyOverrides(null,config,duration),/规划/);
assert.throws(()=>Motion.validateConfig({...config,sensitivity:Infinity},duration));
assert.throws(()=>Motion.validateConfig({...config,unknown:0},duration));
assert.throws(()=>Motion.params("rise",{speed:true}));
assert.throws(()=>Motion.params("rise",{rotation:0}));
assert.throws(()=>Motion.validateConfig({...config,overrides:[edit,{...edit,id:"two"}]},duration),/重叠/);
assert.doesNotThrow(()=>Motion.validateConfig({...config,overrides:[edit,{...edit,id:"two",enabled:false}]},duration));
assert.throws(()=>Motion.validateConfig({...config,overrides:[{...edit,start_us:1}]},duration),/范围/);
const sample=time=>Motion.sampleFrame(result,time);
assert.equal(sample(0).mode,"rise");
assert.equal(sample(4000000).mode,"orbit");
assert.equal(sample(6000000).mode,"orbit");
assert.equal(sample(9000000).mode,"rise");
const halfway=sample(3000000);
assert.equal(halfway.source,"transition");
assert.equal(halfway.components.length,2);
assert.ok(halfway.components.every(component=>Math.abs(component.weight-.5)<1e-12));
assert.equal(sample(4000000).components[0].params.speed,.2);
for(const frame of result.effective){
  assert.ok(frame.components.length<=3);
  assert.ok(Math.abs(frame.components.reduce((sum,c)=>sum+c.weight,0)-1)<1e-12);
}
for(const interpolation of ["linear","smoothstep","smootherstep"]){
  const plan=Motion.applyOverrides(baseline,{...config,overrides:[{...edit,interpolation}]},duration);
  const weight=Motion.sampleFrame(plan,2500000).components.find(c=>c.mode==="orbit").weight;
  assert.ok(Math.abs(weight-({linear:.25,smoothstep:.15625,smootherstep:.103515625}[interpolation]))<1e-12);
}
const hard=Motion.applyOverrides(baseline,{engine:"flow-v1",overrides:[{...edit,transition_in_us:0,transition_out_us:0}]},duration);
assert.equal(Motion.sampleFrame(hard,3999999).mode,"rise");
assert.equal(Motion.sampleFrame(hard,4000000).mode,"orbit");
assert.equal(Motion.sampleFrame(hard,6000000).mode,"orbit");
assert.equal(Motion.sampleFrame(hard,6000001).mode,"rise");
const anglePlan={times_us:[0,10],effective:[350,10].map((angle,index)=>({
  mode:"wave",source:"auto",reason:String(index),components:[{mode:"wave",weight:1,
    params:Motion.params("wave",{direction_deg:angle,rotation:index?-1:1})}]}))};
const angle=Motion.sampleFrame(anglePlan,5);
assert.ok(Math.min(angle.components[0].params.direction_deg,360-angle.components[0].params.direction_deg)<1e-10);
assert.equal(angle.components[0].params.rotation,1);
assert.equal(angle.reason,"0");
console.log("Motion math invariants passed: presets, validation, weights, boundaries, modes, angles, purity.");
