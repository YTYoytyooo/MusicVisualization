"use strict";
// The real system-control function against a minimal DOM surface, no mutations
// to projects/revisions and no model or rendering jobs.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,'../../workbench-web/app.js'),'utf8');
const start=source.indexOf('  function renderSystemControls('),end=source.indexOf('  function comparisonSources(',start);
assert.ok(start>=0&&end>start);
function element(tag='',text=''){
  return {textContent:text,value:'',children:[],style:{},append(...nodes){this.children.push(...nodes);},
    replaceChildren(){this.children=[];},get options(){return this.children;}};
}
const nodes={},state={data:{systems:[]},comparisons:new Set(),sourceSystem:'b01'};
const context={state,$:id=>(nodes[id]??=element()),node:element,document:{createElement:element},
  comparisonSources:()=>state.data.systems,drawTimeline:()=>{}};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const research={evidence:'verified_frozen_record',kind:'research',research:'not_promoted',calibration:'unavailable'};
const checks=[];
function test(name,fn){try{fn();checks.push({name,passed:true});}catch(e){checks.push({name,passed:false,error:e.message});}}
function show(id,status){state.sourceSystem=id;state.data.systems=[{id,label:id,prediction_status:status}];context.renderSystemControls();}
test('B01 requires all verified fields and describes uncalibrated research',()=>{
  show('b01',research);assert.match(nodes['prediction-status-title'].textContent,/研究候选/);
  assert.match(nodes['prediction-status-detail'].textContent,/未校准.*尚未通过/);
});
test('B00 is labeled a research control',()=>{
  show('b00',research);assert.match(nodes['prediction-status-title'].textContent,/研究对照/);
  assert.match(nodes['prediction-status-detail'].textContent,/仅供对照/);
});
test('legacy original values are not given B01 gate claims',()=>{
  show('v2',{evidence:'verified_frozen_record',kind:'legacy',calibration:'not_applied'});
  assert.match(nodes['prediction-status-title'].textContent,/历史模型/);
  assert.match(nodes['prediction-status-detail'].textContent,/未应用校准/);
  assert.doesNotMatch(nodes['prediction-status-detail'].textContent,/尚未通过全部/);
});
test('missing, mismatched and future statuses cannot claim verification',()=>{
  for(const s of [undefined,{}, {...research,evidence:'unverified'}, {...research,calibration:'available'}]){
    show('b01',s);assert.match(nodes['prediction-status-title'].textContent,/未确认/);
  }
  show('v2',{kind:'legacy',evidence:'unverified'});assert.match(nodes['prediction-status-detail'].textContent,/未记录或无法核对/);
});
test('switch redraw updates status; comparison selection never changes driver status',()=>{
  show('b01',research);const title=nodes['prediction-status-title'].textContent;
  const input=nodes['comparison-systems'].children[0].children[0];input.checked=true;input.onchange();
  assert.equal(state.sourceSystem,'b01');assert.equal(nodes['prediction-status-title'].textContent,title);
  show('v2',{kind:'legacy',evidence:'unverified'});assert.match(nodes['prediction-status-title'].textContent,/历史模型/);
});
console.log(JSON.stringify({checks,passed:checks.filter(c=>c.passed).length,total:checks.length},null,2));
if(checks.some(c=>!c.passed))process.exitCode=1;
