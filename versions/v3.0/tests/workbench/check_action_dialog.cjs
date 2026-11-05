"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const source=fs.readFileSync(process.argv[2]||path.resolve(__dirname,"../../workbench-web/app.js"),"utf8");
function fn(name){const start=new RegExp("^  (?:async )?function "+name+"\\(","m").exec(source);assert.ok(start,name);
  const rest=source.slice(start.index+start[0].length),end=/^  (?:async )?function /m.exec(rest);
  return source.slice(start.index,start.index+start[0].length+end.index);}
function setup(){
  const elements={},calls=[];
  const state={data:{revision:{edits:[]}},loading:false,busy:false,loadSerial:1,sourceSystem:"b01",savedSmoothing:{enabled:false},savedMotion:{engine:"legacy"}};
  const get=id=>elements[id]??=(id==="action-dialog"?{
    open:false,returnValue:"",addEventListener(type,cb){this.closed=cb;},showModal(){this.open=true;},
    close(value){if(value!==undefined)this.returnValue=value;this.open=false;this.closed();}
  }:{});
  const c={state,$:get,projectId:()=>"p",revisionId:()=>"r",Promise,clone:structuredClone,
    run:fn=>fn,dirty:()=>true,pending:()=>true,finishPending:()=>{throw Error("invalid input must be discardable");},
    cancelAdjustment:all=>calls.push(["cancel",all]),applyDraft:async(...args)=>calls.push(["apply",...args]),
    cancelEditing:()=>calls.push(["editor"]),notice:msg=>calls.push(["notice",msg])};
  vm.createContext(c);vm.runInContext(fn("dialogChoice")+fn("confirmAction"),c);
  const handler=source.slice(source.indexOf('  $("reset-draft").onclick'),source.indexOf('  $("save-revision").onclick'));
  vm.runInContext(handler,c);return {c,state,get,calls};
}
(async()=>{
  assert.doesNotMatch(source,/\bconfirm\(/,"app actions must not use blocking native confirmation");
  for(const choice of ["cancel",undefined]){const h=setup(),p=h.get("reset-draft").onclick();
    assert.equal(h.get("action-dialog").open,true);assert.deepEqual(h.calls,[]);
    h.get("action-dialog").close(choice);await p;assert.deepEqual(h.calls,[]);}
  {const h=setup(),p=h.get("reset-draft").onclick();h.get("action-dialog").close("accept");await p;
    assert.deepEqual(h.calls.map(x=>x[0]),["cancel","apply","editor","notice"]);assert.equal(h.calls[0][1],true);}
  {const h=setup(),p=h.get("reset-draft").onclick();h.state.loadSerial++;h.get("action-dialog").close("accept");await p;assert.deepEqual(h.calls,[]);}
  {const h=setup(),p=h.c.confirmAction("first","message","ok");assert.equal(await h.c.confirmAction("second","overwrite","ok"),false);
    assert.equal(h.get("action-title").textContent,"first");h.get("action-dialog").close("cancel");assert.equal(await p,false);}
  console.log("PASS: nonblocking confirmation; cancel/Escape, invalid-input reset, stale context, duplicate action");
})().catch(e=>{console.error(e);process.exitCode=1;});
