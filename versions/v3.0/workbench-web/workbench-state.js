/* DOM-independent state boundaries used by the actual workbench and Node checks. */
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.StudioWorkbenchState=api;
})(typeof globalThis==='object'?globalThis:this,function(){
  'use strict';
  function writeDraft(storage,key,payload){
    try{
      if(payload===null)storage.removeItem(key);
      else storage.setItem(key,JSON.stringify(payload));
      return {ok:true};
    }catch(error){return {ok:false,error:error?.message||String(error)};}
  }
  function preserveSelection(values,previous){
    return values.includes(previous)?previous:(values[0]||'');
  }
  function shouldAutoSelectProject(startSerial,currentSerial,hasProject,loading){
    return startSerial===currentSerial&&!hasProject&&!loading;
  }
  function createProjectRequestQueue(){
    const tails=new Map();
    return {
      run(path,method,send){
        // These routes can read the same exclusively locked motion cache.
        // Keep unrelated projects, asset reads and job polling independent.
        const route=path.match(/^\/api\/projects\/([^/?]+)(?:\/(preview|motion-plan|revisions))?(?:\?[^#]*)?$/);
        if(!route||(route[2]?method!=="POST":method!=="GET"))return send();
        const key=route[1],previous=tails.get(key)||Promise.resolve();
        const result=previous.then(send);
        const settled=result.then(()=>{},()=>{});
        tails.set(key,settled);
        settled.then(()=>{if(tails.get(key)===settled)tails.delete(key);});
        return result;
      }
    };
  }
  function createMediaSeeks(){
    const pending=new WeakMap();
    return {
      set(media,time){
        if(!Number.isFinite(media.duration)||!Number.isFinite(time))return;
        const target=Math.max(0,Math.min(media.duration,time));
        // A no-op does not create an acknowledgement that could hide a later user seek.
        // Leave an existing acknowledgement intact until its queued seeked event arrives.
        if(media.currentTime===target&&!media.seeking)return;
        pending.set(media,{src:media.src,time:target});
        if(media.currentTime!==target)media.currentTime=target;
      },
      consume(media){
        const expected=pending.get(media);pending.delete(media);
        // Native controls may supersede a queued programmatic seek. Only consume its
        // matching result, allowing millisecond media-clock rounding, never all seeks.
        return Boolean(expected&&expected.src===media.src&&Math.abs(expected.time-media.currentTime)<=.001);
      },
      clear(media){pending.delete(media);}
    };
  }
  function outputDownloads(systems,sourceSystem,render){
    const source=(systems||[]).find(item=>item.id===sourceSystem);
    return [["下载原始预测（"+sourceSystem+"）",source?.raw_csv_url],["下载当前视频",render?.video_url],
      ["下载逐帧数值",render?.csv_url],["下载生效情绪",render?.effective_csv_url],
      ["下载视觉计划",render?.visual_csv_url],["下载输出记录",render?.manifest_url||render?.render_json_url]];
  }
  async function prepareSwitch(choice,actions){
    if(choice==='cancel')return false;
    if(!['save','keep'].includes(choice))throw new Error('未知的草稿切换选项');
    await actions.finishPending();
    if(choice==='save')await actions.saveRevision();
    else if(actions.persistDraft()!==true)
      throw new Error('无法保存本地草稿，已阻止切换。请保存为正式修订，或修复浏览器存储后重试。');
    return true;
  }
  return {writeDraft,preserveSelection,shouldAutoSelectProject,createProjectRequestQueue,createMediaSeeks,outputDownloads,prepareSwitch};
});
