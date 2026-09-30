const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Run against an already running local Studio server. Only fresh synthetic demos are edited.
const {chromium} = require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const {execFileSync} = require('node:child_process');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const BASE = process.env.STUDIO_URL || 'http://127.0.0.1:8765';
const makeDemo = () => JSON.parse(execFileSync(studioPython,
  ['main.py', 'demo', '--duration', '4',...(process.env.STUDIO_PROJECTS?['--projects',process.env.STUDIO_PROJECTS]:[])], {encoding:'utf8',cwd:codeRoot})).split(/[\\/]/).pop();
const gate = () => {
  let release;
  const promise = new Promise(resolve => { release=resolve; });
  return {promise,release};
};

(async () => {
  const id=makeDemo(), other=makeDemo();
  const browser=await chromium.launch({headless:true,
    executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    page.setDefaultTimeout(20000);
    const errors=[], checks=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('dialog',dialog=>dialog.accept());
    const bootstrap=await (await page.request.get(BASE+'/api/bootstrap')).json();
    const initial=await (await page.request.get(BASE+'/api/projects/'+id)).json();
    const seedEdit={id:'live-arousal',layer:'emotion',field:'arousal',operation:'point_target',
      time_us:2000000,value:.25,transition_in_us:800000,transition_out_us:800000,
      enabled:true,note:'Isolated browser test',interpolation:'linear'};
    const seedResponse=await page.request.post(BASE+'/api/projects/'+id+'/revisions',{
      headers:{'X-Studio-Token':bootstrap.token},data:{base_revision:initial.revision.id,edits:[seedEdit]}});
    assert.ok(seedResponse.ok(),'Could not create isolated seed revision');
    const endpoint=BASE+'/api/projects/'+id+'/preview';
    const load = async project => {
      await page.waitForFunction(() => !document.getElementById('editor').inert);
      await page.locator('#project-list [data-project-id="'+project+'"]').click();
      await page.waitForFunction(project =>
        document.getElementById('project-meta').textContent.includes(project) &&
        !document.getElementById('editor').inert &&
        document.getElementById('editor').getAttribute('aria-busy')==='false',project);
    };
    const settled=(timeout=20000)=>page.waitForFunction(() =>
      document.getElementById('adjustment-state').dataset.state==='ready' &&
      !document.getElementById('draft-state').textContent.includes('临时') &&
      document.getElementById('editor').getAttribute('aria-busy')==='false',null,{timeout});
    const blur=()=>page.evaluate(()=>document.activeElement?.blur());
    const history=async()=> {
      const text=await page.locator('#history-state').textContent();
      return [...text.matchAll(/(\d+) 步/g)].map(m=>Number(m[1])).slice(0,2);
    };
    const draft=async project => {
      const cached=await page.evaluate(project => {
        const key=Object.keys(localStorage).find(k=>k.startsWith('musicvisualization-studio:draft:'+project+':'));
        return key ? JSON.parse(localStorage.getItem(key)).edits : null;
      },project);
      return cached || (await (await page.request.get(BASE+'/api/projects/'+project)).json()).revision.edits;
    };
    const step=async selector => { await page.locator(selector).click(); await settled(); };
    const editFirst=async()=> {
      await page.locator('#edits-list button').filter({hasText:'编辑'}).first().click();
    };
    const setTime=async value=>{
      await page.locator('#playhead').fill(String(value)); await page.locator('#playhead').press('Tab');
    };

    await page.goto(BASE+'/',{waitUntil:'networkidle'});
    await load(id);
    await setTime(2);
    await editFirst();
    assert.equal(await page.locator('#edit-interpolation').inputValue(),'linear','Old edits must remain linear');

    // Hold the server: the full local curve must update before a preview response.
    const hold=gate(), entered=gate();
    await page.route(endpoint,async route=>{entered.release();await hold.promise;await route.continue();});
    await page.locator('#edit-value').fill('0.55');
    await page.locator('#edit-value').fill('0.61');
    await page.locator('#edit-value').fill('0.67');
    await page.waitForFunction(()=>document.getElementById('point-values').textContent.includes('A 0.670'));
    assert.equal((await draft(id))[0].value,.25,'Unconfirmed input must not overwrite persisted draft');
    assert.deepEqual(await history(),[0,0],'Repeated input must not enter history before blur');
    await Promise.race([entered.promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error('No debounce request')),5000))]);
    hold.release();
    await page.unrouteAll({behavior:'wait'});
    await blur(); await settled();
    assert.equal((await draft(id))[0].value,.67);
    assert.deepEqual(await history(),[1,0],'One focus/input/blur session must be one step');
    checks.push('immediate local full-curve preview without server response','one input session equals one history step');

    await page.locator('#edit-in').fill('0.6');
    await page.locator('#edit-in').fill('0.7');
    await blur(); await settled();
    assert.deepEqual(await history(),[2,0],'Following transition input must be a separate step');
    await step('#undo-draft');
    assert.equal((await draft(id))[0].transition_in_us,800000);
    assert.equal((await draft(id))[0].value,.67);
    await step('#undo-draft'); assert.equal((await draft(id))[0].value,.25);
    await step('#redo-draft'); await step('#redo-draft');
    checks.push('two input controls undo independently','undo and redo restore exact edits');
    await blur(); await settled();
    const beforeShortcuts=await draft(id);
    await page.keyboard.press('Control+z'); await settled();
    assert.equal((await draft(id))[0].transition_in_us,800000);
    await page.keyboard.press('Control+y'); await settled();
    assert.deepEqual(await draft(id),beforeShortcuts);
    await page.keyboard.press('Control+z'); await settled();
    await page.keyboard.press('Control+Shift+z'); await settled();
    assert.deepEqual(await draft(id),beforeShortcuts);
    checks.push('Ctrl+Z, Ctrl+Y and Ctrl+Shift+Z outside input controls');

    await editFirst();
    for(const mode of ['smoothstep','smootherstep','linear']){
      await page.locator('#edit-interpolation').focus();
      await page.locator('#edit-interpolation').selectOption(mode);
      await blur(); await settled();
      assert.equal((await draft(id))[0].interpolation,mode);
    }
    checks.push('all three transition modes');

    await page.locator('#edit-note').focus();
    const prevented=await page.locator('#edit-note').evaluate(el=>{
      const event=new KeyboardEvent('keydown',{key:'z',ctrlKey:true,bubbles:true,cancelable:true});
      el.dispatchEvent(event); return event.defaultPrevented;
    });
    assert.equal(prevented,false,'Native input undo intercepted');
    await blur(); await settled();
    checks.push('native input undo retained');

    await editFirst(); await setTime(2); await page.locator('#edit-value').focus();
    const goodText=await page.locator('#point-values').textContent(), goodHistory=await history();
    await page.locator('#edit-value').fill('2');
    await page.waitForFunction(()=>document.getElementById('adjustment-state').dataset.state==='error');
    assert.equal(await page.locator('#save-revision').isDisabled(),true);
    assert.equal(await page.locator('#point-values').textContent(),goodText,'Invalid input changed the last valid curve');
    assert.deepEqual(await history(),goodHistory);
    await page.locator('#edit-value').press('Escape'); await settled();
    assert.equal((await draft(id))[0].value,.67);
    checks.push('invalid input preserves curve and blocks save','Escape restores transaction baseline');

    // Force a stale first response to be an error after the second response succeeds.
    const stale=gate(), staleEntered=gate();
    let previewNumber=0;
    await page.route(endpoint,async route=>{
      if(++previewNumber===1){staleEntered.release();await stale.promise;await route.fulfill({
        status:400,contentType:'application/json',body:JSON.stringify({error:'STALE_RESPONSE_MUST_BE_IGNORED'})});
      }else await route.continue();
    });
    await page.locator('#edit-value').fill('0.7');
    await Promise.race([staleEntered.promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error('No first preview')),5000))]);
    await page.locator('#edit-value').fill('0.71');
    await page.waitForFunction(()=>document.getElementById('adjustment-state').textContent.includes('后台校验通过'));
    stale.release();
    await page.unrouteAll({behavior:'wait'});
    assert.notEqual(await page.locator('#adjustment-state').getAttribute('data-state'),'error');
    assert.ok((await page.locator('#point-values').textContent()).includes('A 0.710'));
    await page.locator('#edit-value').press('Escape'); await settled();
    checks.push('stale validation response cannot overwrite current input');

    // Project switching invalidates old preview responses, not just old DOM controls.
    const switchGate=gate(), switchEntered=gate();
    await page.route(endpoint,async route=>{
      switchEntered.release(); await switchGate.promise;
      await route.fulfill({status:400,contentType:'application/json',body:JSON.stringify({error:'OLD_PROJECT_RESPONSE'})});
    });
    await page.locator('#edit-value').fill('0.73');
    await Promise.race([switchEntered.promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error('No pending preview')),5000))]);
    await load(other);
    switchGate.release(); await page.unrouteAll({behavior:'wait'});
    assert.ok((await page.locator('#project-meta').textContent()).includes(other));
    assert.equal(await page.locator('#edits-list .edit-item').count(),0);
    assert.notEqual(await page.locator('#adjustment-state').getAttribute('data-state'),'error');
    checks.push('switch project discards pending transaction and stale response');

    // A visual curve is not drawn before the real MCTS baseline exists.
    assert.equal(await page.locator('#visual-timeline').getAttribute('data-ready'),'false');
    await page.locator('#visual-field').selectOption('particle_speed');
    await page.waitForFunction(()=>document.getElementById('visual-timeline').dataset.ready==='true',null,{timeout:120000});
    assert.ok(await page.locator('#visual-timeline').getAttribute('data-plan-key'),'Real plan key missing');
    const originalPlanKey=await page.locator('#visual-timeline').getAttribute('data-plan-key');
    const baselineImage=await page.locator('#visual-timeline').evaluate(el=>el.toDataURL());
    assert.equal(await page.locator('#edit-interpolation').inputValue(),'smootherstep','New edits default must be smootherstep');
    await page.locator('#edit-layer').selectOption('visual');
    await page.locator('#edit-value').focus();
    await page.evaluate(()=>{
      const values={'edit-field':'particle_speed','edit-operation':'point_target','edit-start':'2',
        'edit-value':'4','edit-in':'.5','edit-out':'.5','edit-interpolation':'smootherstep'};
      for(const [id,value] of Object.entries(values))document.getElementById(id).value=value;
      document.getElementById('edit-value').dispatchEvent(new Event('input',{bubbles:true}));
    });
    await page.waitForFunction(before=>document.getElementById('visual-timeline').toDataURL()!==before,baselineImage);
    await blur(); await settled();
    assert.equal((await draft(other))[0].field,'particle_speed');
    checks.push('visual preview uses real MCTS baseline','visual input updates its own curve immediately');

    await page.locator('#edit-value').fill('4.5');
    await page.locator('#save-revision').click();
    await page.waitForFunction(()=>document.getElementById('draft-state').textContent.includes('已保存') &&
      document.getElementById('editor').getAttribute('aria-busy')==='false');
    assert.equal((await draft(other))[0].value,4.5,'Save did not include pending transaction snapshot');
    assert.deepEqual(await history(),[0,0]);
    checks.push('save awaits pending transaction and saves exact snapshot');

    // An emotion candidate retaining an enabled visual edit must prepare its own
    // real plan before /preview, otherwise validation and planning deadlock.
    let candidatePlanRequests=0;
    const trackCandidatePlan=request=>{
      if(request.method()==='POST' && request.url()===BASE+'/api/projects/'+other+'/visual-plan') {
        const body=request.postDataJSON();
        if(body.edits.some(e=>e.layer==='emotion'&&e.field==='arousal'&&e.value===.45))
          candidatePlanRequests++;
      }
    };
    page.on('request',trackCandidatePlan);
    await setTime(2);
    await page.locator('#edit-layer').selectOption('emotion');
    await page.locator('#edit-value').focus();
    await page.evaluate(()=>{
      const values={'edit-field':'arousal','edit-operation':'point_target','edit-start':'2',
        'edit-value':'.45','edit-in':'.5','edit-out':'.5','edit-interpolation':'smootherstep'};
      for(const [id,value] of Object.entries(values))document.getElementById(id).value=value;
      document.getElementById('edit-value').dispatchEvent(new Event('input',{bubbles:true}));
    });
    await page.waitForFunction(()=>document.getElementById('point-values').textContent.includes('A 0.450'));
    await blur(); await settled(120000);
    page.off('request',trackCandidatePlan);
    const combinedEdits=await draft(other);
    assert.equal(combinedEdits.filter(e=>e.layer==='visual'&&e.enabled!==false).length,1);
    assert.equal(combinedEdits.find(e=>e.layer==='emotion').value,.45);
    assert.ok(candidatePlanRequests>0,'Candidate emotion never requested its own real plan');
    await page.waitForFunction(()=>document.getElementById('visual-timeline').dataset.ready==='true');
    assert.notEqual(await page.locator('#visual-timeline').getAttribute('data-plan-key'),originalPlanKey);
    await page.locator('#save-revision').click();
    await page.waitForFunction(()=>document.getElementById('draft-state').textContent.includes('已保存')&&
      document.getElementById('editor').getAttribute('aria-busy')==='false');
    const combinedSaved=await (await page.request.get(BASE+'/api/projects/'+other)).json();
    assert.equal(combinedSaved.revision.edits.find(e=>e.layer==='emotion').value,.45);
    assert.equal(combinedSaved.revision.edits.find(e=>e.layer==='visual').value,4.5);
    checks.push('emotion candidate with enabled visual edit obtains real plan and confirms','combined emotion and visual edits save without planning deadlock');

    // A latest (not stale) backend rejection restores the last accepted visual
    // curve while keeping the rejected form value visibly invalid.
    await page.locator('#edits-list .edit-item').filter({hasText:'粒子速度'})
      .getByRole('button',{name:'编辑',exact:true}).click();
    await page.waitForFunction(()=>document.getElementById('visual-timeline').dataset.ready==='true',null,{timeout:120000});
    await page.locator('#edit-value').fill('4.6');
    await page.waitForFunction(()=>document.getElementById('adjustment-state').textContent.includes('后台校验通过'));
    const acceptedVisualImage=await page.locator('#visual-timeline').evaluate(el=>el.toDataURL());
    const otherPreview=BASE+'/api/projects/'+other+'/preview';
    await page.route(otherPreview,route=>route.fulfill({status:400,contentType:'application/json',
      body:JSON.stringify({error:'Intentional current-candidate rejection'})}));
    await page.locator('#edit-value').fill('4.7');
    await page.waitForFunction(()=>document.getElementById('adjustment-state').dataset.state==='error');
    assert.equal(await page.locator('#save-revision').isDisabled(),true);
    assert.equal(await page.locator('#visual-timeline').evaluate(el=>el.toDataURL()),acceptedVisualImage,
      'Rejected visual result remained displayed instead of last backend-accepted curve');
    assert.equal((await draft(other)).find(e=>e.layer==='visual').value,4.5);
    await page.unrouteAll({behavior:'wait'});
    await page.locator('#edit-value').press('Escape'); await settled();
    checks.push('backend rejection restores last accepted visual curve and blocks save');

    await load(id);
    await page.getByRole('button',{name:'恢复草稿',exact:true}).click(); await settled();
    const restoredHistory=await history();
    assert.equal(restoredHistory[0],1);
    await step('#undo-draft'); assert.equal((await draft(id))[0].value,.25);
    await step('#redo-draft');
    checks.push('local draft restore remains undoable');

    await setTime(1.7); await editFirst(); await blur(); await settled();
    await page.locator('#timeline').scrollIntoViewIfNeeded();
    const edit=(await draft(id))[0];
    const geometry=await page.evaluate(async ({id,edit})=>{
      const data=await (await fetch('/api/projects/'+id)).json();
      const p=StudioTimeline.emotionPreview(data.timeline.source_raw,data.timeline.step_us,
        Math.round(data.project.duration*1e6),[edit]);
      const values=[...p.raw,...p.effective].map(row=>row[1]);
      const lowRaw=Math.min(...values,edit.value), highRaw=Math.max(...values,edit.value);
      const mid=(lowRaw+highRaw)/2,span=Math.max(.1,highRaw-lowRaw);
      const low=Math.max(-1,mid-span*.65),high=Math.min(1,mid+span*.65);
      const box=document.getElementById('timeline').getBoundingClientRect();
      return {x:box.x+52+(box.width-67)*edit.time_us/1e6/data.project.duration,
        y:box.y+272-(edit.value-low)/(high-low)*105};
    },{id,edit});
    const beforeDrag=await page.locator('#point-values').textContent(), beforeDragHistory=await history();
    await page.mouse.move(geometry.x,geometry.y); await page.mouse.down();
    await page.mouse.move(geometry.x+30,geometry.y+18,{steps:5});
    await page.waitForFunction(text=>document.getElementById('point-values').textContent!==text,beforeDrag);
    assert.deepEqual(await history(),beforeDragHistory,'Pointermove must not write history');
    await page.mouse.up(); await settled();
    assert.equal((await history())[0],beforeDragHistory[0]+1);
    assert.ok((await draft(id))[0].time_us>2000000);
    checks.push('drag recomputes entire affected curve before pointerup','drag is one history step');

    // Keep prior bounded-history coverage through the public UI.
    for(let i=0;i<51;i++){
      const response=page.waitForResponse(r=>r.url().endsWith('/preview')&&r.request().method()==='POST');
      await page.locator('#edits-list input[type=checkbox]').click();
      assert.ok((await response).ok()); await settled();
    }
    assert.equal((await history())[0],50);
    checks.push('history capped at 50 confirmed operations');
    await page.screenshot({path:validationPath('studio-live-editor.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    const report={status:'passed',projects:[id,other],checks,candidatePlanRequests,
      screenshot:validationPath('studio-live-editor.png'),finishedAt:new Date().toISOString()};
    fs.mkdirSync(validationPath(''),{recursive:true});
    fs.writeFileSync(validationPath('live-editor-report.json'),JSON.stringify(report,null,2)+'\n','utf8');
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
