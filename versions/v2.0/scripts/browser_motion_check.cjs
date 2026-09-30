const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Isolated end-to-end motion editing checks. Run against an already-running Studio.
const {chromium}=require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const {execFileSync}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const BASE=process.env.STUDIO_URL||'http://127.0.0.1:8765';
const makeDemo=()=>JSON.parse(execFileSync(studioPython,
  ['main.py','demo','--duration','12',...(process.env.STUDIO_PROJECTS?['--projects',process.env.STUDIO_PROJECTS]:[])],{encoding:'utf8',cwd:codeRoot})).split(/[\\/]/).pop();
(async()=>{
  const id=makeDemo(),other=makeDemo(),checks=[],errors=[];
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.setDefaultTimeout(30000);page.on('pageerror',error=>errors.push(error.message));page.on('dialog',dialog=>dialog.accept());
  const requests=[];
  page.on('request',request=>{if(request.method()==='POST')requests.push({url:request.url(),data:request.postDataJSON()});});
  const settle=()=>page.waitForFunction(()=>document.getElementById('motion-adjustment-state').dataset.state==='ready'&&
    document.getElementById('editor').getAttribute('aria-busy')==='false'&&
    !document.getElementById('draft-state').textContent.includes('临时'),null,{timeout:120000});
  const load=async(project)=>{
    await page.locator('#project-list [data-project-id="'+project+'"]').click();
    await page.waitForFunction(id=>document.getElementById('project-meta').textContent.includes(id)&&
      !document.getElementById('editor').inert&&document.getElementById('editor').getAttribute('aria-busy')==='false',project);
  };
  const history=()=>page.locator('#history-state').textContent();
  const cached=()=>page.evaluate(id=>JSON.parse(localStorage.getItem('musicvisualization-studio:draft:'+id+':'+
    document.getElementById('revision-select').value)),id);
  const detail=async()=> (await (await page.request.get(BASE+'/api/projects/'+id)).json());
  try{
    await page.goto(BASE);await page.waitForFunction(()=>!document.getElementById('editor').inert);await load(id);
    assert.equal(await page.locator('#motion-engine').inputValue(),'legacy');
    const initial=(await detail()).revision.id;
    await page.locator('#motion-engine').selectOption('flow-v1');await settle();
    await page.waitForFunction(()=>document.getElementById('motion-timeline').dataset.ready==='true');
    assert.equal((await cached()).motion.engine,'flow-v1');
    assert.equal(requests.filter(request=>request.url.endsWith('/visual-plan')).length,0);
    checks.push('old revisions default to legacy; motion engine enables independently of MCTS');

    await page.locator('#motion-new').click();
    await page.locator('#motion-confirm').click();await settle();
    assert.equal(await page.locator('#motion-overrides .edit-item').count(),1);
    for(const mode of ['rise','fall','orbit','spiral','expand','gather','meteor','wave','turbulent']){
      await page.locator('#motion-mode').selectOption(mode);await settle();
      assert.equal((await cached()).motion.overrides[0].mode,mode);
      if(mode==='rise')assert.equal(await page.locator('#motion-param-direction_deg').inputValue(),'270');
      if(mode==='fall')assert.equal(await page.locator('#motion-param-direction_deg').inputValue(),'90');
    }
    await page.locator('#motion-mode').selectOption('spiral');await settle();
    await page.locator('#motion-override-fields details').evaluate(element=>element.open=true);
    const before=await history();
    await page.locator('#motion-param-speed').fill('.211');
    await page.waitForFunction(()=>document.getElementById('motion-frame').textContent.includes('0.211'));
    assert.equal(await history(),before);
    await page.locator('#motion-param-speed').fill('.321');
    await page.waitForFunction(()=>document.getElementById('motion-frame').textContent.includes('0.321'));
    await page.locator('#motion-param-speed').press('Enter');await settle();
    const after=await history();
    assert.equal(Number(after.match(/可撤销 (\d+)/)[1]),Number(before.match(/可撤销 (\d+)/)[1])+1);
    assert.equal((await cached()).motion.overrides[0].params.speed,.321);
    assert.equal(requests.filter(request=>request.url.endsWith('/visual-plan')).length,0);
    checks.push('all nine manual modes and immediate numeric feedback; one input session is one history step');

    await page.locator('#undo-draft').click();await settle();
    assert.notEqual((await cached()).motion.overrides[0].params.speed,.321);
    await page.locator('#redo-draft').click();await settle();
    assert.equal((await cached()).motion.overrides[0].params.speed,.321);
    await page.locator('#motion-param-speed').fill('.9');
    await page.waitForFunction(()=>document.getElementById('motion-adjustment-state').dataset.state==='error');
    assert.equal(await page.locator('#save-revision').isDisabled(),true);
    await page.locator('#motion-param-speed').press('Escape');await settle();
    assert.equal(await page.locator('#motion-param-speed').inputValue(),'0.321');
    await page.locator('#motion-overrides input[type=checkbox]').uncheck();await settle();
    assert.equal((await cached()).motion.overrides[0].enabled,false);
    await page.locator('#undo-draft').click();await settle();
    assert.equal((await cached()).motion.overrides[0].enabled,true);
    await page.locator('#motion-overrides button').filter({hasText:'删除'}).click();await settle();
    assert.equal(await page.locator('#motion-overrides .edit-item').count(),0);
    await page.locator('#undo-draft').click();await settle();
    assert.equal(await page.locator('#motion-overrides .edit-item').count(),1);
    checks.push('motion undo/redo, invalid input rollback, enable/disable and delete recovery');

    // Deliberately let an older real motion-plan HTTP response finish last.
    let releaseOld,seenOld;
    const oldGate=new Promise(resolve=>releaseOld=resolve),oldSeen=new Promise(resolve=>seenOld=resolve);
    const endpoint=BASE+'/api/projects/'+id+'/motion-plan';
    await page.route(endpoint,async route=>{
      if(route.request().postDataJSON().motion.sensitivity===1.2){
        const response=await route.fetch();seenOld();await oldGate;await route.fulfill({response});
      }else await route.continue();
    });
    await page.locator('#motion-sensitivity').fill('1.2');await oldSeen;
    await page.locator('#motion-sensitivity').fill('1.4');
    await page.waitForFunction(()=>document.getElementById('motion-plan-state').dataset.sensitivity==='1.4');
    releaseOld();await page.waitForTimeout(150);
    assert.equal(await page.locator('#motion-plan-state').getAttribute('data-sensitivity'),'1.4');
    await page.locator('#motion-sensitivity').press('Enter');await settle();
    await page.unroute(endpoint);
    checks.push('late automatic-plan response cannot overwrite newer settings');

    await page.locator('#save-revision').click();
    await page.waitForFunction(initial=>document.getElementById('revision-select').value!==initial&&
      !document.getElementById('editor').inert,initial);
    const saved=await detail();
    assert.equal(saved.revision.motion.engine,'flow-v1');
    assert.equal(saved.revision.motion.overrides[0].params.speed,.321);
    await page.reload();await page.waitForFunction(()=>!document.getElementById('editor').inert);await load(id);
    assert.equal(await page.locator('#motion-engine').inputValue(),'flow-v1');
    await page.locator('#revision-select').selectOption(initial);
    await page.waitForFunction(()=>document.getElementById('motion-engine').value==='legacy'&&!document.getElementById('editor').inert);
    await page.locator('#revision-select').selectOption(saved.revision.id);
    await page.waitForFunction(()=>document.getElementById('motion-engine').value==='flow-v1'&&!document.getElementById('editor').inert);
    await page.locator('#motion-overrides button').filter({hasText:'编辑'}).click();
    const previewResponse=page.waitForResponse(response=>response.url().endsWith('/api/projects/'+id+'/previews')&&response.request().method()==='POST');
    await page.locator('#preview-edit').click();
    const response=await previewResponse;assert.ok(response.ok(),await response.text());
    const previewJob=await response.json();
    const previewRequest=requests.filter(request=>request.url.endsWith('/previews')).at(-1);
    assert.equal(previewRequest.data.motion_edit_id,saved.revision.motion.overrides[0].id);
    assert.deepEqual(previewRequest.data.motion,saved.revision.motion);
    assert.equal((await detail()).project.current_revision,saved.revision.id);
    checks.push('motion survives save/reload/history switching; frozen local preview pins motion override without advancing revision');

    // Switching project discards an unconfirmed candidate and ignores its delayed plan response.
    let releaseSwitch,seenSwitch;
    const switchGate=new Promise(resolve=>releaseSwitch=resolve),switchSeen=new Promise(resolve=>seenSwitch=resolve);
    await page.route(endpoint,async route=>{
      if(route.request().postDataJSON().motion.sensitivity===1.6){
        const result=await route.fetch();seenSwitch();await switchGate;await route.fulfill({response:result});
      }else await route.continue();
    });
    await page.locator('#motion-sensitivity').fill('1.6');await switchSeen;await load(other);
    releaseSwitch();await page.waitForTimeout(200);
    assert.equal(await page.locator('#motion-engine').inputValue(),'legacy');
    assert.equal(await page.locator('#motion-plan-state').getAttribute('data-engine'),'legacy');
    await page.unroute(endpoint);await load(id);
    checks.push('project switching rejects stale motion response');
    let completedJob;
    const deadline=Date.now()+240000;
    while(Date.now()<deadline){
      const jobs=(await (await page.request.get(BASE+'/api/jobs')).json()).jobs;
      completedJob=jobs.find(job=>job.id===previewJob.id);
      if(completedJob&&!['queued','running','cancelling'].includes(completedJob.status))break;
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    assert.equal(completedJob?.status,'succeeded',JSON.stringify(completedJob));
    await page.reload();await page.waitForFunction(()=>!document.getElementById('editor').inert);await load(id);
    await page.locator('#render-select').selectOption(completedJob.result.id);
    for(const label of ['下载运动计划 CSV','下载运动计划 JSON']){
      const link=page.getByRole('link',{name:label,exact:true});
      await link.waitFor({state:'visible'});
      assert.ok((await page.request.get(new URL(await link.getAttribute('href'),BASE).href)).ok(),label);
    }
    checks.push('frozen flow preview completes and both motion-plan download links return real files');
    fs.mkdirSync(validationPath(''),{recursive:true});
    await page.evaluate(()=>{scrollTo(0,0);document.querySelectorAll('.sidebar,.inspector').forEach(element=>element.scrollTop=0);});
    await page.screenshot({path:validationPath('motion-workbench.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(validationPath('motion-editor-report.json'),JSON.stringify({project:id,other,checks,errors},null,2));
    console.log(JSON.stringify({project:id,checks,errors},null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
