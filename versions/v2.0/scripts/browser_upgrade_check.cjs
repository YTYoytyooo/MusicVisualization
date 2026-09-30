const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Run from the Studio directory against its already-running local server.
// Creates isolated synthetic projects; never edits existing user projects.
const {chromium}=require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const {execFileSync}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const BASE=process.env.STUDIO_URL||'http://127.0.0.1:8765';
const makeDemo=()=>JSON.parse(execFileSync(studioPython,
  ['main.py','demo','--duration','8',...(process.env.STUDIO_PROJECTS?['--projects',process.env.STUDIO_PROJECTS]:[])],{encoding:'utf8',cwd:codeRoot})).split(/[\\/]/).pop();
(async()=>{
  const id=makeDemo(),checks=[],errors=[];
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.setDefaultTimeout(30000);
  page.on('pageerror',error=>errors.push(error.message));
  page.on('dialog',dialog=>dialog.accept());
  let token;
  const post=async(path,data)=>{
    const response=await page.request.post(BASE+path,{headers:{'X-Studio-Token':token},data});
    assert.ok(response.ok(),path+': '+await response.text());return response.json();
  };
  const detail=async()=> (await page.request.get(BASE+'/api/projects/'+id)).json();
  const settle=()=>page.waitForFunction(()=>
    document.getElementById('adjustment-state').dataset.state==='ready'&&
    !document.getElementById('draft-state').textContent.includes('临时')&&
    document.getElementById('editor').getAttribute('aria-busy')==='false',null,{timeout:180000});
  const load=async()=>{
    await page.locator('#project-list [data-project-id="'+id+'"]').click();
    await page.waitForFunction(id=>document.getElementById('project-meta').textContent.includes(id)&&
      !document.getElementById('editor').inert&&document.getElementById('editor').getAttribute('aria-busy')==='false',id);
  };
  const waitJob=async(job)=>{
    const jobId=job.id||job.job?.id;
    assert.ok(jobId,'Job response must contain id');
    const deadline=Date.now()+240000;
    while(Date.now()<deadline){
      const jobs=(await (await page.request.get(BASE+'/api/jobs')).json()).jobs;
      const current=jobs.find(item=>item.id===jobId);
      if(current&&['succeeded','failed','cancelled','interrupted','diagnostic_only'].includes(current.status)){
        assert.equal(current.status,'succeeded',JSON.stringify(current));return current;
      }
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    throw new Error('Job timeout: '+jobId);
  };
  try{
    token=(await (await page.request.get(BASE+'/api/bootstrap')).json()).token;
    const original=await detail();
    const edit={id:'upgrade-arousal',layer:'emotion',field:'arousal',operation:'point_target',time_us:4000000,
      value:.3,transition_in_us:500000,transition_out_us:500000,interpolation:'smootherstep',enabled:true};
    await post('/api/projects/'+id+'/revisions',{base_revision:original.revision.id,edits:[edit]});
    await page.goto(BASE);await page.waitForFunction(()=>!document.getElementById('editor').inert);
    await load();await settle();
    const baseRevision=(await detail()).revision.id;
    await page.locator('#zoom-in').click();
    assert.equal(Number(await page.locator('#timeline').getAttribute('data-view-end')),4);
    await page.locator('#pan-right').click();
    assert.equal(Number(await page.locator('#timeline').getAttribute('data-view-start')),2);
    await page.locator('#axis-lock').check();
    await page.locator('#waveform').click({position:{x:180,y:38}});
    assert.ok(Number(await page.locator('#playhead').inputValue())>=2);
    await page.locator('#edits-list button').filter({hasText:'编辑'}).first().click();
    await page.locator('#locate-edit').click();await settle();
    const locatedStart=Number(await page.locator('#timeline').getAttribute('data-view-start'));
    const locatedEnd=Number(await page.locator('#timeline').getAttribute('data-view-end'));
    assert.ok(locatedStart<3.5&&locatedStart>2&&locatedEnd>4.5&&locatedEnd<6);
    await page.locator('#view-full').click();
    assert.equal(Number(await page.locator('#timeline').getAttribute('data-view-end')),8);
    assert.match(await page.locator('#history-state').textContent(),/可撤销 0 步/);
    checks.push('timeline zoom/pan/seek/locate/full and axis lock preserve edits/history');

    await page.locator('#smoothing-enabled').check();
    await page.locator('#smoothing-window').fill('0.8');
    await page.locator('#apply-smoothing').click();await settle();
    assert.match(await page.locator('#smoothing-state').textContent(),/已开启/);
    assert.equal(await page.locator('#baseline-legend').isVisible(),true);
    const cached=await page.evaluate(id=>JSON.parse(localStorage.getItem(
      'musicvisualization-studio:draft:'+id+':'+document.getElementById('revision-select').value)),id);
    assert.equal(cached.smoothing.enabled,true);assert.equal(cached.smoothing.window_seconds,.8);
    await page.locator('#undo-draft').click();await settle();
    assert.equal(await page.locator('#smoothing-enabled').isChecked(),false);
    await page.locator('#redo-draft').click();await settle();
    assert.equal(await page.locator('#smoothing-enabled').isChecked(),true);
    await page.locator('#save-revision').click();
    await page.waitForFunction(rev=>document.getElementById('revision-select').value!==rev&&
      !document.getElementById('editor').inert,baseRevision);
    const saved=await detail();
    assert.deepEqual(saved.revision.smoothing,{enabled:true,window_seconds:.8});
    await page.reload();await page.waitForFunction(()=>!document.getElementById('editor').inert);await load();
    assert.equal(await page.locator('#smoothing-enabled').isChecked(),true);
    checks.push('smoothing is undoable, locally persisted and restored from immutable revision');

    // Real immutable preview renders: different start offsets exercise absolute-time comparison.
    const rawJob=await post('/api/projects/'+id+'/previews',{base_revision:saved.revision.id,
      edits:[],smoothing:{enabled:false,window_seconds:.5},start:1,end:5,width:320,height:180,fps:30});
    await waitJob(rawJob);
    const responsePromise=page.waitForResponse(response=>response.url().endsWith('/api/projects/'+id+'/previews')&&
      response.request().method()==='POST');
    await page.locator('#preview-edit').click();
    const previewResponse=await responsePromise;
    assert.ok(previewResponse.ok(),await previewResponse.text());
    await waitJob(await previewResponse.json());
    const afterPreviews=await detail();
    assert.equal(afterPreviews.project.current_revision,saved.revision.id);
    const previews=afterPreviews.renders.filter(render=>render.kind==='preview'&&render.status==='succeeded');
    assert.ok(previews.length>=2);assert.ok(previews.every(render=>render.snapshot_id));
    await load();
    const reference=previews.find(render=>Number(render.actual_start)===1),modified=previews.at(-1);
    assert.ok(reference);
    await page.locator('#render-select').selectOption(modified.id);
    assert.match(await page.locator('#video-version').textContent(),/局部快照.*不是正式修订/);
    await page.locator('#compare-render').selectOption(reference.id);
    await page.locator('#compare-enabled').check();
    await page.waitForFunction(()=>document.getElementById('video').readyState>=1&&
      document.getElementById('compare-video').readyState>=1);
    await page.locator('#playhead').fill('2');await page.locator('#playhead').press('Tab');
    await page.locator('#compare-play').click();await page.waitForTimeout(700);
    await page.locator('#compare-pause').click();
    const media=await page.evaluate(()=>({a:document.getElementById('video').currentTime,
      b:document.getElementById('compare-video').currentTime,am:document.getElementById('video').muted,
      bm:document.getElementById('compare-video').muted}));
    assert.ok(Math.abs(media.a+Number(modified.actual_start)-media.b-Number(reference.actual_start))<.2,JSON.stringify(media));
    assert.equal(Number(media.am)+Number(media.bm),1);
    await page.locator('#compare-sound').selectOption('reference');
    assert.equal(await page.locator('#video').evaluate(video=>video.muted),true);
    assert.equal(await page.locator('#compare-video').evaluate(video=>video.muted),false);
    checks.push('preview snapshots preserve revision; offset video comparison stays synchronized with one audio source');

    // Prepare a real visual baseline, then change emotion repeatedly in one input session.
    let plan=await post('/api/projects/'+id+'/visual-plan',{base_revision:saved.revision.id,
      edits:saved.revision.edits,smoothing:saved.revision.smoothing,consumer_id:'upgrade-test:'+id,purpose:'confirmed'});
    if(plan.status==='queued')await waitJob({id:plan.job_id});
    const visual={id:'upgrade-brightness',layer:'visual',field:'brightness',operation:'point_target',
      time_us:4000000,value:.5,transition_in_us:500000,transition_out_us:500000,enabled:true,interpolation:'smootherstep'};
    await post('/api/projects/'+id+'/revisions',{base_revision:saved.revision.id,edits:[edit,visual],smoothing:saved.revision.smoothing});
    await load();
    const requests=[];
    page.on('request',request=>{
      if(request.method()==='POST'&&/\/(?:visual-plan|plan-release)$/.test(request.url()))
        requests.push({url:request.url(),body:request.postDataJSON()});
    });
    await page.locator('#edits-list button').filter({hasText:'编辑'}).first().click();
    await page.locator('#edit-value').fill('.34');await page.waitForTimeout(280);
    await page.locator('#edit-value').fill('.42');await page.waitForTimeout(280);
    await page.locator('#edit-value').press('Enter');await settle();
    const candidate=await page.evaluate(id=>{
      const key='musicvisualization-studio:draft:'+id+':'+document.getElementById('revision-select').value;
      return JSON.parse(localStorage.getItem(key));
    },id);
    assert.equal(candidate.edits.find(item=>item.id==='upgrade-arousal').value,.42);
    assert.ok(requests.some(request=>request.body.purpose==='temporary'&&Number.isInteger(request.body.consumer_seq)));
    assert.ok(requests.some(request=>request.url.endsWith('/plan-release')&&Number.isInteger(request.body.consumer_seq)));
    assert.ok(requests.some(request=>request.body.purpose==='confirmed'));
    checks.push('latest candidate emotion wins with visual edits; sequenced temporary consumers are released and formal confirmation survives');

    await page.locator('#project-search').fill('no-such-project-upgrade-check');
    assert.equal(await page.locator('#project-list .project-button').count(),0);
    await page.locator('#project-search').fill('');
    assert.ok(await page.locator('#project-list .project-button').count()>0);
    await page.waitForFunction(id=>{
      const rows=[...document.querySelectorAll('#jobs-list .job')];
      return rows.length>0&&rows.every(row=>row.dataset.projectId===id);
    },id);
    const completedVisible=await page.locator('#jobs-list .job').evaluateAll(rows=>rows.filter(row=>
      ['succeeded','failed','cancelled','interrupted','diagnostic_only'].includes(row.dataset.status)).length);
    assert.ok(completedVisible<=5);
    const totalJobs=(await (await page.request.get(BASE+'/api/jobs')).json()).jobs.length;
    await page.locator('#jobs-scope').selectOption('all');
    await page.waitForFunction(total=>Number(document.getElementById('jobs-summary').textContent.match(/\/ (\d+) 项/)?.[1])>=total,totalJobs);
    if(await page.locator('#jobs-more').isVisible()){
      await page.locator('#jobs-more').click();
      await page.waitForFunction(()=>document.getElementById('jobs-more').textContent==='收起');
      assert.ok(await page.locator('#jobs-list .job').count()>5);
      await page.locator('#jobs-more').click();
      await page.waitForFunction(()=>document.getElementById('jobs-more').textContent.startsWith('显示更多'));
    }
    await page.locator('#jobs-scope').selectOption('current');
    await page.waitForFunction(id=>[...document.querySelectorAll('#jobs-list .job')].every(row=>row.dataset.projectId===id),id);
    checks.push('project search and scoped compact job history, expand/collapse and all-project filter');
    await page.locator('#jobs-list button').filter({hasText:'日志'}).first().click();
    await page.waitForSelector('#job-log-dialog[open]');
    await page.locator('#close-log').click();
    fs.mkdirSync(validationPath(''),{recursive:true});
    await page.evaluate(async()=>{document.querySelectorAll('.sidebar,.inspector').forEach(element=>element.scrollTop=0);
      scrollTo(0,0);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
    await page.screenshot({path:validationPath('upgrade-workbench.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(async()=>{document.querySelectorAll('.sidebar,.inspector').forEach(element=>element.scrollTop=0);
      scrollTo(0,0);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
    await page.screenshot({path:validationPath('upgrade-workbench-mobile.png'),fullPage:true});
    const overflow=await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(element=>{
      const rect=element.getBoundingClientRect();
      return rect.width&&rect.right>innerWidth+2;
    }).map(element=>({tag:element.tagName,id:element.id,className:String(element.className),
      width:element.getBoundingClientRect().width,right:element.getBoundingClientRect().right})).slice(0,20));
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),JSON.stringify(overflow));
    checks.push('job log dialog and narrow-screen layout');
    assert.deepEqual(errors,[]);
    fs.mkdirSync(validationPath(''),{recursive:true});
    fs.writeFileSync(validationPath('upgrade-editor-report.json'),JSON.stringify({project:id,checks,errors},null,2));
    console.log(JSON.stringify({project:id,checks,errors},null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
