const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Read-only browser benchmark with synthetic API responses; never changes a project.
const {chromium} = require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const {writeFileSync} = require('node:fs');
const M = require('../web/timeline-math.js');
const BASE = 'http://127.0.0.1:8765';
(async () => {
  const browser = await chromium.launch({headless:true,
    executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const results=[];
  try {
    for (const seconds of [180,600]) {
      const page=await browser.newPage({viewport:{width:1440,height:1100}});
      const raw=Array.from({length:seconds*10},(_,i)=>[.2*Math.sin(i/50),.3+.1*Math.cos(i/70),0,0,0]);
      const project={id:'p-browser-benchmark',name:'SYNTHETIC PREVIEW BENCHMARK',
        duration:seconds,duration_us:seconds*1000000,current_revision:'r000000'};
      const timeline={...M.emotionPreview(raw,100000,project.duration_us,[]),source_raw:raw,step_us:100000};
      await page.route('**/api/**',async route=>{
        const path=new URL(route.request().url()).pathname;
        let data;
        if(path==='/api/bootstrap')data={token:'benchmark-only',projects:[project]};
        else if(path==='/api/jobs')data={jobs:[]};
        else if(path.endsWith('/preview'))data=M.emotionPreview(raw,100000,project.duration_us,route.request().postDataJSON().edits);
        else data={project,revision:{id:'r000000',edits:[]},revisions:[{id:'r000000'}],renders:[],timeline};
        await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)});
      });
      await page.route('**/media/**',route=>route.fulfill({status:204}));
      await page.goto(BASE+'/',{waitUntil:'networkidle'});
      await page.waitForFunction(()=>!document.getElementById('editor').inert);
      await page.locator('#edit-field').selectOption('arousal');
      const timings=await page.evaluate(async seconds=>{
        const $=id=>document.getElementById(id);
        $('edit-start').value=seconds/2;
        $('playhead').value=seconds/2;
        $('playhead').dispatchEvent(new Event('change',{bubbles:true}));
        const values=[.61,.62,.63,.64,.65,.66];
        const samples=[];
        for(const value of values){
          const begin=performance.now();
          $('edit-value').value=value;
          $('edit-value').dispatchEvent(new Event('input',{bubbles:true}));
          await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
          if(!$('point-values').textContent.includes('A '+value.toFixed(3)))throw new Error('Local preview did not update');
          samples.push(performance.now()-begin);
        }
        return samples.slice(1); // warm-up is not part of the reported sample set
      },seconds);
      timings.sort((a,b)=>a-b);
      results.push({seconds,median_ms:timings[2],max_ms:timings[4],samples:timings.length});
      await page.close();
    }
  } finally {await browser.close();}
  const report={status:'passed',created_at:new Date().toISOString(),
    scope:'headless Edge, 1440x1100, synthetic 3/10-minute source; input through local calculation and canvas draw to next animation frame; excludes backend/network/planning/video',results};
  writeFileSync(validationPath('browser-preview-performance.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
