const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Read-only final presentation smoke; reuses an isolated acceptance project.
const {chromium}=require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
  const {project}=JSON.parse(fs.readFileSync(validationPath('upgrade-editor-report.json'),'utf8'));
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://127.0.0.1:8765/');
    await page.locator('#project-list [data-project-id="'+project+'"]').click();
    await page.waitForFunction(()=>[...document.querySelectorAll('#jobs-list .job strong')].some(e=>e.textContent==='视觉规划'));
    for(const text of await page.locator('#jobs-list .job[data-status="succeeded"] .hint').allTextContents()){
      const match=text.match(/已完成 (\d+) \/ (\d+)/);if(match)assert.equal(match[1],match[2]);
    }
    await page.getByRole('button',{name:'日志',exact:true}).first().click();
    await page.waitForSelector('#job-log-dialog[open]');await page.locator('#close-log').click();
    await page.evaluate(async()=>{scrollTo(0,0);document.querySelectorAll('.sidebar,.inspector').forEach(e=>e.scrollTop=0);
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
    await page.screenshot({path:validationPath('upgrade-final-ui.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Final UI labels, completed totals and 16KiB log dialog passed');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
