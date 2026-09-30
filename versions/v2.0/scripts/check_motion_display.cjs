const {validationPath, studioPython, codeRoot} = require('./workspace.cjs');
// Read-only screenshot/pixel regression against a pre-existing acceptance project.
const {chromium}=require(process.env.STUDIO_PLAYWRIGHT || 'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
  const base=process.env.STUDIO_URL,id=process.env.STUDIO_PROJECT;
  assert.ok(base&&id);
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(()=>{
      window.motionLabels=[];
      const original=CanvasRenderingContext2D.prototype.fillText;
      CanvasRenderingContext2D.prototype.fillText=function(...args){
        if(this.canvas.id==='motion-timeline')window.motionLabels.push(args[0]);
        return original.apply(this,args);
      };
    });
    await page.goto(base);
    await page.waitForFunction(()=>!document.getElementById('editor').inert);
    await page.locator('#project-list [data-project-id="'+id+'"]').click();
    await page.waitForFunction(id=>document.getElementById('project-meta').textContent.includes(id)&&
      document.getElementById('motion-timeline').dataset.ready==='true',id);
    await page.locator('#motion-overrides button').filter({hasText:'编辑'}).click();
    await page.evaluate(()=>{
      window.motionLabels=[];
      const input=document.getElementById('playhead');input.value='4';input.dispatchEvent(new Event('change'));
    });
    const labels=await page.evaluate(()=>window.motionLabels.filter(text=>text==='螺旋运动'));
    assert.equal(labels.length,2,'One full label per continuous mode on each of the two tracks');
    const colors=await page.locator('#motion-timeline').evaluate(canvas=>{
      const ctx=canvas.getContext('2d'),dpr=devicePixelRatio||1,seen=new Set();
      for(let x=55;x<canvas.clientWidth-18;x++){
        const pixel=ctx.getImageData(Math.round(x*dpr),Math.round(28*dpr),1,1).data;
        seen.add([...pixel].join(','));
      }
      return [...seen];
    });
    assert.ok(colors.length<=4,'Uniform band has repeated sample seams: '+colors.length);
    await page.locator('#motion-timeline').screenshot({path:validationPath('motion-track.png')});
    await page.waitForFunction(()=>document.getElementById('video').readyState>=2&&!document.getElementById('video').seeking);
    await page.evaluate(()=>{scrollTo(0,0);document.querySelectorAll('.sidebar,.inspector').forEach(element=>element.scrollTop=0);});
    await page.screenshot({path:validationPath('motion-workbench.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    const report={status:'passed',project:id,labels,bandColors:colors.length,errors,checkedAt:new Date().toISOString()};
    fs.writeFileSync(validationPath('motion-display-report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
