const {chromium}=require('C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1200,height:1300},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file:///D:/AI%20project/nnunet/work/region_partition_inspection_20260929/inspector.html');
 await page.locator('#fixed-region-inspector[data-ready=true]').waitFor({timeout:30000});
 await page.locator('#fri-pair').selectOption('4');
 const detail=await page.locator('#fri-detail').innerText();
 if(!detail.includes('2,496')||!detail.includes('75.24'))throw Error('Wrong actual widest region '+detail);
 await page.screenshot({path:'work/region_partition_inspection_20260929/overview.png',fullPage:true});
 const before=await page.locator('#fri-g0').evaluate(c=>c.toDataURL());
 const rect=await page.locator('#fri-g0').boundingBox();
 await page.mouse.move(rect.x+100,rect.y+100);await page.mouse.down();await page.mouse.move(rect.x+140,rect.y+125,{steps:4});await page.mouse.up();
 await page.waitForFunction(s=>document.getElementById('fri-g0').toDataURL()!==s,before);
 await page.locator('#fri-stage').selectOption('1');
 if(!(await page.locator('#fri-detail').innerText()).includes('2차'))throw Error('Stage did not update');
 await page.locator('#fri-z').fill('17');await page.locator('#fri-z').dispatchEvent('input');
 if(!(await page.locator('#fri-zlabel').innerText()).startsWith('17'))throw Error('CT slice failed');
 await page.locator('#fri-edge').selectOption('all');
 for(let i=0;i<8;i++){
  await page.locator('#fri-pair').selectOption(''+i);
  const roles=await page.locator('#fri-role option').evaluateAll(options=>options.map(o=>o.value));
  for(const role of roles){await page.locator('#fri-role').selectOption(role);if(await page.locator('#fixed-region-inspector').getAttribute('data-error'))throw Error('Role failed '+i+' '+role);}
 }
 await page.locator('#fri-pair').selectOption('4');await page.locator('#fri-stage').selectOption('0');await page.locator('#fri-edge').selectOption('selected');
 await page.setViewportSize({width:380,height:1100});
 const layout=await page.locator('#fixed-region-inspector').evaluate(r=>({width:r.clientWidth,scroll:r.scrollWidth}));
 if(layout.scroll>layout.width+2)throw Error('Narrow overflow '+JSON.stringify(layout));
 await page.screenshot({path:'work/region_partition_inspection_20260929/mobile.png',fullPage:true});
 const report={errors,detail,layout,checks:['8 records x 5 roles','actual 2496-node region and 75.24mm bbox','synchronized drag rotation','two scale selector','slice control','all edge rendering','mobile width']};
 await page.goto('file:///D:/AI%20project/nnunet/work/region_partition_inspection_20260929/inline-preview.html');
 const frame=page.frameLocator('iframe');
 await frame.locator('#fixed-region-inspector[data-ready=true]').waitFor();
 await frame.locator('#fri-stage').selectOption('1');
 if(!(await frame.locator('#fri-detail').innerText()).includes('2,496'))throw Error('Inline real data mismatch');
 await page.setViewportSize({width:1024,height:1100});
 await frame.locator('#fri-stage').selectOption('0');
 await page.screenshot({path:'work/region_partition_inspection_20260929/inline.png',fullPage:true});
 report.checks.push('sandboxed inline decode/render/control');
 fs.writeFileSync('work/region_partition_inspection_20260929/visual-check.json',JSON.stringify(report,null,2));
 await browser.close();console.log(JSON.stringify(report));if(errors.length)throw Error(errors.join(';'));
})();
