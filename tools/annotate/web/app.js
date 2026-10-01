'use strict';
const $=id=>document.getElementById(id), audio=$('audio');
const state={reviewer:localStorage.getItem('va-reviewer')||'me',doc:null,songs:[],token:'',ready:false,dirty:false,take:null,version:0,saving:null,values:{valence:null,arousal:null}};
const uid=()=>crypto.randomUUID(), fmt=t=>`${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,'0')}`;
function notice(text,error=false){$('notice').hidden=false;$('notice').textContent=text;$('notice').className=error?'error':'';}
const run=fn=>async(...args)=>{try{await fn(...args);}catch(e){notice(e.message,true);}};
async function api(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':state.token},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error);return data;}
const draftKey=()=>`va-continuous-draft:${state.reviewer}:${state.doc.source.song_id}`;
function remember(){if(!state.doc||!state.dirty)return;try{localStorage.setItem(draftKey(),JSON.stringify(state.doc));}catch(e){notice('浏览器草稿空间不足，请停止记录并保存到磁盘。',true);}}
function changed(){state.dirty=true;state.version++;$('save-state').textContent='有未保存记录';}
async function save(){
 if(!state.doc||!state.ready)return false;
 if(state.saving){await state.saving;if(state.dirty)return save();return true;}
 if(!state.dirty)return true;
 remember();const doc=state.doc,version=state.version;
 state.saving=(async()=>{const saved=await api('/api/save',{...doc,base_revision:doc.revision,audio_sha256:doc.source.audio_sha256,song_id:doc.source.song_id,reviewer:state.reviewer});doc.revision=saved.revision;const entry=state.songs.find(x=>x.id===doc.source.song_id);if(entry)Object.assign(entry,{completed:saved.completed,has_annotations:!!(saved.takes.length||saved.transitions.length),unrated:saved.takes.reduce((n,t)=>n+t.points.filter(p=>p.status==='annotated'&&!p.confidence).length,0)});renderLibrary();state.dirty=state.version!==version;
 if(!state.dirty){doc.takes=saved.takes;renderDetails();draw();localStorage.removeItem(draftKey());$('save-state').textContent=`已保存 r${doc.revision}`;}else remember();})();
 try{await state.saving;return true;}finally{state.saving=null;}
}
function songName(name){return name.replace(/^imports\//,'').replace(/-[a-f0-9]{64}(?=\.[^.]+$)/,'');}
function renderLibrary(){const query=$('search').value.toLowerCase(),filter=$('library-filter').value;$('library').replaceChildren();
 const finished=state.songs.filter(s=>s.completed?.valence&&s.completed?.arousal).length;$('library-summary').textContent=`${state.songs.length} 首音乐 · ${finished} 首已完成`;
 for(const song of state.songs){const done=song.completed?.valence&&song.completed?.arousal;if(!song.name.toLowerCase().includes(query)||(filter==='complete'&&!done)||(filter==='unfinished'&&done))continue;
 const b=document.createElement('button'),title=document.createElement('span'),badge=document.createElement('span'),detail=document.createElement('small');b.className='song'+(state.doc?.source.song_id===song.id?' selected':'');title.className='song-title';title.textContent=songName(song.name);badge.className='song-status '+(done?'complete':song.has_annotations?'progress':'');badge.textContent=done?'✓ 已完成':song.has_annotations?'进行中':'待标注';
 detail.textContent=`V ${song.completed?.valence?'已完成':'未完成'} · A ${song.completed?.arousal?'已完成':'未完成'}`+(song.unrated?' · 把握待评估':'');b.append(title,badge,detail);b.onclick=run(()=>openSong(song));$('library').append(b);}}
async function refresh(){const result=await api('/api/library?reviewer='+encodeURIComponent(state.reviewer));state.songs=result.songs;state.token=result.token;$('output-location').textContent=result.output;renderLibrary();}
async function openSong(song){
 if(state.loading)return;state.loading=true;try{
 stop(false);endReplay();audio.pause();if(state.dirty)await save();state.ready=false;
 const doc=await api('/api/song/'+song.id+'?reviewer='+encodeURIComponent(state.reviewer));state.doc=doc;state.dirty=false;
 const raw=localStorage.getItem(draftKey());if(raw){const draft=JSON.parse(raw);if(draft.revision===doc.revision&&draft.source.audio_sha256===doc.source.audio_sha256){if(confirm('恢复此歌上次未保存的连续标注草稿？')){state.doc=draft;changed();}}else notice('有旧版本浏览器草稿，未自动恢复；磁盘数据优先。',true);}
 state.doc.completed??={valence:false,arousal:false};state.doc.takes=effectiveTracks(state.doc.takes);state.values={valence:null,arousal:null};$('uncertain').checked=false;$('confidence').value='';paintRatings();
 $('song-name').textContent=songName(song.name);for(const id of ['artist','style','note'])$(id).value=state.doc[id]||'';
 $('workspace').hidden=false;$('empty').hidden=true;$('editor').disabled=true;$('record').disabled=true;
 $('save-state').textContent=state.dirty?'草稿待保存':`已保存 r${doc.revision}`;audio.src='/audio/'+song.id;audio.load();renderDetails();renderLibrary();draw();
 }finally{state.loading=false;}
}
audio.onloadedmetadata=()=>{if(!state.doc)return;if(!Number.isFinite(audio.duration)||audio.duration<=0){notice('音频时长无效',true);return;}state.doc.duration=audio.duration;state.viewStart=0;state.viewSpan=audio.duration;$('review-start').value=0;$('review-end').value=audio.duration.toFixed(3);state.ready=true;$('editor').disabled=false;$('record').disabled=false;draw();};
audio.onerror=()=>{stop(false);state.ready=false;notice('音频无法播放，请使用浏览器支持的音频格式。',true);};
function playbackValues(time){const values={valence:null,arousal:null};for(const take of state.doc?.takes||[]){const points=take.points;if(!points.length||time<points[0].time||time>points.at(-1).time)continue;let lo=0,hi=points.length;while(lo<hi){const mid=(lo+hi)>>1;if(points[mid].time<=time)lo=mid+1;else hi=mid;}const point=points[Math.max(0,lo-1)];for(const axis of take.axes)values[axis]=point.status==='annotated'?point[axis]:null;}return values;}
function paintRatings(){const values=state.replay?playbackValues(audio.currentTime):state.values;for(const [axis,id] of [['valence','v-value'],['arousal','a-value']]){const v=values[axis];$(id).textContent=v===null?(state.replay?'此处未标注':'尚未选择'):v.toFixed(2);$(axis).value=v??0;$(axis).disabled=!!state.replay;}document.querySelectorAll('[data-axis] button').forEach(b=>b.disabled=!!state.replay);$('va-plane').classList.toggle('replaying',!!state.replay);$('va-plane').setAttribute('aria-disabled',!!state.replay);
 const p=$('point');p.hidden=Object.values(values).every(v=>v===null);p.classList.toggle('partial',Object.values(values).includes(null));p.title=values.valence===null?'仅有唤醒标注':values.arousal===null?'仅有效价标注':'';p.style.left=(((values.valence??0)+1)*50)+'%';p.style.top=((1-(values.arousal??0))*50)+'%';}
function paintTransport(){const playing=!audio.paused;$('record').textContent=state.take||state.replay&&playing?'暂停并保存（空格）':'开始标注（空格）';$('record').setAttribute('aria-pressed',!!state.take);$('record').disabled=!state.ready||!!state.starting;$('replay').textContent=state.replay?(playing?'暂停复核':'继续复核'):'复核回放';$('replay').setAttribute('aria-pressed',!!state.replay);}
function endReplay(){state.replay=false;paintRatings();paintTransport();}
async function toggleRecord(){if(state.starting)return;if(state.take||state.replay&&!audio.paused){sample();stop(false);audio.pause();await save();paintTransport();return;}endReplay();await start();}
async function toggleReplay(){if(!state.ready||state.starting)return;if(state.replay&&!audio.paused){audio.pause();paintTransport();return;}if(!state.doc.takes.length)throw Error('请先完成一些标注，再复核回放。');sample();stop(false);await save();state.replay=true;paintRatings();await audio.play();paintTransport();}
$('replay').onclick=run(toggleReplay);
function update(axis,value){if(state.replay)return;state.values[axis]=Math.max(-1,Math.min(1,Number(value)));paintRatings();sample();}
for(const axis of ['valence','arousal'])$(axis).oninput=e=>update(axis,e.target.value);
document.querySelectorAll('[data-axis] button').forEach(b=>b.onclick=()=>update(b.parentElement.dataset.axis,b.dataset.value));
const plane=$('va-plane');function move(e){if(state.replay)return;const r=plane.getBoundingClientRect();state.values={valence:Math.max(-1,Math.min(1,(e.clientX-r.left)/r.width*2-1)),arousal:Math.max(-1,Math.min(1,1-(e.clientY-r.top)/r.height*2))};paintRatings();sample();}
plane.onpointerdown=e=>{plane.setPointerCapture(e.pointerId);move(e);};plane.onpointermove=e=>{if(plane.hasPointerCapture(e.pointerId))move(e);};
function selectedAxes(){return (state.axisMode||'both')==='both'?['valence','arousal']:[state.axisMode];}
function validRating(){return $('uncertain').checked||selectedAxes().every(a=>state.values[a]!==null);}
function effectiveTracks(takes){
 let tracks=[];
 for(const take of takes){if(!take.points.length)continue;for(const axis of take.axes||['valence','arousal']){
 const start=take.points[0].time,end=take.points.at(-1).time,next=[];
 for(const old of tracks){if(old.axes[0]!==axis||old.points.at(-1).time<start||old.points[0].time>end){next.push(old);continue;}
 for(const points of [old.points.filter(p=>p.time<start),old.points.filter(p=>p.time>end)])if(points.length)next.push({...old,id:uid(),points});}
 next.push({id:take.axes?.length===1?take.id:uid(),axes:[axis],points:take.points.map(p=>({...p,valence:axis==='valence'?p.valence:null,arousal:axis==='arousal'?p.arousal:null}))});tracks=next;
 }}return tracks;
}
document.querySelectorAll('#axis-mode button').forEach(button=>button.onclick=()=>{if(state.take||state.starting)return;state.axisMode=button.dataset.mode;document.querySelectorAll('#axis-mode button').forEach(b=>b.setAttribute('aria-pressed',b===button));});
function sample(){
 if(!state.take||audio.paused||audio.seeking)return;
 if(!validRating()){stop(false);notice('记录已停止：请明确选择当前维度的数值。',true);return;}
 const t=audio.currentTime,points=state.take.points,last=points.at(-1);
 if(last&&t-last.time>1){stop(false);notice('播放或页面更新中断，记录已停止；未填补空白。');return;}
 if(last&&t<=last.time+.001)return;
 const unsure=$('uncertain').checked;points.push({time:t,status:unsure?'uncertain':'annotated',valence:unsure||!state.take.axes.includes('valence')?null:state.values.valence,arousal:unsure||!state.take.axes.includes('arousal')?null:state.values.arousal,confidence:unsure?'':$('confidence').value});changed();draw();
}
async function start(){if(!state.ready||state.take||state.starting)return;if(!validRating())throw Error('请选择当前标注维度的数值，或勾选不确定。');if(audio.playbackRate!==1)throw Error('请使用正常速度播放后记录。');state.starting=true;try{await audio.play();}finally{state.starting=false;}state.take={id:uid(),axes:selectedAxes(),points:[]};$('axis-mode').disabled=true;state.doc.takes.push(state.take);for(const axis of state.take.axes)state.doc.completed[axis]=false;sample();paintTransport();}
function stop(persist=true){if(state.take){const take=state.take;state.take=null;if(!take.points.length)state.doc.takes=state.doc.takes.filter(x=>x!==take);state.doc.takes=effectiveTracks(state.doc.takes);$('axis-mode').disabled=false;remember();renderDetails();draw();}paintTransport();if(persist&&state.dirty)save().catch(e=>notice(e.message,true));}
$('record').onclick=run(toggleRecord);
for(const event of ['pause','seeking','ratechange','ended'])audio.addEventListener(event,()=>{stop();paintRatings();draw();});audio.addEventListener('play',paintTransport);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();audio.pause();}});
setInterval(()=>{sample();$('media-time').textContent=fmt(audio.currentTime);},250);setInterval(remember,1000);setInterval(()=>{if(state.ready){if(state.replay)paintRatings();if(!audio.paused)draw();}},100);
$('confidence').onchange=sample;$('uncertain').onchange=sample;
for(const id of ['artist','style','note'])$(id).oninput=()=>{state.doc[id]=$(id).value;changed();remember();};
function setWindow(start,span){if(!state.ready)return;const d=state.doc.duration;state.viewSpan=Math.min(d,Math.max(Math.min(1,d),span));state.viewStart=Math.max(0,Math.min(d-state.viewSpan,start));draw();}
function zoom(factor,anchor=.5){const span=state.viewSpan||state.doc.duration,center=state.viewStart+span*anchor,next=Math.min(state.doc.duration,Math.max(Math.min(1,state.doc.duration),span*factor));setWindow(center-next*anchor,next);}
$('zoom-in').onclick=()=>{if(state.ready)zoom(.5);};$('zoom-out').onclick=()=>{if(state.ready)zoom(2);};$('zoom-reset').onclick=()=>setWindow(0,state.doc?.duration);
const timeline=$('timeline');timeline.addEventListener('wheel',e=>{if(!state.ready)return;e.preventDefault();const r=timeline.getBoundingClientRect();zoom(e.deltaY<0?.8:1.25,(e.clientX-r.left)/r.width);},{passive:false});
let drag=null;timeline.onpointerdown=e=>{if(!state.ready)return;timeline.setPointerCapture(e.pointerId);drag={x:e.clientX,start:state.viewStart,moved:false};};timeline.onpointermove=e=>{if(!drag)return;const delta=e.clientX-drag.x;if(Math.abs(delta)>4)drag.moved=true;if(drag.moved)setWindow(drag.start-delta/timeline.getBoundingClientRect().width*state.viewSpan,state.viewSpan);};timeline.onpointerup=e=>{if(!drag)return;if(!drag.moved){const r=timeline.getBoundingClientRect();audio.currentTime=Math.max(0,Math.min(state.doc.duration,state.viewStart+(e.clientX-r.left)/r.width*state.viewSpan));}drag=null;timeline.releasePointerCapture(e.pointerId);draw();};timeline.onpointercancel=()=>drag=null;
function draw(){const canvas=$('timeline'),c=canvas.getContext('2d'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);if(!state.doc?.duration)return;const span=state.viewSpan||state.doc.duration,start=state.viewStart||0,end=start+span,x=t=>(t-start)/span*w,y=v=>(h-24)/2-v*((h-24)/2-12);
 $('time-window').textContent=`${fmt(start)} — ${fmt(end)} / ${fmt(state.doc.duration)}`;$('zoom-in').disabled=span<=Math.min(1,state.doc.duration);$('zoom-out').disabled=span>=state.doc.duration;
 c.strokeStyle='#d9e1d9';c.beginPath();c.moveTo(0,y(0));c.lineTo(w,y(0));c.stroke();c.font='12px Segoe UI';c.fillStyle='#65756b';const ticks=Math.max(2,Math.floor(w/120));for(let i=0;i<=ticks;i++){c.textAlign=i===0?'left':i===ticks?'right':'center';c.fillText(fmt(start+span*i/ticks),w*i/ticks,h-5);}
 c.save();c.beginPath();c.rect(0,0,w,h-22);c.clip();for(const take of state.take?effectiveTracks(state.doc.takes):state.doc.takes)for(const [axis,color] of [['valence','#17694c'],['arousal','#c17a29']]){c.strokeStyle=color;c.lineWidth=2;c.beginPath();let active=false;for(const p of take.points){if(p[axis]===null){active=false;continue;}if(active)c.lineTo(x(p.time),y(p[axis]));else c.moveTo(x(p.time),y(p[axis]));active=true;}c.stroke();}
 c.setLineDash([4,4]);c.strokeStyle='#895782';for(const m of state.doc.transitions){c.beginPath();c.moveTo(x(m.time),0);c.lineTo(x(m.time),h-24);c.stroke();}c.setLineDash([]);c.strokeStyle='#273f36';c.lineWidth=1;c.beginPath();c.moveTo(x(audio.currentTime),0);c.lineTo(x(audio.currentTime),h-24);c.stroke();c.restore();
}
function renderDetails(){
 renderReview();
 $('markers').replaceChildren();for(const m of state.doc.transitions){const chip=document.createElement('button');chip.textContent=`◆ ${fmt(m.time)}`;chip.title=m.note||'编辑转折点';chip.onclick=()=>{sample();stop();audio.pause();state.editMarker=m;$('marker-time').value=m.time.toFixed(3);$('marker-time').max=state.doc.duration;$('marker-note').value=m.note;$('marker-dialog').showModal();};$('markers').append(chip);}
 $('takes').replaceChildren();state.doc.takes.forEach((take,i)=>{const row=document.createElement('div'),text=document.createElement('span'),remove=document.createElement('button');text.textContent=`${(take.axes||['valence','arousal']).map(a=>a==='valence'?'效价':'唤醒').join(' / ')} ${i+1}：${fmt(take.points[0]?.time||0)} — ${fmt(take.points.at(-1)?.time||0)} · ${take.points.length} 点`;remove.textContent='删除这段曲线';remove.onclick=()=>{if(state.take){notice('请先停止记录');return;}if(!confirm('删除这段有效曲线？不会自动恢复被覆盖内容；转折点保持不变。'))return;state.doc.takes=state.doc.takes.filter(t=>t!==take);for(const axis of take.axes||['valence','arousal'])state.doc.completed[axis]=false;changed();remember();renderDetails();draw();};row.append(text,remove);$('takes').append(row);});
}
function mark(){if(!state.ready)return;state.doc.transitions.push({id:uid(),time:audio.currentTime,note:''});changed();remember();renderDetails();draw();}
$('mark').onclick=mark;
document.addEventListener('keydown',e=>{
 const target=e.target,typing=target.isContentEditable||target.closest('textarea,select,input:not([type=range]):not([type=button])');
 if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||typing)return;
 if(e.code==='Space'){e.preventDefault();if(!state.ready||$('marker-dialog').open)return;run(state.replay?toggleReplay:toggleRecord)();}
 else if(e.key.toLowerCase()==='m'&&!$('marker-dialog').open){e.preventDefault();mark();}
},true);
$('save').onclick=run(async()=>{sample();stop(false);await save();});$('reload').onclick=run(refresh);$('search').oninput=renderLibrary;$('library-filter').onchange=renderLibrary;
$('reviewer').value=state.reviewer;$('reviewer').onchange=run(async()=>{if(state.loading){$('reviewer').value=state.reviewer;return;}endReplay();stop(false);audio.pause();try{if(state.dirty)await save();const next=$('reviewer').value.trim().toLowerCase();const result=await api('/api/library?reviewer='+encodeURIComponent(next));state.reviewer=next;state.songs=result.songs;state.token=result.token;state.doc=null;state.ready=false;state.dirty=false;localStorage.setItem('va-reviewer',next);$('workspace').hidden=true;$('empty').hidden=false;renderLibrary();}finally{$('reviewer').value=state.reviewer;}});
document.querySelectorAll('[data-export]').forEach(b=>b.onclick=run(async()=>{sample();stop(false);await save();const kind=b.dataset.export,r=await fetch(`/api/export?reviewer=${encodeURIComponent(state.reviewer)}&kind=${kind}`);if(!r.ok)throw Error((await r.json()).error);const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download=`continuous-${state.reviewer}-${kind}.${kind==='json'?'json':'csv'}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}));
window.addEventListener('beforeunload',e=>{if(state.dirty||state.take){remember();e.preventDefault();e.returnValue='';}});
refresh().catch(e=>notice(e.message,true));

function renderReview(){if(!state.doc)return;for(const axis of ['valence','arousal'])$('complete-'+axis).checked=!!state.doc.completed?.[axis];const points=state.doc.takes.flatMap(t=>t.points);$('review-summary').textContent=`${points.filter(p=>p.status==='annotated'&&!p.confidence).length} 个采样待评估把握程度`;}
$('apply-confidence').onclick=run(async()=>{
 if(!state.ready)return;if(state.take)throw Error('请先暂停记录，再调整已有标注。');
 const start=Number($('review-start').value),end=Number($('review-end').value),axis=$('review-axis').value;
 if(!$('review-start').value||!$('review-end').value||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end>state.doc.duration+.001||start>end)throw Error('请填写歌曲范围内的有效起止时间。');
 let count=0;for(const take of state.doc.takes){if(axis!=='both'&&!take.axes.includes(axis))continue;for(const p of take.points)if(p.time>=start&&p.time<=end&&p.status==='annotated'){p.confidence=$('review-confidence').value;count++;}}
 if(!count)throw Error('这个范围内没有可调整的有效标注。');changed();remember();renderReview();await save();notice(`已更新 ${count} 个采样的把握程度。`);
});
for(const axis of ['valence','arousal'])$('complete-'+axis).onchange=run(async()=>{
 const input=$('complete-'+axis);if(!state.ready||state.take){input.checked=!!state.doc?.completed?.[axis];throw Error('请先暂停记录，再确认完成。');}
 if(input.checked&&!state.doc.takes.some(t=>t.axes.includes(axis)&&t.points.some(p=>p.status==='annotated'))){input.checked=false;throw Error('请先记录这个维度，再标记完成。');}
 state.doc.completed[axis]=input.checked;changed();remember();await save();
});
const drop=$('drop-zone');
async function importFiles(files){if(state.importing)return;state.importing=true;let success=0;const failures=[];try{for(const [index,file] of Array.from(files).entries()){
 $('import-status').textContent=`正在导入 ${index+1} / ${files.length}：${file.name}`;
 if(!/\.(mp3|wav|flac|ogg|m4a|aac)$/i.test(file.name)||file.size>512*1024*1024||!file.size){failures.push(file.name+'：格式或大小不支持');continue;}
 try{const r=await fetch('/api/import?name='+encodeURIComponent(file.name),{method:'POST',headers:{'X-Annotation-Token':state.token,'Content-Type':'application/octet-stream'},body:file});const result=await r.json();if(!r.ok)throw Error(result.error);success++;}catch(e){failures.push(file.name+'：'+e.message);}
 }await refresh();$('import-status').textContent=`已导入 ${success} 首`+(failures.length?`，${failures.length} 首失败`:'，已加入音乐库');if(failures.length)notice(failures.join('；'),true);
 }finally{state.importing=false;$('audio-files').value='';}}
drop.onclick=()=>$('audio-files').click();drop.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('audio-files').click();}};$('audio-files').onchange=run(e=>importFiles(e.target.files));
const libraryPanel=document.querySelector('aside');
for(const event of ['dragenter','dragover'])libraryPanel.addEventListener(event,e=>{e.preventDefault();drop.classList.add('dragging');});libraryPanel.ondragleave=()=>drop.classList.remove('dragging');libraryPanel.ondrop=run(async e=>{e.preventDefault();drop.classList.remove('dragging');await importFiles(e.dataTransfer.files);});
window.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files'))e.preventDefault();});window.addEventListener('drop',e=>{if(e.dataTransfer.types.includes('Files'))e.preventDefault();});

$('marker-apply').onclick=run(async()=>{const m=state.editMarker,t=Number($('marker-time').value);if(!m)return;if(!$('marker-time').value||!Number.isFinite(t)||t<0||t>state.doc.duration)throw Error('转折时间超出歌曲范围');m.time=t;m.note=$('marker-note').value;changed();remember();renderDetails();draw();await save();$('marker-dialog').close();});
$('marker-delete').onclick=run(async()=>{if(!state.editMarker)return;state.doc.transitions=state.doc.transitions.filter(m=>m!==state.editMarker);changed();remember();renderDetails();draw();await save();$('marker-dialog').close();});

window.addEventListener('resize',draw);
