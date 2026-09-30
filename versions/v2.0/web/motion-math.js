"use strict";
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  root.StudioMotion=api;
})(typeof globalThis!=="undefined"?globalThis:this,function(){
  const labels={rise:"轻盈上升",fall:"缓慢下沉",orbit:"环流绕圈",spiral:"螺旋运动",expand:"向外扩散",
    gather:"向内汇聚",meteor:"流星掠过",wave:"波浪平移",turbulent:"湍流运动"};
  const bounds={speed:[0,.5],coherence:[0,1],turbulence:[0,1],direction_deg:[0,360],center_x:[0,1],
    center_y:[0,1],radius:[.05,.48],rotation:[-1,1],radial:[-1,1],pulse:[0,1],trail_seconds:[0,2]};
  const common={speed:.12,coherence:.92,turbulence:.08,direction_deg:0,center_x:.5,center_y:.5,
    radius:.28,rotation:1,radial:0,pulse:.2,trail_seconds:1};
  const presets={
    rise:{speed:.10,direction_deg:270,trail_seconds:.7},
    fall:{speed:.07,direction_deg:90,turbulence:.05,trail_seconds:1.2},
    orbit:{speed:.13,coherence:.98,turbulence:.03,radius:.30,pulse:.12,trail_seconds:1.4},
    spiral:{speed:.18,coherence:.94,turbulence:.07,radial:.25,pulse:.25,trail_seconds:1.5},
    expand:{speed:.19,radial:1,pulse:.45,trail_seconds:1},
    gather:{speed:.09,radial:-1,pulse:.15,trail_seconds:1.2},
    meteor:{speed:.30,direction_deg:45,coherence:.98,turbulence:.02,pulse:.35,trail_seconds:1},
    wave:{speed:.10,turbulence:.03,pulse:.15,trail_seconds:1},
    turbulent:{speed:.13,coherence:.25,turbulence:.75,pulse:.35,trail_seconds:.8}};
  const defaults={engine:"legacy",sensitivity:1,min_hold_seconds:6,transition_seconds:2,overrides:[]};
  const copy=value=>JSON.parse(JSON.stringify(value));
  const object=value=>value!==null&&typeof value==="object"&&!Array.isArray(value);
  function finite(value,name,low,high){
    if(typeof value!=="number"||!Number.isFinite(value)||value<low||value>high)
      throw new Error(name+" 必须为 "+low+"～"+high+" 之间的有限数值");
    return value;
  }
  function micros(value,name){
    if(!Number.isSafeInteger(value)||value<0)throw new Error(name+" 必须为非负整数微秒");
    return value;
  }
  function params(mode,input){
    if(!Object.hasOwn(labels,mode))throw new Error("未知运动模式");
    input=input??{};
    if(!object(input)||Object.keys(input).some(key=>!Object.hasOwn(bounds,key)))throw new Error("运动参数含未知字段");
    const result={...common,...presets[mode]};
    for(const [key,value] of Object.entries(input))result[key]=finite(value,key,...bounds[key]);
    if(![-1,1].includes(result.rotation))throw new Error("旋转方向只能为 -1 或 1");
    result.direction_deg%=360;return result;
  }
  function validateConfig(input,duration_us){
    input=input??{};
    if(!object(input)||Object.keys(input).some(key=>!Object.hasOwn(defaults,key)))throw new Error("运动配置含未知字段");
    const result={...copy(defaults),...copy(input)};
    if(!["legacy","flow-v1"].includes(result.engine))throw new Error("运动引擎必须为 legacy 或 flow-v1");
    finite(result.sensitivity,"运动敏感度",.5,2);finite(result.min_hold_seconds,"效果最短保持",4,12);
    finite(result.transition_seconds,"自动过渡",1,4);
    if(result.transition_seconds>result.min_hold_seconds)throw new Error("过渡时间不可超过最短保持时间");
    if(!Array.isArray(result.overrides)||result.overrides.length>1000)throw new Error("运动覆盖必须为列表，最多 1000 项");
    const ids=new Set(),occupied=[],allowed=["id","enabled","mode","start_us","end_us","transition_in_us",
      "transition_out_us","interpolation","note","params"];
    result.overrides=result.overrides.map(original=>{
      if(!object(original)||Object.keys(original).some(key=>!allowed.includes(key)))throw new Error("运动覆盖含未知字段");
      const item={...original};
      if(typeof item.id!=="string"||!item.id||item.id.length>100||ids.has(item.id))throw new Error("运动覆盖 ID 必须唯一且非空");
      ids.add(item.id);
      if(!Object.hasOwn(item,"enabled"))item.enabled=true;
      if(typeof item.enabled!=="boolean")throw new Error("运动覆盖 enabled 必须为布尔值");
      for(const key of ["start_us","end_us"])item[key]=micros(item[key],key);
      for(const key of ["transition_in_us","transition_out_us"])item[key]=micros(Object.hasOwn(item,key)?item[key]:2000000,key);
      if(item.end_us<=item.start_us)throw new Error("运动覆盖结束时间必须晚于开始时间");
      const left=item.start_us-item.transition_in_us,right=item.end_us+item.transition_out_us;
      if(left<0||duration_us!=null&&right>duration_us)throw new Error("运动覆盖及过渡范围必须位于歌曲内");
      if(!Object.hasOwn(item,"interpolation"))item.interpolation="smootherstep";
      if(!["linear","smoothstep","smootherstep"].includes(item.interpolation))throw new Error("未知运动过渡方式");
      item.params=params(item.mode,item.params);item.note=String(item.note??"").slice(0,2000);
      if(item.enabled){
        for(const previous of occupied){
          const a=previous.start_us-previous.transition_in_us,b=previous.end_us+previous.transition_out_us;
          if(left<b&&a<right||left===b&&item.transition_in_us===0&&previous.transition_out_us===0||
            right===a&&item.transition_out_us===0&&previous.transition_in_us===0)
            throw new Error("运动覆盖及过渡区间重叠，请先调整或禁用已有覆盖");
        }
        occupied.push(item);
      }
      return item;
    });
    return result;
  }
  function componentsMix(entries){
    const groups=new Map();
    for(const component of entries){
      if(component.weight<=1e-12)continue;
      if(!groups.has(component.mode))groups.set(component.mode,[]);
      groups.get(component.mode).push(component);
    }
    const output=[];
    for(const [mode,parts] of groups){
      const weight=parts.reduce((sum,c)=>sum+c.weight,0),result={};
      for(const key of Object.keys(bounds)){
        if(key==="rotation"){
          let winner=parts[0];for(const part of parts)if(part.weight>winner.weight)winner=part;
          result[key]=winner.params[key];
        }else if(key==="direction_deg"){
          const origin=parts[0].params[key];
          const value=parts.reduce((sum,c)=>sum+(origin+(((c.params[key]-origin+180)%360+360)%360)-180)*c.weight,0)/weight;
          result[key]=(value%360+360)%360;
        }else result[key]=parts.reduce((sum,c)=>sum+c.params[key]*c.weight,0)/weight;
      }
      output.push({mode,weight,params:result});
    }
    const total=output.reduce((sum,c)=>sum+c.weight,0);
    for(const c of output)c.weight/=total;
    return output;
  }
  function frame(components,source,reason){
    let dominant=components[0];for(const c of components)if(c.weight>dominant.weight)dominant=c;
    return {mode:dominant?.mode??null,source,reason,components};
  }
  function sampleFrame(plan,time_us,kind="effective"){
    const times=plan?.times_us,frames=plan?.[kind]||plan?.auto;
    if(!times?.length||frames?.length!==times.length)throw new Error("运动时间线为空或长度不匹配");
    finite(time_us,"采样时间",0,Number.MAX_SAFE_INTEGER);
    if(time_us<=times[0])return copy(frames[0]);
    if(time_us>=times[times.length-1])return copy(frames[frames.length-1]);
    let lo=0,hi=times.length-1;
    while(hi-lo>1){const mid=(lo+hi)>>1;if(times[mid]<=time_us)lo=mid;else hi=mid;}
    if(times[lo]===time_us)return copy(frames[lo]);
    const mix=(time_us-times[lo])/(times[hi]-times[lo]),a=frames[lo],b=frames[hi];
    return {...copy(a),...frame(componentsMix([...a.components.map(c=>({...c,weight:c.weight*(1-mix)})),
      ...b.components.map(c=>({...c,weight:c.weight*mix}))]),a.source,a.reason)};
  }
  const easing=(x,mode)=>mode==="linear"?x:mode==="smoothstep"?x*x*(3-2*x):x*x*x*(x*(x*6-15)+10);
  function influence(edit,time){
    if(time>=edit.start_us&&time<=edit.end_us)return 1;
    if(time<edit.start_us&&edit.transition_in_us&&time>=edit.start_us-edit.transition_in_us)
      return easing((time-edit.start_us+edit.transition_in_us)/edit.transition_in_us,edit.interpolation);
    if(time>edit.end_us&&edit.transition_out_us&&time<=edit.end_us+edit.transition_out_us)
      return 1-easing((time-edit.end_us)/edit.transition_out_us,edit.interpolation);
    return 0;
  }
  function applyOverrides(plan,config,duration_us){
    config=validateConfig(config,duration_us);
    const modes=Object.fromEntries(Object.keys(labels).map(mode=>[mode,{label:labels[mode],params:params(mode)}]));
    if(config.engine==="legacy")return {config,times_us:[],auto:[],effective:[],segments:[],modes};
    if(!plan?.times_us?.length||!plan.auto?.length)throw new Error("真实自动运动规划尚未准备，不能用旧规划代替。");
    const points=new Set(plan.times_us);
    for(const edit of config.overrides.filter(edit=>edit.enabled)){
      const left=edit.start_us-edit.transition_in_us,right=edit.end_us+edit.transition_out_us;
      for(const value of [left,edit.start_us,edit.end_us,right,
        ...(edit.transition_in_us===0?[edit.start_us-1]:[]),...(edit.transition_out_us===0?[edit.end_us+1]:[])])
        if(value>=0&&value<=duration_us)points.add(value);
      for(let i=1;i<8;i++){
        points.add(Math.floor(left+edit.transition_in_us*i/8));
        points.add(Math.floor(edit.end_us+edit.transition_out_us*i/8));
      }
    }
    const times_us=[...points].sort((a,b)=>a-b),auto=[],effective=[];
    for(const time of times_us){
      const baseline=sampleFrame(plan,time,"auto");auto.push(baseline);
      let value=copy(baseline);
      for(const edit of config.overrides){
        if(!edit.enabled)continue;const weight=influence(edit,time);if(weight<=1e-12)continue;
        const full=weight>=1-1e-12,reason=full?(edit.note||"人工指定："+labels[edit.mode]):"人工过渡："+labels[edit.mode];
        value={...(weight>=1?{}:copy(baseline)),...frame(componentsMix([...baseline.components.map(c=>({...c,weight:c.weight*(1-weight)})),
          {mode:edit.mode,weight,params:edit.params}]),full?"manual":"transition",reason),override_id:edit.id};
        break;
      }
      effective.push(value);
    }
    const segments=[];
    for(let i=0;i<times_us.length;i++){
      const value=effective[i],previous=segments[segments.length-1],end=times_us[i+1]??duration_us;
      if(end<=times_us[i])continue;
      if(previous&&previous.mode===value.mode&&previous.source===value.source)
        previous.end_us=end;
      else{
        segments.push({start_us:times_us[i],end_us:end,mode:value.mode,source:value.source,reason:value.reason});
      }
    }
    if(segments.length)segments[segments.length-1].end_us=duration_us;
    return {config,times_us,auto,effective,segments,modes};
  }
  return {validateConfig,applyOverrides,sampleFrame,labels,bounds,params,defaults:copy(defaults)};
});
