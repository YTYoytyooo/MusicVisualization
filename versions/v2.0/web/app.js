"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const state = {
    token: "", projects: [], data: null, edits: [], saved: "[]",
    time: 0, preview: null, selectedRender: null, busy: false,
    terminalJobs: new Set(), loadSerial: 0, polling: false, editingId: null,
    plotGeometry: [], handles: [], drag: null, loading: false,
    undoStack: [], redoStack: [], tx: null, confirmQueue: [], finalizing: null,
    confirmPromise: null, confirmEpoch: 0, requestSeq: 0, txCounter: 0, localFrame: null,
    debounceTimer: null, confirmedPreview: null, visualPlan: null, visualPreview: null,
    planSignature: null, planJob: null, planRequest: false, planSeq: 0,
    candidatePlans: new Map(), pageId: crypto.randomUUID(), temporaryConsumers: new Set(),consumerSeq:0,
    smoothing: {enabled:false,window_seconds:.5}, savedSmoothing: {enabled:false,window_seconds:.5},
    viewport: {start:0,end:null}, axisLocked:false, axisCache:{},
    compare:null, comparing:false, syncingMedia:false,jobsExpanded:false,
    motion:{engine:"legacy",sensitivity:1,min_hold_seconds:6,transition_seconds:2,overrides:[]},
    savedMotion:{engine:"legacy",sensitivity:1,min_hold_seconds:6,transition_seconds:2,overrides:[]},
    motionPlan:null,motionSignature:null,motionPreview:null,motionSeq:0,motionTimer:null,
    motionTx:null,motionFinal:null,motionEditingId:null,motionFrame:null,motionError:""
  };
  const fieldLabels = {
    valence: "Valence 效价", arousal: "Arousal 唤醒度",
    hue_base: "基础色相", hue_range: "色相范围", saturation: "饱和度",
    brightness: "画面亮度", particle_count: "粒子数量",
    particle_speed: "粒子速度", field_turbulence: "湍流强度",
    trail_length: "粒子拖尾长度（帧）"
  };
  const operations = { point_target: "单点目标", interval_set: "区间固定", interval_offset: "区间偏移" };
  const fieldRanges = {
    valence: [-1, 1], arousal: [-1, 1], hue_base: [0, 360],
    hue_range: [20, 120], saturation: [.3, 1], brightness: [.1, .9],
    particle_count: [50, 500], particle_speed: [.5, 8],
    field_turbulence: [0, 1], trail_length: [5, 60]
  };
  const terminalStates = ["success", "succeeded", "completed", "complete", "done", "diagnostic_only", "failed", "cancelled", "canceled", "interrupted"];
  const statusLabels = {queued:"排队中",running:"处理中",cancelling:"取消中",succeeded:"已完成",failed:"失败",cancelled:"已取消",interrupted:"已中断",diagnostic_only:"仅诊断输出"};
  const stageLabels = {queued:"等待执行",starting:"启动任务",loading:"读取音频",embedding:"提取 CLAP 特征",inference:"模型推理",saving:"保存分析",planning:"规划视觉参数","audio-motion-features":"提取音频运动特征",rendering:"生成画面",encoding:"合并音视频",succeeded:"已完成",analyzed:"分析完成"};
  const jobLabels = {analyze:"音频分析",plan:"视觉规划",preview:"片段预览",render:"视频生成"};
  const clone = value => JSON.parse(JSON.stringify(value));
  const defaultMotion=()=>({engine:"legacy",sensitivity:1,min_hold_seconds:6,transition_seconds:2,overrides:[]});
  const dirty = () => JSON.stringify(state.edits) !== state.saved ||
    JSON.stringify(state.smoothing) !== JSON.stringify(state.savedSmoothing)||
    JSON.stringify(state.motion)!==JSON.stringify(state.savedMotion);
  const projectId = () => state.data.project.id;
  const revisionId = () => state.data.revision.id;
  const urlId = value => encodeURIComponent(value);
  const projectRoute = () => "/api/projects/" + urlId(projectId());
  const videoOffset = () => Number(state.selectedRender?.actual_start ?? state.selectedRender?.start ?? state.selectedRender?.settings?.start ?? 0);
  const draftKey = (id, revision) => "musicvisualization-studio:draft:" + id + ":" + revision;
  const HISTORY_LIMIT = 50;
  const pending = () => Boolean(state.localFrame !== null || state.tx?.changed ||
    state.confirmQueue.length || state.finalizing||state.motionTx?.changed||state.motionFinal||state.motionFrame!==null);
  const durationUs = () => Math.round(state.data.project.duration * 1e6);
  const visibleEdits = () => state.tx?.lastValidEdits || (state.tx?.valid ? state.tx.edits :
    state.confirmQueue.length ? state.confirmQueue[state.confirmQueue.length-1].edits :
    state.finalizing?.edits || state.edits);
  const emotionSignature = edits => JSON.stringify([state.smoothing,edits.filter(e => e.layer === "emotion" && e.enabled !== false)
    .map(e => [e.id,e.field,e.operation,e.value,e.time_us ?? null,e.start_us ?? null,e.end_us ?? null,
      e.transition_in_us || 0,e.transition_out_us || 0,e.interpolation || "linear"])]);
  function pushHistory(stack, edits, smoothing = state.smoothing,motion=state.motion) {
    stack.push({edits:clone(edits),smoothing:clone(smoothing),motion:clone(motion)});
    if (stack.length > HISTORY_LIMIT) stack.splice(0, stack.length - HISTORY_LIMIT);
  }
  function localEmotion(edits) {
    const source = state.data?.timeline?.source_raw;
    const step = state.data?.timeline?.step_us;
    if (!Array.isArray(source) || !Number.isFinite(step))
      throw new Error("完整原始采样尚未加载，不能用显示用抽样数据代替。请刷新项目。");
    if (!window.StudioTimeline) throw new Error("本地曲线计算模块未加载，请刷新页面。");
    return StudioTimeline.emotionPreview(source, step, durationUs(), edits, state.smoothing);
  }
  function transactionContext(tx) {
    return state.data && !state.loading && tx.loadSerial === state.loadSerial &&
      tx.project === projectId() && tx.revision === revisionId();
  }
  function adjustmentStatus() {
    const el = $("adjustment-state"), tx = state.tx;
    const status = tx?.error ? "error" : state.localFrame !== null || state.finalizing || state.confirmQueue.length ||
      (tx?.changed && tx.remoteStatus !== "valid") ? "pending" : "ready";
    el.dataset.state = status;
    el.textContent = tx?.error || (tx?.remoteStatus === "planning" || state.finalizing?.remoteStatus === "planning"
      ? "已有视觉修改：正在准备候选情绪的真实规划，完成后自动继续校验"
      : state.finalizing || state.confirmQueue.length
      ? "正在按顺序确认调整 · 未确认数据不会保存到草稿"
      : state.localFrame !== null ? "正在计算本地即时预览…"
      : tx?.changed ? (tx.remoteStatus === "valid"
        ? "后台校验通过 · 离开输入框 / Enter 确认这一步"
        : "本地即时预览 · 等待后台校验")
      : "就绪 · 输入即时预览，每个输入会话计为一步");
    $("cancel-adjustment").hidden = !pending();
    const edit = tx?.candidate || visibleEdits().find(e => e.id === state.editingId);
    if (edit) {
      const start = (edit.time_us ?? edit.start_us) - (edit.transition_in_us || 0);
      const end = (edit.time_us ?? edit.end_us) + (edit.transition_out_us || 0);
      $("impact-range").textContent = "影响范围 " + clock(Math.max(0,start)/1e6) + " — " +
        clock(Math.min(durationUs(),end)/1e6) + " · " + (edit.interpolation || "linear");
    } else $("impact-range").textContent = "黄色浅色区域标出本次调整的影响范围。";
  }
  function beginTransaction(field) {
    if (!state.data || state.loading || state.busy) return null;
    if(state.motionTx?.changed||state.motionFinal)throw new Error("请先确认或取消运动调整，再修改数值。");
    if (state.tx) {
      if (state.tx.field === field || !state.tx.valid) return state.tx;
      queueCurrentTransaction();
    }
    const prior = state.confirmQueue[state.confirmQueue.length-1] || state.finalizing;
    const base = clone(prior?.edits || state.edits);
    const editId = state.editingId || prior?.editId || "edit-" + crypto.randomUUID();
    state.tx = {
      id: ++state.txCounter, field, editId, base, edits: base,
      project: projectId(), revision: revisionId(), loadSerial: state.loadSerial,
      valid: true, changed: false, error: "", remoteStatus: "idle",
      lastAcceptedEdits: clone(state.edits), smoothing:clone(state.smoothing)
    };
    return state.tx;
  }
  function evaluateInput() {
    const tx = state.tx;
    if (!tx || !transactionContext(tx)) return;
    state.requestSeq++;
    clearTimeout(state.debounceTimer);
    try {
      for (const id of ["edit-start","edit-value","edit-in","edit-out",
        ...($("edit-operation").value === "point_target" ? [] : ["edit-end"])]) {
        if ($(id).value.trim() === "" || !Number.isFinite(Number($(id).value)))
          throw new Error("请输入完整且有限的数值；当前保留最后有效曲线。");
      }
      const candidate = getEdit();
      tx.candidate = candidate;
      const next = tx.base.some(e => e.id === tx.editId)
        ? tx.base.map(e => e.id === tx.editId ? candidate : e) : [...tx.base, candidate];
      const normalized = StudioTimeline.validateEdits(next, durationUs());
      const edits = Array.isArray(normalized) ? normalized : next;
      const nextSignature=emotionSignature(edits);
      if(tx.planSignature && tx.planSignature!==nextSignature) releaseTemporary(tx);
      tx.planSignature=nextSignature;
      const emotion = localEmotion(edits);
      let visual = null;
      if (candidate.layer === "visual") {
        if (!state.visualPlan || state.planSignature !== emotionSignature(edits))
          throw new Error("真实视觉规划基线尚未准备好。请等待规划完成，不能用模拟曲线代替。");
        visual = StudioTimeline.visualPreview(state.visualPlan, durationUs(), edits);
      }
      tx.edits = edits; tx.valid = true; tx.error = "";
      tx.lastValidEdits = clone(edits);
      tx.changed = JSON.stringify(tx.base) !== JSON.stringify(edits);
      tx.remoteStatus = "pending";
      state.preview = emotion;
      scheduleMotionPlan(edits,state.motion);
      if (visual) state.visualPreview = visual;
      drawTimeline();
      state.debounceTimer = setTimeout(() => validateTemporary(tx), 200);
    } catch (error) {
      releaseTemporary(tx);
      tx.valid = false; tx.changed = true; tx.error = error.message; tx.remoteStatus = "error";
    }
    updateDraft();
  }
  function scheduleInput(field) {
    if (!state.data || state.loading || state.busy) return;
    try { beginTransaction(field); }
    catch (error) {
      if (state.tx) { state.tx.valid = false; state.tx.error = error.message; state.tx.changed = true; }
      updateDraft(); return;
    }
    if (state.localFrame !== null) cancelAnimationFrame(state.localFrame);
    state.localFrame = requestAnimationFrame(() => {
      state.localFrame = null;
      evaluateInput();
    });
    // A scheduled frame is already pending work. Do not expose the old "ready"
    // state to a blur/save action or an accessibility client before rAF executes.
    updateDraft();
  }
  function flushInput() {
    if (state.localFrame !== null) {
      cancelAnimationFrame(state.localFrame); state.localFrame = null; evaluateInput();
    }
  }
  async function validateTemporary(tx) {
    if (state.tx !== tx || !tx.valid || !tx.changed || !transactionContext(tx)) return;
    const seq = ++state.requestSeq, edits = clone(tx.edits), snapshot = JSON.stringify(edits);
    const current = () => state.tx === tx && seq === state.requestSeq && transactionContext(tx) &&
      snapshot === JSON.stringify(tx.edits);
    tx.remoteStatus = "validating"; updateDraft();
    try {
      const plan = await candidatePlanFor(edits, tx);
      if (!current()) return;
      tx.remoteStatus = "validating"; updateDraft();
      await api("/api/projects/" + urlId(tx.project) + "/preview",
        {base_revision:tx.revision, edits, smoothing:tx.smoothing});
      if (!current()) return;
      tx.remoteStatus = "valid"; tx.error = "";
      tx.lastAcceptedEdits = clone(edits);
      if (plan) adoptVisualPlan(plan, edits);
    } catch (error) {
      if (!current()) return;
      tx.valid = false; tx.error = error.message; tx.remoteStatus = "error";
      // Rejection must not leave the rejected candidate displayed as accepted.
      // Keep the last server-accepted curve for this input session.
      tx.lastValidEdits = clone(tx.lastAcceptedEdits || state.edits);
      state.preview = localEmotion(tx.lastValidEdits);
      scheduleMotionPlan(tx.lastValidEdits,state.motion);
    }
    updateDraft(); drawTimeline();
  }
  function queueCurrentTransaction() {
    flushInput();
    const tx = state.tx;
    if (!tx) return;
    if (!tx.changed) { state.tx = null; updateDraft(); return; }
    if (!tx.valid) throw new Error(tx.error || "请先修正或取消当前调整。");
    state.requestSeq++; clearTimeout(state.debounceTimer);
    state.tx = null;
    state.confirmQueue.push(clone(tx));
    startConfirmationWorker();
    updateDraft();
  }
  function startConfirmationWorker() {
    if (state.confirmPromise || !state.confirmQueue.length) return;
    const serial = state.loadSerial, epoch = state.confirmEpoch;
    state.confirmPromise = (async () => {
      while (state.confirmQueue.length && serial === state.loadSerial && epoch === state.confirmEpoch) {
        const tx = state.confirmQueue.shift();
        state.finalizing = tx; updateDraft();
        try {
          const plan = await candidatePlanFor(tx.edits, tx, "confirmed");
          if (!transactionContext(tx) || epoch !== state.confirmEpoch) return;
          tx.remoteStatus = "validating"; updateDraft();
          await api("/api/projects/" + urlId(tx.project) + "/preview",
            {base_revision:tx.revision, edits:tx.edits, smoothing:tx.smoothing});
          if (!transactionContext(tx) || epoch !== state.confirmEpoch) return;
          const oldEmotion = emotionSignature(state.edits);
          if (JSON.stringify(state.edits) !== JSON.stringify(tx.edits)) {
            pushHistory(state.undoStack, state.edits); state.redoStack = [];
          }
          state.edits = clone(tx.edits);
          state.confirmedPreview = localEmotion(state.edits);
          if (!state.editingId) state.editingId = tx.editId;
          persistDraft(); renderEdits();
          if (plan) adoptVisualPlan(plan, state.edits);
          else if (oldEmotion !== emotionSignature(state.edits)) invalidateVisualPlan();
          if (!state.tx?.changed && !state.confirmQueue.length) state.preview = state.confirmedPreview;
          scheduleMotionPlan(state.edits,state.motion);
        } catch (error) {
          if (!transactionContext(tx) || epoch !== state.confirmEpoch) return;
          state.confirmQueue = []; state.requestSeq++;
          if (state.localFrame !== null) cancelAnimationFrame(state.localFrame);
          state.localFrame = null; clearTimeout(state.debounceTimer);
          state.tx = {...tx, base:clone(state.edits), lastValidEdits:clone(state.edits),
            lastAcceptedEdits:clone(state.edits), valid:false, error:
            "调整未确认，后续待确认输入已取消：" + error.message, remoteStatus:"error"};
          state.preview = state.confirmedPreview;
          break;
        } finally {
          releaseTemporary(tx);
          if (serial === state.loadSerial && epoch === state.confirmEpoch) {
            state.finalizing = null; updateDraft(); drawTimeline();
          }
        }
      }
    })().finally(() => {
      if (serial !== state.loadSerial || epoch !== state.confirmEpoch) return;
      state.confirmPromise = null; updateDraft();
      if (state.confirmQueue.length) startConfirmationWorker();
    });
  }
  async function finishPending() {
    await finishMotion();
    queueCurrentTransaction();
    while (state.confirmPromise) await state.confirmPromise;
    if (state.tx?.error) throw new Error(state.tx.error);
  }
  function cancelAdjustment(all = false) {
    if(all)cancelMotion();
    const cancelQueued = all || (!state.tx && (state.confirmQueue.length || state.finalizing));
    state.requestSeq++; clearTimeout(state.debounceTimer);
    if (state.localFrame !== null) cancelAnimationFrame(state.localFrame);
    state.localFrame = null;
    if(state.tx)releaseTemporary(state.tx);
    state.tx = null;
    if (cancelQueued) {
      for(const tx of state.confirmQueue)releaseTemporary(tx);
      if(state.finalizing)releaseTemporary(state.finalizing);
      state.confirmEpoch++;
      state.confirmQueue = []; state.finalizing = null; state.confirmPromise = null;
    }
    const base = state.confirmQueue[state.confirmQueue.length-1]?.edits || state.finalizing?.edits || state.edits;
    if (state.data) {
      try { state.preview = localEmotion(base); } catch { state.preview = state.confirmedPreview; }
      const edit = base.find(e => e.id === state.editingId);
      if (edit) fillEditForm(edit);
    }
    updateDraft(); drawTimeline();
    if(state.data&&!state.loading)scheduleMotionPlan(base,state.motion);
  }
  function invalidateVisualPlan() {
    state.planSeq++; state.planRequest = false; state.planJob = null;
    state.visualPlan = null; state.visualPreview = null; state.planSignature = null;
    if (state.visualRequested) requestVisualPlan();
  }
  function adoptVisualPlan(plan, edits) {
    state.planSeq++; state.planRequest = false; state.planJob = null;
    state.visualPlan = plan; state.planSignature = emotionSignature(edits);
    state.visualPreview = StudioTimeline.visualPreview(plan, durationUs(), edits);
    $("visual-plan-state").textContent = "真实规划基线与后端校验已就绪 · " + (plan.key || "");
  }
  function planRecordCurrent(record) {
    return state.data && !state.loading && record.loadSerial === state.loadSerial &&
      record.project === projectId() && record.revision === revisionId();
  }
  function consumerId(context,purpose) {
    return state.pageId+":"+context.loadSerial+":"+context.project+":"+(context.id||"operation")+":"+purpose;
  }
  function releaseConsumer(id) {
    if(!id||!state.temporaryConsumers.delete(id))return;
    fetch("/api/plan-release",{method:"POST",keepalive:true,
      headers:{"Content-Type":"application/json","X-Studio-Token":state.token},
      body:JSON.stringify({consumer_id:id,consumer_seq:++state.consumerSeq})}).catch(()=>{});
  }
  function releaseTemporary(context) {
    releaseConsumer(consumerId(context,"temporary"));
  }
  function prunePlanCache() {
    const completed=[...state.candidatePlans.values()].filter(r=>r.finished&&r.plan)
      .sort((a,b)=>(a.lastUsed||0)-(b.lastUsed||0));
    let count=completed.length;
    for(const record of completed){
      if(count<=8)break;
      if(record.users||record.plan===state.visualPlan)continue;
      state.candidatePlans.delete(record.key);count--;
    }
  }
  function rejectCandidatePlan(record, error) {
    if (record.finished) return;
    record.finished = true;
    state.candidatePlans.delete(record.key);
    record.reject(error);
  }
  async function fetchCandidatePlan(record) {
    if (record.inFlight || record.finished) return;
    if (!planRecordCurrent(record)) {
      rejectCandidatePlan(record, new Error("项目已切换，旧候选规划已取消。")); return;
    }
    record.inFlight = true;
    try {
      const result = await api("/api/projects/" + urlId(record.project) + "/visual-plan",
        {base_revision:record.revision, edits:record.edits,smoothing:record.smoothing,
          consumer_id:record.consumerId,consumer_seq:record.consumerSeq,purpose:record.purpose});
      if (!planRecordCurrent(record)) {
        rejectCandidatePlan(record, new Error("项目已切换，旧候选规划已取消。")); return;
      }
      if(result.status==="superseded")throw new Error("旧候选已被新的输入替代。");
      if (result.status === "ready") {
        record.plan = result.plan; record.finished = true; record.jobId = null;
        record.resolve(result.plan);
      } else if (result.status === "queued" && result.job_id) record.jobId = result.job_id;
      else throw new Error("候选视觉规划返回了无法识别的状态。");
    } catch (error) { rejectCandidatePlan(record, error); }
    finally { record.inFlight = false; }
  }
  async function candidatePlanFor(edits, context, purpose="temporary") {
    // Ordinary emotion-only previews never wait for MCTS.
    if (!edits.some(e => e.layer === "visual" && e.enabled !== false)) return null;
    const signature = emotionSignature(edits);
    if (purpose==="temporary" && state.visualPlan && state.planSignature === signature) return state.visualPlan;
    if (!transactionContext(context)) throw new Error("候选调整的项目上下文已过期。");
    context.remoteStatus = "planning";
    $("visual-plan-state").textContent = "已有视觉修改：正在准备候选情绪的真实规划，完成后自动校验。";
    updateDraft();
    const consumer=consumerId(context,purpose);
    if(purpose==="temporary")state.temporaryConsumers.add(consumer);
    const key = context.loadSerial + ":" + context.project + ":" + context.revision + ":" + signature+":"+consumer;
    let record = state.candidatePlans.get(key);
    if (!record) {
      record = {key, project:context.project, revision:context.revision,
        loadSerial:context.loadSerial, signature, edits:clone(edits),
        smoothing:clone(context.smoothing||state.smoothing), consumerId:consumer,consumerSeq:++state.consumerSeq,purpose,
        plan:null, jobId:null, inFlight:false, finished:false,users:0,lastUsed:Date.now()};
      record.promise = new Promise((resolve,reject) => { record.resolve=resolve; record.reject=reject; });
      state.candidatePlans.set(key,record);
      fetchCandidatePlan(record);
    }
    record.users++;record.lastUsed=Date.now();
    try{return await record.promise;}
    finally{record.users--;record.lastUsed=Date.now();prunePlanCache();}
  }
  function discardCandidatePlans() {
    for(const id of [...state.temporaryConsumers])releaseConsumer(id);
    for (const record of state.candidatePlans.values())
      if (!record.finished) rejectCandidatePlan(record,new Error("项目已切换，旧候选规划已取消。"));
    state.candidatePlans.clear();
  }
  async function requestVisualPlan() {
    if (!state.data || state.loading || state.planRequest) return;
    state.visualRequested = true;
    const signature = emotionSignature(state.edits);
    if (state.visualPlan && state.planSignature === signature) { drawVisual(); return; }
    const serial = state.loadSerial, seq = ++state.planSeq, id = projectId(), rev = revisionId();
    state.planRequest = true;
    $("visual-plan-state").textContent = "正在准备真实 MCTS 规划基线…";
    try {
      const result = await api(projectRoute() + "/visual-plan", {base_revision:rev, edits:clone(state.edits),
        smoothing:clone(state.smoothing),consumer_id:state.pageId+":"+serial+":baseline",consumer_seq:++state.consumerSeq,purpose:"confirmed"});
      if (serial !== state.loadSerial || seq !== state.planSeq || id !== projectId() ||
          rev !== revisionId() || signature !== emotionSignature(state.edits)) return;
      if (result.status === "ready") {
        state.visualPlan = result.plan; state.planSignature = signature; state.planJob = null;
        state.visualPreview = StudioTimeline.visualPreview(result.plan, durationUs(), visibleEdits());
        $("visual-plan-state").textContent = "真实规划基线已就绪 · " + (result.plan.key || "");
        if (state.tx?.candidate?.layer === "visual") scheduleInput(state.tx.field);
      } else {
        state.planJob = result.job_id;
        $("visual-plan-state").textContent = "真实规划排队 / 计算中，完成后自动读取。";
      }
    } catch (error) {
      if (serial === state.loadSerial && seq === state.planSeq)
        $("visual-plan-state").textContent = "规划暂不可用：" + error.message;
    } finally {
      if (serial === state.loadSerial && seq === state.planSeq) state.planRequest = false;
      drawVisual();
    }
  }
  function persistDraft() {
    if (!state.data) return;
    const key = draftKey(projectId(), revisionId());
    try {
      if (dirty()) localStorage.setItem(key, JSON.stringify({
        project_id: projectId(), base_revision: revisionId(), edits: state.edits,
        smoothing:state.smoothing,motion:state.motion,saved_at: new Date().toISOString()
      }));
      else localStorage.removeItem(key);
    } catch {
      notice("浏览器无法保存本地草稿（存储已满或受限制）。请及时保存为正式修订，关闭页面可能丢失草稿。", true);
    }
  }
  function offerSavedDraft() {
    if (state.loading || !state.data) return;
    const id = projectId(), revision = revisionId(), key = draftKey(id, revision), serial = state.loadSerial;
    const currentContext = () => !state.loading && !state.busy && !pending() && state.data &&
      state.loadSerial === serial && projectId() === id && revisionId() === revision;
    let stored, draft, invalid = false;
    try {
      stored = localStorage.getItem(key);
      if (!stored) return;
      draft = JSON.parse(stored);
      if (!draft || draft.project_id !== id || draft.base_revision !== revision || !Array.isArray(draft.edits))
        throw new Error("Invalid draft");
      if (JSON.stringify(draft.edits) === state.saved &&
          JSON.stringify(draft.smoothing||{enabled:false,window_seconds:.5})===JSON.stringify(state.savedSmoothing)&&
          JSON.stringify(draft.motion||defaultMotion())===JSON.stringify(state.savedMotion)) {
        localStorage.removeItem(key); return;
      }
    } catch { invalid = true; }
    const el = $("notice");
    el.replaceChildren(node("span", invalid
      ? "此修订的本地草稿无法读取。后台数据未被修改；可以丢弃损坏草稿后继续。"
      : "发现此项目 / 修订的未保存本地草稿。恢复前将由服务器校验，不会覆盖原始预测。"));
    el.className = invalid ? "error" : ""; el.hidden = false;
    if (!invalid) {
      const restore = node("button", "恢复草稿");
      restore.onclick = run(async () => {
        if (!currentContext()) return;
        await applyDraft(draft.edits,"edit",draft.smoothing||{enabled:false,window_seconds:.5},draft.motion||defaultMotion());
        notice("已恢复并校验本地草稿。请保存为新修订后生成视频。");
      });
      el.append(restore);
    }
    const discard = node("button", invalid ? "丢弃损坏草稿" : "丢弃本地草稿");
    discard.onclick = () => {
      if (!currentContext()) return;
      try { localStorage.removeItem(key); notice("已丢弃此浏览器中的草稿，后台修订未改动。"); }
      catch { notice("浏览器不允许清除存储。可以继续编辑并保存正式修订。", true); }
    };
    el.append(discard);
  }

  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function notice(message, error = false) {
    const el = $("notice");
    el.replaceChildren(node("span", message));
    const close = node("button", "×");
    close.setAttribute("aria-label", "关闭通知");
    close.onclick = () => { el.hidden = true; };
    el.append(close);
    el.className = error ? "error" : "";
    el.hidden = false;
  }
  async function api(path, body) {
    if(body!==undefined && /\/(?:preview|previews|revisions|visual-plan)$/.test(path) &&
        body.smoothing===undefined)body={...body,smoothing:clone(state.smoothing)};
    if(body!==undefined&&/\/(?:preview|previews|revisions|motion-plan)$/.test(path)&&body.motion===undefined)
      body={...body,motion:clone(state.motion)};
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json", "X-Studio-Token": state.token },
      body: body === undefined ? undefined : JSON.stringify(body)
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error("本地服务返回了无法识别的响应（" + response.status + "）。请检查服务日志。"); }
    if (!response.ok) {
      const msg = result.error || result.detail || result.message || ("请求失败（" + response.status + "）");
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return result;
  }
  function run(fn) {
    return async event => {
      if (event) event.preventDefault();
      try { await fn(event); }
      catch (error) { notice(error.message || String(error), true); }
    };
  }
  function clock(seconds) {
    const ms = Math.max(0, Math.round(Number(seconds || 0) * 1000));
    return String(Math.floor(ms / 60000)).padStart(2, "0") + ":" +
      String(Math.floor(ms / 1000) % 60).padStart(2, "0") + "." + String(ms % 1000).padStart(3, "0");
  }
  function motionAutoSignature(edits,motion){
    return emotionSignature(edits)+JSON.stringify([motion.engine,motion.sensitivity,motion.min_hold_seconds,motion.transition_seconds]);
  }
  const shownMotion=()=>state.motionTx?.valid?state.motionTx.config:state.motion;
  function adoptMotionPlan(plan,edits,motion){
    state.motionPlan=plan;state.motionSignature=motionAutoSignature(edits,motion);
    state.motionPreview=plan;state.motionError="";
  }
  function scheduleMotionPlan(edits=visibleEdits(),motion=shownMotion()){
    if(!state.data||state.loading)return;
    clearTimeout(state.motionTimer);state.motionSeq++;
    const signature=motionAutoSignature(edits,motion);
    if(motion.engine==="legacy"){
      state.motionPreview={times_us:[],auto:[],effective:[],segments:[]};drawMotion();return;
    }
    if(state.motionPlan&&signature===state.motionSignature){
      try{state.motionPreview=StudioMotion.applyOverrides(state.motionPlan,motion,durationUs());state.motionError="";}
      catch(error){state.motionError=error.message;}
      drawMotion();return;
    }
    state.motionPreview=null;state.motionError="";drawMotion();
    const snapshot=clone(edits),config=clone(motion),serial=state.loadSerial,seq=state.motionSeq;
    const id=projectId(),rev=revisionId();
    state.motionTimer=setTimeout(async()=>{
      try{
        const plan=await api("/api/projects/"+urlId(id)+"/motion-plan",
          {base_revision:rev,edits:snapshot,smoothing:clone(state.smoothing),motion:config});
        if(seq!==state.motionSeq||serial!==state.loadSerial||state.loading||
          id!==projectId()||rev!==revisionId()||signature!==motionAutoSignature(visibleEdits(),shownMotion()))return;
        adoptMotionPlan(plan,snapshot,config);drawMotion();
      }catch(error){
        if(seq===state.motionSeq&&serial===state.loadSerial){state.motionError=error.message;drawMotion();}
      }
    },200);
  }
  function updateMotionStatus(){
    const tx=state.motionTx,el=$("motion-adjustment-state");
    el.dataset.state=tx?.error?"error":state.motionFinal||tx?.changed?"pending":"ready";
    el.textContent=tx?.error|| (state.motionFinal?"正在确认运动调整…":tx?.changed?
      "本地运动预览 · 离开输入框 / Enter 确认一步":"就绪 · 输入即时更新运动轨道");
  }
  function fillMotionForm(){
    const config=state.motion;
    $("motion-engine").value=config.engine;$("motion-sensitivity").value=config.sensitivity;
    $("motion-hold").value=config.min_hold_seconds;$("motion-transition").value=config.transition_seconds;
    const edit=config.overrides.find(e=>e.id===state.motionEditingId);
    $("motion-override-fields").hidden=!edit;$("motion-end-edit").hidden=!edit;
    if(!edit){state.motionEditingId=null;return;}
    $("motion-mode").value=edit.mode;
    for(const [id,key] of [["motion-start","start_us"],["motion-end","end_us"],["motion-in","transition_in_us"],["motion-out","transition_out_us"]])
      $(id).value=edit[key]/1e6;
    $("motion-interpolation").value=edit.interpolation||"smootherstep";$("motion-note").value=edit.note||"";
    for(const key of Object.keys(StudioMotion.bounds))$("motion-param-"+key).value=edit.params?.[key]??"";
  }
  function readMotionForm(){
    const number=id=>{
      const raw=$(id).value;if(raw.trim()===""||!Number.isFinite(Number(raw)))throw new Error("运动输入需为完整的有限数值。");
      return Number(raw);
    };
    const config=clone(state.motionTx?.base||state.motion);
    config.engine=$("motion-engine").value;config.sensitivity=number("motion-sensitivity");
    config.min_hold_seconds=number("motion-hold");config.transition_seconds=number("motion-transition");
    if(state.motionEditingId){
      const params={};
      for(const key of Object.keys(StudioMotion.bounds)){
        if($("motion-param-"+key).value.trim()!=="")params[key]=number("motion-param-"+key);
      }
      const previous=config.overrides.find(e=>e.id===state.motionEditingId);
      const edit={id:state.motionEditingId,mode:$("motion-mode").value,enabled:previous?.enabled!==false,
        start_us:Math.round(number("motion-start")*1e6),end_us:Math.round(number("motion-end")*1e6),
        transition_in_us:Math.round(number("motion-in")*1e6),transition_out_us:Math.round(number("motion-out")*1e6),
        interpolation:$("motion-interpolation").value,note:$("motion-note").value,params};
      config.overrides=previous?config.overrides.map(e=>e.id===edit.id?edit:e):[...config.overrides,edit];
    }
    return StudioMotion.validateConfig(config,durationUs());
  }
  function motionInput(){
    if(!state.data||state.loading||state.busy||state.motionFinal)return;
    if(state.tx?.changed||state.confirmQueue.length||state.finalizing){
      notice("请先确认或取消当前数值调整，再修改运动。",true);return;
    }
    if(!state.motionTx)state.motionTx={base:clone(state.motion),config:clone(state.motion),valid:true,changed:false,
      serial:state.loadSerial,id:projectId(),revision:revisionId()};
    if(state.motionFrame!==null)cancelAnimationFrame(state.motionFrame);
    state.motionFrame=requestAnimationFrame(()=>{state.motionFrame=null;evaluateMotion();});
  }
  function evaluateMotion(){
    const tx=state.motionTx;if(!tx)return;
    try{
      tx.config=readMotionForm();tx.valid=true;tx.error="";
      tx.changed=JSON.stringify(tx.base)!==JSON.stringify(tx.config);
      scheduleMotionPlan(state.edits,tx.config);
    }catch(error){tx.valid=false;tx.changed=true;tx.error=error.message;}
    updateDraft();
  }
  function cancelMotion(){
    state.motionSeq++;clearTimeout(state.motionTimer);
    if(state.motionFrame!==null)cancelAnimationFrame(state.motionFrame);
    state.motionFrame=null;state.motionTx=null;state.motionError="";
    if(state.data){fillMotionForm();scheduleMotionPlan(state.edits,state.motion);}
    updateDraft();drawMotion();
  }
  async function finishMotion(){
    if(state.motionFrame!==null){cancelAnimationFrame(state.motionFrame);state.motionFrame=null;evaluateMotion();}
    if(state.motionFinal){await state.motionFinal;return;}
    const tx=state.motionTx;
    if(!tx?.changed){state.motionTx=null;return;}
    if(!tx.valid)throw new Error(tx.error||"请先修正运动设置。");
    state.motionTx=null;
    state.motionFinal=(async()=>{
      try{
        await applyDraft(state.edits,"edit",state.smoothing,tx.config);
      }catch(error){
        if(tx.serial===state.loadSerial&&!state.loading){
          state.motionTx={...tx,valid:false,error:"运动调整未确认："+error.message,changed:true};
          scheduleMotionPlan(state.edits,state.motion);
        }
        throw error;
      }finally{state.motionFinal=null;updateDraft();}
    })();
    await state.motionFinal;
  }
  function renderMotionEdits(){
    const list=$("motion-overrides");list.replaceChildren();
    if(!state.motion.overrides.length){list.append(node("p","尚无人工运动覆盖。","hint"));return;}
    for(const edit of state.motion.overrides){
      const row=node("div",undefined,"edit-item"+(edit.enabled===false?" off":""));row.dataset.motionId=edit.id;
      const enabled=document.createElement("input");enabled.type="checkbox";enabled.checked=edit.enabled!==false;
      enabled.setAttribute("aria-label","启用运动 "+edit.id);
      enabled.onchange=run(async()=>{
        if(pending()){enabled.checked=!enabled.checked;throw new Error("请先确认或取消当前调整。");}
        const next=clone(state.motion);next.overrides.find(e=>e.id===edit.id).enabled=enabled.checked;
        try{await applyDraft(state.edits,"edit",state.smoothing,next);}
        catch(error){enabled.checked=!enabled.checked;throw error;}
      });row.append(enabled);
      const description=node("div",undefined,"edit-description");
      description.append(node("strong",StudioMotion.labels[edit.mode]),node("p",clock(edit.start_us/1e6)+" — "+clock(edit.end_us/1e6)+
        " · 进入 "+edit.transition_in_us/1e6+"s / 退出 "+edit.transition_out_us/1e6+"s"));
      if(edit.note)description.append(node("p",edit.note));row.append(description);
      const select=node("button","编辑","quiet");
      select.onclick=run(async()=>{await finishPending();state.motionEditingId=edit.id;fillMotionForm();
        seek(edit.start_us/1e6);$("motion-mode").focus();});
      const remove=node("button","删除","quiet danger");
      remove.onclick=run(async()=>{
        if(pending())throw new Error("请先确认或取消当前调整。");
        const next=clone(state.motion);next.overrides=next.overrides.filter(e=>e.id!==edit.id);
        await applyDraft(state.edits,"edit",state.smoothing,next);
      });row.append(select,remove);list.append(row);
    }
  }
  function drawMotion(){
    const canvas=$("motion-timeline");if(!canvas)return;
    const ctx=canvas.getContext("2d"),w=Math.max(280,canvas.clientWidth),h=150,dpr=devicePixelRatio||1;
    canvas.width=Math.round(w*dpr);canvas.height=h*dpr;ctx.scale(dpr,dpr);
    const motion=shownMotion(),plan=state.motionPreview;
    const signature=state.data?motionAutoSignature(visibleEdits(),motion):null;
    const ready=motion.engine==="flow-v1"&&plan?.times_us?.length&&signature===state.motionSignature;
    canvas.dataset.ready=String(Boolean(ready));
    const status=$("motion-plan-state");
    status.dataset.engine=motion.engine;
    status.dataset.sensitivity=ready?String(plan.config?.sensitivity??motion.sensitivity):"";
    status.textContent=state.motionError?"运动规划暂不可用："+state.motionError:motion.engine==="legacy"?
      "旧引擎 · 既有运动不变，人工运动覆盖保留但不生效。":!ready?
      "自动运动规划待确认 · 不显示过期自动轨道。":"自动运动规划已就绪 · 人工覆盖独立于情绪与视觉数值修改。";
    ctx.font="11px sans-serif";ctx.fillStyle="#a3b2c9";
    if(!ready){ctx.fillText(motion.engine==="legacy"?"切换多模式引擎以开启独立运动轨道。":"正在等待匹配当前情绪与设置的运动规划…",12,62);
      $("motion-frame").textContent="暂无当前时间的已匹配运动数据。";return;}
    const colors={rise:"#69b7a6",fall:"#6d8da9",orbit:"#a293c2",spiral:"#b88cc4",expand:"#d3a968",
      gather:"#75a68e",meteor:"#d78978",wave:"#70abc1",turbulent:"#a38c78"};
    const range=viewRange(),left=52,right=w-15;
    for(const [key,top,label] of [["auto",25,"自动"],["effective",80,"生效"]]){
      ctx.fillStyle="#a3b2c9";ctx.fillText(label,4,top+19);
      ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,34);ctx.clip();
      const segments=[],opacity=key==="auto"?.65:.88;
      // Preblend to opaque colors: neighboring samples may overlap one pixel
      // after rounding, but must not accumulate alpha into visible seams.
      const background=getComputedStyle(canvas.closest(".card")).backgroundColor.match(/[\d.]+/g);
      const base=background?.length>=3?background.slice(0,3).map(Number):[20,29,43];
      const fills=Object.fromEntries(Object.entries(colors).map(([mode,color])=>{
        const rgb=[1,3,5].map((position,index)=>Math.round(
          parseInt(color.slice(position,position+2),16)*opacity+base[index]*(1-opacity)));
        return [mode,"rgb("+rgb.join(",")+")"];
      }));
      for(let i=0;i<plan.times_us.length;i++){
        const start=plan.times_us[i]/1e6,end=(plan.times_us[i+1]??durationUs())/1e6;
        if(end<=start||end<=range.start||start>=range.end)continue;
        const frame=plan[key][i],x=Math.floor(viewX(Math.max(start,range.start),left,right)),
          x2=Math.ceil(viewX(Math.min(end,range.end),left,right)),width=x2-x;
        const previous=segments[segments.length-1];
        if(previous&&previous.mode===frame.mode&&previous.end===start)previous.end=end;
        else segments.push({mode:frame.mode,start,end});
        let heightStart=top;
        for(const c of frame.components){
          ctx.fillStyle=fills[c.mode]||"#64748b";
          const nextHeight=heightStart+c.weight*34;
          ctx.fillRect(x,Math.floor(heightStart),width,Math.ceil(nextHeight)-Math.floor(heightStart));
          heightStart=nextHeight;
        }
      }
      // Paint text only after every band is complete. One intact label per
      // contiguous dominant-mode segment; omit labels that cannot fit.
      for(const segment of segments){
        const x1=Math.max(left,viewX(segment.start,left,right)),x2=Math.min(right,viewX(segment.end,left,right));
        const text=StudioMotion.labels[segment.mode]||segment.mode,textWidth=ctx.measureText(text).width;
        if(x2-x1<textWidth+12)continue;
        const textX=x1+(x2-x1-textWidth)/2;
        ctx.strokeStyle="#101826";ctx.lineWidth=3;ctx.lineJoin="round";
        ctx.strokeText(text,textX,top+21);
        ctx.fillStyle="#edf3fb";ctx.fillText(text,textX,top+21);
      }
      ctx.restore();
    }
    const x=viewX(state.time,left,right);
    if(x>=left&&x<=right){ctx.strokeStyle="#ffd393";ctx.beginPath();ctx.moveTo(x,15);ctx.lineTo(x,122);ctx.stroke();}
    ctx.fillStyle="#a3b2c9";ctx.fillText(clock(range.start),left,140);ctx.fillText(clock(range.end),right-70,140);
    const current=StudioMotion.sampleFrame(plan,Math.round(state.time*1e6));
    $("motion-frame").textContent=clock(state.time)+" · "+({auto:"自动",manual:"人工",transition:"过渡"}[current.source]||current.source)+
      " · "+current.components.map(c=>StudioMotion.labels[c.mode]+" "+Math.round(c.weight*100)+"% / 速度 "+c.params.speed.toFixed(3)+
        " / 一致性 "+c.params.coherence.toFixed(2)+" / 湍流 "+c.params.turbulence.toFixed(2)+
        " / 方向 "+c.params.direction_deg.toFixed(1)+"° / 中心 "+c.params.center_x.toFixed(2)+","+c.params.center_y.toFixed(2)+
        " / 半径 "+c.params.radius.toFixed(2)+" / 旋转 "+c.params.rotation+" / 径向 "+c.params.radial.toFixed(2)+
        " / 脉动 "+c.params.pulse.toFixed(2)+" / 拖尾 "+c.params.trail_seconds.toFixed(2)+"s").join(" ｜ ");
    $("motion-frame").dataset.mode=current.mode;$("motion-frame").dataset.source=current.source;
  }
  function updateDraft() {
    const changed = dirty();
    const historical = state.data && revisionId() !== state.data.project.current_revision;
    const blocked = state.loading || state.busy || !state.data;
    $("editor").classList.toggle("disabled", state.loading || !state.data);
    $("editor").inert = state.loading || !state.data;
    $("editor").setAttribute("aria-disabled", String(state.loading || !state.data));
    $("editor").setAttribute("aria-busy", String(state.loading || state.busy));
    $("revision-select").disabled = blocked;
    $("project-list").querySelectorAll("button").forEach(button => { button.disabled = state.loading || state.busy; });
    $("draft-state").textContent = state.loading ? "正在载入项目…" : pending() ? "临时预览 · 尚待确认" :
      changed ? "草稿已修改 · 尚未保存" : state.data ? "已保存 · " + revisionId() : "尚未选择项目";
    $("draft-state").classList.toggle("dirty", changed);
    $("save-revision").disabled = (!changed && !historical && !pending()) || blocked || Boolean(state.tx?.error||state.motionTx?.error);
    $("save-revision").textContent = pending() ? "确认当前调整并保存" : historical ? "以此草稿恢复为新版本" : "保存为新修订";
    $("render-form").querySelector("button").disabled = changed || pending() || blocked;
    $("edit-form").querySelector("button").disabled = blocked;
    $("edit-form").querySelectorAll("input,select,textarea").forEach(control => { control.disabled = blocked||Boolean(state.motionTx?.changed||state.motionFinal); });
    $("reset-draft").disabled = blocked;
    $("preview-edit").disabled=blocked || Boolean(state.tx?.error||state.motionTx?.error);
    $("motion-form").querySelectorAll("input,select,button").forEach(control=>{
      control.disabled=blocked||Boolean(state.tx?.changed||state.finalizing||state.confirmQueue.length);
    });
    $("motion-cancel").hidden=!state.motionTx?.changed;
    $("apply-smoothing").disabled=blocked;
    $("smoothing-enabled").disabled=blocked;
    $("smoothing-window").disabled=blocked;
    $("smoothing-state").textContent=(state.smoothing.enabled ? "已开启 · "+state.smoothing.window_seconds+" 秒窗口" : "未开启")+
      " · 原始预测只读；人工目标精确生效。";
    $("undo-draft").disabled = blocked || Boolean(state.drag) || (!state.undoStack.length && !pending());
    $("redo-draft").disabled = blocked || Boolean(state.drag) || pending() || !state.redoStack.length;
    $("history-state").textContent = "可撤销 " + state.undoStack.length + " 步 · 可重做 " +
      state.redoStack.length + " 步 · 最多 50 步；输入框保留原生撤销。";
    updateVideoLabel();
    adjustmentStatus();
    updateMotionStatus();
  }
  function updateVideoLabel() {
    const r = state.selectedRender;
    if(r?.kind==="preview"){
      $("video-version").textContent="正在显示局部快照："+(r.snapshot_id||r.id)+
        " · 基于 "+r.revision+" 的冻结草稿 · 不是正式修订输出"+
        (dirty()||pending()?" ｜ 当前草稿可能与此快照不同":"");
      return;
    }
    $("video-version").textContent = r
      ? "正在显示正式视频：" + r.id + " · 修订 " + r.revision +
        (dirty() || pending() ? "　｜　数值已变化，此视频尚未更新" : r.revision !== revisionId() ? "　｜　与当前查看的修订不同" : "")
      : "尚无视频。曲线修改不会自动生成视频。";
  }
  async function refreshProjects() {
    const result = await api("/api/bootstrap");
    state.token = result.token;
    state.projects = result.projects || [];
    renderProjects();
  }
  function renderProjects() {
    const list = $("project-list");
    list.replaceChildren();
    if (!state.projects.length) list.append(node("p", "还没有项目。从下方添加音乐开始。", "muted"));
    const query=$("project-search").value.trim().toLocaleLowerCase();
    const projects=state.projects.filter(p=>(p.name||p.id).toLocaleLowerCase().includes(query));
    if(state.projects.length&&!projects.length)list.append(node("p","没有匹配的项目。清空搜索可查看全部项目。","muted"));
    for (const p of projects) {
      const b = node("button", p.name || p.id, "project-button" + (state.data && p.id === projectId() ? " active" : ""));
      b.dataset.projectId = p.id;
      b.disabled = state.loading || state.busy;
      b.append(node("small", (p.duration ? clock(p.duration) : "待分析") + " · " + (p.current_revision || "分析中")));
      b.onclick = run(async () => {
        if (state.loading || state.busy || state.drag) return;
        if ((dirty() || pending()) && !confirm("切换将丢弃未确认的临时调整；已确认草稿保存在本地。继续吗？")) return;
        await loadProject(p.id);
      });
      list.append(b);
    }
  }
  async function loadProject(id, revision) {
    const serial = ++state.loadSerial;
    state.loading = true;
    state.motionSeq++;clearTimeout(state.motionTimer);
    discardCandidatePlans();
    cancelAdjustment(true);
    state.planSeq++; state.planRequest = false; state.planJob = null;
    state.visualPlan = null; state.visualPreview = null; state.planSignature = null;
    state.visualRequested = false;
    // Remove old restore/discard actions immediately, before any awaited request.
    $("notice").replaceChildren();
    $("notice").hidden = true;
    $("audio").pause();
    $("video").pause();
    updateDraft();
    try {
      const data = await api("/api/projects/" + urlId(id) + (revision ? "?revision=" + urlId(revision) : ""));
      if (serial !== state.loadSerial) return;
      state.data = data;
      state.jobsExpanded=false;
      state.edits = clone(data.revision.edits || []);
      state.smoothing=clone(data.revision.smoothing||{enabled:false,window_seconds:.5});
      state.savedSmoothing=clone(state.smoothing);
      state.motion=clone(data.revision.motion||defaultMotion());
      state.savedMotion=clone(state.motion);
      state.motionPlan=data.timeline.motion||null;state.motionPreview=state.motionPlan;
      state.motionSignature=state.motionPlan?motionAutoSignature(state.edits,state.motion):null;
      state.motionError="";state.motionEditingId=null;
      fillMotionForm();renderMotionEdits();
      $("smoothing-enabled").checked=state.smoothing.enabled;
      $("smoothing-window").value=state.smoothing.window_seconds;
      state.saved = JSON.stringify(state.edits);
      state.preview = data.timeline;
      state.confirmedPreview = data.timeline;
      try { state.preview = state.confirmedPreview = localEmotion(state.edits); } catch { /* Read-only view remains available. */ }
      state.undoStack = []; state.redoStack = [];
      cancelEditing();
      state.time = 0;
      state.viewport={start:0,end:data.project.duration};state.axisCache={};
      updateViewportLabel();
      state.compare=null;state.comparing=false;
      $("compare-video").pause();$("compare-video").removeAttribute("src");
      $("compare-enabled").checked=false;
      $("comparison-panel").hidden=true;
      $("playhead").value = "0";
      $("clock").textContent = clock(0);
      $("project-title").textContent = data.project.name || id;
      $("project-meta").textContent = clock(data.project.duration) + " · 项目 " + id + " · 原始预测只读";
      const select = $("revision-select");
      select.replaceChildren();
      for (const r of data.revisions || [data.revision]) {
        const o = node("option", r.id + (r.parent ? " ← " + r.parent : " · 原始"));
        o.value = r.id;
        select.append(o);
      }
      select.value = data.revision.id;
      $("audio").src = "/media/" + urlId(id) + "/audio";
      $("playhead").max = data.project.duration;
      renderEdits();
      renderOutputs();
      drawTimeline();
      await refreshProjects();
    } finally {
      if (serial === state.loadSerial) {
        state.loading = false;
        if (state.data) $("revision-select").value = revisionId();
        updateDraft();
      }
    }
    if (serial === state.loadSerial) offerSavedDraft();
    if(serial===state.loadSerial)pollJobs();
  }
  function safeMediaUrl(value) {
    if (!value) return null;
    try {
      const u = new URL(value, location.origin);
      return u.origin === location.origin && ["http:", "https:"].includes(u.protocol) ? u.href : null;
    } catch { return null; }
  }
  function renderOutputs() {
    const select = $("render-select");
    const completed = (state.data.renders || []).filter(r =>
      r.video_url && ["success", "succeeded", "completed", "complete", "done"].includes(String(r.status).toLowerCase()));
    select.replaceChildren(node("option", completed.length ? "选择生成结果" : "尚无生成视频"));
    select.firstChild.value = "";
    for (const r of completed) {
      const o = node("option", (r.kind==="preview"?"局部快照 · ":"正式输出 · ")+r.id + " · " + r.revision);
      o.value = r.id;
      select.append(o);
    }
    const matching = completed.filter(r => r.revision === revisionId());
    const r = completed.find(r=>r.id===state.selectedRender?.id) || matching[matching.length - 1] || completed[completed.length - 1];
    select.value = r ? r.id : "";
    if(state.selectedRender?.id!==r?.id || !$("video").getAttribute("src"))selectRender(select.value);
    const references=$("compare-render");
    references.replaceChildren(node("option","选择已有原版 / 参考视频"));references.firstChild.value="";
    for(const render of completed) {
      const option=node("option",render.id+" · "+render.revision+(render.kind==="preview"?" · 快照":""));
      option.value=render.id;references.append(option);
    }
    references.value=state.compare?.id||"";
  }
  function selectRender(id) {
    state.selectedRender = (state.data.renders || []).find(r => r.id === id) || null;
    const r = state.selectedRender;
    const video = $("video");
    video.pause();
    const src = r && safeMediaUrl(r.video_url);
    video.hidden = !src;
    $("video-empty").hidden = Boolean(src);
    if (src) video.src = src;
    else { video.removeAttribute("src"); video.load(); }
    const links = $("download-links");
    links.replaceChildren();
    const downloads=[["下载当前视频",r?.video_url],["下载逐帧数值",r?.csv_url]];
    if(r&&(r.motion_engine||r.motion?.engine)==="flow-v1"){
      const prefix="/media/"+urlId(projectId())+"/renders/"+urlId(r.id)+"/";
      downloads.push(["下载运动计划 CSV",r.motion_csv_url||prefix+"motion_plan.csv"],
        ["下载运动计划 JSON",r.motion_json_url||prefix+"motion_plan.json"]);
    }
    for (const [label, value] of downloads) {
      const href = safeMediaUrl(value);
      if (href) {
        const a = node("a", label);
        a.href = href;
        a.setAttribute("download", "");
        links.append(a);
      }
    }
    updateVideoLabel();
    updateComparison();
  }
  function renderBounds(render,media) {
    const start=Number(render?.actual_start??render?.start??render?.settings?.start??0);
    const declared=render?.actual_end??render?.end??render?.settings?.end;
    const end=declared==null ? start+(Number.isFinite(media.duration)?media.duration:0) : Number(declared);
    return {start,end};
  }
  function comparisonRange() {
    if(!state.selectedRender||!state.compare)return null;
    const a=renderBounds(state.selectedRender,$("video")),b=renderBounds(state.compare,$("compare-video"));
    return {start:Math.max(a.start,b.start),end:Math.min(a.end,b.end),a,b};
  }
  function updateComparison() {
    state.comparing=$("compare-enabled").checked;
    $("comparison-panel").hidden=!state.comparing;
    const audible=$("compare-sound").value;
    $("video").muted=state.comparing&&audible==="reference";
    $("compare-video").muted=!state.comparing||audible!=="reference";
    if(!state.comparing)$("compare-video").pause();
    const range=comparisonRange();
    $("compare-status").textContent=!range ? "请选择两个已有视频；不会自动生成或假设原版。" :
      range.end<=range.start ? "两段视频尚无可播放的重叠区间，请等待媒体载入或更换参考。" :
      "共同歌曲时间 "+clock(range.start)+" — "+clock(range.end)+" · 仅一侧有声";
    $("compare-play").disabled=!range||range.end<=range.start;
  }
  function syncComparison(absolute,play=false) {
    if(!state.comparing||state.syncingMedia)return;
    const range=comparisonRange();if(!range||range.end<=range.start)return;
    state.syncingMedia=true;
    try {
      const time=Math.max(range.start,Math.min(range.end,absolute));
      for(const [media,bounds] of [[$("video"),range.a],[$("compare-video"),range.b]]) {
        if(Number.isFinite(media.duration)&&Math.abs(media.currentTime-(time-bounds.start))>.09)
          media.currentTime=Math.max(0,time-bounds.start);
      }
      if(absolute>=range.end){$("video").pause();$("compare-video").pause();}
      else if(play)$("compare-video").play().catch(()=>notice("参考视频未能播放，请检查生成结果。",true));
    } finally {state.syncingMedia=false;}
  }
  function viewRange() {
    const duration=state.data?.project.duration||1;
    const start=Math.max(0,Math.min(duration,state.viewport.start||0));
    const end=Math.max(start+.001,Math.min(duration,state.viewport.end??duration));
    return {start,end,span:end-start};
  }
  const viewX=(time,left,right)=>{const range=viewRange();return left+(time-range.start)/range.span*(right-left);};
  function updateViewportLabel() {
    const range=viewRange();
    $("view-start").value=range.start.toFixed(3);$("view-end").value=range.end.toFixed(3);
    $("viewport-state").textContent=clock(range.start)+" — "+clock(range.end);
    $("timeline").dataset.viewStart=range.start;$("timeline").dataset.viewEnd=range.end;
  }
  function setViewport(start,end) {
    if(!state.data||state.drag)return;
    const duration=state.data.project.duration;
    if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)throw new Error("视图终点必须大于起点。");
    const span=Math.min(duration,Math.max(Math.min(.1,duration),end-start));
    start=Math.max(0,Math.min(duration-span,start));
    state.viewport={start,end:start+span};updateViewportLabel();drawTimeline();
  }
  function zoomViewport(factor) {
    const range=viewRange(),center=state.time>=range.start&&state.time<=range.end?state.time:(range.start+range.end)/2;
    setViewport(center-range.span/factor/2,center+range.span/factor/2);
  }
  function locateEdit() {
    const edit=visibleEdits().find(e=>e.id===state.editingId);
    if(!edit)throw new Error("请先在修改记录中选择一条修改，再定位它的影响范围。");
    const start=((edit.time_us??edit.start_us)-(edit.transition_in_us||0))/1e6;
    const end=((edit.time_us??edit.end_us)+(edit.transition_out_us||0))/1e6;
    const margin=Math.max(.2,(end-start)*.15);setViewport(start-margin,end+margin);
    seek((edit.time_us??edit.start_us)/1e6);
  }
  function populateFields() {
    const emotion = $("edit-layer").value === "emotion";
    $("edit-field").replaceChildren();
    for (const field of Object.keys(fieldLabels).filter(f =>
      emotion ? ["valence", "arousal"].includes(f) : !["valence", "arousal"].includes(f))) {
      const o = node("option", fieldLabels[field]);
      o.value = field;
      $("edit-field").append(o);
    }
    updateFieldHint();
  }
  function updateFieldHint() {
    const field = $("edit-field").value, range = fieldRanges[field];
    $("field-hint").textContent = "目标值范围 " + range[0] + "～" + range[1] +
      "；偏移不得使结果越界。" + (field === "trail_length" ? "单位为帧，30 帧约 1 秒。" : "");
  }
  function updateOperation() {
    const point = $("edit-operation").value === "point_target";
    $("end-label").hidden = point; $("edit-end").required = !point;
    $("start-label").textContent = point ? "目标时间（秒）" : "开始时间（秒）";
  }
  function cancelEditing() {
    if (state.tx) cancelAdjustment();
    state.editingId = null;
    $("edit-interpolation").value = "smootherstep";
    $("apply-edit").textContent = "添加修改";
    $("cancel-edit").hidden = true;
    drawTimeline();
  }
  function fillEditForm(edit) {
    $("edit-layer").value = edit.layer; populateFields();
    $("edit-field").value = edit.field; updateFieldHint();
    $("edit-operation").value = edit.operation; updateOperation();
    $("edit-start").value = (edit.time_us ?? edit.start_us) / 1e6;
    $("edit-end").value = (edit.end_us ?? edit.time_us) / 1e6;
    $("edit-value").value = edit.value;
    $("edit-in").value = (edit.transition_in_us || 0) / 1e6;
    $("edit-out").value = (edit.transition_out_us || 0) / 1e6;
    $("edit-note").value = edit.note || "";
    $("edit-interpolation").value = edit.interpolation || "linear";
  }
  function editRecord(edit, focus = true) {
    state.editingId = edit.id;
    fillEditForm(edit);
    $("apply-edit").textContent = "确认调整";
    $("cancel-edit").hidden = false;
    if (focus) $("edit-value").focus();
    drawTimeline();
  }
  function renderEdits() {
    const list = $("edits-list");
    list.replaceChildren();
    if (!state.edits.length) list.append(node("p", "没有人工修改，保留模型原始预测。", "muted"));
    for (const edit of state.edits) {
      const row = node("div", undefined, "edit-item" + (edit.enabled === false ? " off" : ""));
      const enabled = document.createElement("input");
      enabled.type = "checkbox";
      enabled.checked = edit.enabled !== false;
      enabled.setAttribute("aria-label", "启用修改 " + edit.id);
      enabled.onchange = run(async () => {
        if (pending()) {
          enabled.checked = !enabled.checked;
          throw new Error("请先确认或取消当前调整，再启用 / 禁用记录。");
        }
        const next = clone(state.edits);
        next.find(e => e.id === edit.id).enabled = enabled.checked;
        try { await applyDraft(next); }
        catch (e) { enabled.checked = !enabled.checked; throw e; }
      });
      row.append(enabled);
      const description = node("div", undefined, "edit-description");
      description.append(node("strong", (fieldLabels[edit.field] || edit.field) + " · " +
        (operations[edit.operation] || edit.operation) + " → " + edit.value));
      const start = (edit.time_us ?? edit.start_us) / 1e6;
      const end = edit.operation === "point_target" ? null : edit.end_us / 1e6;
      description.append(node("p", clock(start) + (end === null ? "" : " — " + clock(end)) +
        " · 进入 " + (edit.transition_in_us || 0) / 1e6 + "s / 退出 " + (edit.transition_out_us || 0) / 1e6 + "s"));
      if (edit.note) description.append(node("p", edit.note));
      row.append(description);
      const modify = node("button", "编辑", "quiet");
      modify.onclick = run(async () => {
        if (state.loading || state.busy) return;
        await finishPending(); editRecord(edit);
        if (edit.layer === "visual") requestVisualPlan();
      });
      row.append(modify);
      const remove = node("button", "删除", "quiet danger");
      remove.onclick = run(async () => {
        if (pending()) throw new Error("请先确认或取消当前调整，再删除记录。");
        await applyDraft(state.edits.filter(e => e.id !== edit.id));
        if (state.editingId === edit.id) cancelEditing();
      });
      row.append(remove);
      list.append(row);
    }
  }
  async function applyDraft(edits, historyAction = "edit", smoothing = state.smoothing,motion=state.motion) {
    if (!state.data) throw new Error("请先选择项目。");
    if (state.loading || state.busy) throw new Error("正在载入或校验，请稍后重试。");
    const previous = clone(state.edits), next = clone(edits), serial = state.loadSerial;
    const previousSmoothing=clone(state.smoothing);
    const previousMotion=clone(state.motion);
    let accepted=false;
    state.smoothing=clone(smoothing);
    state.motion=clone(motion);
    state.busy = true;
    updateDraft();
    try {
      const context = {project:projectId(),revision:revisionId(),loadSerial:serial,
        id:"history-"+crypto.randomUUID(),smoothing:clone(state.smoothing)};
      const unchangedNumeric=JSON.stringify(previous)===JSON.stringify(next)&&JSON.stringify(previousSmoothing)===JSON.stringify(state.smoothing);
      const plan = unchangedNumeric?null:await candidatePlanFor(next, context,"confirmed");
      if (!transactionContext(context)) throw new Error("项目已切换，此次草稿操作未应用。");
      const preview = await api(projectRoute() + "/preview", { base_revision: revisionId(), edits: next });
      if (state.loading || serial !== state.loadSerial) throw new Error("项目已切换，此次草稿操作未应用。");
      if (JSON.stringify(previous) !== JSON.stringify(next) ||
          JSON.stringify(previousSmoothing)!==JSON.stringify(state.smoothing)||
          JSON.stringify(previousMotion)!==JSON.stringify(state.motion)) {
        if (historyAction === "undo") {
          state.undoStack.pop();
          pushHistory(state.redoStack, previous,previousSmoothing,previousMotion);
        } else if (historyAction === "redo") {
          state.redoStack.pop();
          pushHistory(state.undoStack, previous,previousSmoothing,previousMotion);
        } else {
          pushHistory(state.undoStack, previous,previousSmoothing,previousMotion);
          state.redoStack = [];
        }
      }
      const oldEmotion = emotionSignature(state.edits);
      state.edits = next;
      state.preview = state.confirmedPreview = localEmotion(next);
      if(preview.motion)adoptMotionPlan(preview.motion,next,state.motion);
      else scheduleMotionPlan(next,state.motion);
      if (plan) adoptVisualPlan(plan, next);
      else if (oldEmotion !== emotionSignature(next) ||
        JSON.stringify(previousSmoothing)!==JSON.stringify(state.smoothing)) invalidateVisualPlan();
      accepted=true;
      if (state.editingId && !state.edits.some(e => e.id === state.editingId)) cancelEditing();
      persistDraft();
      renderEdits();
      renderMotionEdits();fillMotionForm();
      drawTimeline();
    } finally {
      if(!accepted){state.smoothing=previousSmoothing;state.motion=previousMotion;}
      $("smoothing-enabled").checked=state.smoothing.enabled;
      $("smoothing-window").value=state.smoothing.window_seconds;
      state.busy = false;
      updateDraft();
    }
  }
  async function stepHistory(direction) {
    if (!state.data || state.loading || state.busy || state.drag) return;
    if (state.tx?.error) cancelAdjustment();
    if(state.motionTx?.error)cancelMotion();
    await finishPending();
    const stack = direction === "undo" ? state.undoStack : state.redoStack;
    if (!stack.length) return;
    const snapshot=stack[stack.length-1];
    await applyDraft(snapshot.edits,direction,snapshot.smoothing,snapshot.motion||defaultMotion());
    cancelEditing();
    notice(direction === "undo" ? "已撤销上一步草稿修改。" : "已重做草稿修改。");
  }
  function getEdit() {
    const operation = $("edit-operation").value;
    const seconds = id => Math.round(Number($(id).value) * 1e6);
    const edit = {
      id: state.tx?.editId || state.editingId || ("edit-" + crypto.randomUUID()),
      layer: $("edit-layer").value, field: $("edit-field").value, operation,
      value: Number($("edit-value").value),
      transition_in_us: seconds("edit-in"), transition_out_us: seconds("edit-out"),
      enabled: (state.tx?.base || state.edits).find(e => e.id === (state.tx?.editId || state.editingId))?.enabled !== false,
      note: $("edit-note").value.trim(), interpolation: $("edit-interpolation").value
    };
    if (operation === "point_target") edit.time_us = seconds("edit-start");
    else { edit.start_us = seconds("edit-start"); edit.end_us = seconds("edit-end"); }
    return edit;
  }
  function seek(time, source) {
    const duration = state.data ? state.data.project.duration : 0;
    state.time = Math.min(duration, Math.max(0, Number(time) || 0));
    $("playhead").value = state.time.toFixed(3);
    $("clock").textContent = clock(state.time);
    if (!source) {
      const media = !$("audio").paused || $("video").hidden ? $("audio") : $("video");
      if (Number.isFinite(media.duration)) {
        const offset = media === $("video") ? videoOffset() : 0;
        media.currentTime = Math.max(0, Math.min(media.duration, state.time - offset));
      }
    }
    if(state.comparing&&source!=="audio")syncComparison(state.time,!$("video").paused);
    drawTimeline();
  }
  function drawTimeline() {
    drawMotion();
    drawWaveform();
    drawVisual();
    const canvas = $("timeline"), ctx = canvas.getContext("2d");
    const w = Math.max(280, canvas.clientWidth), h = 310, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = h * dpr;
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const t = state.preview;
    $("baseline-legend").hidden=!state.smoothing.enabled||!t?.baseline;
    state.plotGeometry = []; state.handles = [];
    if (!t || !t.times || !t.times.length) {
      ctx.fillStyle = "#a3b2c9"; ctx.font = "13px sans-serif";
      ctx.fillText("分析完成后，这里将显示真实预测曲线。", 20, 70);
      return;
    }
    const duration = state.data.project.duration || t.times[t.times.length - 1] || 1;
    const left = 52, right = w - 15, plotWidth = right - left;
    let sample = 0;
    for (let i = 0; i < t.times.length && t.times[i] <= state.time + 1e-8; i++) sample = i;
    for (let channel = 0; channel < 2; channel++) {
      const top = 22 + channel * 145, bottom = top + 105;
      let low = Infinity, high = -Infinity;
      for (const series of [t.raw, t.effective,...(state.smoothing.enabled&&t.baseline?[t.baseline]:[])]) for (const row of series || []) {
        const value = row[channel];
        if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); }
      }
      for (const edit of visibleEdits().filter(e => e.enabled !== false && e.layer === "emotion" &&
        e.operation === "point_target" && e.field === (channel === 0 ? "valence" : "arousal"))) {
        low = Math.min(low, edit.value); high = Math.max(high, edit.value);
      }
      if (!Number.isFinite(low)) { low = -1; high = 1; }
      const span = Math.max(.1, high - low), mid = (low + high) / 2;
      low = Math.max(-1, mid - span * .65); high = Math.min(1, mid + span * .65);
      if (high - low < .01) high = low + .1;
      if(state.axisLocked&&state.axisCache[channel])({low,high}=state.axisCache[channel]);
      else state.axisCache[channel]={low,high};
      if (state.drag && state.drag.channel === channel) {
        low = state.drag.geometry.low; high = state.drag.geometry.high;
      }
      const y = v => bottom - (v - low) / (high - low) * (bottom - top);
      state.plotGeometry[channel] = {left, right, top, bottom, low, high, duration,view:viewRange()};
      shadeImpact(ctx, "emotion", channel === 0 ? "valence" : "arousal", left, right, top, bottom, duration);
      ctx.font = "11px sans-serif"; ctx.fillStyle = "#d9e4f6";
      ctx.fillText(channel === 0 ? "Valence · 效价" : "Arousal · 唤醒度", left, top - 8);
      ctx.lineWidth = 1;
      for (let n = 0; n < 3; n++) {
        const value = low + (high - low) * n / 2, py = y(value);
        ctx.strokeStyle = "#2b3a50"; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
        ctx.fillStyle = "#a3b2c9"; ctx.fillText(value.toFixed(2), 4, py + 4);
      }
      const indices = drawingIndices(t.times, t.raw.map(row=>row[channel]),
        t.effective.map(row=>row[channel]), plotWidth, "emotion",
        channel === 0 ? "valence" : "arousal",
        state.smoothing.enabled&&t.baseline?t.baseline.map(row=>row[channel]):null);
      ctx.save();ctx.beginPath();ctx.rect(left,top,plotWidth,bottom-top);ctx.clip();
      const seriesStyles=[["raw","#90a3bf",1.5]];
      if(state.smoothing.enabled&&t.baseline)seriesStyles.push(["baseline","#b39ccd",1]);
      seriesStyles.push(["effective","#78e0ca",2]);
      for (const [key, color, width] of seriesStyles) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(key === "raw" ? [4, 3] : key==="baseline"?[2,5]:[]);
        ctx.beginPath(); let started = false;
        for (const i of indices) {
          const value = t[key]?.[i]?.[channel];
          if (!Number.isFinite(value)) continue;
          const px = viewX(t.times[i],left,right);
          if (!started) { ctx.moveTo(px, y(value)); started = true; }
          else ctx.lineTo(px, y(value));
        }
        ctx.stroke(); ctx.setLineDash([]);
      }
      for (let edit of visibleEdits().filter(e => e.enabled !== false && e.layer === "emotion" &&
        e.field === (channel === 0 ? "valence" : "arousal"))) {
        const x = viewX((edit.time_us ?? edit.start_us)/1e6,left,right);
        if(x<left||x>right)continue;
        if (edit.operation === "point_target") {
          const py = y(edit.value), selected = edit.id === state.editingId;
          ctx.fillStyle = selected ? "#ffffff" : "#ffd393"; ctx.strokeStyle = "#c79745"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, py, selected ? 7 : 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          if(py>=top&&py<=bottom)state.handles.push({editId:edit.id, x, y:py, channel});
        } else {
          ctx.fillStyle = "#ffd393"; ctx.beginPath(); ctx.arc(x, top + 3, 3, 0, Math.PI * 2); ctx.fill();
        }
      }
      const x = viewX(state.time,left,right);
      ctx.strokeStyle = "#f6cf8d"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#a3b2c9";
      for (let n = 0; n < 5; n++) {
        const tx = left + plotWidth * n / 4;
        const range=viewRange(),time=range.start+range.span*n/4;
        ctx.fillText(clock(time).slice(0,range.span<10?8:5), Math.min(w - 40, tx - 12), bottom + 17);
      }
    }
    const raw = t.raw?.[sample], effective = t.effective?.[sample];
    const fmt = v => Number.isFinite(v) ? v.toFixed(3) : "—";
    $("point-values").textContent = clock(state.time) + "　V " + fmt(effective?.[0]) +
      "（原始 " + fmt(raw?.[0]) + "）　A " + fmt(effective?.[1]) + "（原始 " + fmt(raw?.[1]) + "）";
  }
  function drawWaveform() {
    const canvas = $("waveform"), ctx = canvas.getContext("2d");
    const w = Math.max(280, canvas.clientWidth), h = 76, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w*dpr); canvas.height = h*dpr;
    ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
    const data = state.data?.timeline?.waveform || state.preview?.waveform;
    ctx.font = "11px sans-serif"; ctx.fillStyle = "#a3b2c9";
    if (!data || !data.times?.length || !data.peaks?.length) {
      ctx.fillText("暂无音频波形数据，仍可使用音频播放器和下方曲线。", 12, 40);
      return;
    }
    const left = 52, right = w-15, duration = state.data.project.duration || 1;
    ctx.strokeStyle = "#2b3a50"; ctx.beginPath(); ctx.moveTo(left,38); ctx.lineTo(right,38); ctx.stroke();
    ctx.strokeStyle = "#6389a2"; ctx.lineWidth = Math.max(1,(right-left)/data.times.length);
    ctx.beginPath();
    for (let i=0; i<data.times.length; i++) {
      const peak=Math.min(1,Math.max(0,Number(data.peaks[i])||0));
      const x=viewX(data.times[i],left,right), height=peak*29;
      if(x<left||x>right)continue;
      ctx.moveTo(x,38-height); ctx.lineTo(x,38+height);
    }
    ctx.stroke(); ctx.fillStyle="#a3b2c9"; ctx.fillText("波形",4,42);
    const x=viewX(state.time,left,right);
    if(x<left||x>right)return;
    ctx.strokeStyle="#f6cf8d"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,4); ctx.lineTo(x,72); ctx.stroke();
  }
  function shadeImpact(ctx, layer, field, left, right, top, bottom, duration) {
    const edit = state.tx?.candidate || visibleEdits().find(e => e.id === state.editingId);
    if (!edit || edit.layer !== layer || edit.field !== field || edit.enabled === false) return;
    const start = Math.max(0, (edit.time_us ?? edit.start_us) - (edit.transition_in_us || 0))/1e6;
    const end = Math.min(durationUs(), (edit.time_us ?? edit.end_us) + (edit.transition_out_us || 0))/1e6;
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    ctx.fillStyle = "rgba(255, 211, 147, 0.10)";
    const x1=Math.max(left,viewX(start,left,right)),x2=Math.min(right,viewX(end,left,right));
    if(x2>=x1)ctx.fillRect(x1,top,Math.max(1,x2-x1),bottom-top);
  }
  function drawingIndices(times, raw, effective, width, layer, field, baseline=null) {
    const mandatory = [];
    for (const edit of visibleEdits()) {
      if (edit.enabled === false || edit.layer !== layer || edit.field !== field) continue;
      const start = edit.time_us ?? edit.start_us, end = edit.time_us ?? edit.end_us;
      mandatory.push(start/1e6, end/1e6,
        (start-(edit.transition_in_us||0))/1e6, (end+(edit.transition_out_us||0))/1e6);
    }
    const range=viewRange();
    let first=0,last=times.length-1;
    while(first<last&&times[first+1]<range.start)first++;
    while(last>first&&times[last-1]>range.end)last--;
    mandatory.push(range.start,range.end);
    if (typeof StudioTimeline.plotIndices === "function")
      return StudioTimeline.plotIndices(times.slice(first,last+1),
        [raw.slice(first,last+1),effective.slice(first,last+1),...(baseline?[baseline.slice(first,last+1)]:[])],
        Math.max(1,Math.floor(width)), mandatory).map(index=>index+first);
    // Older math assets may still be cached; preserve the full curve until refreshed.
    return Array.from({length:times.length},(_,i)=>i);
  }
  function drawVisual() {
    const canvas = $("visual-timeline");
    if (!canvas) return;
    const ctx=canvas.getContext("2d"), w=Math.max(280,canvas.clientWidth), h=190, dpr=window.devicePixelRatio||1;
    canvas.width=Math.round(w*dpr); canvas.height=h*dpr; ctx.scale(dpr,dpr);
    ctx.font="11px sans-serif"; ctx.fillStyle="#a3b2c9";
    const edits=visibleEdits();
    if (!state.data || !state.visualPlan || state.planSignature !== emotionSignature(edits)) {
      ctx.fillText(state.tx?.candidate?.layer === "emotion"
        ? "情绪临时调整中：确认后更新真实规划，不伪造视觉基线。"
        : "等待真实视觉规划基线。请选择字段以开始准备。", 15,85);
      canvas.dataset.ready="false"; return;
    }
    try {
      state.visualPreview=StudioTimeline.visualPreview(state.visualPlan,durationUs(),edits);
    } catch (error) {
      ctx.fillText("视觉曲线暂不可用："+error.message,15,85); canvas.dataset.ready="false"; return;
    }
    const p=state.visualPreview, field=$("visual-field").value, index=p.fields.indexOf(field);
    if(index<0){ctx.fillText("此规划不含所选字段。",15,85);canvas.dataset.ready="false";return;}
    canvas.dataset.ready="true"; canvas.dataset.planKey=state.visualPlan.key||"";
    const left=52,right=w-15,top=20,bottom=155,duration=state.data.project.duration||1;
    let low=Infinity,high=-Infinity;
    for(const rows of [p.raw,p.effective]) for(const row of rows) {
      low=Math.min(low,row[index]);high=Math.max(high,row[index]);
    }
    const span=Math.max((fieldRanges[field][1]-fieldRanges[field][0])*.04,high-low),mid=(low+high)/2;
    low=mid-span*.65;high=mid+span*.65;
    const axisKey="visual:"+field;
    if(state.axisLocked&&state.axisCache[axisKey])({low,high}=state.axisCache[axisKey]);
    else state.axisCache[axisKey]={low,high};
    const y=value=>bottom-(value-low)/(high-low)*(bottom-top);
    shadeImpact(ctx,"visual",field,left,right,top,bottom,duration);
    for(let i=0;i<3;i++){
      const value=low+(high-low)*i/2,py=y(value);
      ctx.fillStyle="#a3b2c9";ctx.fillText(value.toFixed(2),3,py+4);
      ctx.strokeStyle="#2b3a50";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();
    }
    const indices=drawingIndices(p.times,p.raw.map(row=>row[index]),p.effective.map(row=>row[index]),
      right-left,"visual",field);
    ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
    for(const [key,color] of [["raw","#90a3bf"],["effective","#78e0ca"]]){
      ctx.strokeStyle=color;ctx.lineWidth=key==="raw"?1.5:2;ctx.setLineDash(key==="raw"?[4,3]:[]);
      ctx.beginPath();
      indices.forEach((i,position)=>{const x=viewX(p.times[i],left,right);
        if(position===0)ctx.moveTo(x,y(p[key][i][index]));else ctx.lineTo(x,y(p[key][i][index]));});
      ctx.stroke();ctx.setLineDash([]);
    }
    ctx.strokeStyle="#f6cf8d";ctx.lineWidth=1;ctx.beginPath();
    const px=viewX(state.time,left,right);ctx.moveTo(px,top);ctx.lineTo(px,bottom);ctx.stroke();ctx.restore();
    const range=viewRange();
    ctx.fillStyle="#a3b2c9";ctx.fillText(clock(range.start),left,bottom+20);ctx.fillText(clock(range.end),right-70,bottom+20);
  }
  async function pollJobs() {
    if (document.hidden || state.polling) return;
    state.polling = true;
    try {
      const data = await api("/api/jobs");
      const jobs = data.jobs || [], list = $("jobs-list");
      const owner=job=>job.project_id||job.payload?.project_id||job.result?.project_id;
      const active=job=>!terminalStates.includes(String(job.status||"").toLowerCase());
      const current=state.data?.project.id,scope=$("jobs-scope").value;
      const filtered=jobs.filter(job=>scope==="all"||(current&&owner(job)===current))
        .sort((a,b)=>String(b.created_at||"").localeCompare(String(a.created_at||"")));
      const visible=new Set((state.jobsExpanded?filtered:filtered.filter((job,index)=>index<5||active(job))).map(job=>job.id));
      const others=jobs.filter(job=>active(job)&&owner(job)!==current).length;
      $("jobs-other-active").hidden=scope==="all"||!others;
      $("jobs-other-active").textContent="其他项目 / 尚未建成项目还有 "+others+" 项活动任务。选择“所有项目”可查看，不会因筛选而停止。";
      $("jobs-summary").textContent="显示 "+visible.size+" / "+filtered.length+" 项 · "+
        (state.jobsExpanded?"完整历史":"最近 5 项及所有活动任务")+" · 每 2 秒刷新";
      $("jobs-more").hidden=filtered.length<=visible.size&&!state.jobsExpanded;
      $("jobs-more").textContent=state.jobsExpanded?"收起":"显示更多（"+filtered.length+"）";
      list.replaceChildren();
      if (!visible.size) list.append(node("p",scope==="current"?"当前项目暂无任务。可切换“所有项目”查看其他任务。":"尚无任务。","muted"));
      let changed = false;
      for (const job of jobs) {
        const status = String(job.status || "unknown").toLowerCase(), terminal = terminalStates.includes(status);
        for (const record of state.candidatePlans.values()) {
          if (record.jobId !== job.id || !terminal || record.finished) continue;
          record.jobId = null;
          if (["succeeded","completed","success"].includes(status)) fetchCandidatePlan(record);
          else rejectCandidatePlan(record,new Error("候选视觉规划未完成：" + (job.error || status)));
        }
        if (state.planJob === job.id && terminal) {
          state.planJob = null;
          if (["succeeded","completed","success"].includes(status)) requestVisualPlan();
          else $("visual-plan-state").textContent = "真实规划未完成：" + (job.error || status);
        }
        if (terminal && !state.terminalJobs.has(job.id)) { state.terminalJobs.add(job.id); changed = true; }
        // Filters affect only presentation; every job still drives plan completion above.
        if(!visible.has(job.id))continue;
        const row = node("div", undefined, "job"), info = node("div");
        row.dataset.jobId=job.id;row.dataset.projectId=owner(job)||"";row.dataset.status=status;
        info.append(node("strong", job.name || jobLabels[job.kind] || job.kind || job.type || job.id));
        info.append(node("p", job.error || job.message || stageLabels[job.stage] || job.stage || job.id)); row.append(info);
        const progressBox = node("div"), p = node("progress"); p.max = 1;
        const value = job.ratio ?? job.progress?.ratio ?? job.progress;
        if (typeof value === "number") p.value = value > 1 ? value / 100 : value;
        else if (terminal) p.value = 1;
        progressBox.append(p); progressBox.append(node("p", statusLabels[status] || status, "job-state")); row.append(progressBox);
        if(job.details?.total){
          const unit=job.details.unit||({embedding:"音频片段",inference:"情绪采样",planning:"节点",rendering:"帧"}[job.stage])||"项";
          const done=status==="succeeded" ? job.details.total : job.details.done;
          progressBox.append(node("p","已完成 "+done+" / "+job.details.total+" "+unit,"hint"));
        }
        const actions=node("div",undefined,"job-actions");
        if (!terminal) {
          const cancel = node("button",status==="cancelling"?"取消中…":"取消","quiet");
          cancel.disabled=status==="cancelling";
          cancel.onclick = run(async () => {
            cancel.disabled=true;cancel.textContent="取消中…";
            try{await api("/api/jobs/" + urlId(job.id) + "/cancel", {});}
            catch(error){cancel.disabled=false;cancel.textContent="取消";throw error;}
          });
          actions.append(cancel);
        }
        const log=node("button","日志","quiet");
        log.onclick=run(async()=>{
          const result=await api("/api/jobs/"+urlId(job.id)+"/log?limit=16384");
          $("job-log").textContent=(result.truncated?"（仅显示最近日志）\n":"")+(result.text||"此任务暂无日志。");
          $("job-log-dialog").showModal();
        });actions.append(log);
        if(["failed","cancelled","canceled","interrupted"].includes(status)){
          const retry=node("button","重试","quiet");
          retry.onclick=run(async()=>{
            retry.disabled=true;
            try{await api("/api/jobs/"+urlId(job.id)+"/retry",{});
              notice("已按原任务的冻结参数创建重试任务。");}
            finally{retry.disabled=false;}
          });actions.append(retry);
        }
        row.append(actions);
        list.append(row);
      }
      if (changed) {
        await refreshProjects();
        if (state.data && !state.loading && !state.busy) {
          const id = projectId(), rev = revisionId();
          const fresh = await api(projectRoute() + "?revision=" + urlId(rev));
          if (!state.loading && projectId() === id && revisionId() === rev) {
            state.data.renders = fresh.renders; renderOutputs();
          }
        }
      }
    } catch (error) {
      $("jobs-list").replaceChildren(node("p", "任务状态暂时无法读取：" + error.message, "danger"));
    } finally { state.polling = false; }
  }

  $("refresh-projects").onclick = run(refreshProjects);
  $("motion-form").addEventListener("input",event=>{
    if(event.target.matches("input,select"))motionInput();
  });
  $("motion-form").addEventListener("focusout",event=>{
    if(!event.target.matches("input,select")||event.relatedTarget?.closest("#motion-cancel,#project-list,#revision-select"))return;
    run(finishMotion)();
  });
  $("motion-form").addEventListener("keydown",event=>{
    if(event.key==="Enter"&&event.target.matches("input,select")){
      event.preventDefault();run(finishMotion)();
    }
  });
  $("motion-form").addEventListener("change",event=>{
    if(!event.target.matches("select"))return;
    if(event.target.id==="motion-mode"){
      // A mode switch is a new preset, not the old mode with a new label.
      // In particular, rise (270°) -> fall must immediately restore 90°.
      const defaults=StudioMotion.params(event.target.value);
      for(const key of Object.keys(StudioMotion.bounds))$("motion-param-"+key).value=defaults[key];
    }
    motionInput();run(finishMotion)();
  });
  $("motion-form").onsubmit=run(async()=>{if(!state.motionTx)motionInput();await finishMotion();});
  $("motion-cancel").onclick=()=>cancelMotion();
  $("motion-end-edit").onclick=run(async()=>{await finishMotion();state.motionEditingId=null;fillMotionForm();});
  $("motion-new").onclick=run(async()=>{
    await finishPending();
    if(state.motion.engine!=="flow-v1")throw new Error("请先选择“多模式 · 情绪驱动”引擎。");
    state.motionEditingId="motion-"+crypto.randomUUID();
    const duration=state.data.project.duration;
    $("motion-override-fields").hidden=false;$("motion-end-edit").hidden=false;
    $("motion-mode").value="rise";$("motion-start").value=(duration/3).toFixed(3);
    $("motion-end").value=(duration*2/3).toFixed(3);
    $("motion-in").value=Math.min(2,Math.floor(duration/3*1000)/1000);
    $("motion-out").value=Math.min(2,Math.floor(duration/3*1000)/1000);
    $("motion-interpolation").value="smootherstep";$("motion-note").value="";
    for(const key of Object.keys(StudioMotion.bounds))$("motion-param-"+key).value="";
    seek(Number($("motion-start").value));
    motionInput();
  });
  $("project-search").oninput=renderProjects;
  $("jobs-scope").onchange=()=>{state.jobsExpanded=false;pollJobs();};
  $("jobs-more").onclick=()=>{state.jobsExpanded=!state.jobsExpanded;pollJobs();};
  $("zoom-in").onclick=run(()=>zoomViewport(2));
  $("zoom-out").onclick=run(()=>zoomViewport(.5));
  $("pan-left").onclick=run(()=>{const r=viewRange();setViewport(r.start-r.span*.5,r.end-r.span*.5);});
  $("pan-right").onclick=run(()=>{const r=viewRange();setViewport(r.start+r.span*.5,r.end+r.span*.5);});
  $("view-full").onclick=run(()=>setViewport(0,state.data.project.duration));
  $("locate-edit").onclick=run(locateEdit);
  $("view-apply").onclick=run(()=>setViewport(Number($("view-start").value),Number($("view-end").value)));
  $("axis-lock").onchange=()=>{state.axisLocked=$("axis-lock").checked;drawTimeline();};
  $("apply-smoothing").onclick=run(async()=>{
    const config={enabled:$("smoothing-enabled").checked,window_seconds:Number($("smoothing-window").value)};
    if(!Number.isFinite(config.window_seconds)||config.window_seconds<.1||config.window_seconds>5)
      throw new Error("平滑窗口应为 0.1～5 秒。");
    await finishPending();await applyDraft(state.edits,"edit",config);
    notice("平滑配置已校验，作为独立一步加入草稿；保存修订后持久生效。");
  });
  $("preview-edit").onclick=run(async()=>{
    if(!state.data||state.loading||state.busy)return;
    await finishPending();
    if(!confirm("将冻结当前草稿并生成局部预览视频，不改变已保存修订。继续吗？"))return;
    state.busy=true;updateDraft();
    try {
      await api(projectRoute()+"/previews",{base_revision:revisionId(),edits:clone(state.edits),
        smoothing:clone(state.smoothing),motion:clone(state.motion),
        ...(state.motionEditingId?{motion_edit_id:state.motionEditingId}:{edit_id:state.editingId||null})});
      notice("局部预览任务已提交。完成后可在视频列表选择快照；正式修订未改变。");
      await pollJobs();
    }finally{state.busy=false;updateDraft();}
  });
  $("compare-enabled").onchange=updateComparison;
  $("compare-sound").onchange=updateComparison;
  $("compare-render").onchange=()=>{
    const render=state.data?.renders.find(r=>r.id===$("compare-render").value);
    state.compare=render||null;
    const media=$("compare-video");media.pause();
    const src=safeMediaUrl(render?.video_url);
    if(src)media.src=src;else{media.removeAttribute("src");media.load();}
    updateComparison();
  };
  $("compare-play").onclick=run(async()=>{
    const range=comparisonRange();
    if(!range||range.end<=range.start)throw new Error("请先选择具有共同歌曲时间区间的两个视频。");
    $("audio").pause();
    const time=state.time>=range.start&&state.time<range.end?state.time:range.start;
    seek(time);syncComparison(time,true);await $("video").play();
  });
  $("compare-pause").onclick=()=>{$("video").pause();$("compare-video").pause();};
  $("close-log").onclick=()=>$("job-log-dialog").close();
  $("create-form").onsubmit = run(async event => {
    const form = event.target, button = form.querySelector("button"), data = new FormData(form);
    button.disabled = true;
    try {
      await api("/api/projects", {
        name: String(data.get("name") || "").trim(), audio_path: String(data.get("audio_path")).trim(),
        model_path: String(data.get("model_path")).trim(), seed: Number(data.get("seed"))
      });
      notice("分析任务已提交。完成后请在左侧选择项目，再编辑或生成视频。");
      await pollJobs(); await refreshProjects();
    } finally { button.disabled = false; }
  });
  $("revision-select").onchange = run(async event => {
    const next = event.target.value;
    if (state.loading || state.busy || state.drag || ((dirty() || pending()) &&
        !confirm("切换将丢弃未确认调整；已确认草稿可在返回时恢复。继续吗？"))) {
      event.target.value = revisionId(); return;
    }
    await loadProject(projectId(), next);
  });
  $("edit-layer").onchange = () => {
    populateFields();
    if ($("edit-layer").value === "visual") {
      $("visual-field").value = $("edit-field").value; requestVisualPlan();
    }
    scheduleInput("edit-layer");
  };
  $("edit-field").onchange = () => {
    updateFieldHint();
    if ($("edit-layer").value === "visual") {
      $("visual-field").value = $("edit-field").value; requestVisualPlan();
    }
    scheduleInput("edit-field");
  };
  $("edit-operation").onchange = updateOperation;
  $("edit-interpolation").onchange = run(() => {
    // A native select commits a choice on change, including keyboard choices and
    // programmatic selectOption. Unlike text entry, it need not gain/lose focus.
    scheduleInput("edit-interpolation");
    flushInput();
    if (state.tx?.valid) queueCurrentTransaction();
  });
  $("cancel-edit").onclick = run(async () => { await finishPending(); cancelEditing(); });
  $("cancel-adjustment").onclick = () => { cancelAdjustment(); notice("已取消当前未确认调整，恢复本次输入前的基线。"); };
  $("undo-draft").onclick = run(() => stepHistory("undo"));
  $("redo-draft").onclick = run(() => stepHistory("redo"));
  $("edit-form").onsubmit = run(async () => {
    if (!state.data || state.loading || state.busy) return;
    if (!state.tx && !state.confirmPromise) { beginTransaction("submit"); evaluateInput(); }
    await finishPending();
    notice("本次调整已确认。保存为新修订后即可生成视频。");
  });
  $("reset-draft").onclick = run(async () => {
    await finishPending();
    if (!dirty() || confirm("放弃当前未保存的修改，恢复所选修订？")) {
      await applyDraft(clone(state.data.revision.edits || []),"edit",state.savedSmoothing,state.savedMotion);
      cancelEditing();
    }
  });
  $("save-revision").onclick = run(async () => {
    if (!state.data || state.loading || state.busy ||
        (!dirty() && !pending() && revisionId() === state.data.project.current_revision)) return;
    if (revisionId() !== state.data.project.current_revision &&
        !confirm("正在查看历史修订。保存将以这份草稿创建新的当前版本，不覆盖已有历史。继续吗？")) return;
    await finishPending();
    state.busy = true; updateDraft();
    try {
      const snapshot = clone(state.edits);
      const oldDraftKey = draftKey(projectId(), revisionId());
      const result = await api(projectRoute() + "/revisions", { base_revision: state.data.project.current_revision, edits: snapshot });
      try { localStorage.removeItem(oldDraftKey); } catch { /* The committed revision remains safe on disk. */ }
      await loadProject(projectId(), result.id || result.revision?.id);
      notice("已保存新修订。原始预测和已有视频保持不变。");
    } finally { state.busy = false; updateDraft(); }
  });
  $("render-form").onsubmit = run(async event => {
    if (!state.data || state.loading || state.busy) return;
    if (dirty() || pending()) throw new Error("请先确认调整并保存草稿，再生成该修订的视频。");
    const form = new FormData(event.target), [width, height] = $("render-size").value.split(",").map(Number);
    const start = Number(form.get("start"));
    const end = String(form.get("end") || "").trim() === "" ? null : Number(form.get("end"));
    await api(projectRoute() + "/renders", { revision: revisionId(), mode: form.get("mode"), start, end, width, height, fps: 30 });
    notice("已提交视频生成任务。本次输出锁定当前修订，后续编辑不会改变它。"); await pollJobs();
  });
  $("render-select").onchange = event => selectRender(event.target.value);
  $("playhead").onchange = event => seek(event.target.value);
  $("edit-form").addEventListener("focusin", event => {
    if (event.target.matches("input,select,textarea")) {
      try { beginTransaction(event.target.id); } catch (error) { notice(error.message, true); }
    }
  });
  $("edit-form").addEventListener("input", event => {
    if (event.target.matches("input,select,textarea")) scheduleInput(event.target.id);
  });
  $("edit-form").addEventListener("focusout", event => {
    if (!event.target.matches("input,select,textarea")) return;
    const next = event.relatedTarget;
    if (next?.closest("#cancel-adjustment,#project-list,#revision-select")) return;
    try { queueCurrentTransaction(); } catch { updateDraft(); }
  });
  $("edit-form").addEventListener("keydown", event => {
    if (event.key !== "Enter" || event.isComposing || !event.target.matches("input,select,textarea")) return;
    event.preventDefault();
    run(async () => { await finishPending(); notice("本次输入已确认。"); })();
  });
  $("visual-field").onchange = () => { requestVisualPlan(); drawVisual(); };
  $("visual-field").onfocus = () => requestVisualPlan();
  $("use-playhead").onclick = () => {
    $("edit-start").value = state.time.toFixed(3);
    $("edit-end").value = Math.min(state.data.project.duration, state.time + 1).toFixed(3);
    $("edit-start").focus();
    scheduleInput("edit-start");
  };
  function seekFromCanvas(event) {
    if (!state.data || state.loading) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const range=viewRange();
    seek(range.start+(event.clientX - rect.left - 52) / (rect.width - 67) * range.span);
  }
  $("waveform").onclick = seekFromCanvas;
  $("motion-timeline").onclick=seekFromCanvas;
  $("visual-timeline").onclick = seekFromCanvas;
  $("timeline").addEventListener("pointerdown", event => {
    if (!state.data || state.loading || state.busy || event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX-rect.left, y = event.clientY-rect.top;
    const handle = state.handles.find(h => Math.hypot(h.x-x,h.y-y) <= 12);
    if (!handle) { seekFromCanvas(event); return; }
    if (pending()) { notice("请先确认或取消当前输入，再拖动关键帧。"); return; }
    const edit = state.edits.find(e => e.id === handle.editId);
    if (!edit) return;
    event.preventDefault();
    editRecord(edit, false);
    beginTransaction("pointer");
    state.drag = {
      editId: edit.id, original: clone(edit),
      channel: handle.channel, geometry: {...state.plotGeometry[handle.channel]}, pointerId: event.pointerId
    };
    updateDraft();
    event.currentTarget.setPointerCapture(event.pointerId);
    drawTimeline();
  });
  $("timeline").addEventListener("pointermove", event => {
    const drag = state.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect(), g = drag.geometry;
    const x=event.clientX-rect.left, y=event.clientY-rect.top;
    const time = Math.max(0, Math.min(g.duration,g.view.start+(x-g.left)/(g.right-g.left)*g.view.span));
    const value = Math.max(-1,Math.min(1,g.high-(y-g.top)/(g.bottom-g.top)*(g.high-g.low)));
    $("edit-start").value = time.toFixed(3);
    $("edit-value").value = value.toFixed(4);
    scheduleInput("pointer");
  });
  $("timeline").addEventListener("pointerup", run(async event => {
    const drag = state.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    flushInput();
    state.drag = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    try {
      await finishPending();
      const updated = state.edits.find(e => e.id === drag.editId);
      if (updated) editRecord(updated, false);
      notice("关键帧调整已校验并写入本地草稿。请保存修订后生成视频。");
    } catch (error) {
      cancelAdjustment();
      editRecord(drag.original, false);
      drawTimeline();
      throw new Error("此次拖动未生效，已回退：" + error.message);
    }
  }));
  $("timeline").addEventListener("pointercancel", event => {
    const drag = state.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    state.drag = null;
    cancelAdjustment();
    editRecord(drag.original, false);
    updateDraft();
    drawTimeline();
  });
  $("audio").addEventListener("timeupdate", () => {
    if (!$("audio").paused) seek($("audio").currentTime, "audio");
  });
  $("video").addEventListener("timeupdate", () => {
    if (!$("video").paused) seek($("video").currentTime + videoOffset(), "video");
  });
  $("audio").addEventListener("play", () => $("video").pause());
  $("video").addEventListener("play", () => $("audio").pause());
  $("video").addEventListener("play",()=>syncComparison($("video").currentTime+videoOffset(),true));
  $("video").addEventListener("pause",()=>{$("compare-video").pause();});
  $("video").addEventListener("seeked",()=>{
    if(!state.syncingMedia)seek($("video").currentTime+videoOffset(),"video");
  });
  for(const id of ["video","compare-video"])$(id).addEventListener("loadedmetadata",updateComparison);
  window.addEventListener("pagehide",()=>{for(const id of [...state.temporaryConsumers])releaseConsumer(id);});
  window.addEventListener("beforeunload", event => {
    if (dirty() || pending()) { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("keydown", event => {
    if(event.key==="Escape"&&state.motionTx?.changed&&!state.loading&&!state.busy){
      event.preventDefault();cancelMotion();notice("已取消本次未确认运动调整。");return;
    }
    if (event.key === "Escape" && pending() && !state.loading && !state.busy) {
      event.preventDefault();
      state.drag = null; cancelAdjustment();
      notice("已取消当前未确认调整。");
      return;
    }
    if (event.defaultPrevented || event.isComposing || event.altKey ||
        (!event.ctrlKey && !event.metaKey) || !state.data ||
        state.loading || state.busy || state.drag) return;
    const target = event.target;
    if (target instanceof Element && (target.closest("input, textarea, select") ||
        target.isContentEditable || target.closest('[contenteditable]:not([contenteditable="false"])'))) return;
    const key = event.key.toLowerCase();
    let direction = null;
    if (key === "z") direction = event.shiftKey ? "redo" : "undo";
    else if (key === "y" && event.ctrlKey && !event.metaKey && !event.shiftKey) direction = "redo";
    if (!direction) return;
    const stack = direction === "undo" ? state.undoStack : state.redoStack;
    if (!stack.length && !(direction === "undo" && pending())) return;
    event.preventDefault();
    run(() => stepHistory(direction))();
  });
  window.addEventListener("resize", drawTimeline);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) pollJobs(); });
  populateFields();
  for(const [mode,label] of Object.entries(StudioMotion.labels)){
    const option=node("option",label);option.value=mode;$("motion-mode").append(option);
  }
  const motionParamLabels={speed:"速度（短边/秒）",coherence:"方向一致性",turbulence:"湍流强度",
    direction_deg:"方向（度）",center_x:"中心 X",center_y:"中心 Y",radius:"半径（短边比例）",
    rotation:"旋转方向（-1 / 1）",radial:"径向（负内 / 正外）",pulse:"脉动强度",trail_seconds:"拖尾（秒）"};
  for(const [key,bounds] of Object.entries(StudioMotion.bounds)){
    const label=node("label",motionParamLabels[key]),input=document.createElement("input");
    input.id="motion-param-"+key;input.type="number";input.step=key==="rotation"?"2":"any";
    input.min=bounds[0];input.max=bounds[1];input.placeholder="模式预设";
    label.append(input);$("motion-params").append(label);
  }
  for (const field of Object.keys(fieldLabels).filter(f => !["valence","arousal"].includes(f))) {
    const option = node("option", fieldLabels[field]); option.value = field;
    $("visual-field").append(option);
  }
  updateDraft();
  run(async () => {
    await refreshProjects(); await pollJobs();
    if (state.projects.length) await loadProject(state.projects[0].id);
  })();
  setInterval(pollJobs, 2000);
})();
