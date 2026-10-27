"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const audio = $("audio");
  const modelKeys = ["b01", "b00", "v2"];
  const colors = { b01: "#70e0cc", b00: "#ad9cf0", v2: "#8b9bad" };
  const names = { b01: "B01 等权集成", b00: "B00 基线", v2: "旧 V2" };
  const activeStatuses = new Set(["registered", "queued", "pending", "running", "cancelling", "starting"]);
  const successStatuses = new Set(["completed", "succeeded", "done"]);
  const recoverableStatuses = new Set(["failed", "cancelled", "interrupted"]);
  const statusNames = { registered: "已登记", queued: "排队中", pending: "待处理", running: "进行中", starting: "准备中", cancelling: "正在取消", completed: "已完成", succeeded: "已完成", done: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断", created: "已创建", ready: "结果就绪" };
  const stageNames = { queued: "等待执行", verify_manifest: "核对模型清单与校验值", native_decode: "原生解码与时间轴检查", decode: "解码音频", decoding: "解码音频", features: "提取固定特征", extracting: "提取固定特征", feature_extraction: "提取固定特征", inference: "模型推理", predict: "模型推理", predict_b01: "B01 模型推理", predict_b00: "B00 基线推理", predict_v2: "旧 V2 对照推理", legacy_cpu_decode: "旧 V2 · CPU 解码", legacy_cpu_features: "旧 V2 · CPU 特征提取", legacy_cpu_head: "旧 V2 · CPU 预测", acoustic: "计算声学属性", save_raw_result: "保存原始预测与凭证", save: "保存结果", saving: "保存结果", export: "生成预览视频", rendering: "渲染视频", completed: "任务完成", cancelled: "任务已取消" };
  const state = { server: null, project: null, result: null, selectedAudio: null, projectId: null, jobId: null, mode: "raw", smoothing: 0, smoothCache: new Map(), loopA: null, loopB: null, looping: false, busy: false, connected: false, pollTimer: null, projectRequest: 0, finishedJobs: new Set(), refreshing: false };
  const timeSteps = new WeakMap();
  let lastReadout = 0;
  let dirtyCharts = true;
  let lastChartTime = -1;
  let lastVisualTime = -1;
  let lastVisualKey = "";

  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function clock(value, decimal = false) {
    if (!finite(value) || value < 0) value = 0;
    const hour = Math.floor(value / 3600);
    const minute = Math.floor(value / 60) % 60;
    const second = Math.floor(value % 60);
    return `${hour ? `${hour}:` : ""}${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}${decimal ? `.${Math.floor(value * 10) % 10}` : ""}`;
  }
  function duration() {
    const resultDuration = state.result?.audio?.duration_seconds;
    const projectDuration = state.project?.audio?.duration_seconds;
    return finite(resultDuration) ? resultDuration : finite(projectDuration) ? projectDuration : finite(audio.duration) ? audio.duration : 0;
  }
  function message(text, type = "error") {
    const element = $("notice");
    element.textContent = String(text || "");
    element.classList.toggle("info", type === "info");
    element.hidden = !text;
  }
  function setConnection(connected) {
    state.connected = connected;
    $("connection").textContent = connected ? "本地服务已连接" : "连接已断开";
    $("connection").classList.toggle("online", connected);
    $("connection").classList.toggle("offline", !connected);
    updateActions();
  }
  async function api(path, body) {
    const options = { cache: "no-store", headers: { Accept: "application/json" } };
    if (body !== undefined) {
      if (!state.server?.token) throw new Error("尚未连接本地服务，请稍后重试。");
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.headers["X-V3-Token"] = state.server.token;
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    let data;
    try { data = await response.json(); } catch (_) { throw new Error(`服务返回非 JSON 内容（HTTP ${response.status}）。`); }
    if (!response.ok || data.error) {
      const detail = data.error?.message || data.error || data.message || `HTTP ${response.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }
  function selectedJob() {
    const jobs = state.server?.jobs || [];
    const projectJobs = jobs.filter((job) => job.project_id === state.projectId).sort((a, b) => (Date.parse(b.updated_at || b.registered_at) || 0) - (Date.parse(a.updated_at || a.registered_at) || 0));
    return jobs.find((job) => job.id === state.jobId) || projectJobs.find((job) => activeStatuses.has(job.status)) || projectJobs[0] || null;
  }
  function anyActive() { return (state.server?.jobs || []).some((job) => activeStatuses.has(job.status)); }
  function renderLibrary() {
    const list = $("library-list");
    const query = $("library-search").value.trim().toLocaleLowerCase();
    const library = state.server?.library || [];
    const filtered = library.filter((item) => `${item.name || ""} ${item.path || ""}`.toLocaleLowerCase().includes(query));
    const fragment = document.createDocumentFragment();
    filtered.forEach((item) => {
      const option = document.createElement("option");
      option.value = String(item.id);
      option.textContent = item.name || item.path || item.id;
      option.title = item.path || item.name || item.id;
      option.selected = item.id === state.selectedAudio;
      fragment.append(option);
    });
    list.replaceChildren(fragment);
    if (!filtered.some((item) => item.id === state.selectedAudio)) list.selectedIndex = -1;
    $("library-count").textContent = query ? `${filtered.length} / ${library.length}` : String(library.length);
    updateSelectedFile();
  }
  function updateSelectedFile() {
    const item = (state.server?.library || []).find((entry) => entry.id === state.selectedAudio);
    $("selected-file").textContent = item ? `${item.name || item.path}${finite(item.duration_seconds) ? ` · ${clock(item.duration_seconds)}` : ""}` : "选择一首音频开始";
    $("selected-file").title = item?.path || "";
    updateActions();
  }
  function renderProjects() {
    const container = $("projects");
    const projects = state.server?.projects || [];
    const fragment = document.createDocumentFragment();
    [...projects].sort((a,b)=>String(b.created_at || '').localeCompare(String(a.created_at || ''))).forEach((project) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "project-button";
      button.classList.toggle("selected", project.id === state.projectId);
      button.setAttribute("aria-pressed", String(project.id === state.projectId));
      const icon = document.createElement("span");
      icon.className = "project-icon";
      icon.textContent = project.has_result ? "◈" : "◇";
      icon.setAttribute("aria-hidden", "true");
      const info = document.createElement("span");
      info.className = "project-info";
      const name = document.createElement("span");
      name.className = "project-name";
      name.textContent = project.name || project.id;
      const status = document.createElement("span");
      status.className = "project-state";
      status.textContent = project.has_result ? "结果已保存" : statusNames[project.status] || project.status || "待分析";
      if (project.includes_v2) status.textContent += " · 含旧 V2";
      if (project.created_at) status.textContent += " · " + new Date(project.created_at).toLocaleString("zh-CN", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
      info.append(name, status);
      button.append(icon, info);
      button.title = `${project.name || project.id}\n${project.id}`;
      button.addEventListener("click", () => loadProject(project.id).catch((error) => message(error.message)));
      fragment.append(button);
    });
    if (!projects.length) {
      const empty = document.createElement("p");
      empty.className = "empty-note";
      empty.textContent = "尚无项目。完成首次分析后会保存在这里。";
      fragment.append(empty);
    }
    container.replaceChildren(fragment);
  }
  function renderJob() {
    const job = selectedJob();
    $("job-panel").hidden = !job;
    if (!job) return;
    const exporting = /export|render|video/.test(`${job.kind || ""} ${job.type || ""} ${job.stage || ""}`);
    $("job-title").textContent = `${exporting ? "视频导出" : "分析任务"} · ${statusNames[job.status] || job.status}`;
    const systemLabel = names[job.system] ? ` · ${names[job.system]}` : "";
    $("job-stage").textContent = `${stageNames[job.stage] || job.stage || ""}${systemLabel}`;
    const progress = typeof job.progress === "object" && job.progress !== null ? job.progress.fraction : job.progress;
    if (finite(progress)) $("job-progress").value = clamp(progress > 1 ? progress / 100 : progress, 0, 1);
    else if (successStatuses.has(job.status)) $("job-progress").value = 1;
    else $("job-progress").removeAttribute("value");
    const counts = [];
    if (finite(job.frames_done)) counts.push(`本阶段已处理 ${job.frames_done.toLocaleString()}${finite(job.frames_total) ? ` / ${job.frames_total.toLocaleString()}` : ""} 帧`);
    if (finite(job.members_done)) counts.push(`已完成 ${job.members_done.toLocaleString()}${finite(job.members_total) ? ` / ${job.members_total.toLocaleString()}` : ""} 个模型成员`);
    $("job-progress").setAttribute("aria-valuetext", counts.join("；") || (finite(progress) ? `本阶段 ${Math.round(clamp(progress > 1 ? progress / 100 : progress, 0, 1) * 100)}%` : successStatuses.has(job.status) ? "任务已完成" : "阶段进行中，总进度尚不可计算"));
    const detail = job.error?.message || job.error || job.message || job.detail || "";
    const detailText = detail ? String(typeof detail === "object" ? JSON.stringify(detail) : detail) : activeStatuses.has(job.status) ? "任务按真实阶段更新；不定长阶段不估算总百分比。" : `任务标识：${job.id}`;
    $("job-detail").textContent = [...counts, detailText].join(" · ");
    $("cancel-job").hidden = !activeStatuses.has(job.status) || job.status === "cancelling";
    $("resume-job").hidden = !recoverableStatuses.has(job.status);
    $("cancel-job").disabled = state.busy || !state.connected;
    $("resume-job").disabled = state.busy || !state.connected || anyActive();
    if (exporting && successStatuses.has(job.status) && job.project_id === state.projectId) {
      $("download-video").hidden = false;
      $("download-video").href = downloadUrl("preview.mp4");
    }
  }
  async function refreshState(manual = false) {
    if (state.refreshing) return;
    state.refreshing = true;
    clearTimeout(state.pollTimer);
    try {
      const previousLibrary = JSON.stringify(state.server?.library || []);
      const previousProjects = JSON.stringify(state.server?.projects || []);
      state.server = await api("/api/state");
      setConnection(true);
      if (previousLibrary !== JSON.stringify(state.server.library || [])) renderLibrary();
      if (previousProjects !== JSON.stringify(state.server.projects || [])) renderProjects();
      $("model-name").textContent = state.server.model?.description || state.server.model?.id || "B01 · 三种子等权集成";
      const job = selectedJob();
      if (job && successStatuses.has(job.status) && !state.finishedJobs.has(job.id)) {
        state.finishedJobs.add(job.id);
        if (job.project_id === state.projectId) await loadProject(state.projectId, true);
      }
      if (!state.projectId && state.server.projects?.length) {
        const last = state.server.projects.find((project) => project.has_result) || state.server.projects[0];
        await loadProject(last.id);
      }
      renderJob();
      if (manual) message("已刷新本地项目与任务状态。", "info");
    } catch (error) {
      setConnection(false);
      if (manual || !state.server) message(`无法连接本地服务：${error.message}`);
    } finally {
      state.refreshing = false;
      state.pollTimer = setTimeout(() => refreshState(), anyActive() ? 2000 : 10000);
    }
  }
  async function loadProject(id, preservePlayback = false) {
    const request = ++state.projectRequest;
    const data = await api(`/api/project?id=${encodeURIComponent(id)}`);
    if (request !== state.projectRequest) return;
    const changed = state.projectId !== id;
    state.projectId = id;
    state.project = data.project || {};
    state.result = data.result || null;
    state.smoothCache.clear();
    if (changed) {
      state.jobId = null;
      state.loopA = null;
      state.loopB = null;
      state.looping = false;
      $("download-video").hidden = true;
      message("");
      updateLoop();
    }
    const name = state.project.name || state.result?.audio?.name || state.project.audio?.name || id;
    $("project-title").textContent = name;
    $("project-subtitle").textContent = state.result ? "查看原始预测、独立声学属性与同步艺术预览。" : "项目已建立，等待完整分析结果。";
    $("project-id").textContent = id;
    $("project-status").textContent = state.result ? "结果已保存" : statusNames[state.project.status] || "等待分析";
    $("project-status").classList.toggle("ready", !!state.result);
    if (changed || !preservePlayback && !audio.getAttribute("src")) {
      audio.pause();
      audio.src = `/audio?id=${encodeURIComponent(id)}`;
      audio.load();
      $("export-start").value = "0";
      $("export-end").value = String(Math.min(15, duration() || 15));
    }
    renderProjects();
    renderJob();
    updateResult();
  }
  function updateActions() {
    const available = !!state.result && !!state.projectId;
    $("analyze").disabled = !state.selectedAudio || !state.connected || state.busy || anyActive();
    $("analyze").title = anyActive() ? "已有任务正在运行，完成或取消后可创建下一任务。" : "使用固定模型分析已有音频";
    $("export-video").disabled = !available || !state.connected || state.busy || anyActive();
    $("use-current").disabled = !available;
    const hasAudio = !!audio.getAttribute("src");
    $("set-loop-a").disabled = !hasAudio;
    $("set-loop-b").disabled = !hasAudio;
    $("toggle-loop").disabled = !hasAudio;
    for (const [id, file] of [["download-raw", "raw.csv"], ["download-display", "display.csv"], ["download-bundle", "bundle.json"]]) {
      const link = $(id);
      link.classList.toggle("disabled", !available);
      link.setAttribute("aria-disabled", String(!available));
      if (available) link.href = downloadUrl(file);
      else link.removeAttribute("href");
    }
  }
  function effectiveSmoothing() { return state.mode === "display" ? state.smoothing : 0; }
  function downloadUrl(file) {
    return `/download?id=${encodeURIComponent(state.projectId)}&file=${encodeURIComponent(file)}&smoothing_seconds=${effectiveSmoothing()}`;
  }
  function updateResult() {
    modelKeys.forEach((key) => {
      const system = state.result?.systems?.[key];
      const present = Array.isArray(system?.times_seconds) && system.times_seconds.length > 0;
      $(`${key}-availability`).textContent = present ? `${system.times_seconds.length.toLocaleString()} 点` : "未计算";
      $(`show-${key}`).disabled = !present;
    });
    const available = modelKeys.some((key) => state.result?.systems?.[key]?.times_seconds?.length);
    $("visual-empty").hidden = available;
    $("curves-empty").hidden = available;
    $("acoustic-empty").hidden = !!state.result?.acoustic?.times_seconds?.length;
    $("sample-count").textContent = `${modelKeys.filter((key) => state.result?.systems?.[key]?.times_seconds?.length).length} 个模型序列 · 保存原生采样`;
    $("smoothing").disabled = state.mode !== "display";
    $("curve-caption").textContent = state.mode === "raw" ? "原始预测 · 未应用显示平滑" : state.smoothing === 0.5 ? "0.5 秒窗对 0.5 秒原生采样保持原值" : state.smoothing ? `仅显示平滑 ${state.smoothing} 秒 · 原始文件不变` : "显示曲线 · 平滑关闭";
    $("visual-mode").textContent = effectiveSmoothing() ? `显示平滑 ${effectiveSmoothing()} 秒` : "原始预测映射";
    $("export-note").textContent = effectiveSmoothing() ? `下一次视频导出与当前显示 CSV 使用 ${effectiveSmoothing()} 秒平滑；已生成的视频和原始 CSV 不变。` : "下一次视频导出使用原始预测；已生成的视频与原始 CSV 不变。";
    const exportLimit = duration();
    if (exportLimit) {
      $("export-start").max = String(exportLimit);
      $("export-end").max = String(exportLimit);
    }
    updateActions();
    dirtyCharts = true;
    lastVisualKey = "";
    renderReadouts();
  }
  function medianStep(times) {
    if (!times || times.length < 2) return 0.5;
    if (timeSteps.has(times)) return timeSteps.get(times);
    const differences = [];
    for (let index = 1; index < Math.min(times.length, 1000); index++) {
      const delta = times[index] - times[index - 1];
      if (delta > 0 && finite(delta)) differences.push(delta);
    }
    differences.sort((a, b) => a - b);
    const step = differences.length ? differences[0] : 0.5;
    timeSteps.set(times, step);
    return step;
  }
  function smoothValues(system, seconds) {
    if (!seconds) return system.raw_va;
    const times = system.times_seconds;
    const values = system.raw_va;
    const maxGap = medianStep(times) * 1.5;
    const output = values.map(() => [null, null]);
    for (let axis = 0; axis < 2; axis++) {
      let start = 0;
      while (start < times.length) {
        if (!finite(values[start]?.[axis])) { start++; continue; }
        let end = start + 1;
        while (end < times.length && finite(values[end]?.[axis]) && times[end] - times[end - 1] <= maxGap + 1e-8) end++;
        let left = start, right = start, sum = 0;
        for (let index = start; index < end; index++) {
          while (right < end && times[right] <= times[index] + seconds / 2 + 1e-8) sum += values[right++][axis];
          while (left < right && times[left] < times[index] - seconds / 2 - 1e-8) sum -= values[left++][axis];
          output[index][axis] = right > left ? sum / (right - left) : values[index][axis];
        }
        start = end;
      }
    }
    return output;
  }
  function valuesFor(key) {
    const system = state.result?.systems?.[key];
    if (!system) return null;
    const seconds = effectiveSmoothing();
    if (!seconds) return system.raw_va;
    const cacheKey = `${key}:${seconds}`;
    if (!state.smoothCache.has(cacheKey)) state.smoothCache.set(cacheKey, smoothValues(system, seconds));
    return state.smoothCache.get(cacheKey);
  }
  function interpolate(times, values, time, axis = null) {
    if (!times?.length || !values?.length || !finite(time)) return null;
    const at = (index) => axis === null ? values[index] : values[index]?.[axis];
    if (time <= times[0]) return finite(at(0)) ? at(0) : null;
    if (time >= times[times.length - 1]) return finite(at(times.length - 1)) ? at(times.length - 1) : null;
    let lo = 0, hi = times.length - 1;
    while (hi - lo > 1) { const mid = Math.floor((lo + hi) / 2); if (times[mid] <= time) lo = mid; else hi = mid; }
    if (time === times[lo] || lo === hi) return finite(at(lo)) ? at(lo) : null;
    if (time === times[hi]) return finite(at(hi)) ? at(hi) : null;
    const a = at(lo), b = at(hi), delta = times[hi] - times[lo];
    if (!finite(a) || !finite(b) || delta <= 0 || delta > medianStep(times) * 1.5) return null;
    return a + (b - a) * (time - times[lo]) / delta;
  }
  function currentValues() {
    const time = audio.currentTime || 0;
    const key = modelKeys.find((candidate) => state.result?.systems?.[candidate]?.times_seconds?.length);
    const system = key ? state.result.systems[key] : null;
    const acoustic = state.result?.acoustic;
    return { key, time, v: system ? interpolate(system.times_seconds, valuesFor(key), time, 0) : null, a: system ? interpolate(system.times_seconds, valuesFor(key), time, 1) : null, dv: system ? interpolate(system.times_seconds, system.seed_disagreement_std, time, 0) : null, da: system ? interpolate(system.times_seconds, system.seed_disagreement_std, time, 1) : null, energy: acoustic ? interpolate(acoustic.times_seconds, acoustic.rms_dbfs, time) : null, centroid: acoustic ? interpolate(acoustic.times_seconds, acoustic.spectral_centroid_hz, time) : null };
  }
  function signed(value) { return finite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(3)}` : "—"; }
  function unitReadout(id, value, unit, decimals) {
    const element = $(id);
    element.replaceChildren(document.createTextNode(`${finite(value) ? value.toFixed(decimals) : "—"} `));
    const small = document.createElement("small"); small.textContent = unit; element.append(small);
  }
  function renderReadouts() {
    const current = currentValues();
    $("sample-time").textContent = clock(current.time, true);
    $("time-label").textContent = `${clock(current.time)} / ${clock(duration())}`;
    $("readout-v").textContent = signed(current.v);
    $("readout-a").textContent = signed(current.a);
    ["v", "a"].forEach((axis) => { const marker = $(`meter-${axis}`); marker.style.opacity = finite(current[axis]) ? "1" : "0"; marker.style.left = `${(clamp(current[axis] || 0, -1, 1) + 1) * 50}%`; });
    unitReadout("readout-energy", current.energy, "dBFS", 1);
    unitReadout("readout-centroid", current.centroid, "Hz", 0);
    $("readout-disagreement").textContent = `V ${finite(current.dv) ? current.dv.toFixed(3) : "—"} / A ${finite(current.da) ? current.da.toFixed(3) : "—"}`;
    $("visual-model").textContent = current.key ? names[current.key] : "等待模型结果";
  }
  function fitCanvas(canvas) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth));
    const height = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
  }
  function drawVisual() {
    const { ctx, width, height } = fitCanvas($("visual-canvas"));
    ctx.fillStyle = "#0d1625"; ctx.fillRect(0, 0, width, height);
    const current = currentValues();
    if (!finite(current.v) || !finite(current.a) || !finite(current.energy)) return;
    const e = clamp((current.energy + 60) / 60, 0, 1);
    const a = (current.a + 1) / 2;
    const t = current.time;
    for (let index = 0; index < 80; index++) {
      const phi = index * 2.3999632297;
      const angle = phi + t * (0.12 + 0.2 * a);
      const radius = (30 + (index % 17) / 17 * Math.min(width, height) * 0.34) * (0.7 + 0.3 * e);
      const x = width / 2 + Math.cos(angle) * radius;
      const y = height / 2 + Math.sin(angle * 0.87) * radius;
      const hue = ((current.v + 1) * 120 + index % 30) % 360;
      const size = 1 + index % 3 + 3 * e;
      ctx.fillStyle = `hsla(${hue}, 68%, 66%, ${0.45 + 0.5 * e})`;
      ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2); ctx.fill();
    }
  }
  function frame(ctx, width, top, height, title, low, high, total, unit) {
    const left = 57, right = width - 17, bottom = top + height;
    const mapX = (time) => left + clamp(time / (total || 1), 0, 1) * (right - left);
    const mapY = (value) => bottom - (value - low) / (high - low) * height;
    ctx.font = '10px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = "left"; ctx.fillStyle = "#b6c8d9"; ctx.fillText(title, left, top - 12);
    ctx.textAlign = "right"; ctx.fillStyle = "#6e8199"; ctx.fillText(unit, right, top - 12);
    [low, (low + high) / 2, high].forEach((value) => {
      const y = mapY(value); ctx.strokeStyle = value === 0 ? "#3b4b61" : "#263449"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.fillStyle = "#7387a0"; ctx.textAlign = "right"; ctx.fillText(Math.abs(value) >= 100 ? String(Math.round(value)) : value.toFixed(value % 1 ? 1 : 0), left - 9, y + 3);
    });
    for (let index = 0; index <= 4; index++) {
      const time = total * index / 4, x = mapX(time);
      ctx.strokeStyle = "#223044"; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      ctx.fillStyle = "#687d96"; ctx.textAlign = index === 0 ? "left" : index === 4 ? "right" : "center"; ctx.fillText(clock(time), x, bottom + 18);
    }
    return { left, right, top, bottom, mapX, mapY };
  }
  function line(ctx, box, times, values, color, axis = null) {
    if (!times?.length || !values?.length) return;
    ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top - 1, box.right - box.left, box.bottom - box.top + 2); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = color === colors.b01 ? 1.65 : 1.15; ctx.globalAlpha = color === colors.b01 ? 1 : 0.8;
    ctx.beginPath();
    let drawing = false, previousTime = null;
    const gap = medianStep(times) * 1.5;
    for (let index = 0; index < times.length; index++) {
      const value = axis === null ? values[index] : values[index]?.[axis];
      if (!finite(value) || !finite(times[index])) { drawing = false; continue; }
      const x = box.mapX(times[index]), y = box.mapY(value);
      if (!drawing || previousTime !== null && times[index] - previousTime > gap) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      drawing = true; previousTime = times[index];
    }
    ctx.stroke(); ctx.restore();
  }
  function cursor(ctx, box, total) {
    if (!state.result || !total) return;
    const x = box.mapX(audio.currentTime || 0);
    ctx.strokeStyle = "#d5e7ee99"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, box.top); ctx.lineTo(x, box.bottom); ctx.stroke(); ctx.setLineDash([]);
    if (state.loopA !== null && state.loopB !== null && state.loopB > state.loopA) {
      ctx.fillStyle = state.looping ? "#70e0cc15" : "#70e0cc08"; ctx.fillRect(box.mapX(state.loopA), box.top, box.mapX(state.loopB) - box.mapX(state.loopA), box.bottom - box.top);
    }
  }
  function drawEmotion() {
    const { ctx, width, height } = fitCanvas($("emotion-canvas"));
    ctx.clearRect(0, 0, width, height);
    const total = duration();
    const chartHeight = (height - 91) / 2;
    [0, 1].forEach((axis) => {
      const box = frame(ctx, width, 30 + axis * (chartHeight + 49), chartHeight, axis === 0 ? "V · 效价" : "A · 唤醒", -1, 1, total, "−1 / +1");
      [...modelKeys].reverse().forEach((key) => { const system = state.result?.systems?.[key]; if (system && $(`show-${key}`).checked) line(ctx, box, system.times_seconds, valuesFor(key), colors[key], axis); });
      cursor(ctx, box, total);
    });
  }
  function drawAcoustic() {
    const { ctx, width, height } = fitCanvas($("acoustic-canvas"));
    ctx.clearRect(0, 0, width, height);
    const total = duration(), acoustic = state.result?.acoustic;
    const chartHeight = (height - 84) / 2;
    let minimumEnergy = -80;
    for (const value of acoustic?.rms_dbfs || []) if (finite(value) && value < minimumEnergy) minimumEnergy = value;
    minimumEnergy = Math.floor(minimumEnergy / 20) * 20;
    const energyBox = frame(ctx, width, 25, chartHeight, "RMS · 能量", minimumEnergy, 0, total, "dBFS");
    line(ctx, energyBox, acoustic?.times_seconds, acoustic?.rms_dbfs, "#e8bc7d");
    cursor(ctx, energyBox, total);
    const centroids = acoustic?.spectral_centroid_hz || [];
    let maxCentroid = 4000;
    for (const value of centroids) if (finite(value) && value > maxCentroid) maxCentroid = value;
    maxCentroid = Math.ceil(maxCentroid / 1000) * 1000;
    const centroidBox = frame(ctx, width, 25 + chartHeight + 47, chartHeight, "频谱重心 · 明亮度", 0, maxCentroid, total, "Hz");
    line(ctx, centroidBox, acoustic?.times_seconds, centroids, "#7aa8de");
    cursor(ctx, centroidBox, total);
  }
  function updateLoop() {
    $("loop-label").textContent = `A ${state.loopA === null ? "—" : clock(state.loopA, true)} · B ${state.loopB === null ? "—" : clock(state.loopB, true)}`;
    $("toggle-loop").setAttribute("aria-pressed", String(state.looping));
    $("toggle-loop").textContent = state.looping ? "循环开启" : "循环关闭";
    dirtyCharts = true;
  }
  function seek(time) { if (duration()) audio.currentTime = clamp(time, 0, duration()); dirtyCharts = true; }
  async function runAction(action) {
    if (state.busy) return;
    state.busy = true; message(""); updateActions(); renderJob();
    try { await action(); await refreshState(); } catch (error) { message(error.message); } finally { state.busy = false; updateActions(); renderJob(); }
  }
  $("library-search").addEventListener("input", renderLibrary);
  $("library-list").addEventListener("change", () => { state.selectedAudio = $("library-list").value || null; updateSelectedFile(); });
  $("refresh").addEventListener("click", () => refreshState(true));
  $("analyze").addEventListener("click", () => runAction(async () => {
    const result = await api("/api/analyze", { audio_id: state.selectedAudio, device: $("device").value, include_v2: $("include-v2").checked });
    await loadProject(result.project_id);
    state.jobId = result.job_id;
  }));
  $("cancel-job").addEventListener("click", () => runAction(async () => { const job = selectedJob(); if (job) await api("/api/cancel", { job_id: job.id }); }));
  $("resume-job").addEventListener("click", () => runAction(async () => { const job = selectedJob(); if (job) { const response = await api("/api/resume", { job_id: job.id }); state.jobId = response.job_id || job.id; state.finishedJobs.delete(state.jobId); } }));
  document.querySelectorAll('input[name="curve-mode"]').forEach((input) => input.addEventListener("change", () => { state.mode = input.value; updateResult(); }));
  $("smoothing").addEventListener("change", () => { state.smoothing = Number($("smoothing").value); updateResult(); });
  modelKeys.forEach((key) => $(`show-${key}`).addEventListener("change", () => { dirtyCharts = true; }));
  $("set-loop-a").addEventListener("click", () => { state.loopA = audio.currentTime; if (state.loopB !== null && state.loopB <= state.loopA) { state.loopB = null; state.looping = false; } updateLoop(); });
  $("set-loop-b").addEventListener("click", () => { const time = audio.currentTime; if (state.loopA === null) state.loopA = 0; if (time <= state.loopA) { message("循环终点 B 必须晚于起点 A。", "info"); return; } state.loopB = time; updateLoop(); });
  $("toggle-loop").addEventListener("click", () => { if (state.loopA === null || state.loopB === null || state.loopB <= state.loopA) { message("先在播放器中设置 A、B 两个循环位置。", "info"); return; } state.looping = !state.looping; if (state.looping && (audio.currentTime < state.loopA || audio.currentTime >= state.loopB)) seek(state.loopA); updateLoop(); });
  audio.addEventListener("loadedmetadata", () => { updateResult(); renderReadouts(); });
  audio.addEventListener("timeupdate", () => { if (state.looping && state.loopB !== null && audio.currentTime >= state.loopB) audio.currentTime = state.loopA; renderReadouts(); });
  audio.addEventListener("ended", () => { if (state.looping && state.loopA !== null) { audio.currentTime = state.loopA; audio.play().catch((error) => message(error.message)); } });
  audio.addEventListener("error", () => { if (audio.getAttribute("src")) message("浏览器暂时无法播放这份音频。请检查本地服务与项目音频是否可用；这不会更改已保存的分析结果。"); });
  for (const id of ["emotion-canvas", "acoustic-canvas"]) {
    $(id).addEventListener("click", (event) => { const bounds = $(id).getBoundingClientRect(); seek((event.clientX - bounds.left - 57) / (bounds.width - 74) * duration()); });
    $(id).addEventListener("keydown", (event) => { if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); seek(audio.currentTime + (event.key === "ArrowLeft" ? -1 : 1)); } });
  }
  $("use-current").addEventListener("click", () => { const start = Math.min(audio.currentTime, Math.max(0, duration() - 0.1)); $("export-start").value = start.toFixed(1); $("export-end").value = Math.min(duration(), start + 15).toFixed(1); });
  $("export-video").addEventListener("click", () => runAction(async () => {
    const start = Number($("export-start").value), end = Number($("export-end").value);
    if (!finite(start) || !finite(end) || start < 0 || end <= start || end > duration() + 0.001 || end - start > 30.001) throw new Error("请选择音频范围内、终点晚于起点且不超过 30 秒的片段。");
    $("download-video").hidden = true;
    const result = await api("/api/export", { project_id: state.projectId, smoothing_seconds: effectiveSmoothing(), start, end });
    state.jobId = result.job_id;
  }));
  document.querySelectorAll("a.download").forEach((link) => link.addEventListener("click", (event) => { if (link.getAttribute("aria-disabled") === "true") event.preventDefault(); }));
  function tick(timestamp) {
    const time = audio.currentTime || 0;
    const visualKey = `${state.projectId}:${state.mode}:${state.smoothing}`;
    if (time !== lastVisualTime || visualKey !== lastVisualKey || dirtyCharts) { drawVisual(); lastVisualTime = time; lastVisualKey = visualKey; }
    if (dirtyCharts || Math.abs(time - lastChartTime) > 0.1) { drawEmotion(); drawAcoustic(); dirtyCharts = false; lastChartTime = time; }
    if (timestamp - lastReadout > 100) { renderReadouts(); lastReadout = timestamp; }
    if (state.looping && state.loopB !== null && time >= state.loopB && !audio.paused) audio.currentTime = state.loopA;
    requestAnimationFrame(tick);
  }
  const resizeObserver = new ResizeObserver(() => { dirtyCharts = true; lastVisualKey = ""; });
  [$("visual-canvas"), $("emotion-canvas"), $("acoustic-canvas")].forEach((canvas) => resizeObserver.observe(canvas));
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { dirtyCharts = true; refreshState(); } });
  updateActions();
  refreshState();
  requestAnimationFrame(tick);
})();
