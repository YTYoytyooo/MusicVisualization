> 目录整理（2026-09-30）：代码现位于 `versions/v2.0`；项目与输出位于工作区 `data/v2.0`，模型位于 `models`。下文旧绝对路径及历史验收记录保留作为来源记录，当前启动与路径说明见工作区根目录 `README.md`。

# 多模式粒子运动：实施与接口

本轮范围：情绪驱动的规则运动、平滑切换、人工时间段覆盖、即时轨道反馈、撤销/修订/预览、逐帧数值与检查点。保留旧运动引擎、模型、原始预测和历史输出，不自动重新训练。

## 兼容与数据合同

- 修订新增可选 `motion`；不含此字段的旧修订按旧引擎解释。创建修订或预览时省略字段或传入 null，继承基准修订配置；无基准时才采用旧引擎。配置结构：`{engine: "legacy" | "flow-v1", sensitivity: 1.0, min_hold_seconds: 6.0, transition_seconds: 2.0, overrides: []}`。
- 敏感度范围 0.5–2，最短保持 4–12 秒，自动过渡 1–4 秒且不超过保持时间。字段必须有限；未知字段拒绝。
- 模式：`rise` 上升、`fall` 下沉、`orbit` 环流、`spiral` 螺旋、`expand` 扩散、`gather` 汇聚、`meteor` 流星、`wave` 波浪、`turbulent` 湍流。
- 人工覆盖结构：`{id, enabled, mode, start_us, end_us, transition_in_us, transition_out_us, interpolation, note, params}`。模式仅使用区间，不按数值插值。进入/退出默认 2 秒，支持范围必须在歌曲内；已启用覆盖的支持范围不得重叠。
- params 为可选数值字段：`speed`（短边/秒，0–0.5）、`coherence`（0–1）、`turbulence`（0–1）、`direction_deg`（屏幕角度，右0/下90/左180/上270）、`center_x/center_y`（0–1）、`radius`（短边比例0.05–0.48）、`rotation`（-1/1）、`radial`（-1–1）、`pulse`（0–1）、`trail_seconds`（0–2）。缺省由模式预设补齐。
- 有效单帧结构：`{mode, source, reason, components:[{mode, weight, params}]}`。权重非负且和为1。普通自动切换最多两项；人工过渡叠加自动过渡时可暂时三项以保留连续性。source 为 auto/manual/transition。
- `timeline.motion`：`{config, times_us, auto, effective, segments, modes}`。只给 flow-v1 计算轨道；旧引擎返回空数组。segments 含 start_us/end_us/mode/source/reason。
- 新接口 `POST /api/projects/:id/motion-plan` 使用 `{base_revision, edits, smoothing, motion}`，返回上述 motion 对象。修订、preview 校验、冻结预览均接受 motion。视觉 MCTS 任务仍不依赖 motion；运动规划单独执行。
- 数值 timeline 仍由原有 edits/smoothing 计算。前端能依据已获取 auto 帧和当前 motion 在本地重算覆盖与轨道；V/A 或平滑变化后的自动形态需后台重新确认，不能以旧 auto 冒充新规划。

## 引擎接口

- `VideoRenderer.render_frame(..., motion=None)` 新增可选末尾参数。None 严格进入旧路径。
- flow-v1 单独按粒子位置求解析向量，不创建整屏向量场；目标速度跟随、限加速度/转向、模式出生策略、按秒渐隐拖尾，规则运动不依赖湍流。
- 新动态状态全部纳入 export_state/restore_state，包含模式相关相位/粒子半径/出生代数/拖尾/RNG；恢复前验证。旧引擎状态仍能使用。

## 集成顺序与验收

1. 固定接口，运动引擎与网页编辑独立实现；主任务负责规划/持久化/HTTP/渲染集成。
2. 验证纯方向、圆形稳定、过渡、重生与2秒拖尾边界；同种子和检查点重放一致。
3. 验证情绪候选、保持/迟滞、有限变化增强、特征对齐与人工覆盖，旧配置默认行为不变。
4. 验证网页即时轨道、撤销重做、修订刷新、冻结预览和实际视频/CSV。
5. 执行原回归与新增测试，生成隔离验收输出和真实音乐视频；结果不足时明确记录，不将启动任务等同完成。

## 实现位置

- `motion_schema.py`：共享合同、默认值与严格校验；`web/motion-math.js` 为网页即时计算对应实现。
- `studio/motion_features.py`：真实音频轻量特征、对齐与校验缓存；`studio/motion.py`：自动决策、迟滞、人工覆盖与混合。
- `motion_renderer.py`：九模式独立粒子模拟；`renderer.py`：兼容入口与完整检查点。
- `studio/store.py` / `studio/pipeline.py` / `studio/server.py`：修订、冻结预览、API、生成与逐帧导出；`main.py edit --motion` 支持配置文件。
- `web/app.js` / `web/index.html` / `web/style.css`：运动轨道、参数表单、事务撤销、草稿/版本与数值下载。
- `tests/test_motion_*.py`、`scripts/check_motion_math.cjs`、`scripts/browser_motion_check.cjs`、`scripts/verify_motion_release.py`：数值、物理、HTTP、浏览器与实际 MP4 验收。

操作说明见 [多模式运动工作台](MOTION_WORKBENCH.md)。实施文件存在不代表验证通过；新鲜执行结果见 [验收记录](VERIFICATION.md)。
