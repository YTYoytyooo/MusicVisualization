# ============================================================
# mcts.py
# ------------------------------------------------------------
# 使用蒙特卡洛树搜索（MCTS）在 8 维视觉参数空间中
# 为当前音乐情感找到最匹配的视觉状态（VisualState）。
#
# 为什么用 MCTS 而非直接映射？
#   情感 → 视觉的映射存在多目标冲突（颜色/速度/密度/湍流
#   都需要同时满足），无法用简单线性公式求解。
#   MCTS 通过模拟-回溯迭代，在离散-连续混合空间中
#   探索并聚焦于高奖励区域，比随机搜索效率高得多。
#
# 单次搜索性能：<5ms（纯 numpy 浮点运算，无神经网络调用）
# ============================================================

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ── VisualState：8 维视觉参数空间 ─────────────────────────────

@dataclass
class VisualState:
    """
    描述某一时刻视觉效果的 8 个参数，是 MCTS 的"状态"。
    每个字段均有对应的合法范围（BOUNDS），
    mutate() 方法在邻域内随机扰动生成子状态，供 MCTS 展开使用。
    """
    hue_base: float = 180.0         # 主色相（度），0-360，冷暖色调的核心控制
    hue_range: float = 60.0         # 粒子色相的散布范围（度），20-120
    saturation: float = 0.8         # 颜色饱和度，0.3-1.0，高值=鲜艳
    brightness: float = 0.5         # 背景及粒子亮度基准，0.1-0.9
    particle_count: int = 200       # 活跃粒子数量，50-500
    particle_speed: float = 2.0     # 粒子运动速度倍率，0.5-8.0
    field_turbulence: float = 0.3   # 向量场湍流强度，0.0=平滑流动，1.0=随机混沌
    trail_length: int = 20          # 轨迹保留帧数，5-60，长轨迹=流畅感

    # 各字段的合法取值范围，供 mutate() 和 _emotion_init() 参考
    BOUNDS = {
        'hue_base':         (0.0,   360.0),
        'hue_range':        (20.0,  120.0),
        'saturation':       (0.3,   1.0),
        'brightness':       (0.1,   0.9),
        'particle_count':   (50,    500),
        'particle_speed':   (0.5,   8.0),
        'field_turbulence': (0.0,   1.0),
        'trail_length':     (5,     60),
    }

    def mutate(self, rng: np.random.Generator, sigma: float = 0.15) -> 'VisualState':
        """
        在当前状态邻域内生成一个随机变异状态，用于 MCTS 展开子节点。

        sigma 控制扰动幅度：sigma=0.15 意味着扰动标准差
        为该字段取值范围的 15%（例如 hue_range 范围 100°，标准差 15°）。
        所有字段在扰动后 clip 到合法范围。

        hue_base 特殊处理：色相是环形空间（0° = 360°），
        直接加 delta 后取模 360，不 clip，避免红色端（0°/360°）出现边界突变。
        """
        b = VisualState.BOUNDS

        def _perturb_float(val: float, lo: float, hi: float) -> float:
            delta = (hi - lo) * sigma * rng.standard_normal()
            return float(np.clip(val + delta, lo, hi))

        def _perturb_int(val: int, lo: int, hi: int) -> int:
            delta = (hi - lo) * sigma * rng.standard_normal()
            return int(np.clip(round(val + delta), lo, hi))

        # 色相在 360° 环形空间内扰动，取模而非 clip 以支持跨 0°/360° 的平滑过渡
        hue_lo, hue_hi = b['hue_base']
        hue_delta = (hue_hi - hue_lo) * sigma * rng.standard_normal()
        new_hue = (self.hue_base + hue_delta) % 360.0

        return VisualState(
            hue_base=new_hue,
            hue_range=_perturb_float(self.hue_range, *b['hue_range']),
            saturation=_perturb_float(self.saturation, *b['saturation']),
            brightness=_perturb_float(self.brightness, *b['brightness']),
            particle_count=_perturb_int(self.particle_count, *b['particle_count']),
            particle_speed=_perturb_float(self.particle_speed, *b['particle_speed']),
            field_turbulence=_perturb_float(self.field_turbulence, *b['field_turbulence']),
            trail_length=_perturb_int(self.trail_length, *b['trail_length']),
        )


# ── MCTSNode：MCTS 搜索树的节点 ───────────────────────────────

@dataclass
class MCTSNode:
    """
    MCTS 搜索树中的单个节点，记录对应的 VisualState 及统计数据。

    n_visits：该节点被访问（模拟）的次数，越多表示被探索越充分
    q_value：该节点累积的奖励总和，除以 n_visits 得到平均奖励
    """
    state: VisualState
    parent: Optional['MCTSNode'] = field(default=None, repr=False)
    children: List['MCTSNode'] = field(default_factory=list, repr=False)
    n_visits: int = 0
    q_value: float = 0.0

    def uct_score(self, c: float = 1.414) -> float:
        """
        UCT（Upper Confidence Bound for Trees）分数，平衡探索与利用。

        公式：Q/N + c × √(ln(N_parent) / N)
          - Q/N：exploitation（利用）：该节点的平均奖励，高=质量好
          - c × √(...)：exploration（探索）：访问少的节点有加分，
            鼓励 MCTS 探索未充分搜索的区域

        c = √2 ≈ 1.414 是理论最优的探索系数（UCB1 的标准值）。
        未被访问的节点返回 inf，确保每个子节点至少被模拟一次。
        """
        if self.n_visits == 0:
            return float('inf')  # 未访问节点优先级最高
        if self.parent is None or self.parent.n_visits == 0:
            return self.q_value / self.n_visits  # 根节点无父节点，不加探索项
        exploitation = self.q_value / self.n_visits
        exploration = c * math.sqrt(math.log(self.parent.n_visits) / self.n_visits)
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """是否为叶节点（尚未展开子节点）。"""
        return len(self.children) == 0

    def best_child(self) -> 'MCTSNode':
        """选择 UCT 分数最高的子节点（用于 Selection 阶段）。"""
        return max(self.children, key=lambda c: c.uct_score())

    def best_action(self) -> 'MCTSNode':
        """
        选择访问次数最多的子节点作为最终结果（而非 UCT 最高）。
        访问次数更稳定：UCT 分数会随探索加分波动，
        而多次模拟后最优节点自然积累最高访问次数。
        """
        return max(self.children, key=lambda c: c.n_visits)


# ── 奖励函数：情感 → 视觉参数的映射准则 ──────────────────────

def _color_match(hue_base: float, valence: float) -> float:
    """
    根据情感效价（valence）计算色相匹配分数。

    映射规则（基于色彩心理学）：
      正效价（开心/积极）→ 暖色目标：30-60°（橙黄色）
        valence=1.0 → target=30°（橙色）
        valence=0.0 → target=60°（黄色）
      负效价（悲伤/消极）→ 冷色目标：180-240°（青蓝色）
        valence=0.0 → target=180°（青色）
        valence=-1.0 → target=240°（蓝色）

    angular_diff 使用圆弧距离（min(diff, 360-diff)）处理 0°/360° 环绕。
    分数 = 1 - 角度差/180，满分 1.0 = 完全匹配目标色相。
    """
    if valence >= 0:
        target_hue = 30.0 + (1.0 - valence) * 30.0  # 范围 [30°, 60°] 橙→黄
    else:
        target_hue = 180.0 + (-valence) * 60.0        # 范围 [180°, 240°] 青→蓝

    diff = abs(hue_base - target_hue)
    angular_diff = min(diff, 360.0 - diff)  # 取较短的圆弧距离
    return 1.0 - angular_diff / 180.0


def _speed_match(particle_speed: float, arousal: float) -> float:
    """
    根据唤醒度（arousal）计算粒子速度匹配分数。

    高唤醒（arousal=1）→ 目标速度 8.0（快速飞舞）
    低唤醒（arousal=-1）→ 目标速度 0.5（缓慢漂移）
    线性插值：target = 0.5 + 7.5 × ((arousal+1)/2)
    """
    target = 0.5 + 7.5 * ((arousal + 1.0) / 2.0)
    return max(0.0, 1.0 - abs(particle_speed - target) / 7.5)


def _density_match(particle_count: int, arousal: float) -> float:
    """
    根据唤醒度计算粒子密度匹配分数。

    高唤醒 → 粒子密集（500个），低唤醒 → 稀疏（50个）。
    视觉上：活跃段落粒子爆炸式增多，安静段落只有寥寥几粒。
    """
    target = 50.0 + 450.0 * ((arousal + 1.0) / 2.0)
    return max(0.0, 1.0 - abs(particle_count - target) / 450.0)


def _turbulence_match(field_turbulence: float, trail_length: int,
                      arousal: float, valence: float) -> float:
    """
    根据情感计算湍流度和轨迹长度的匹配分数（合并评估）。

    湍流度（60% 权重）：
      高唤醒 + 负效价（如激烈的小调摇滚）→ 最大湍流
      公式：target_turb = clip(0.5 + 0.5×arousal - 0.3×valence, 0, 1)
      → arousal 提升湍流，valence 降低湍流（正面情感趋向流畅）

    轨迹长度（40% 权重）：
      正效价 → 长轨迹（5+55×(valence+1)/2，最长 60 帧）→ 流畅连续感
      负效价 → 短轨迹（最短 5 帧）→ 破碎感
    """
    # 湍流目标：高唤醒且低效价时最混乱
    target_turb = float(np.clip(0.5 + 0.5 * arousal - 0.3 * valence, 0.0, 1.0))
    turb_score = 1.0 - abs(field_turbulence - target_turb)

    # 轨迹长度目标：正效价 → 长轨迹（流畅），负效价 → 短轨迹（破碎）
    target_trail = int(5 + 55 * ((valence + 1.0) / 2.0))
    trail_score = max(0.0, 1.0 - abs(trail_length - target_trail) / 55.0)

    return 0.6 * turb_score + 0.4 * trail_score


def _reward(state: VisualState, valence: float, arousal: float) -> float:
    """
    综合奖励函数：评估 VisualState 与当前情感的匹配程度。

    权重分配反映各维度的感知重要性：
      色相（40%）：颜色是最直接的情感传达，权重最高
      速度（25%）：运动节奏感，与 arousal 直接关联
      密度（20%）：视觉"热闹程度"，也由 arousal 驱动
      湍流/轨迹（15%）：细腻的氛围感，权重较低但不可忽视

    返回值在 [0, 1] 范围内（各子函数均返回 [0,1]）。
    """
    return (
        0.40 * _color_match(state.hue_base, valence)
        + 0.25 * _speed_match(state.particle_speed, arousal)
        + 0.20 * _density_match(state.particle_count, arousal)
        + 0.15 * _turbulence_match(state.field_turbulence, state.trail_length, arousal, valence)
    )


# ── MCTS 搜索主类 ─────────────────────────────────────────────

class MCTS:
    """
    蒙特卡洛树搜索，在 8 维 VisualState 空间中找到高奖励状态。

    每次调用 search() 执行 n_iter=200 次迭代，每次迭代包含：
      1. Selection：沿 UCT 最高路径向下，找到未展开的叶节点
      2. Expansion：对叶节点展开 branching=5 个变异子节点
      3. Rollout：从子节点出发随机模拟 3 步，计算平均奖励
      4. Backpropagation：将奖励沿父链反向累积

    连贯性设计：
      _last_state 保存上次搜索结果，作为下次搜索的起点，
      确保视觉效果在相邻时间点之间平滑过渡，不出现跳变。
    """
    def __init__(self, n_iter: int = 200, branching: int = 5, c: float = 1.414):
        self.n_iter = n_iter      # 每次搜索的迭代次数（越多质量越高，越慢）
        self.branching = branching  # 每次展开的子节点数（树的宽度）
        self.c = c                # UCT 探索系数（√2 为理论最优值）
        self._last_state: Optional[VisualState] = None  # 上次搜索结果（连贯性）
        self._rng = np.random.default_rng()  # 固定种子会使每首歌结果一致

    def _emotion_init(self, valence: float, arousal: float) -> VisualState:
        """
        根据情感值创建一个"情感感知"的初始 VisualState，
        作为 MCTS 第一次搜索的根节点起点。

        不使用完全随机初始化，而是基于情感规则构造初始点，
        使 MCTS 从接近最优解的位置开始搜索，大幅提升收敛速度。
        （随机初始化需要更多迭代才能找到同等质量的解）

        色相规则（与 _color_match 目标一致）：
          正效价 → 暖色（30-60°）
          负效价 → 冷色（180-240°）
        其余参数均随 arousal 线性缩放（a_norm 为归一化到 [0,1] 的唤醒度）。
        """
        if valence >= 0:
            hue = 30.0 + (1.0 - valence) * 30.0   # 30-60°（橙黄）
        else:
            hue = 180.0 + (-valence) * 60.0         # 180-240°（青蓝）

        a_norm = (arousal + 1.0) / 2.0  # arousal 从 [-1,1] 归一化到 [0,1]
        return VisualState(
            hue_base=hue,
            hue_range=20.0 + a_norm * 100.0,        # 高唤醒→色彩丰富（宽色相范围）
            saturation=0.5 + a_norm * 0.5,           # 高唤醒→高饱和
            brightness=0.2 + a_norm * 0.6,           # 高唤醒→高亮度
            particle_count=int(50 + a_norm * 450),   # 高唤醒→粒子密集
            particle_speed=0.5 + a_norm * 7.5,       # 高唤醒→快速运动
            field_turbulence=float(np.clip(0.5 + 0.5 * arousal - 0.3 * valence, 0.0, 1.0)),
            trail_length=int(5 + ((valence + 1.0) / 2.0) * 55.0),  # 正效价→长轨迹
        )

    def search(
        self,
        valence: float,
        arousal: float,
        init_state: Optional[VisualState] = None,  # 外部指定起点（用于连贯性）
    ) -> VisualState:
        """
        执行 MCTS 搜索，返回与当前情感最匹配的 VisualState。

        起点优先级：init_state > _last_state > _emotion_init()
        这种设计保证了：
          - 首次调用：用情感感知的初始点，快速找到合理解
          - 后续调用：从上次结果出发，保持视觉连贯性
          - 外部强制指定：main.py 传入 prev_state，控制过渡
        """
        if init_state is not None:
            start = init_state
        elif self._last_state is not None:
            start = self._last_state
        else:
            start = self._emotion_init(valence, arousal)
        root = MCTSNode(state=start)

        for _ in range(self.n_iter):
            # 1. Selection：沿 UCT 最高分路径向下到叶节点
            leaf = self._select(root)

            # 2. Expansion：若叶节点已被访问过，展开子节点（否则直接 rollout 根节点）
            if leaf.n_visits > 0:
                children = self._expand(leaf)
                leaf = children[0]  # 取第一个子节点进行 rollout

            # 3. Rollout：从当前节点随机模拟若干步，评估潜在奖励
            r = self._rollout(leaf.state, valence, arousal)

            # 4. Backpropagation：将奖励沿父链反向传递，更新统计数据
            self._backpropagate(leaf, r)

        # 搜索树为空（未展开任何子节点）时直接返回起点
        if root.is_leaf():
            self._last_state = start
            return start

        # 返回访问次数最多的子节点（最稳定的最优解）
        best = root.best_action()
        self._last_state = best.state
        return best.state

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection 阶段：从根节点沿 UCT 最高分路径向下，
        直到到达叶节点（未展开的节点）。
        """
        while not node.is_leaf():
            node = node.best_child()  # 递归选择 UCT 最高的子节点
        return node

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expansion 阶段：对节点展开 branching 个变异子节点。
        每个子节点是当前节点状态的随机邻域扰动。
        """
        children = []
        for _ in range(self.branching):
            new_state = node.state.mutate(self._rng)  # 随机变异
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)
            children.append(child)
        return children

    def _rollout(self, state: VisualState, valence: float, arousal: float, depth: int = 3) -> float:
        """
        Rollout 阶段：从当前状态出发随机模拟 depth 步，
        计算沿途奖励的均值作为该节点的"预期价值"。

        sigma=0.2 比 mutate() 默认的 0.15 更大，
        使 rollout 的探索范围更广，评估该邻域的平均质量，
        而非只看一步之内的局部最优。
        """
        total = _reward(state, valence, arousal)  # 当前状态的奖励
        current = state
        for _ in range(depth):
            current = current.mutate(self._rng, sigma=0.2)  # 随机游走
            total += _reward(current, valence, arousal)
        return total / (depth + 1)  # 对 depth+1 步取均值（含起点）

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation 阶段：将奖励沿父链向上传递，
        更新途经所有节点的 n_visits 和 q_value。
        这使 Selection 阶段的 UCT 分数得以正确反映各节点的历史表现。
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.n_visits += 1
            current.q_value += reward
            current = current.parent  # 向上追溯直到根节点
