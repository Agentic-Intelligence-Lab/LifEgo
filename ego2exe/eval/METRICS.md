# ego2exe 评估指标手册

`eval_ego2exe.py` 输出的每一个数字是什么、怎么算的、该怎么读。字段名与
`eval_report.json` 里的键名一一对应，方便直接对着 JSON 查。用法见
[README.md](README.md)；本文档只讲指标含义，不讲怎么跑。

## 真机和 ego 比较的是哪个点

两边比的都是 TCP 位姿，但**不能直接读真机自己上报的 `poses.tcp_pose`**——它和
ego 定义的 TCP 不是同一个物理点。

- 真机自带的 `poses.tcp_pose` 是**夹爪中间**：实测（对着一条真机 jsonl 反算）
  确认 `tcp_pose = flange_pose + R_flange @ [0.13, 0, 0]`（法兰局部 +X 方向偏移
  0.13m，姿态与法兰完全相同）——这个偏移量对应的是 `tcp_offset_flange_frame`
  这个 metadata 字段。
- ego2exe 全链路（mink IK 的 `site:tcp`、hand2gripper 的 EEF 约定）定义的 TCP
  是**指尖**：`ego2exe/README.md` 里写明 `p_tcp = p_flange + R_flange @ [0.18,
  0, 0]`，同一根轴，只是量值从 0.13 改成了 0.18（2026-08-18 由 0.13 修正为
  0.18）。

两个点相差恰好 0.05m，沿同一根轴，量级和之前分析里反复出现的几十毫米系统性偏移
接近，不是可以忽略的误差。所以 `load_real_jsonl()` **默认不读 `poses.tcp_pose`**，
而是用 `poses.flange_pose` 现场按 ego 的定义重新算一遍
（`traj_metrics.flange_to_tip_pose()`）：

```
p_tip = p_flange + R_flange @ [tcp_offset_m, 0, 0]     # 默认 tcp_offset_m = 0.18
R_tip = R_flange
```

对应 `--pose-key`（默认 `tcp_tip_pose`）和 `--tcp-offset-m`（默认 0.18，NERO 专用，
换机械臂要跟着改）两个 CLI 参数。`--pose-key tcp_pose` 仍然保留，读机器人自己上报
的夹爪中心点，只用来对比/排查，不是默认行为。

在 `stack_object_horizontal`（31 条真机、30 条 ego 池化）上实测这个修正的效果——
姿态完全没动（这个修正只改位置，不改朝向），**位置误差单靠这一个修正就降了**：

| | 修正前（读 `tcp_pose`，夹爪中心 vs ego 指尖） | 修正后（`tcp_tip_pose`，指尖 vs 指尖） |
|---|---|---|
| `rho_pos` 均值 | 4.12 | **3.17** |
| `rho_offset` 均值 | 5.33 | **2.88** |

`rho_offset`（放置误差比值）几乎腰斩——说明之前归因给"hand2gripper/外参标定"的
那部分位置误差里，有相当一部分其实是这个 5cm 的定义错位，不是感知或标定的锅。
`rho_rot` 基本不变（7.59 → 7.56，符合预期：这个修正只沿 flange 局部 X 挪动了
位置，姿态 `R_tip = R_flange` 没有变化），姿态那部分之前的分析（33–40° 常量偏差
等）仍然成立，两个问题相互独立，互不解释。

## 核心设计：一切都是比值

```
rho = D(ego, real) / D(real, real)
```

绝对毫米数不能跨任务比较——同一套 pipeline 在 `stack_object_horizontal` 上噪声底是
17.0mm，在 `stack_bowl` 上是 57mm，纯粹因为参照集质量不同。`rho` 可以比较，
**`rho → 1` 意味着 ego 轨迹和真机的差距，已经和两条真机之间的天然差距一样大**，
再往下优化就是在拟合参照集本身解析不了的噪声。

分母（噪声底）一律用 **留一（LOO）** 口径算：从 N 条真机里留一条出来，算它和其余
N−1 条的距离，遍历 N 次取均值。这样待测量（1 条 ego vs N 条真机）和参照量
（1 条真机 vs N−1 条真机）用的是同一个估计量，比值才有意义，还顺带给出 z-score。

L0 的门禁**只报告，不阻断**——不管过没过，L1–L3 照算。任何"全局修正"类的数字
**只用留一验证的那一列**作数，自身拟合的残差只用来展示过拟合有多大，不能当效果。

---

## 评估范围：全局 / 分段 / 健康过滤

L0（门禁）、L2b（锚点）、L2c（分段）**永远看全局整条 episode**——锚点是特定事件
处的绝对位姿，分段本来就是按事件切出来的自然区间，两者都自带"该看哪一段"的
逻辑，不需要额外的范围控制。

L1、L2a、以及 L3 里除 `anchor_distributions` 之外的部分（`dispersion`、
`energy_test`、`manifold_overlap`、`global_correction`、`c2st`）都建立在同一份
`DistanceBook` 上，**共享同一个"范围"**，由两件事共同决定：

**1. Grasp 健康过滤（默认开启）。** 真机集合里出现次数最多的 grasp 翻转次数
（`mode_events`）当作这个任务的"正常事件数"；哪条 episode（真机或 ego）的翻转
次数跟这个众数不一致，就从 L1/L2a/L3 里排除。这不是新引入的假设——L2b/L2c
（`l2_anchors`/`l2_segments`）本来就用同一个"众数过滤"逻辑判断某条 episode
的锚点/分段能不能算，这里只是把 L1/L2a/L3 补齐到跟它们一致的口径。审计当前
代码发现的正是这个不一致：**修这个之前，一条 grasp 信号全程卡死的 episode（比如
`action_grasp` 全程恒为 1，从未翻转）会正常出现在 L1 的表格里、拖进 L3 的
dispersion/global_correction 计算，不会有任何提示**——L0 的
`grasp_events_match` 会显示 `[FAIL]`，但 L1 表格、`rho_pos`/`rho_rot` 的
mean/std、`global_correction` 的拟合结果都会不动声色地把它算进去。`--no-
grasp-health-filter` 可以关掉这个过滤，回到旧行为，用来对比"这条坏数据到底把
数字带偏了多少"。

**2. Segment 范围（默认关闭 = 全局）。** `--segment-start`/`--segment-end`
用跟 `--anchors` 一样的 0-based 事件索引，把 L1/L2a/L3 的比较范围收窄到两个
事件之间的一段——比如"从第一次张开到第一次闭合再到第二次张开"就是
`--segment-start 0 --segment-end 2`（只需要点出两端，中间那次翻转不用管）。
`traj_metrics.crop_to_segment()` 把每条轨迹按自己的事件位置切一段出来，重新
探测这段内部的 grasp 事件，再喂给 `DistanceBook`——L1/L2a/L3 的其余代码完全
不需要知道"现在是分段模式"，因为它们本来就只认从 `DistanceBook` 里传出来的
轨迹，不关心这些轨迹的长度或来源。哪条 episode 解析不出这两个边界事件（比如
比全局众数还少翻转的、或者压根没有 grasp 信号的），会被丢弃并打印原因，不会
用错误的边界硬凑。

两者可以叠加：健康过滤先排除明显坏掉的 episode，segment 范围再从剩下的里
裁剪。SUMMARY 区块开头会打印"这次 L1/L2a/L3 实际用了几条"，跟 L0 表格里
"全部加载了几条"分开显示，避免把两个不同的分母搞混。

⚠️ **分段会改变噪声底，所以跨 variant 直接比 `rho` 是有陷阱的。** 噪声底是在
裁剪后的那一段上重新算的，段越短通常越小（lean 实测：full 17.2mm、seg0-2
15.0mm、seg1-2 14.2mm）。于是 `rho` 变大可能只是因为分母变小了，不代表绝对
误差真的变差。跨 variant 比较时**必须同时看 `D_pos_mm` / `D_rot_deg`**（绝对值）
才能分清是分子动了还是分母动了。报表在分段模式下会主动打这条提醒，
`compare_reports.py` 检测到噪声底不一致时也会警告。

另外，segment 的**物理含义会随任务变化**。`events[1]->events[2]` 在两个任务上
都是"夹爪闭合→张开"（持物搬运段），但覆盖的进度区间不同：horizontal 是
32%→64%，lean 是 39%→65%。报表现在会直接算出并打印这个区间：

```
segment scope: events[1] -> events[2]  (L1 / L2a / L3 only)
  close -> open   32% -> 64% of the episode (measured on the 31 real episodes)
```

---

## 什么是 DTW（动态时间规整）

后面几乎所有距离指标——`D_pos_mm`、`rho_pos`、L2a 的 `rho_abs`/`rho_shape`、
L3 的 `dispersion_ratio`/`energy_test`/`manifold_overlap`/`global_correction`
——都建立在同一个原语上：`traj_metrics.py` 里的 `dtw_dist` / `dtw_pairs`。
不先讲清楚 DTW 在算什么，后面这些字段就只是一串数字。

**要解决的问题**：两条轨迹都先用 `resample()` 各自变成 200 个点（按自己的
0→1 归一化进度取点），这一步抹掉了"总时长不同"的差异（ego 7.5s、真机 19s）。
但**点数一样不等于进度对齐**——ego 可能在前 10% 的点里就走完了"伸手接近"这个
阶段，真机因为遥操作动作更谨慎，同样的"伸手接近"要占掉前 30% 的点。如果直接
拿第 20 个点比第 20 个点，比的其实是 ego 的"已经摸到物体"和真机的"还在半路"——
两条轨迹形状明明很像，这么比却会算出一个很大的误差。

**DTW 解决的正是这个"两边节奏不同步"的问题**：允许把一个点对齐到对方的多个
点（或反过来），只要求对齐关系整体单调（不能时间倒流），在这个约束下找一条
让"每一对对齐点之间距离之和"最小的路径。可以想象成一个 200×200 的网格，
每个格子 `(i, j)` 存着 `‖X_i − Y_j‖`，DTW 要找一条从左下角 `(0,0)` 走到右上角
`(199,199)`、每步只能向右/向上/右上斜走的路径，使路径经过的格子代价之和最小：

```
        R0  R1  R2  R3  R4  R5
      ┌───┬───┬───┬───┬───┬───┐
  E0  │ ● │   │   │   │   │   │
      ├───┼───┼───┼───┼───┼───┤
  E1  │   │ ● │ ● │   │   │   │   ← E1 这一个点同时对上了 R1 和 R2
      ├───┼───┼───┼───┼───┼───┤
  E2  │   │   │   │ ● │ ● │   │   ← E2 同时对上了 R3 和 R4
      ├───┼───┼───┼───┼───┼───┤
  E3  │   │   │   │   │   │ ● │
      └───┴───┴───┴───┴───┴───┘
```
（真实计算里网格是 200×200，这里画小方便看。）逐点强行按索引对齐（沿主对角线
走）遇到节奏不同步就会算错；DTW 这条弯曲的路径，本质就是在补偿"ego 这一段动作
比真机快，那一段又比真机慢"这种局部节奏差异，同时仍然保证不会把开头和结尾对反。

**具体到代码**：`_dtw_cost` 用动态规划算最优路径的总代价（`D[i,j] = C[i,j] +
min(D[i-1,j], D[i,j-1], D[i-1,j-1])`），`dtw_dist` 把这个总代价除以
`(len(X)+len(Y))/2`，得到"沿最优对齐路径的平均逐点距离"——这样算出来的单位就是
米（报表里转 mm），可以直接和噪声底比。`_dtw_path` 额外把这条最优路径的下标
对 `(pa, pb)` 找出来（`dtw_pairs`），别的地方需要知道"ego 的这个点该对真机的
哪个点"时就用它——比如姿态误差就是沿着**位置**的 DTW 路径去配对姿态角，而不是
单独再对姿态跑一次 DTW（我们想问的是"手在它所在的那个位置上，朝向对不对"，
不是"这个朝向在整条轨迹里有没有出现过"）。DP 是 O(200×200)，几十条 ego 对几十
条真机要算上千对，这也是为什么这两个函数都用 numba 编译。

**什么时候不用 DTW**：L2b（接触锚点）和 L2c（分段）刻意不用。锚点是抓取/释放
那一帧的绝对位姿，本身就是"这一个点对那一个点"，不存在节奏问题；分段是按
grasp 事件切出来的短区间，两头都被事件钉住，区间内部按比例（`np.linspace`）
取点误差已经足够小，专门为每个短段跑一次 DTW 反而是浪费。DTW 只用在没有天然
对应关系、需要靠算法自己找对齐的地方——也就是 L1/L2a 的整体轨迹比较，以及
L3 里所有建立在这份整体距离矩阵之上的指标。

---

## 快查表

| 层 | 指标 | JSON 路径 | 越小/越接近多少越好 |
|---|---|---|---|
| L0 | 5 个门禁 pass/fail | `L0.<episode>.<gate>.pass` | 全 `true` |
| L1 | `rho_pos` / `rho_rot` / `rho_se3` | `L1.per_ego.<episode>` | → 1 |
| L2a | `rho_offset` / `rho_shape` | `L2_offset_shape.per_ego.<episode>` | → 1 |
| L2b | 每个锚点的 `err_pos_mm` / `err_rot_deg` | `L2_anchors.<episode>.anchors[i]` | → 噪声底 |
| L2c | 每段的 `rho` | `L2_segments.<episode>.segments[i]` | → 1，各段接近 |
| L3 | `dispersion_ratio` | `L3.dispersion.<field>` | → 1 |
| L3 | 置换检验 `p_value` | `L3.energy_test`, `L3.c2st` | → 大（>0.05） |
| L3 | `precision` / `coverage` | `L3.manifold_overlap` | → 1 |
| L3 | 全局修正 loo 残差 | `L3.global_correction.{pos_mm,rot_deg}.loo` | → 噪声底 |

---

## L0：门禁（零对齐成本，只报告不阻断）

对每条 ego episode 单独算，不需要和真机做任何配对。

| 门禁 | 判据 | 说明 |
|---|---|---|
| `grasp_events_match` | ego 的 grasp 翻转次数 == 真机众数 | 次数不等时 L2b/L2c/L3 的锚点类指标会因为对不齐事件而报 `unavailable`，不是被跳过，是算不出来 |
| `workspace_entry_rate` | 落在真机 bbox（外扩 `--bbox-margin`，默认 2cm）内的帧占比 ≥ 99% | `per_axis_outside` 给出逐轴越界比例，能看出是哪根轴的问题 |
| `table_penetration_rate` | 低于真机 z 最小值的帧占比 ≤ 1% | 单独于 workspace_entry 之外报，因为穿透台面是最不能容忍的一类越界 |
| `valid_frame_rate` | 导出 CSV 里 `valid=True` 的帧占比 ≥ 95% | 来自 WiLoR/hand2gripper 自己标的置信度 |
| `speed_feasibility` | 换算成 mm/s 后超过真机 p99 速度的帧占比 ≤ 2% | ⚠️ 人比遥操作天然快 ~2.5 倍，这个门禁**大概率会失败**，失败率本身不太能区分 pipeline 好坏，仅供参考，不要当硬指标 |

⚠️ `value` 字段的"好方向"**不统一**：`table_penetration_rate`/`speed_feasibility`
是违规率，越小越好；`workspace_entry_rate`/`valid_frame_rate` 是通过率，越大越好；
`grasp_events_match` 是 0/1 的过关失败指示。所以 SUMMARY 里 `l0_aggregate()`
汇总时，除了 `value` 的 mean/min/max，**另外单独给每个门禁数一个 `failed` 计数**
（`X/N`，直接读 `pass` 字段，不受 `value` 极性影响）——只看 `value` 的均值容易
把"通过率"类门禁的高值误读成异常。

---

## L1：总览比值（每条 ego 一行）

在真机集合上先做位置和姿态两个噪声底（都是 LOO）：

- `floors.abs` — DTW 距离（按步数归一化），metres → 报表里转 mm
- `floors.rot` — 沿位置 DTW 对应点算的测地角均值，度

对每条 ego：

| 字段 | 含义 |
|---|---|
| `D_pos_mm` | ego 到真机集合的平均 DTW 距离 |
| `D_rot_deg` | 同一条 DTW 路径上的姿态测地角均值 |
| `rho_pos` = `D_pos_mm / floors.abs` | 位置比值 |
| `rho_rot` = `D_rot_deg / floors.rot` | 姿态比值 |
| `rho_se3` = `sqrt(rho_pos · rho_rot)` | 几何平均，单一追踪标量 |
| `z_pos` | `(D_pos - floor_mean) / floor_std`，位置误差的标准分 |

用几何平均而不是算术平均，是因为 `rho_pos` 和 `rho_rot` 都已经是无量纲比值，
几何平均不需要再编造一个 mm↔度 的换算系数。

---

## L2a：位置误差分解——放置 vs 形状

把 DTW 距离拆成两个近似正交的分量，用来判断"该修标定还是该修感知"：

| 字段 | 计算方式 |
|---|---|
| `D_abs_mm` | 同 L1 的 `D_pos_mm` |
| `D_shape_mm` | 两边先各自减去自己的质心，再算 DTW —— 去质心后剩下的形状差异 |
| `D_offset_mm` | ego 质心到真机质心均值的距离 —— 整体平移量 |
| `rho_offset` / `rho_shape` | 各自除以对应的 LOO 噪声底 |
| `centroid_offset_mm` | 3 维偏移向量（不只是模长），可以直接读出该往哪个方向修 |

**读法**：`rho_offset` 远大于 `rho_shape` → 标定/外参/TCP offset 一类的问题，
改一个变换就能大幅改善；两者都大 → 感知/手部估计噪声，没有单一变换能救。

`offset_floor()` 用的是 `‖c_i − mean(其余 c)‖` 而不是两两质心距离的均值——
两者在 Jensen 不等式下不相等，后者会系统性偏大，把 `rho_offset` 拉低。

---

## L2b：接触锚点——唯一的绝对真值

没有手部位姿真值，但物体在一个 layout 内位置固定，所以**真机在夹爪翻转瞬间的
TCP 位姿，就是该接触位姿在 base 系下的一次测量**。这是整套指标体系里唯一不依赖
"真机内部离散度"的绝对误差来源。

锚点标签**从信号本身推导**（`Traj.event_dirs`），不是按事件索引的奇偶数猜测：
奇偶假设在起始状态是"闭合"的任务上会把每个锚点标反。每个锚点报告：

| 字段 | 含义 |
|---|---|
| `label` | 如 `E2_close`，闭合/张开来自信号方向 |
| `grasp_real_before_after` / `grasp_ego_before_after` | 翻转前后的 `action_grasp` 值，人工核对用 |
| `direction_agrees_with_ego` | ego 在这个锚点的翻转方向是否与真机一致；`false` 时这一行的误差数字没有意义，因为两边描述的不是同一个物理事件 |
| `real_dir_unanimous` | 31 条真机在这个锚点的翻转方向是否完全一致（应当总是 `true`） |
| `floor_pos_mm` / `floor_rot_deg` | 真机锚点云到中位数的平均距离/角度——这个锚点自己的噪声底 |
| `err_pos_mm` / `err_vec_mm` | ego 位置到真机中位数的距离/带符号误差向量 |
| `err_rot_deg` | ego 姿态到真机测地中位数（`_geodesic_median_rotation`）的角度 |
| `rho_pos` / `rho_rot` | 各自除以该锚点的噪声底 |
| `phase_ego_pct` / `phase_real_pct_mean±std` / `phase_z` | 事件发生在 episode 的百分比进度，`phase_z` 是标准化后的时序偏差 |

**不是每个 toggle 都是任务相关的**——比如"张爪准备接近"和"收尾闭合复位"就不是。
用 `--anchors` 传入索引子集只保留关心的那些；不传则全报。

---

## L2c：分段比值——误差集中在哪个阶段

用 grasp 事件把 episode 切成 `n_events + 1` 段（S0 = 起始到第一个事件，
Sn = 最后一个事件到结尾），每段内部按比例重采样成固定点数，段内单独算：

| 字段 | 含义 |
|---|---|
| `floor_mm` | 该段内真机两两平均距离 |
| `err_mm` | 该段内 ego 到每条真机的平均距离 |
| `rho` | 两者之比 |

自由空间的误差和接触阶段的误差不该被平均掉——这是为什么单独按段报，而不是只看
L1 的整体 `rho_pos`。经验上 S0（初始接近段）是误差最集中的一段。

---

## L3：多条 ego 才有意义的集合级指标

L1/L2 回答"这一条 ego 离真机流形有多远"。L3 回答三类只有多条 ego 才能问的问题。

### 1. `dispersion_ratio`——差距是系统性偏差还是 pipeline 噪声

```
dispersion_ratio = mean(ego 内部两两距离) / mean(真机内部两两距离)
```

对 `abs` / `shape` / `offset` / `rot` 四个字段各算一个。**这是整个 L3 里信息量
最大的一个数**：

- **≈ 1**：pipeline 本身可重复，ego 和真机的差距是一个恒定偏差，某个变换就能修掉
- **≫ 1**：ego 内部彼此的差异比真机内部还大，pipeline 本身在抖，先降方差，
  拟合任何全局修正都不会有真实收益

⚠️ **一个已经在真实数据上验证过的陷阱**：如果把多个操作员的 ego 数据混在一起算
`dispersion_ratio`，它会被操作员之间的系统性差异（比如姿态修正常量本身因人而
异）冲高，被误读成"pipeline 噪声大"。实测：三个操作员各自算 `rot` 的
`dispersion_ratio` 分别是 1.03 / 1.18 / 1.49（都接近 1，说明各自的 pipeline
很稳），三人混合后变成 2.52。**这不是噪声变大了，是被当成噪声的其实是操作员间
差异。** 多操作员场景下，这个比值要按操作员分开算，混合数据上算出来的会误导后续
"该不该拟合一个全局修正"的判断。

### 2. 能量距离置换检验、C2ST——两个集合还能不能被区分

- **`energy_test`**：`E = 2·mean(D_xy) − mean(D_xx) − mean(D_yy)`，直接在
  已算好的距离矩阵上算，`E=0` 当且仅当两个集合同分布。配一个置换检验给出
  `p_value`，不需要任何分布假设——n≈30 时这一点很重要。
- **`c2st`**：用轨迹的空间特征（位置均值/展幅/标准差/路径长/7 个相位点的形状/
  平均姿态及其离散度，见 `_traj_features`）训一个 5 折交叉验证的逻辑回归去区分
  ego 与真机，报 `auc` 和置换 `p_value`。**默认不把时长/速度当特征**
  （`include_timing=False`）——人比遥操作快 ~2.5 倍是设计使然，让分类器用这个
  信息会让测试轻易饱和，饱和的是我们不关心的东西。`--c2st-timing` 可以打开它。
  需要每类 ≥5 条 episode 才会算，样本不足时 `available: false`。

`p > 0.05` / `AUC → 0.5` 是理想终态，但在 `rho` 还很大（比如 4–6）时这两个值
**会直接饱和**（实测 AUC = 1.000, p = 0.005，达到置换次数决定的最小 p 值）——
这时候它们只能告诉你"确实还差得远"，没有更多信息量，要等 `rho` 降到 2 左右才
开始有区分度。

### 3. `manifold_overlap`——ego 落没落在真机的"正常范围"里

```
precision = 有多少条 ego 落在某条真机的 k-NN 半径内   （"像不像真机"）
coverage  = 有多少条真机被某条 ego 覆盖到               （"覆没覆盖真机的多样性"）
```

`k` 默认 3，半径取真机集合里每条到其第 k 近邻的距离。一个 pipeline 可以靠"永远
输出同一种轨迹"刷高 precision、同时 coverage 很低——所以两个数要一起看。
`precision=0` 时额外报 `nearest_real_mm`（每条 ego 离最近真机还差多远），
让"差得远"这件事至少有个量级，而不是一个信息量为零的 0。

### 4. `global_correction`——一个共享变换能买到多少

在所有 ego episode 上联合拟合**一个平移 + 一个局部旋转**：

| 字段 | 含义 |
|---|---|
| `fitted_translation_mm` / `fitted_rotation_deg` / `fitted_rotation_axis_local` | 拟合出的共享变换 |
| `per_ego_rotation_deg` | 每条 ego 各自拟合出的旋转幅度——这组数字的离散度就是"这个常量到底有多常量"的直接证据 |
| `rotation_spread_deg` | 各条到共享旋转的平均测地距离 |
| `pos_mm` / `rot_deg` 各带 `raw` / `self_fit` / `loo` 三列 | **只有 `loo` 列可以当部署后的预期收益**；`self_fit` 是在同一批数据上拟合又验证，必然更好看，只用来对比展示过拟合有多大 |

留一验证：对每条 ego，用**其余** ego 拟合出的变换去修正它、再打分——这样这个
数字才是"如果我现在采用这个修正，下一条新数据大概能改善多少"的诚实估计。

### 5. `anchor_distributions`——锚点误差是偏移了还是也更抖了

和 L2b 的区别：L2b 是单条 ego 对真机中位数的误差，这里是 **ego 云 vs 真机云**
的整体比较，需要 ≥3 条 ego 才有统计意义（不足时 `ego_spread_mm`/`spread_ratio`
报 `null`，不给一个基于 1-2 个样本、没有信息量的数字）：

| 字段 | 含义 |
|---|---|
| `mean_offset_mm` / `mean_offset_norm_mm` | 两个云的质心差 |
| `mahalanobis` | 质心差按真机云的协方差归一化后的马氏距离——比欧氏距离更能反映"这个偏移相对真机本身的重复性来说大不大" |
| `real_spread_mm` / `ego_spread_mm` | 两个云各自到自己质心的平均距离 |
| `spread_ratio` | `ego_spread / real_spread`；≈1 说明只是整体偏移了，≫1 说明 ego 云本身也比真机云更分散（不只是偏了，还更不稳） |
| `n_ego_dir_agrees` | 有多少条 ego 在这个锚点的翻转方向和真机一致 |

---

## SUMMARY 区块是怎么汇总的

跑多条 ego 时，报表末尾的 SUMMARY 用三个聚合函数把逐条数据收成一张表：

- **`l3_aggregate(per_ego, keys)`**（`metrics_l3.py`）：对给定字段算
  mean/std/median/min/max，并标出 `worst_episode`/`best_episode`——L1
  （`rho_pos`/`rho_rot`/`rho_se3`）、L2a（`rho_abs`/`rho_shape`/`rho_offset`）
  都用它。
- **`l0_aggregate(l0)`**（`eval_ego2exe.py`）：把每条 episode 每个门禁的
  `value` 取出来算 mean/min/max，再单独用 `pass` 字段数一个 `failed`（`X/N`）
  计数附在旁边——SUMMARY 表头的"N gate-failure(s) total"是这些 `failed` 计数
  跨全部门禁、全部 episode 的合计，不是某一个门禁的失败数，读表时以每一行自己
  的 `failed` 列为准。
- **`l2_anchor_aggregate(l2a)`**（`eval_ego2exe.py`）：按锚点 `label` 分组，
  把每条 episode 在该锚点的 `err_pos_mm`/`err_rot_deg` 聚合成一行。

这三者的输出分别对应 `eval_report.json` 里的 `L0_aggregate` /
`L1_aggregate` / `L2_offset_shape_aggregate` / `L2_anchors_aggregate`。

**控制台默认只打摘要。** 逐条 episode 的表格（L0 门禁、L1、L2a、L2b 锚点）需要
`--verbose` 才显示——30 条 ego 时逐条明细占了 614 行里的约 450 行，实际发生过
把 `grasp_events_match [FAIL]` 淹没掉的情况。默认模式只列出"有门禁失败的
episode 及失败项"，明细始终完整写进 `eval_report.json`，不丢信息。

---

## 跨报告对比：`compare_reports.py`

```bash
python ego2exe/eval/compare_reports.py outputs/eval/stack_object_*__all__*
python ego2exe/eval/compare_reports.py --metric rho_rot,D_rot_deg outputs/eval/*
```

自动拉表并处理两个人肉对照时容易踩的坑：

**① 恒等列自动折叠。** `rho_pos` / `rho_offset` / `rho_shape` 在同一 task+variant
下跨 hand2gripper 方案是**逐位相同**的——因为 hand2gripper 只改姿态不改位置
（已在 horizontal 上逐帧验证）。这些列在对比算法时不含任何信息，工具会把它们
从表里折叠掉并单独说明，避免有人盯着不会动的数字找差异。`--show-constant`
可以展开。

**② 噪声底不一致时警告。** 跨 variant 对比时如果各行的 `floor_mm` 不同，会提示
`rho` 的分母不一样，建议加 `--metric D_pos_mm,D_rot_deg` 一并看绝对值。

**③ schema 版本校验。** 报告带 `schema_version` 字段，版本不符的会被跳过并提示
重跑，而不是静默把不同口径的数字混在一张表里。历史上 `outputs/eval/` 曾经并存
过三代格式（早期报告没有 `scope` / `run` 块），这类混比很难事后发现。

标 `*` 的是该列最优值（越小越好）。

---

## 已知的使用陷阱

1. **必须按 layout 配对。** 真机跨 layout 混算会把噪声底虚高数倍（实测
   `stack_bowl` 4 条中途挪过物体，混算后噪声底从 ~20mm 涨到 57mm），所有 `rho`
   都会被相应稀释。
2. **噪声底需要足够的真机样本**，`--min-real`（默认 10）以下会打警告——低于
   10 条时的噪声底本身波动很大，`rho` 只能当粗略参考。
3. **不要把 DTW 和 Fréchet 当两个独立证据**——两者强相关（Toohey & Duckham），
   本工具只算 DTW，是刻意的选择。
4. **多操作员数据不要直接混合算 `dispersion_ratio`**——见上面 L3 第 1 条,
   会把操作员间差异误记成 pipeline 噪声。
5. **`global_correction` 只信 `loo` 列**，`self_fit` 列只用来看过拟合有多大，
   不能当效果汇报。
6. **`rho → 1` 不等于策略能学会**——这套指标是快速开发循环用的代理，最终判据
   仍是 replay 成功率和策略迁移增益。
7. **`speed_feasibility` 门禁失败是预期内的**，人比遥操作快是设计使然，不要
   把它当作 pipeline 质量的信号去追。
8. **grasp 健康过滤默认开启，L1/L2a/L3 用的样本数可能比 L0 显示的加载总数少**——
   SUMMARY 开头的"scope: X/Y ego"就是在提醒这件事。跨实验对比 `rho` 时，如果
   两次跑的健康过滤排除数不一样，样本集其实不完全相同，先看这一行再下结论。
9. **`--segment-start`/`--segment-end` 只影响 L1/L2a/L3**，L2b/L2c 的锚点/
   分段表格不会跟着变——想看某一段单独的锚点误差，直接读 L2c 对应的 `S<n>`
   行，不需要也不应该用 segment 参数去凑。
10. **`--segment-start 0` 是第一次夹爪翻转，不是第 0 帧**——实测这次翻转在
    episode 的 10–24% 处，所以传 `0` 会把接近段切掉。想从视频真正的开头开始，
    **不要传这个参数**（留空才是 episode 起始）。
11. **跨 variant 比 `rho` 要同时看绝对值**——见上面"评估范围"一节，分段会改变
    噪声底这个分母。
12. **`--no-c2st` 在 `rho` 还大的时候可以常开**——C2ST 在 `rho` 降到 2 附近之前
    恒为 AUC 1.000，没有信息量却要跑几百次交叉验证，是纯开销。

---

## 性能

单次评估（31 真机 + 29 ego）约 **16 秒**。两处优化：

- **真机侧距离缓存**：real-real 的 465–496 对 DTW 在同一参照集上是固定的，按
  `(real_dir, pose_key, tcp_offset_m, segment, n_resample)` 缓存到
  `outputs/_cache/real_dtw/`，缓存里存了 episode 名单并逐一校验，真机集增删后
  自动失效重算。`--no-cache` 可关闭。
- **锚点真机侧统计只算一次**：`anchor_reference()` 把每个锚点的真机中位位姿、
  噪声底、相位统计提出来算一遍给所有 ego 共用。之前每条 ego 都重算一遍，其中
  测地中位数是 O(n²) 的纯 Python 循环，profile 显示占了总耗时的 27%；同时这些
  值也不再在 JSON 里按 episode 重复存储，改为 `L2_anchors_reference` 存一份。
