# LifEgo Nero 流程（WiLoR → HumanEgo EEF → TCP IK）

已跑通主线：从 RGB egocentric 视频得到手/EEF 轨迹与二元 grasp，在 MuJoCo 中让 **Nero `site:tcp`** 跟踪 HumanEgo EEF，并用真机 JSONL 做左右对照回放。

默认在仓库根目录执行。文档中以 `ego_nero_easy` 为例；也可用 `nero_pick_place` 等同名 session（见文末）。

```bash
cd LifEgo
PY=/home/ymq/miniconda3/envs/lifego/bin/python
```

## 安装（首次）

```bash
PREDOWNLOAD=0 SKIP_HAND=0 SKIP_HARDWARE=1 bash setup.sh
```

WiLoR 需要 PyTorch 动态库时：

```bash
export LD_LIBRARY_PATH=$(
  $PY -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
):$LD_LIBRARY_PATH
```

默认 WiLoR 权重：`.cache/wilor_mini`。

依赖本机 URDF（建场景时）：

```text
/home/ymq/code/agx_arm_urdf/nero
```

## 坐标系与 grasp（必读）

仿真 **`site:tcp`（tool-centric）**：

```text
R_tcp = R_flange
p_tcp = p_flange + R_flange @ [0.13, 0, 0]    # link7 +X，朝夹爪尖端
```

与法兰、尖端在同一条线上。IK 跟踪的是该帧，**不是** JSONL 里控制器旧定义  
`p = flange + R @ [0, 0, 0.13]`（法兰 +Z）。

HumanEgo IK 输入使用**轴校正后**的 EEF：

```text
outputs/<session>/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
```

该 JSON 每帧含：

| 字段 | 含义 |
|--|--|
| `T_ee_in_base` | EEF 在 robot base 下的 4×4 |
| `grasp` | **二元**夹爪状态：`0`=开，`1`=闭（WiLoR 拇指–食指距离比） |

IK 将 `grasp` 映射为夹爪开口宽：

| grasp | 默认宽度 | 参数 |
|--|--|--|
| 0 开 | `0.1 m` | `--gripper-open-m` |
| 1 闭 | `0.0 m` | `--gripper-closed-m` |

---

## 1. WiLoR 处理 RGB 视频

```bash
$PY patches/process_examples_wilor.py --videos examples/ego_nero_easy.mp4
```

权重目录非默认时：

```bash
$PY patches/process_examples_wilor.py \
  --videos examples/ego_nero_easy.mp4 \
  --wilor-pretrained-dir .cache/wilor_mini
```

输出：

```text
outputs/ego_nero_easy/preprocess/all_data/<frame>/{rgb.png,aria_cam_rgb.json,wilor_hands.json}
outputs/ego_nero_easy/preprocess/vis/wilor_hands_vis.mp4
```

`wilor_hands.json` 中 `hand_r.grasp_state` 即为二元 grasp（0/1）。

## 2. 导出 HumanEgo EEF（机器人基座系）

```bash
$PY patches/export_robot_eef_from_wilor.py \
  --session outputs/ego_nero_easy \
  --out outputs/ego_nero_easy/robot_eef_scene_camera
```

输出（含 `grasp`）：

```text
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.csv
```

## 3. EEF 局部轴校正

对齐机器人坐标系轴向（保留 `grasp`）：

```bash
$PY patches/apply_humanego_eef_axis_correction.py \
  --input outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json \
  --out outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected
```

后续步骤**只用**轴校正结果：

```text
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
```

## 4. 构建 Nero MuJoCo 场景

```bash
$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --out outputs/mujoco_nero_scene
```

有真机 JSONL 时可附带轨迹点（可选）：

```bash
$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene
```

输出：

```text
outputs/mujoco_nero_scene/scene.xml
outputs/mujoco_nero_scene/meshes/*.stl
```

`scene.xml`：`site:tcp` 为 tool-centric；含 HumanEgo 路径点 / 姿态帧与 `humanego_eef_marker` mocap。

## 5. 解 IK：`site:tcp` 跟踪 HumanEgo EEF

```bash
$PY patches/solve_nero_eef_ik.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --target-name tcp \
  --out outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

（`--target-name tcp` 为默认。）无显示器时可加 `--gl-backend osmesa`。

`npz` 中与回放相关的字段：

| 字段 | 内容 |
|--|--|
| `joint_qpos` | 7 轴关节解 |
| `gripper_width_m` | 由 `grasp` 映射的开口序列 |
| `grasp` | 二元 0/1 |
| `target_pos_m` / `target_quat_xyzw` | HumanEgo EEF 目标 |
| `pos_err_m` / `ang_err_deg` | TCP 跟踪误差 |

输出：

```text
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.json
```

## 6. 回放 IK（单臂）

```bash
# mp4
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_nero_eef_ik.mp4

# viewer
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --viewer
```

青色 mocap = HumanEgo EEF 目标；臂由 `joint_qpos` 驱动，夹爪跟 `gripper_width_m`（来自 grasp）。

## 7. 左右对照：真机 vs IK（推荐）

生成双平台场景（左 = 真机关节 + 真机夹爪，右 = IK 关节 + HumanEgo grasp），**默认循环**回放；两侧显示 HumanEgo EEF 路径与当前目标，便于看执行差异。

```bash
# 交互（循环）
$PY patches/compare_realbot_humanego_eef.py --viewer \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz

# 导出 mp4（默认 2 遍；--once 单遍）
$PY patches/compare_realbot_humanego_eef.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_vs_ik.mp4
```

图例：

| 标记 | 含义 |
|--|--|
| 青色 | HumanEgo EEF 路径 + 当前 EEF 帧（两侧） |
| 品红 | `site:tcp` FK |
| 橙 | `site:gripper_tip` FK |

说明：

- 右平台默认沿 base **+Y** 偏移 `0.9 m`（`--side-offset-y`）。
- 每次会生成 dual MJCF：`scene_dual_compare.xml`；改间距后勿加 `--reuse-dual-scene`。
- 真机与 IK 时长可能不同：按**归一化进度**对齐到真机时间轴。
- 旧 IK（常数夹爪）若无 `grasp` 字段，可传 `--eef <axis_corrected.json>` 回填开合。

---

## 流程一览

| 步骤 | 脚本 | 关键产物 |
|--|--|--|
| 1 WiLoR | `process_examples_wilor.py` | `preprocess/`（含 `grasp_state`） |
| 2 导出 EEF | `export_robot_eef_from_wilor.py` | `robot_eef_scene_camera/`（含 `grasp`） |
| 3 轴校正 | `apply_humanego_eef_axis_correction.py` | `robot_eef_scene_camera_axis_corrected/` |
| 4 场景 | `build_nero_mujoco_scene.py` | `mujoco_nero_scene/scene.xml` |
| 5 IK | `solve_nero_eef_ik.py` | `nero_eef_ik/nero_eef_ik.npz`（关节 + grasp 宽度） |
| 6 单臂回放 | `replay_nero_eef_ik_mujoco.py` | `replays/*_nero_eef_ik.mp4` |
| 7 左右对照 | `compare_realbot_humanego_eef.py` | `scene_dual_compare.xml` + 对照 mp4/viewer |

## 示例 session：`nero_pick_place`

人手视频 + 真机 JSONL：

```text
examples/nero_pick_place_human.mp4
examples/nero_pick_place_realbot.jsonl
```

```bash
SESSION=outputs/nero_pick_place_human

$PY patches/process_examples_wilor.py --videos examples/nero_pick_place_human.mp4

$PY patches/export_robot_eef_from_wilor.py \
  --session $SESSION --out $SESSION/robot_eef_scene_camera

$PY patches/apply_humanego_eef_axis_correction.py \
  --input $SESSION/robot_eef_scene_camera/robot_eef_trajectory.json \
  --out $SESSION/robot_eef_scene_camera_axis_corrected

$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef $SESSION/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/nero_pick_place_realbot.jsonl \
  --out $SESSION/mujoco_nero_scene

$PY patches/solve_nero_eef_ik.py \
  --scene $SESSION/mujoco_nero_scene/scene.xml \
  --eef $SESSION/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --out $SESSION/nero_eef_ik/nero_eef_ik.npz \
  --gl-backend osmesa

$PY patches/compare_realbot_humanego_eef.py --viewer \
  --scene $SESSION/mujoco_nero_scene/scene.xml \
  --realbot examples/nero_pick_place_realbot.jsonl \
  --ik $SESSION/nero_eef_ik/nero_eef_ik.npz
```

对照视频示例路径：

```text
outputs/nero_pick_place_human/mujoco_nero_scene/replays/nero_pick_place_realbot_vs_ik.mp4
```

## 其他脚本

| 脚本 | 用途 |
|--|--|
| `estimate_scale_apriltag.py` | AprilTag 检测 + PnP / VGGT 深度尺度 `s` / 可选 `T_cam_in_base` 与 EEF 修正 |
| `run_vggt_omega_infer.py` | VGGT-Ω 本地推理 smoke test（深度/相机姿态；后续做物体尺寸尺度校正） |
| `replay_nero_realbot_data_mujoco.py` | 仅真机关节回放；隐藏场景内 HumanEgo 烘焙轨迹；品红 TCP / 橙 tip |
| `replay_humanego_eef_mujoco.py` | 只动 mocap 看 EEF |
| `view_nero_zero_pose_frames.py` | 零位三色 viewer（法兰 / TCP / 尖端） |
| `nero_tcp_frames.py` | 坐标约定辅助 |
| `apply_realbot_tcp_orientation_correction.py` | 实验，非主流程 |

仅真机：

```bash
$PY patches/replay_nero_realbot_data_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_data.mp4
```

VGGT-Ω 推理 smoke test（需 CUDA，权重在 `thirdparty/vggt-omega/weights/VGGT-Omega/`）：

```bash
$PY patches/run_vggt_omega_infer.py \
  --images outputs/nero_pick_place_human/preprocess/all_data \
  --max-frames 8 --frame-stride 20 \
  --out outputs/vggt_omega/smoke_nero_pick_place
```

AprilTag 尺度 / 外参（需真实 tag 边长 `--tag-size-m`）：

```bash
$PY patches/estimate_scale_apriltag.py \
  --session outputs/nero_pick_place_human \
  --tag-size-m 0.08 \
  --tag-id 1 \
  --vggt-npz outputs/nero_pick_place_human/vggt_omega/predictions.npz \
  --out outputs/nero_pick_place_human/apriltag_calib
```

若已知 tag 在 robot base 下的位姿，加 `--tag-in-base-json` 或 `--tag-origin-base`，并可配合：

```bash
  --eef-in  .../robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --eef-out .../robot_eef_apriltag_corrected \
  --apply-scale
```
