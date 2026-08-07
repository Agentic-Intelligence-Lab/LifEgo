# LifEgo Nero 流程（WiLoR → HumanEgo EEF → TCP IK）

已跑通主线：从 RGB  egocentric 视频得到手/EEF 轨迹，在 MuJoCo 中让 **Nero `site:tcp`** 跟踪 HumanEgo EEF。

默认在仓库根目录执行；示例 session 为 `ego_nero_easy`。

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

## 坐标系（必读）

仿真 **`site:tcp`（tool-centric）**：

```text
R_tcp = R_flange
p_tcp = p_flange + R_flange @ [0.13, 0, 0]    # link7 +X，朝夹爪尖端
```

与法兰、尖端在同一条线上。IK 跟踪的是该帧，**不是** JSONL 里控制器旧定义  
`p = flange + R @ [0, 0, 0.13]`（法兰 +Z）。

HumanEgo IK 输入使用**轴校正后**的 EEF：

```text
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
```

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

## 2. 导出 HumanEgo EEF（机器人基座系）

```bash
$PY patches/export_robot_eef_from_wilor.py \
  --session outputs/ego_nero_easy \
  --out outputs/ego_nero_easy/robot_eef_scene_camera
```

输出：

```text
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json
```

## 3. EEF 局部轴校正

对齐机器人坐标系轴向：

```bash
$PY patches/apply_humanego_eef_axis_correction.py \
  --input outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json \
  --out outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected
```

输出（后续步骤均用此文件）：

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

`scene.xml` 中 `site:tcp` 为 tool-centric（法兰 +X 0.13 m）。

## 5. 解 IK：`site:tcp` 跟踪 HumanEgo EEF

```bash
$PY patches/solve_nero_eef_ik.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --target-name tcp \
  --out outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

（`--target-name tcp` 为默认。）无显示器时可加 `--gl-backend osmesa`。

输出：

```text
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.json
```

## 6. 回放 IK

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

青色 mocap 为 HumanEgo EEF 目标；机械臂由 IK 关节驱动，`site:tcp` 应对齐目标。

---

## 一键对照（ego_nero_easy）

| 步骤 | 脚本 | 关键产物 |
|--|--|--|
| 1 WiLoR | `process_examples_wilor.py` | `preprocess/` |
| 2 导出 EEF | `export_robot_eef_from_wilor.py` | `robot_eef_scene_camera/` |
| 3 轴校正 | `apply_humanego_eef_axis_correction.py` | `robot_eef_scene_camera_axis_corrected/` |
| 4 场景 | `build_nero_mujoco_scene.py` | `mujoco_nero_scene/scene.xml` |
| 5 IK | `solve_nero_eef_ik.py` | `nero_eef_ik/nero_eef_ik.npz` |
| 6 回放 | `replay_nero_eef_ik_mujoco.py` | `replays/*_nero_eef_ik.mp4` |

## 其他脚本

主链路之外、需要时再看源码 docstring：

| 脚本 | 用途 |
|--|--|
| `compare_realbot_humanego_eef.py` | 左右双平台：左真机关节、右 IK 解，默认可循环比对 |
| `replay_humanego_eef_mujoco.py` | 只动 mocap 看 EEF |
| `replay_nero_realbot_data_mujoco.py` | 真机关节回放 + 当前 TCP/尖端 |
| `view_nero_zero_pose_frames.py` | 零位三色 viewer（法兰/TCP/尖端） |
| `nero_tcp_frames.py` | 坐标约定辅助 |
| `apply_realbot_tcp_orientation_correction.py` | 实验，非主流程 |

真机关节 + 当前 tool 帧：

```bash
$PY patches/replay_nero_realbot_data_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_data.mp4
```

左右并排：左真机 / 右 IK 解（默认循环）：

```bash
$PY patches/compare_realbot_humanego_eef.py --viewer \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```
