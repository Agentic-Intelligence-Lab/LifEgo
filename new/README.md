# LifEgo New Pipeline

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

This directory contains the cleaned Nero pipeline code. New work should stay
here unless an old `patches/` entry point must be kept for compatibility.

## Scope

The current pipeline covers:

1. RGB video to WiLoR hand reconstruction.
2. WiLoR hand geometry to robot EEF targets.
3. MuJoCo replay for EEF markers, IK results, and real-bot comparison.
4. Shared calibrated assets and fixed MuJoCo scenes.

RL training code is still in `patches/` and uses the EEF/IK outputs produced by
this pipeline.

## Files

`assets.py`
: Shared calibration and fixed asset paths. It stores camera intrinsics,
camera-to-base extrinsics, hand-to-EEF transforms, axis correction, and:

```python
MUJOCO_NERO_SCENE
MUJOCO_NERO_DUAL_SCENE
```

`assets/mujoco_nero_scene/`
: Fixed MuJoCo scenes and meshes. Replay scripts load these scenes directly and
do not generate XML at runtime.

`preprocess_wilor_hands.py`
: Runs WiLoR on one RGB video. It writes per-frame `rgb.png`,
`wilor_hands.json`, and smoothed `wilor_hands_processed.json`.

`hand2gripper.py`
: Converts reconstructed hand keypoints to gripper/EEF targets. HumanEgo-specific
logic is contained in `HumanEgoMode`.

`preprocess_export_eef.py`
: Converts processed WiLoR hands into robot-base EEF trajectories. It reads
camera calibration from `assets.py`. Axis correction is enabled by default and
can be disabled with `--no-axis-correction`.

`replay_eef_mujoco.py`
: Shows only exported EEF target markers in the fixed single-arm scene. The
robot is not driven.

`replay_ik_mujoco.py`
: Loads an IK `.npz`, drives robot joints in the fixed single-arm scene, and
shows the target EEF marker.

`replay_realbot_mujoco.py`
: Loads the fixed dual scene. The left robot follows real-bot JSONL data. The
right robot optionally follows an IK `.npz`; without `--ik`, it stays static.

`replay_utils_mujoco.py`
: Shared MuJoCo/OpenCV runtime helpers for replay scripts.

`tmp_compare_eef_replay.py`
: Temporary comparison script for old-vs-new EEF trajectories.

`build_nero_mujoco_scene.py`
: Legacy scene builder kept for reference. The replay path now uses fixed scenes
from `assets/mujoco_nero_scene/`.

## Common Commands

Run WiLoR on one video:

```bash
/home/ymq/miniconda3/envs/lifego/bin/python new/preprocess_wilor_hands.py \
  --video examples/ego_nero_easy.mp4 \
  --out outputs/new_pipeline
```

Export EEF targets:

```bash
/home/ymq/miniconda3/envs/lifego/bin/python new/preprocess_export_eef.py \
  --session outputs/new_pipeline/ego_nero_easy \
  --out outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera
```

Replay EEF markers only:

```bash
/home/ymq/miniconda3/envs/lifego/bin/python new/replay_eef_mujoco.py \
  --viewer --gl-backend glfw
```

Replay IK:

```bash
/home/ymq/miniconda3/envs/lifego/bin/python new/replay_ik_mujoco.py \
  --viewer --gl-backend glfw
```

Replay real-bot data with optional IK comparison:

```bash
/home/ymq/miniconda3/envs/lifego/bin/python new/replay_realbot_mujoco.py \
  --viewer --gl-backend glfw \
  --ik outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

Without `--viewer`, replay scripts render MP4 files instead of opening a window.

## Main Outputs

WiLoR preprocessing:

```text
outputs/<session>/preprocess/all_data/<frame>/rgb.png
outputs/<session>/preprocess/all_data/<frame>/wilor_hands.json
outputs/<session>/preprocess/all_data/<frame>/wilor_hands_processed.json
outputs/<session>/preprocess/wilor_hands_config.json
```

EEF export:

```text
outputs/<session>/robot_eef_scene_camera/robot_eef_trajectory.json
outputs/<session>/robot_eef_scene_camera/robot_eef_trajectory.jsonl
outputs/<session>/robot_eef_scene_camera/robot_eef_trajectory.csv
```

IK replay expects:

```text
outputs/<session>/nero_eef_ik/nero_eef_ik.npz
```

Required IK keys:

```text
joint_qpos
target_pos_m
target_quat_xyzw
```

Optional IK keys:

```text
time_s
grasp
gripper_width_m
pos_err_m
ang_err_deg
```

