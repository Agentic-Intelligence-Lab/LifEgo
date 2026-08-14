# LifEgo New Pipeline

默认在仓库根目录执行。文档中以 `ego_nero_easy` 为例；也可用 `nero_pick_place` 等同名 session（见文末）。

```bash
cd LifEgo
PY=python
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
3. EEF retargeting to Nero joint IK.
4. MuJoCo replay for EEF markers, IK results, and real-bot comparison.
5. mjlab residual-control RL on top of the mink IK trajectory.
6. Shared calibrated assets and fixed MuJoCo scenes.

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

`retarget_with_scipy.py`
: Solves Nero IK from exported EEF targets with SciPy least-squares. This is the
cleaned version of the old IK script and is kept as a baseline/reference solver.

`retarget_with_mink.py`
: Solves the same IK retargeting problem with mink. It writes the same `.npz`
schema as the SciPy solver. Use this as the default retargeter for the RL
pipeline.

`rl_env_mjlab.py`
: Defines the mjlab residual-control environment, registers
`Mjlab-Nero-IK-Residual`, and stores its PPO config. The action is an offset on
top of the mink IK joint trajectory plus a gripper-width offset.

`rl_train_mjlab.py`
: Training entry point. It registers the Nero task, then calls mjlab's train
script.

`rl_play_mjlab.py`
: Playback/viewer entry point. Run this only from a local graphics session, not
from SSH.

`rl_eval_mjlab.py`
: Headless checkpoint evaluation for zero policy or trained policies.

`replay_eef_mujoco.py`
: Shows only exported EEF target markers in the fixed single-arm scene. The
robot is not driven.

`replay_ik_mujoco.py`
: Loads an IK `.npz`, drives robot joints in the fixed single-arm scene, and
shows the target EEF marker.

`replay_realbot_mujoco.py`
: Loads the fixed dual scene. The left robot follows real-bot JSONL data. The
right robot optionally follows an IK `.npz`; without `--ik`, it stays static.

`utils_replay.py`
: Shared MuJoCo/OpenCV runtime helpers for replay scripts.

`utils_retarget.py`
: Shared MuJoCo, trajectory, and IK result helpers for retargeting scripts.

`tmp_compare_eef_replay.py`
: Temporary comparison script for old-vs-new EEF trajectories.

`build_nero_mujoco_scene.py`
: Legacy scene builder kept for reference. The replay path now uses fixed scenes
from `assets/mujoco_nero_scene/`.

## Common Commands

Run WiLoR on one video:

```bash
python ego2exe/preprocess_wilor_hands.py \
  --video examples/ego_nero_easy.mp4 \
  --out outputs/new_pipeline
```

Export EEF targets:

```bash
python ego2exe/preprocess_export_eef.py \
  --session outputs/new_pipeline/ego_nero_easy \
  --out outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera
```

Retarget EEF targets with mink:

```bash
python ego2exe/retarget_with_mink.py
```

Retarget EEF targets with SciPy baseline:

```bash
python ego2exe/retarget_with_scipy.py
```

Check the mjlab RL scene:

```bash
python ego2exe/rl_env_mjlab.py --check scene
```

Train the mjlab residual policy:

```bash
python ego2exe/rl_train_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --agent.max-iterations 500 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --log-root outputs/new_pipeline/ego_nero_easy/rl_logs
```

Evaluate checkpoints headlessly:

```bash
python ego2exe/rl_eval_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --include-zero \
  --steps 240 \
  --num-envs 1024 \
  --out outputs/new_pipeline/ego_nero_easy/rl_logs/eval.json
```

Open the mjlab viewer from a local graphics session:

```bash
python ego2exe/rl_play_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --agent zero \
  --viewer native
```

Replay EEF markers only:

```bash
python ego2exe/replay_eef_mujoco.py \
  --viewer --gl-backend glfw
```

Replay IK:

```bash
python ego2exe/replay_ik_mujoco.py \
  --viewer --gl-backend glfw
```

Replay real-bot data with optional IK comparison:

```bash
python ego2exe/replay_realbot_mujoco.py \
  --viewer --gl-backend glfw \
  --ik outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

Without `--viewer`, replay scripts render MP4 files instead of opening a window.
Do not start viewer commands from an SSH session.

## Retarget Solver Choice

`retarget_with_mink.py` is the preferred solver for the current RL setup. On a
60-frame smoke test from `outputs/new_pipeline/ego_nero_easy`, mink produced
lower position error and smoother joint motion than the SciPy baseline, while
SciPy produced slightly lower orientation error.

```text
solver  pos mean/max mm  ang mean/max deg  |dq| mean/max  |ddq| mean/max
mink    0.32 / 0.51      3.12 / 5.04       0.015 / 0.029  0.004 / 0.013
scipy   5.30 / 8.58      2.09 / 3.36       0.015 / 0.038  0.005 / 0.044
```

For downstream RL, mink is a better initial IK source because it tracks EEF
position more accurately and has fewer large joint accelerations. Use SciPy when
comparing against the older least-squares behavior.

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

RL defaults:

```text
task: Mjlab-Nero-IK-Residual
scene: ego2exe/assets/mujoco_nero_scene/scene.xml
eef: outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
ik: outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

RL observation/action/reward:

```text
action: 8D residual = 7 arm joint offsets + 1 gripper width offset
actor obs: 52D = joint pos 9 + joint vel 9 + IK reference 17 + joint target error 9 + last action 8
critic obs: same as actor
reward tracking: site:tcp position/orientation tracking, weight 1.0
reward action_smoothness: action_rate_l2, weight -0.01
termination: timeout
```
