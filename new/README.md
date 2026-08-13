# LifEgo New Pipeline

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

