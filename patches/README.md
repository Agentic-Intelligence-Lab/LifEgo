# 安装

```bash
cd LifEgo
PREDOWNLOAD=0 SKIP_HAND=0 SKIP_HARDWARE=1 bash setup.sh
```

# 关键运行命令

以下命令默认在仓库根目录执行。推荐使用当前项目环境：

```bash
PY=/home/ymq/miniconda3/envs/lifego/bin/python
```

## 1. 使用 WiLoR 处理 RGB 视频

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

默认 WiLoR 权重目录：

```text
.cache/wilor_mini
```

运行单个示例视频：

```bash
$PY patches/process_examples_wilor.py --videos examples/ego_nero_easy.mp4
```

运行多个示例视频：

```bash
$PY patches/process_examples_wilor.py \
  --videos examples/ego_nero_easy.mp4 examples/ego_nero_h.mp4 examples/ego_nero_v.mp4
```

如果权重目录不在默认位置，可以显式指定：

```bash
$PY patches/process_examples_wilor.py \
  --videos examples/ego_nero_easy.mp4 \
  --wilor-pretrained-dir .cache/wilor_mini
```

主要输出：

```text
outputs/ego_nero_easy/preprocess/all_data/<frame>/rgb.png
outputs/ego_nero_easy/preprocess/all_data/<frame>/aria_cam_rgb.json
outputs/ego_nero_easy/preprocess/all_data/<frame>/wilor_hands.json
outputs/ego_nero_easy/preprocess/vis/wilor_hands_vis.mp4
```

## 2. 从 WiLoR 结果导出机器人 EEF 轨迹

```bash
$PY patches/export_robot_eef_from_wilor.py \
  --session outputs/ego_nero_easy \
  --out outputs/ego_nero_easy/robot_eef_scene_camera
```

主要输出：

```text
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.jsonl
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.csv
```

## 3. 可选：修正 EEF 局部坐标轴

```bash
$PY patches/apply_humanego_eef_axis_correction.py \
  --input outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json \
  --out outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected
```

主要输出：

```text
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.jsonl
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.csv
```

## 4. 对比真实机器人轨迹和 HumanEgo/WiLoR EEF

```bash
$PY patches/compare_realbot_humanego_eef.py \
  --humanego outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --real-pose-key tcp_pose \
  --out outputs/ego_nero_easy/compare_realbot_humanego
```

主要输出：

```text
outputs/ego_nero_easy/compare_realbot_humanego/comparison_summary.json
outputs/ego_nero_easy/compare_realbot_humanego/comparison_timeseries.csv
outputs/ego_nero_easy/compare_realbot_humanego/positions_absolute.png
outputs/ego_nero_easy/compare_realbot_humanego/positions_relative_to_start.png
outputs/ego_nero_easy/compare_realbot_humanego/orientation_errors.png
```

## 5. 构建 NERO MuJoCo 场景

依赖本机 URDF 资源：

```text
/home/ymq/code/agx_arm_urdf/nero
```

构建场景：

```bash
$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene
```

主要输出：

```text
outputs/mujoco_nero_scene/scene.xml
outputs/mujoco_nero_scene/README.md
outputs/mujoco_nero_scene/meshes/*.stl
```

## 6. 回放 HumanEgo/WiLoR EEF

生成 MP4：

```bash
$PY patches/replay_humanego_eef_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_humanego_eef_scene_camera_axis_corrected.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_humanego_eef_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --viewer
```

## 7. 回放真实机器人关节

生成 MP4：

```bash
$PY patches/replay_nero_realbot_joints_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_follower_joints.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_nero_realbot_joints_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --viewer
```

## 8. 解 NERO EEF IK

精简 IK 脚本读取 HumanEgo/WiLoR 导出的 EEF 轨迹，求解 `joint1..joint7`，
默认让 MuJoCo 场景里的 `site:jaw_parallel_flange` 跟踪目标位姿。

```bash
$PY patches/solve_nero_eef_ik.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --out outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

主要输出：

```text
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.json
```

调试少量帧：

```bash
$PY patches/solve_nero_eef_ik.py \
  --end 10 \
  --out /tmp/nero_eef_ik_debug.npz \
  --progress-every 1
```

## 9. 回放 NERO EEF IK

生成 MP4：

```bash
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_nero_eef_ik.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --viewer
```