# LifEgo

LifEgo 将第一视角 RGB 视频转换为 Nero 机械臂可执行的示教轨迹。当前整理后的主流程代码在
`ego2exe/` 下，`patches/` 保留旧实验代码和参考记录。

## Installation

默认在仓库根目录执行：

```bash
cd LifEgo
PY=/home/ymq/miniconda3/envs/lifego/bin/python
```

安装软件环境：

```bash
PREDOWNLOAD=0 SKIP_HAND=0 SKIP_HARDWARE=1 bash setup.sh
```

如果 WiLoR 找不到 PyTorch 动态库，设置：

```bash
export LD_LIBRARY_PATH=$(
  $PY -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
):$LD_LIBRARY_PATH
```

默认本地资产：

```text
.cache/wilor_mini                 # WiLoR 默认权重缓存
./thirdparty/agx_arm_urdf/nero  # 构建 Nero scene 使用的 URDF / mesh
```

当前流程使用固定 MuJoCo scene：

```text
ego2exe/assets/mujoco_nero_scene/
```

真机控制依赖 `datacollection/pyAgxArm`，只应在连接 Nero CAN 总线的机器上运行。当前已验证的低层控制路径是
`move_j` 关节位置控制；轨迹回放刻意不使用 `move_js`，因为该模式是无平滑的快速响应控制，风险较高。

## 坐标系与标定约定

Nero base 坐标系：

```text
+z：向上
+y：机器人右侧
-x：机器人前方
```

相机 optical 坐标系遵循 OpenCV：

```text
+x：图像右方
+y：图像下方
+z：光轴前方
```

当前仿真、IK 和回放使用 tool-centric TCP：

```text
R_tcp = R_flange
p_tcp = p_flange + R_flange @ [0.13, 0, 0]
```

也就是说，TCP 位于法兰局部 `+X` 方向 13 cm 处，朝向夹爪尖端。旧控制器日志中可能使用
`p = flange + R @ [0, 0, 0.13]`，这不是当前 IK 目标使用的 TCP 定义。

IK 输入应使用轴校正后的 EEF：

```text
outputs/<session>/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
```

每个有效 EEF record 包含：

```text
T_ee_in_base  # EEF 在 Nero base 下的 4x4 位姿
grasp         # 二元夹爪状态：0=开，1=闭
```

二元 `grasp` 映射为夹爪宽度：

```text
0 -> 0.10 m open
1 -> 0.00 m closed
```

相机内参、`T_cam_in_base`、TCP 偏移、EEF 轴校正和固定 scene 路径统一写在：

```text
ego2exe/assets.py
```

当前 RealSense 场景相机外参由 AprilTag 标定得到。简化流程：

```text
1. 用真机 TCP 触碰每个 AprilTag 的 top_left 角点。
2. 将这些 base 坐标点保存到 examples/calib/tag_corners_base.json。
3. 在 RGB 标定图中检测 AprilTag 像素角点。
4. 通过重投影误差联合优化 T_cam_in_base 和每个 tag 的 yaw。
```

每个 tag 只需要触碰一个物理点；其余三个角点由 tag 实际边长和平放在桌面上的假设推断。触点采集时
`tcp_offset` 必须对应真实接触点。当前 AgxGripper 默认接触参考为法兰坐标系下 `[0.13, 0, 0]`。

## ego2exe 流程

`ego2exe/` 是当前整理后的主流程：单条第一视角 RGB 视频先经过 WiLoR 重建手部 3D 模型，再转换为
Nero base 坐标系下的 EEF 轨迹，随后用 mink 求初始 IK，并输出 EEF/IK 的 MuJoCo 回放视频。

一键预处理示例：

```bash
PY=/home/ymq/miniconda3/envs/lifego/bin/python \
ego2exe/scripts/preprocess.sh examples/ego_nero_easy.mp4
```

默认输出：

```text
outputs/new_pipeline/<video_stem>/preprocess/all_data/<frame>/
outputs/new_pipeline/<video_stem>/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
outputs/new_pipeline/<video_stem>/nero_eef_ik/nero_eef_ik.npz
outputs/new_pipeline/<video_stem>/replays/<video_stem>_eef.mp4
outputs/new_pipeline/<video_stem>/replays/<video_stem>_ik.mp4
```

核心脚本：

```text
ego2exe/preprocess_wilor_hands.py   # RGB video -> WiLoR hand reconstruction
ego2exe/preprocess_export_eef.py    # WiLoR hands -> robot-base EEF targets
ego2exe/retarget_with_mink.py       # EEF targets -> Nero IK, 当前默认
ego2exe/retarget_with_scipy.py      # 旧版 SciPy IK baseline
ego2exe/replay_eef_mujoco.py        # 只显示 EEF target marker
ego2exe/replay_ik_mujoco.py         # 驱动仿真机器人关节并显示 target marker
ego2exe/replay_realbot_mujoco.py    # 真机 JSONL 与 IK 轨迹对比回放
```

不加 `--viewer` 时，回放脚本只渲染 MP4；需要打开 MuJoCo viewer 时，应在本地图形会话中运行，不要在
SSH 会话中启动。

真机回放脚本默认 dry-run，不会连接机械臂：

```bash
python ego2exe/replay_ik_nero.py \
  --ik outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

实际下发 Nero 控制命令必须显式加 `--execute`。脚本使用 `pyAgxArm` 的 `move_j` 按 waypoint 回放，
不使用高风险的 `move_js` 快速响应模式。

如果需要先把真机从任意当前位置缓慢移动到 ego 视频第一帧 EEF，可开启 `move_p` 预对齐：

```bash
python ego2exe/replay_ik_nero.py \
  --ik outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --end 20 \
  --stride 2 \
  --prealign-move-p \
  --prealign-speed-percent 5 \
  --speed-percent 10 \
  --execute
```

该阶段会用第一帧 TCP target 和 `tcp_offset=[0.13, 0, 0]` 反算 `move_p` 所需的 flange pose。到位后脚本会等待
输入 `READY`，再开始 `move_j` 回放。

## RL 流程

RL 阶段不从零学习 IK，而是在 mink 初始 IK 轨迹上学习 residual offset，用来优化 EEF tracking 和动作平滑性。
当前任务注册在 mjlab：

```text
task: Mjlab-Nero-IK-Residual
scene: ego2exe/assets/mujoco_nero_scene/scene.xml
eef: outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
ik: outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

当前环境配置：

```text
action: 8D = 7 arm joint residual offsets + 1 gripper width offset
actor obs: 52D
critic obs: 52D
reward tracking: TCP position/orientation tracking, weight 1.0
reward action_smoothness: action_rate_l2, weight -0.01
termination: timeout
parallel envs: train 默认 1024，play 默认 1
```

检查 mjlab scene：

```bash
python ego2exe/rl_env_mjlab.py --check scene
```

训练：

```bash
python ego2exe/rl_train_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --agent.max-iterations 500 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --log-root outputs/new_pipeline/ego_nero_easy/rl_logs
```

无 viewer 评估：

```bash
python ego2exe/rl_eval_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --include-zero \
  --steps 240 \
  --num-envs 1024 \
  --out outputs/new_pipeline/ego_nero_easy/rl_logs/eval.json
```

本地图形会话中查看 viewer：

```bash
python ego2exe/rl_play_mjlab.py \
  Mjlab-Nero-IK-Residual \
  --agent zero \
  --viewer native
```
