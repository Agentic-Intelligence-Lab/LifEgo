# Real Robot Episode Data Collector

面向头戴相机、松灵 NERO 七轴机械臂和 BrainCo Revo 2 Dexterous Hand 的 episode-based 数据采集工程。原始数据格式适合后续转换到 ACT、Diffusion Policy、LeRobot、OpenPI/pi0.5 等策略训练流程。

## 安装

```bash
cd /Users/bochen/Documents/data_collection
python -m venv .venv
source .venv/bin/activate
pip install -r real_robot_data_collector/requirements.txt
```

macOS 首次使用摄像头时，需要在系统设置里允许终端、IDE 或 Codex 访问摄像头和屏幕录制。

BrainCo Revo 2 真实硬件采集需要安装官方 SDK：

```bash
pip3 install bc-stark-sdk==2.0.2
```

官方示例仓库：`BrainCoTech/brainco-hand-sdk`。

## 采集

无硬件测试：

```bash
python real_robot_data_collector/collect_headcam_robot_episodes.py \
  --camera-id 0 \
  --output-dir ./data_test \
  --arm-adapter dummy \
  --hand-adapter dummy
```

真实硬件接入后的示例：

```bash
python real_robot_data_collector/collect_headcam_robot_episodes.py \
  --camera-id 0 \
  --output-dir ./data \
  --image-format jpg \
  --task-name pick_cube \
  --language-instruction "pick up the red cube and place it into the box" \
  --arm-adapter agilex_nero \
  --arm-config configs/nero.json \
  --hand-adapter brainco_revo2 \
  --hand-config configs/brainco_revo2.json
```

启动后会立即打开相机并显示实时画面。键盘控制：

- `SPACE`：在 `IDLE` 下开始新 episode，在 `RECORDING` 下停止并保存。
- `q`：退出。若正在录制，会先安全保存当前 episode。

常用参数：

- `--camera-id`：OpenCV camera id，默认 `0`。
- `--image-format`：`jpg` 或 `png`，默认 `jpg`。
- `--display-width --display-height`：仅影响预览窗口。
- `--save-width --save-height`：影响落盘图像尺寸。
- `--target-fps`：限制采集循环频率。
- `--dry-run`：只显示和打印状态，不落盘。
- `--overwrite`：允许覆盖同名 episode。默认禁止覆盖。
- `--show-preview / --no-show-preview`：默认显示预览；关闭预览后没有 OpenCV 键盘控制，通常只用于集成测试。

## 状态机

采集状态由 `StateMachine` 管理：

- `IDLE`：相机打开并预览，不保存数据。
- `RECORDING`：保存图像、时间戳、机器人状态、动作和 frame index。
- `SAVING`：写 `arrays.npz`、`metadata.json`、`quality_report.json`，并更新根目录 `manifest.json`。该状态下空格会被忽略。

主循环只负责读取相机、查询 arm/hand adapter、交给状态机处理键盘事件和刷新显示。

## 原始数据目录

```text
data/
  manifest.json
  episode_000001/
    images/
      head/
        000000.jpg
        000001.jpg
    timestamps.txt
    episode.jsonl
    arrays.npz
    metadata.json
    quality_report.json
```

`episode.jsonl` 每行是一帧：

```json
{
  "frame_index": 0,
  "timestamp_unix": 1710000000.123,
  "timestamp_monotonic": 12345.678,
  "camera_timestamp_unix": 1710000000.123,
  "camera_timestamp_msec": 0.0,
  "image_head": "images/head/000000.jpg",
  "observation": {
    "arm_joint_positions": [0, 0, 0, 0, 0, 0, 0],
    "arm_joint_velocities": [0, 0, 0, 0, 0, 0, 0],
    "arm_joint_torques": [0, 0, 0, 0, 0, 0, 0],
    "arm_end_effector_pose": [0, 0, 0, 0, 0, 0, 1],
    "hand_joint_positions": [0, 0, 0, 0, 0, 0],
    "hand_joint_velocities": [0, 0, 0, 0, 0, 0],
    "hand_joint_currents_or_forces": [0, 0, 0, 0, 0, 0],
    "hand_joint_positions_raw": [0, 0, 0, 0, 0, 0],
    "hand_joint_velocities_raw": [0, 0, 0, 0, 0, 0],
    "hand_joint_currents_raw": [0, 0, 0, 0, 0, 0]
  },
  "action": {
    "arm_action": [0, 0, 0, 0, 0, 0, 0],
    "hand_action": [0, 0, 0, 0, 0, 0],
    "full_action": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }
}
```

若某些 SDK 暂不支持速度、力矩、末端位姿或 action，系统用 `NaN` 填充，并在 `metadata.json` 中记录 `state_source` / `action_source`。`episode.jsonl` 使用 Python JSON 的 `NaN` token；`arrays.npz` 存储真实 `np.nan`。

`arrays.npz` 包含：

- `timestamps_unix`: `[T]`
- `timestamps_monotonic`: `[T]`
- `arm_joint_positions`: `[T, 7]`
- `arm_joint_velocities`: `[T, 7]`
- `arm_joint_torques`: `[T, 7]`
- `arm_end_effector_pose`: `[T, 7]`
- `hand_joint_positions`: `[T, 6]`
- `hand_joint_velocities`: `[T, 6]`
- `hand_joint_currents_or_forces`: `[T, 6]`
- `hand_joint_positions_raw`: `[T, 6]`
- `hand_joint_velocities_raw`: `[T, 6]`
- `hand_joint_currents_raw`: `[T, 6]`
- `actions`: `[T, 13]`
- `image_paths`: `[T]`

BrainCo Revo 2 raw 字段按官方 SDK 0-1000 标尺保存，同时生成训练友好的 normalized 字段：

- `hand_joint_positions = hand_joint_positions_raw / 1000.0`
- `hand_joint_velocities = hand_joint_velocities_raw / 1000.0`
- `hand_joint_currents_or_forces = hand_joint_currents_raw / 1000.0`

ACT / Diffusion Policy / LeRobot / OpenPI 导出默认使用 normalized `hand_joint_positions` 与 NERO 七轴机械臂拼接：

```text
state/qpos = arm_joint_positions[0:7] + hand_joint_positions[0:6]
action = arm_action[0:7] + hand_action[0:6]
```

## 硬件 SDK 接入

真实 SDK 只应写在具体 adapter 中：

- `real_robot_data_collector/adapters/agilex_nero_arm.py`
- `real_robot_data_collector/adapters/brainco_revo2_hand.py`

不要把 vendor SDK 调用写进采集主循环。需要实现：

- `connect()`
- `disconnect()`
- `get_state()`
- `get_action()`

如果控制栈拿不到真实 action，可以让 `get_action()` 返回 NaN。ACT 和其他导出器支持使用 `next_qpos` 作为 supervised action。

BrainCo Revo 2 配置文件示例：

```json
{
  "protocol": "modbus",
  "port": "/dev/ttyUSB0",
  "baudrate": 460800,
  "slave_id": 126,
  "auto_detect": true,
  "poll_hz": 100,
  "hand_side": "right"
}
```

`slave_id` 可以用十进制 `126`，对应常见十六进制写法 `0x7e`；如使用另一只手或另一套固件地址，请按 BrainCo 官方示例确认 `0x7e` / `0x7f` 对应关系。

`BrainCoRevo2HandAdapter` 使用后台 asyncio event loop 线程按 `poll_hz` 读取最近状态，主 OpenCV 采集循环只读取缓存状态，适合比 `asyncio.run()` 单次读取更稳定的同步采集主循环。

## 导出 ACT HDF5

```bash
python scripts/export_act_hdf5.py \
  --input-dir ./data \
  --output ./exports/act_dataset.hdf5 \
  --image-size 224 224 \
  --action-policy next_qpos
```

单 episode 输出根路径包含：

- `/observations/images/head`: `uint8 [T, H, W, 3]`
- `/observations/qpos`: `float32 [T, 13]`
- `/action`: `float32 [T, 13]`

`qpos = arm_joint_positions[7] + hand_joint_positions[6]`。`--action-policy next_qpos` 使用 `action[t] = qpos[t + 1]`，最后一帧复用倒数第二帧 action。

## 导出 Diffusion Policy Zarr

```bash
python scripts/export_diffusion_policy_zarr.py \
  --input-dir ./data \
  --output ./exports/dataset.zarr \
  --image-size 224 224 \
  --compress
```

输出结构：

```text
dataset.zarr/
  data/
    img
    state
    action
  meta/
    episode_ends
```

`state` 和 `action` 均为 13 维，`episode_ends` 是累计结束 index。

## 导出 LeRobot / OpenPI 可映射目录

```bash
python scripts/export_lerobot_dataset.py \
  --input-dir ./data \
  --output-dir ./exports/lerobot_normalized
```

导出字段：

- `observation.images.head`
- `observation.state`
- `action`
- `timestamp`
- `episode_index`
- `frame_index`
- `task_index`
- `language_instruction`

OpenPI/pi0.5 通常还需要在具体训练 config 中写数据映射。本导出器只生成规范化、可映射的数据目录，不硬编码远程训练环境。

## 测试

```bash
pytest real_robot_data_collector/tests
```

测试会创建临时 episode，检查必需文件、shape、manifest 更新和 schema 校验。
