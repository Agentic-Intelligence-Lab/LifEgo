# 真机回放命令

默认在仓库根目录执行：

```bash
cd /home/ymq/code/LifEgo
PY=/home/ymq/miniconda3/envs/lifego/bin/python
IK=outputs/new_pipeline/ego_hand1/nero_eef_ik_flat_x0/nero_eef_ik.npz
```

所有真机命令默认需要 `--execute`，运行后还会要求输入 `EXECUTE` 二次确认。下面命令默认 CAN 通道为 `can0`。

## 1. move_p 对齐首帧 + move_j 回放 IK

这个入口先用 `move_p` 将 TCP 对齐到 ego 第一帧 EEF，确认后再用 `move_j` 按 IK 关节轨迹回放。由于 `move_p`
到达首帧 EEF 后不一定落在同一个 IK 关节分支，建议保留 `--approach`。

短段低速测试：

```bash
$PY ego2exe/replay_ik_nero.py \
  --ik $IK \
  --channel can0 \
  --end 20 \
  --prealign-move-p \
  --prealign-speed-percent 5 \
  --approach \
  --speed-percent 10 \
  --command-settle-s 0.25 \
  --execute
```

完整回放：

```bash
$PY ego2exe/replay_ik_nero.py \
  --ik $IK \
  --channel can0 \
  --prealign-move-p \
  --prealign-speed-percent 5 \
  --approach \
  --speed-percent 10 \
  --command-settle-s 0.25 \
  --execute
```

如果中途因为关节误差略超阈值停止，可放宽：

```bash
--joint-tol-deg 2.5 --motion-timeout 45
```

## 2. 纯 move_p 回放 EEF

这个入口只读取 `.npz` 中的 `target_pos_m` / `target_quat_xyzw`，每帧都用 `move_p` 控制 TCP，不使用
`joint_qpos`，也不调用 `move_j` / `move_js`。

短段低速测试：

```bash
$PY ego2exe/replay_ik_nero_move_p.py \
  --ik $IK \
  --channel can0 \
  --end 20 \
  --speed-percent 5 \
  --execute
```

完整回放：

```bash
$PY ego2exe/replay_ik_nero_move_p.py \
  --ik $IK \
  --channel can0 \
  --speed-percent 5 \
  --execute
```

默认每帧会等待 TCP 到达：

```text
--wait --pos-tol-m 0.04 --rot-tol-deg 10
```

如果要更接近连续下发 EEF 命令，而不是每帧等到位：

```bash
$PY ego2exe/replay_ik_nero_move_p.py \
  --ik $IK \
  --channel can0 \
  --speed-percent 5 \
  --no-wait \
  --command-settle-s 0.15 \
  --execute
```
