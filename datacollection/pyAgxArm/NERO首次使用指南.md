

## 十、Jetson 快速命令备忘

```bash
# 1. 查 USB CAN 是哪一路
for i in $(ip -br link show type can | awk '{print $1}'); do
  echo -n "$i: "; ethtool -i $i 2>/dev/null | grep bus-info
done


sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 2. 激活 can1（每次重启后需重做）
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up

# 3. 验证
ip -details link show can0
candump can0

# 4. Python 连接（先 Ctrl+C 停掉 candump）
python3 first_test.py

# 0. 回关节零位
python3 go_home.py

# 5. 关节运动（第 7 关节 +15°）
python3 motion_test.py

# 6. 验证 move_p + 上位机点位
python3 move_p_verify.py
python3 move_p_goto.py
```

---

## 十一、回零与继续测试

### 回到原点

Nero **没有** `go_home()` 专用 API。回关节零位用：

```python
robot.set_speed_percent(20)
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
robot.move_j([0.0] * robot.joint_nums)  # 7 轴全 0 rad
```

一键执行：

```bash
cd ~/Downloads/pyAgxArm-master && python3 go_home.py
```

> 零位 = 各关节角 0 rad（竖直向上姿态）。若当前位置较远，回零前确认周围无障碍物。

### 当前进度

| 步骤 | 状态 |
|------|------|
| 读状态 / `move_j` 试机 | ✅ |
| `move_p_verify.py`（z 下移 5cm） | 待完整跑通 |
| `move_p_goto.py`（上位机点位） | 待测 |

### 推荐测试顺序（复制粘贴）

```bash
cd ~/Downloads/pyAgxArm-master

# 0. 若不在竖直零位，先回零
python3 go_home.py

# 1. 验证 move_p（下移 5cm → 回到本次起点）
python3 move_p_verify.py

# 2. 再回零，从已知姿态测上位机点位
python3 go_home.py
python3 move_p_goto.py
```

`move_p_goto.py` 跑完后若要回到运行前位姿，把文件里 `RETURN_TO_START = True` 再执行一次。

若 `move_p_goto.py` 物理已动但报失败，看位置偏差打印；固件 1.10 的 `motion_status=1` 可忽略。

若 `arm_status=无解` 且**完全没动**，检查上位机坐标是法兰还是 TCP、RPY 顺序是否与 SDK 一致。