#!/usr/bin/env python3
"""Nero leader teach pendant -> Revo2Touch hand bridge.

The Nero leader/follower arms remain linked directly by CAN.  This program
does not enable either arm, change its mode, or send any AgxGripper command.
It only:

1. reads leader/follower joint feedback for monitoring;
2. reads the leader teach-pendant command on CAN 0x159;
3. maps that opening to a 0..1000 Revo2 grasp target;
4. optionally sends the target to the Revo2 hand over Modbus RTU.

Preview is the default.  Revo2 motion requires the explicit ``--execute``
flag, which makes testing the CAN mapping safe and unambiguous.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from teleop_mapping import (
    DEFAULT_MAX_ANGLE_DEG,
    DEFAULT_MAX_POSITION,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_OPEN_POSE,
    GripperToGraspMapper,
    parse_hand_pose as parse_shared_hand_pose,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "brainco-hand-sdk", "python"))
from common_imports import libstark, logger  # noqa: E402


DEFAULT_CAN_CHANNEL = "can1"
DEFAULT_HAND_PORT = "auto"
DEFAULT_HAND_SLAVE_ID = 0x7F
DEFAULT_CONTROL_HZ = 20.0
DEFAULT_MOTION_DURATION_MS = 80
CTRL_STALE_SECONDS = 0.5


def parse_hand_pose(value: str) -> tuple[int, int, int, int, int, int]:
    try:
        return parse_shared_hand_pose(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map Nero leader teach-pendant opening to a Revo2Touch hand"
    )
    parser.add_argument("-c", "--channel", default=DEFAULT_CAN_CHANNEL)
    parser.add_argument(
        "--port",
        default=DEFAULT_HAND_PORT,
        help="Revo2 serial port, or 'auto' to scan ttyUSB devices",
    )
    parser.add_argument(
        "--slave-id",
        type=lambda value: int(value, 0),
        default=DEFAULT_HAND_SLAVE_ID,
        help="Revo2 slave id, decimal or 0x-prefixed (default: 0x7F)",
    )
    parser.add_argument("--hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds; 0 means until Ctrl+C",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=DEFAULT_MAX_RANGE_M,
        help="Teach-pendant fully-open width in metres; device config overrides it",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=DEFAULT_MAX_ANGLE_DEG,
        help="Teach-pendant angle corresponding to a fully closed hand",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=DEFAULT_MAX_POSITION,
        help="Maximum Revo2 normalized close position (0..1000)",
    )
    parser.add_argument(
        "--open-pose",
        type=parse_hand_pose,
        default=DEFAULT_OPEN_POSE,
        help="Six Revo2 positions for fully open (default: 0,800,0,0,0,0)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="Low-pass factor in (0,1]; 1 disables smoothing",
    )
    parser.add_argument(
        "--deadband",
        type=int,
        default=5,
        help="Do not resend targets that differ by fewer normalized units",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the Revo2 hand (default is preview only)",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect to Revo2; inspect CAN input/mapping only",
    )
    args = parser.parse_args()

    if args.hz <= 0:
        parser.error("--hz must be greater than 0")
    if args.duration < 0:
        parser.error("--duration cannot be negative")
    if args.max_range <= 0:
        parser.error("--max-range must be greater than 0")
    if args.max_angle <= 0:
        parser.error("--max-angle must be greater than 0")
    if not 0 <= args.max_position <= 1000:
        parser.error("--max-position must be in 0..1000")
    if not 0 < args.smoothing <= 1:
        parser.error("--smoothing must be in (0,1]")
    if args.deadband < 0:
        parser.error("--deadband cannot be negative")
    if args.execute and args.no_hand:
        parser.error("--execute cannot be combined with --no-hand")
    return args


class ArmHandBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mapper = GripperToGraspMapper(
            max_range_m=args.max_range,
            max_angle_deg=args.max_angle,
            max_position=args.max_position,
            open_pose=args.open_pose,
        )
        self.robot: Any = None
        self.end_effector: Any = None
        self.hand_client: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nero-can")
        self._running = False
        self._loop_count = 0
        self._start_time = 0.0
        self._filtered_grasp: Optional[float] = None
        self._last_sent_position: Optional[list[int]] = None
        self._last_ctrl_can_timestamp: Optional[float] = None
        self._last_ctrl_seen_monotonic = 0.0

    # All pyAgxArm API calls stay on one executor thread.
    def _create_arm_stack(self) -> tuple[Any, Any]:
        config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.V112,
            interface="socketcan",
            channel=self.args.channel,
        )
        robot = AgxArmFactory.create_arm(config)
        # Register the 0x159/0x2A8 parser before the CAN receive thread starts.
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.connect()
        return robot, effector

    def _read_arm_state(self) -> dict[str, Any]:
        leader = self.robot.get_leader_joint_angles()
        follower = self.robot.get_joint_angles()
        ctrl = self.end_effector.get_gripper_ctrl_states()

        result: dict[str, Any] = {
            "leader": None if leader is None else list(leader.msg),
            "leader_hz": 0.0 if leader is None else float(leader.hz),
            "follower": None if follower is None else list(follower.msg),
            "follower_hz": 0.0 if follower is None else float(follower.hz),
            "ctrl": None,
        }
        if ctrl is None:
            return result

        # pyAgxArm currently applies 1e-6 to 0x159 for both modes.  Rebuild
        # the signed int32 payload so angle mode can use its real 1e-3 deg scale.
        status_code = int(ctrl.msg.status_code)
        raw_value = int(round(float(ctrl.msg.value) * 1_000_000.0))
        mode = "angle" if status_code & 0x04 else "width"
        value = raw_value * (1e-3 if mode == "angle" else 1e-6)
        result["ctrl"] = {
            "raw_value": raw_value,
            "value": value,
            "unit": "deg" if mode == "angle" else "m",
            "mode": mode,
            "force_n": float(ctrl.msg.force),
            "status_code": status_code,
            "can_timestamp": float(ctrl.timestamp),
            "hz": float(ctrl.hz),
        }
        return result

    async def setup_arm(self) -> None:
        loop = asyncio.get_running_loop()
        self.robot, self.end_effector = await loop.run_in_executor(
            self._executor, self._create_arm_stack
        )
        await asyncio.sleep(0.35)
        firmware = await loop.run_in_executor(self._executor, self.robot.get_firmware)
        teach_param = await loop.run_in_executor(
            self._executor,
            lambda: self.end_effector.get_gripper_teaching_pendant_param(
                timeout=1.0, min_interval=0.0
            ),
        )
        if teach_param is not None:
            self.mapper.update_max_range(teach_param.msg.max_range_config)
            logger.info(
                "Teach pendant: max_range=%.3f m range=%s%% friction=%s",
                self.mapper.max_range_m,
                teach_param.msg.teaching_range_per,
                teach_param.msg.teaching_friction,
            )
        state = await loop.run_in_executor(self._executor, self._read_arm_state)
        logger.info(
            "Nero: firmware=%s leader=%s follower=%s teach_0x159=%s",
            firmware,
            "OK" if state["leader"] is not None else "NONE",
            "OK" if state["follower"] is not None else "NONE",
            "OK" if state["ctrl"] is not None else "NONE",
        )
        if state["leader"] is None or state["ctrl"] is None:
            raise RuntimeError("Missing Nero leader or teach-pendant CAN feedback")

    async def setup_hand(self) -> None:
        if self.args.no_hand:
            logger.info("Revo2 connection skipped (--no-hand)")
            return

        ports = (
            sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyHAND*"))
            if self.args.port == "auto"
            else [self.args.port]
        )
        detected = None
        detected_port = None
        for port in ports:
            devices = await libstark.auto_detect(False, port, "Modbus")
            for device in devices:
                if device.slave_id == self.args.slave_id:
                    detected = device
                    detected_port = port
                    break
            if detected is not None:
                break
        if detected is None:
            raise RuntimeError(
                f"Revo2 slave 0x{self.args.slave_id:02X} not found on {ports}"
            )

        self.hand_client = await libstark.init_from_detected(detected)
        info = await self.hand_client.get_device_info(self.args.slave_id)
        unit_mode = await self.hand_client.get_finger_unit_mode(self.args.slave_id)
        if self.args.execute and unit_mode != libstark.FingerUnitMode.Normalized:
            await self.hand_client.set_finger_unit_mode(
                self.args.slave_id, libstark.FingerUnitMode.Normalized
            )
            unit_mode = await self.hand_client.get_finger_unit_mode(self.args.slave_id)
        if self.args.execute and unit_mode != libstark.FingerUnitMode.Normalized:
            raise RuntimeError(f"Revo2 unit mode is not normalized: {unit_mode}")
        status = await self.hand_client.get_motor_status(self.args.slave_id)
        logger.info("Revo2 @ %s: %s", detected_port, info.description)
        logger.info("Revo2 unit mode: %s", unit_mode)
        logger.info("Revo2 current positions: %s", list(status.positions))
        if self.args.execute:
            logger.info("EXECUTE mode: teach pendant will command the Revo2 hand")
        else:
            logger.info("PREVIEW mode: Revo2 is connected read-only; no position commands")

    def _ctrl_is_fresh(self, ctrl: Optional[dict[str, Any]], now: float) -> bool:
        if ctrl is None:
            return False
        timestamp = ctrl["can_timestamp"]
        if timestamp != self._last_ctrl_can_timestamp:
            self._last_ctrl_can_timestamp = timestamp
            self._last_ctrl_seen_monotonic = now
        return now - self._last_ctrl_seen_monotonic <= CTRL_STALE_SECONDS

    def _mapped_grasp(self, ctrl: dict[str, Any]) -> float:
        raw_grasp = self.mapper.grasp(ctrl["mode"], ctrl["value"])
        if self._filtered_grasp is None:
            self._filtered_grasp = raw_grasp
        else:
            alpha = self.args.smoothing
            self._filtered_grasp += alpha * (raw_grasp - self._filtered_grasp)
        return self._filtered_grasp

    async def _send_if_needed(self, positions: list[int]) -> bool:
        if not self.args.execute:
            return False
        target = positions
        if (
            self._last_sent_position is not None
            and max(
                abs(current - previous)
                for current, previous in zip(target, self._last_sent_position)
            ) < self.args.deadband
        ):
            return False
        await self.hand_client.set_finger_positions_and_durations(
            self.args.slave_id,
            positions,
            [DEFAULT_MOTION_DURATION_MS] * 6,
        )
        self._last_sent_position = list(target)
        return True

    async def run(self) -> None:
        self._running = True
        self._start_time = time.monotonic()
        interval = 1.0 / self.args.hz
        loop = asyncio.get_running_loop()
        next_tick = self._start_time
        next_status = self._start_time

        while self._running:
            now = time.monotonic()
            elapsed = now - self._start_time
            if self.args.duration > 0 and elapsed >= self.args.duration:
                break
            if now < next_tick:
                await asyncio.sleep(next_tick - now)
                continue

            state = await loop.run_in_executor(self._executor, self._read_arm_state)
            ctrl = state["ctrl"]
            fresh = self._ctrl_is_fresh(ctrl, now)
            positions: Optional[list[int]] = None
            grasp: Optional[float] = None
            sent = False
            if fresh and ctrl is not None:
                grasp = self._mapped_grasp(ctrl)
                positions = self.mapper.positions(grasp)
                if not self.args.no_hand:
                    sent = await self._send_if_needed(positions)

            self._loop_count += 1
            if now >= next_status:
                leader = state["leader"]
                leader_text = (
                    "NONE"
                    if leader is None
                    else "[" + ", ".join(f"{math.degrees(v):.1f}" for v in leader) + "]deg"
                )
                if ctrl is None:
                    ctrl_text = "NONE"
                else:
                    ctrl_text = f"{ctrl['value']:.5f}{ctrl['unit']} {ctrl['hz']:.0f}Hz"
                target_text = "--" if positions is None else str(positions)
                mode_text = "EXEC" if self.args.execute else "PREVIEW"
                print(
                    f"\r[{mode_text}] leader={leader_text}  teach={ctrl_text}  "
                    f"fresh={'Y' if fresh else 'N'}  grasp="
                    f"{'--' if grasp is None else f'{grasp:.3f}'}  target={target_text}  "
                    f"sent={'Y' if sent else 'N'}   ",
                    end="",
                    flush=True,
                )
                next_status = now + 0.5

            next_tick += interval
            now = time.monotonic()
            if now - next_tick > interval * 5:
                next_tick = now + interval

    async def shutdown(self) -> None:
        self._running = False
        print()
        # Intentionally do not open/close the hand on shutdown.  It holds its
        # last commanded state and no surprise motion is generated.
        if self.hand_client is not None:
            try:
                libstark.modbus_close(self.hand_client)
            except Exception as exc:
                logger.error("Revo2 disconnect error: %s", exc)
            self.hand_client = None
        if self.robot is not None:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self.robot.disconnect)
            except Exception as exc:
                logger.error("Nero disconnect error: %s", exc)
            self.robot = None
        self._executor.shutdown(wait=False)
        logger.info("Bridge stopped after %d ticks; no AgxGripper commands sent", self._loop_count)


async def async_main(args: argparse.Namespace) -> int:
    bridge = ArmHandBridge(args)
    try:
        await bridge.setup_arm()
        await bridge.setup_hand()
        await bridge.run()
        return 0
    except Exception as exc:
        logger.error("Bridge failed: %s", exc, exc_info=True)
        return 1
    finally:
        await bridge.shutdown()


def main() -> int:
    args = parse_args()
    libstark.init_logging()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
