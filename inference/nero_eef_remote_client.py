#!/usr/bin/env python3
"""Nero EEF remote client for camera capture and real-robot move_p control."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "thirdparty" / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.nero_eef_common import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CHANNEL,
    DEFAULT_COMMAND_SETTLE_S,
    DEFAULT_CONTROL_HZ,
    DEFAULT_DEFAULT_GRASP,
    DEFAULT_ENABLE_TIMEOUT,
    DEFAULT_EXECUTE_CHUNK_STEPS,
    DEFAULT_FIRMWARE,
    DEFAULT_GRIPPER_CLOSED_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_OPEN_M,
    DEFAULT_GRIPPER_THRESHOLD,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_REALSENSE_FPS,
    DEFAULT_REALSENSE_HEIGHT,
    DEFAULT_REALSENSE_WIDTH,
    DEFAULT_INTERFACE,
    DEFAULT_MAX_STEP_M,
    DEFAULT_MOTION_TIMEOUT,
    DEFAULT_POS_TOL_M,
    DEFAULT_ROT_TOL_DEG,
    DEFAULT_SPEED_PERCENT,
    DEFAULT_STANDBY_JSONL,
    DEFAULT_STANDBY_SPEED_PERCENT,
    DEFAULT_STANDBY_TIMEOUT,
    DEFAULT_TCP_OFFSET_M,
    DEFAULT_TASK_PROMPT,
    DEFAULT_XYZ_MAX,
    DEFAULT_XYZ_MIN,
    action_to_tcp_pose,
    clip_target_action,
    format_pose,
    grasp_to_width,
    import_robot_helpers,
    make_frame_source,
    move_to_standby,
    pose_to_state,
    print_standby,
    read_gripper_state,
    resolve_standby_pose,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def connect_policy_client(args: argparse.Namespace):
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    print(f"Connected to Nero EEF policy server at {args.host}:{args.port}")
    print(f"Server metadata: {metadata}")
    return client, metadata


def request_action_chunk(client, image: np.ndarray, state: np.ndarray, args: argparse.Namespace, request_index: int) -> np.ndarray:
    result = client.infer(
        {
            "request_id": f"client-{request_index:06d}",
            "image": np.asarray(image, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "prompt": args.prompt,
            "actions_per_inference": args.execute_chunk_steps,
        }
    )
    actions = np.asarray(result["actions"], dtype=np.float32)
    timing = result.get("policy_timing") or {}
    print(
        f"infer request={result.get('request_id')} actions={actions.shape} "
        f"policy={float(timing.get('infer_ms', 0.0)):.1f}ms"
    )
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise RuntimeError(f"server returned invalid actions shape: {actions.shape}")
    return actions


def confirm_execute(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print()
    print("About to deploy remote OpenPI inference on the real Nero arm.")
    print(f"  server={args.host}:{args.port}")
    print(f"  channel={args.channel} interface={args.interface} firmware={args.firmware}")
    print(f"  control_steps={args.control_steps} control_hz={args.control_hz} speed={args.speed_percent}%")
    text = input("Type DEPLOY to continue > ").strip()
    if text != "DEPLOY":
        raise RuntimeError("User cancelled deployment")


def run_dry(args: argparse.Namespace, standby: dict) -> int:
    client, _metadata = connect_policy_client(args)
    source = make_frame_source(args)
    try:
        tcp_pose = standby.get("tcp_pose") or standby["flange_pose"]
        standby_grasp = float((standby.get("gripper") or {}).get("state_grasp", args.default_grasp))
        state = pose_to_state(tcp_pose, standby_grasp)
        image = source.read()
        chunk = request_action_chunk(client, image, state, args, 1)
        print("===== dry-run first remote inference =====")
        print(f"input state: {np.round(state, 5).tolist()}")
        print(f"predicted chunk shape: {chunk.shape}")
        for i, raw_action in enumerate(chunk):
            action = clip_target_action(raw_action, state, args)
            print(f"chunk[{i}] action: {np.round(action, 5).tolist()}")
            print(f"chunk[{i}] tcp:    {format_pose(action_to_tcp_pose(action))}")
        print("dry-run only. Add --execute to command the robot.")
        return 0
    finally:
        source.close()


def run_execute(args: argparse.Namespace, standby: dict) -> int:
    helpers = import_robot_helpers()
    rt = helpers["import_pyagx_runtime"]()
    client, _metadata = connect_policy_client(args)
    source = make_frame_source(args)
    robot = None
    gripper = None
    try:
        print("===== connect / enable =====")
        print(f"Connecting NERO: channel={args.channel}, interface={args.interface}, firmware={args.firmware}")
        robot = helpers["connect_and_enable_robot"](rt, args)
        robot.set_tcp_offset([float(args.tcp_offset_m[0]), float(args.tcp_offset_m[1]), float(args.tcp_offset_m[2]), 0, 0, 0])

        if args.gripper:
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            standby_grasp = float((standby.get("gripper") or {}).get("state_grasp", args.default_grasp))
            standby_width = grasp_to_width(
                standby_grasp,
                open_m=args.gripper_open_m,
                closed_m=args.gripper_closed_m,
                threshold=args.gripper_threshold,
                mode=args.gripper_mode,
            )
            helpers["command_gripper"](gripper, standby_width, force=args.gripper_force)

        if not move_to_standby(robot, standby["flange_pose"], args):
            return 1
        input("Standby reached. Press ENTER to start policy inference, or Ctrl-C to cancel.")

        robot.set_speed_percent(max(1, min(100, args.speed_percent)))
        period = 1.0 / max(args.control_hz, 1.0e-6)
        executed = 0
        request_index = 0
        last_grasp = float((standby.get("gripper") or {}).get("state_grasp", args.default_grasp))
        t0 = time.monotonic()
        print("===== remote policy control =====")
        while executed < args.control_steps:
            current_tcp = helpers["read_tcp_pose"](robot)
            current_grasp = read_gripper_state(gripper, last_grasp)
            state = pose_to_state(current_tcp, current_grasp)
            image = source.read()
            request_index += 1
            chunk = request_action_chunk(client, image, state, args, request_index)

            for chunk_i, raw_action in enumerate(chunk[: args.execute_chunk_steps]):
                if executed >= args.control_steps:
                    break
                action = clip_target_action(raw_action, state, args)
                tcp_pose = action_to_tcp_pose(action)
                if args.position_only:
                    tcp_pose[3:6] = list(current_tcp[3:6])
                flange_pose = robot.get_tcp2flange_pose(tcp_pose)
                width = grasp_to_width(
                    float(action[7]),
                    open_m=args.gripper_open_m,
                    closed_m=args.gripper_closed_m,
                    threshold=args.gripper_threshold,
                    mode=args.gripper_mode,
                )
                print(
                    f"step {executed + 1:04d}/{args.control_steps} chunk={chunk_i} "
                    f"grasp={float(action[7]):.3f} width={width:.3f}m"
                )
                print(f"current tcp:  {format_pose(current_tcp)}")
                print(f"target tcp:   {format_pose(tcp_pose)}")
                print(f"move_p flange:{format_pose(flange_pose)}")
                robot.move_p(list(flange_pose))
                if args.gripper and gripper is not None:
                    helpers["command_gripper"](gripper, width, force=args.gripper_force)
                last_grasp = float(action[7])

                if args.wait:
                    ok = helpers["wait_until_tcp_reached"](
                        robot,
                        tcp_pose,
                        timeout=args.motion_timeout,
                        pos_tolerance_m=args.pos_tol_m,
                        rot_tolerance_rad=args.rot_tol_rad,
                        check_orientation=not args.position_only,
                        min_wait_s=args.command_settle_s,
                    )
                    if not ok:
                        return 1
                else:
                    time.sleep(args.command_settle_s)

                executed += 1
                if not args.wait:
                    target_elapsed = executed * period
                    sleep_s = target_elapsed - (time.monotonic() - t0)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                current_tcp = helpers["read_tcp_pose"](robot)
        print("===== done =====")
        print(f"final tcp: {format_pose(helpers['read_tcp_pose'](robot))}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Use the physical emergency stop if the arm must stop immediately.")
        return 130
    finally:
        source.close()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--print-standby-only", action="store_true")
    parser.add_argument("--image", default=None, help="Static RGB image for dry-run inference.")
    parser.add_argument("--video", default=None, help="Video file source instead of live camera.")
    parser.add_argument("--camera-backend", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the DEPLOY confirmation prompt.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--steps", type=int, default=60, help="Number of move_p policy commands to send.")
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--gripper", action="store_true")
    args = parser.parse_args()

    args.prompt = DEFAULT_TASK_PROMPT
    args.standby_jsonl = DEFAULT_STANDBY_JSONL
    args.standby_pose = None
    args.interface = DEFAULT_INTERFACE
    args.firmware = DEFAULT_FIRMWARE
    args.enable_timeout = DEFAULT_ENABLE_TIMEOUT
    args.standby_speed_percent = DEFAULT_STANDBY_SPEED_PERCENT
    args.speed_percent = DEFAULT_SPEED_PERCENT
    args.standby_timeout = DEFAULT_STANDBY_TIMEOUT
    args.motion_timeout = DEFAULT_MOTION_TIMEOUT
    args.control_steps = int(args.steps)
    args.control_hz = DEFAULT_CONTROL_HZ
    args.execute_chunk_steps = DEFAULT_EXECUTE_CHUNK_STEPS
    args.max_step_m = DEFAULT_MAX_STEP_M
    args.pos_tol_m = DEFAULT_POS_TOL_M
    args.rot_tol_deg = DEFAULT_ROT_TOL_DEG
    args.command_settle_s = DEFAULT_COMMAND_SETTLE_S
    args.wait = False
    args.wait_standby = True
    args.tcp_offset_m = DEFAULT_TCP_OFFSET_M
    args.xyz_min = DEFAULT_XYZ_MIN
    args.xyz_max = DEFAULT_XYZ_MAX
    args.camera_width = DEFAULT_CAMERA_WIDTH
    args.camera_height = DEFAULT_CAMERA_HEIGHT
    args.camera_fps = DEFAULT_CAMERA_FPS
    args.realsense_width = DEFAULT_REALSENSE_WIDTH
    args.realsense_height = DEFAULT_REALSENSE_HEIGHT
    args.realsense_fps = DEFAULT_REALSENSE_FPS
    args.image_width = DEFAULT_IMAGE_WIDTH
    args.image_height = DEFAULT_IMAGE_HEIGHT
    args.gripper_force = DEFAULT_GRIPPER_FORCE
    args.gripper_open_m = DEFAULT_GRIPPER_OPEN_M
    args.gripper_closed_m = DEFAULT_GRIPPER_CLOSED_M
    args.gripper_threshold = DEFAULT_GRIPPER_THRESHOLD
    args.gripper_mode = "threshold"
    args.default_grasp = DEFAULT_DEFAULT_GRASP
    args.rot_tol_rad = np.deg2rad(args.rot_tol_deg)
    return args


def main() -> int:
    args = parse_args()
    standby = resolve_standby_pose(args)
    print_standby(standby, args)
    if args.print_standby_only:
        return 0
    if not args.execute:
        return run_dry(args, standby)
    confirm_execute(args)
    return run_execute(args, standby)


if __name__ == "__main__":
    sys.exit(main())
