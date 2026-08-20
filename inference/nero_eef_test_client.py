#!/usr/bin/env python3
"""Test the Nero EEF policy server without connecting to the real robot."""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_DEFAULT_GRASP,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_REALSENSE_FPS,
    DEFAULT_REALSENSE_HEIGHT,
    DEFAULT_REALSENSE_WIDTH,
    DEFAULT_STANDBY_JSONL,
    DEFAULT_TASK_PROMPT,
    as_abs,
    action_to_tcp_pose,
    format_pose,
    make_frame_source,
    pose_to_state,
    print_standby,
    resolve_standby_pose,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_STEPS = 10
DEFAULT_HZ = 5.0


def connect_policy_client(args: argparse.Namespace):
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    print(f"Connected to Nero EEF policy server at {args.host}:{args.port}")
    print(f"Server metadata: {metadata}")
    return client


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def request_action_chunk(client, image: np.ndarray, state: np.ndarray, args: argparse.Namespace, step: int) -> dict:
    started = time.perf_counter()
    result = client.infer(
        {
            "request_id": f"test-client-{step:06d}",
            "image": np.asarray(image, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "prompt": args.prompt,
            "actions_per_inference": args.actions_per_inference,
        }
    )
    roundtrip_ms = (time.perf_counter() - started) * 1000.0
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise RuntimeError(f"server returned invalid actions shape: {actions.shape}")
    result["actions"] = actions
    result["client_timing"] = {"roundtrip_ms": roundtrip_ms}
    return result


def write_record(f, payload: dict) -> None:
    if f is None:
        return
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    f.flush()


def run(args: argparse.Namespace) -> int:
    standby = resolve_standby_pose(args)
    print_standby(standby, args)
    tcp_pose = standby.get("tcp_pose") or standby["flange_pose"]
    standby_grasp = float((standby.get("gripper") or {}).get("state_grasp", args.default_grasp))
    state = pose_to_state(tcp_pose, standby_grasp)
    print(f"test state: {np.round(state, 5).tolist()}")

    client = connect_policy_client(args)
    source = make_frame_source(args)

    output_dir = as_abs(args.output_dir) if args.output_dir else None
    jsonl = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl = (output_dir / "requests.jsonl").open("w", encoding="utf-8")
        print(f"writing test records to: {output_dir}")

    period = 1.0 / max(args.hz, 1.0e-6)
    t0 = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            image = source.read()
            result = request_action_chunk(client, image, state, args, step)
            actions = result["actions"]
            first_action = actions[0]
            server_timing = result.get("policy_timing") or {}
            client_timing = result.get("client_timing") or {}
            print(
                f"step {step:04d}/{args.steps} actions={actions.shape} "
                f"policy={float(server_timing.get('infer_ms', 0.0)):.1f}ms "
                f"roundtrip={float(client_timing.get('roundtrip_ms', 0.0)):.1f}ms"
            )
            print(f"  first action: {np.round(first_action, 5).tolist()}")
            print(f"  first tcp:    {format_pose(action_to_tcp_pose(first_action))}")

            image_path = None
            if output_dir is not None and args.save_images:
                image_path = output_dir / f"rgb_{step:06d}.png"
                save_rgb_image(image_path, image)
            write_record(
                jsonl,
                {
                    "step": step,
                    "request_id": result.get("request_id"),
                    "prompt": args.prompt,
                    "state": state,
                    "actions": actions,
                    "first_tcp_pose": action_to_tcp_pose(first_action),
                    "policy_timing": server_timing,
                    "client_timing": client_timing,
                    "image_path": image_path,
                },
            )

            sleep_s = step * period - (time.monotonic() - t0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        return 0
    finally:
        source.close()
        if jsonl is not None:
            jsonl.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--image", default=None, help="Static RGB image source.")
    parser.add_argument("--video", default=None, help="Video file source instead of live camera.")
    parser.add_argument("--camera-backend", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSONL records and images.")
    parser.add_argument("--save-images", action="store_true", help="Save RGB inputs when --output-dir is set.")
    args = parser.parse_args()

    args.prompt = DEFAULT_TASK_PROMPT
    args.standby_jsonl = DEFAULT_STANDBY_JSONL
    args.standby_pose = None
    args.default_grasp = DEFAULT_DEFAULT_GRASP
    args.actions_per_inference = 10
    args.camera_width = DEFAULT_CAMERA_WIDTH
    args.camera_height = DEFAULT_CAMERA_HEIGHT
    args.camera_fps = DEFAULT_CAMERA_FPS
    args.realsense_width = DEFAULT_REALSENSE_WIDTH
    args.realsense_height = DEFAULT_REALSENSE_HEIGHT
    args.realsense_fps = DEFAULT_REALSENSE_FPS
    args.image_width = DEFAULT_IMAGE_WIDTH
    args.image_height = DEFAULT_IMAGE_HEIGHT
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
