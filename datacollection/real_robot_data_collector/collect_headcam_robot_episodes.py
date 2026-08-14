from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from real_robot_data_collector.adapters.agilex_nero_arm import AgilexNeroArmAdapter
from real_robot_data_collector.adapters.arm_base import ArmAdapter, NullArmAdapter
from real_robot_data_collector.adapters.brainco_revo2_hand import BrainCoRevo2HandAdapter
from real_robot_data_collector.adapters.camera import CameraStream
from real_robot_data_collector.adapters.dummy_arm import DummyArmAdapter
from real_robot_data_collector.adapters.dummy_hand import DummyHandAdapter
from real_robot_data_collector.adapters.hand_base import HandAdapter, NullHandAdapter
from real_robot_data_collector.adapters.renv2_hand import DeprecatedReNV2HandAdapter
from real_robot_data_collector.recorder.episode_recorder import EpisodeRecorder
from real_robot_data_collector.recorder.schema import ARM_DOF, HAND_DOF, ArmState, HandState
from real_robot_data_collector.recorder.state_machine import StateAction, StateMachine
from real_robot_data_collector.utils.image_utils import draw_overlay, maybe_resize_for_display
from real_robot_data_collector.utils.time_utils import FrameTimestamp


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_arm_adapter(kind: str, config: Dict[str, Any]) -> ArmAdapter:
    if kind == "dummy":
        return DummyArmAdapter(config)
    if kind == "agilex_nero":
        return AgilexNeroArmAdapter(config)
    if kind == "none":
        return NullArmAdapter(config)
    raise ValueError(f"Unknown arm adapter: {kind}")


def make_hand_adapter(kind: str, config: Dict[str, Any]) -> HandAdapter:
    if kind == "dummy":
        return DummyHandAdapter(config)
    if kind == "brainco_revo2":
        return BrainCoRevo2HandAdapter(config)
    if kind == "renv2":
        return DeprecatedReNV2HandAdapter(config)
    if kind == "none":
        return NullHandAdapter(config)
    raise ValueError(f"Unknown hand adapter: {kind}")


def read_arm_safely(adapter: ArmAdapter, verbose: bool = False) -> Tuple[ArmState, np.ndarray, bool]:
    failed = False
    try:
        state = adapter.get_state()
    except Exception as exc:
        failed = True
        state = ArmState.nan()
        print(f"warning: failed to read arm state: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
    try:
        action = adapter.get_action()
    except Exception as exc:
        failed = True
        action = np.full((ARM_DOF,), np.nan, dtype=np.float32)
        print(f"warning: failed to read arm action: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
    return state, action, failed


def read_hand_safely(adapter: HandAdapter, verbose: bool = False) -> Tuple[HandState, np.ndarray, bool]:
    failed = False
    try:
        state = adapter.get_state()
    except Exception as exc:
        failed = True
        state = HandState.nan(str(exc))
        print(f"warning: failed to read hand state: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
    try:
        action = adapter.get_action()
    except Exception as exc:
        failed = True
        action = np.full((HAND_DOF,), np.nan, dtype=np.float32)
        print(f"warning: failed to read hand action: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
    return state, action, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect episode-based head camera, AgileX NERO arm, and BrainCo Revo 2 hand data."
    )
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--display-width", type=int, default=None)
    parser.add_argument("--display-height", type=int, default=None)
    parser.add_argument("--save-width", type=int, default=None)
    parser.add_argument("--save-height", type=int, default=None)
    parser.add_argument("--task-name", default="default_task")
    parser.add_argument("--language-instruction", default="")
    parser.add_argument("--arm-adapter", choices=["dummy", "agilex_nero", "none"], default="dummy")
    parser.add_argument("--hand-adapter", choices=["dummy", "brainco_revo2", "renv2", "none"], default="dummy")
    parser.add_argument("--arm-config", default=None)
    parser.add_argument("--hand-config", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--show-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def show_frame(window_name: str, frame: np.ndarray, lines: list[str], args: argparse.Namespace) -> None:
    import cv2

    display = maybe_resize_for_display(frame, args.display_width, args.display_height)
    display = draw_overlay(display, lines)
    cv2.imshow(window_name, display)


def main() -> int:
    args = parse_args()
    if args.target_fps is not None and args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")

    import cv2

    camera = CameraStream(camera_id=args.camera_id)
    arm = make_arm_adapter(args.arm_adapter, load_config(args.arm_config))
    hand = make_hand_adapter(args.hand_adapter, load_config(args.hand_config))
    window_name = "real_robot_data_collector"

    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        image_format=args.image_format,
        save_width=args.save_width,
        save_height=args.save_height,
        camera_id=args.camera_id,
        dataset_name=args.output_dir.name,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        arm_adapter_name=args.arm_adapter,
        hand_adapter_name=args.hand_adapter,
        arm_state_source=arm.state_source,
        arm_action_source=arm.action_source,
        hand_state_source=hand.state_source,
        hand_action_source=hand.action_source,
    )
    state_machine = StateMachine(
        recorder=recorder,
        output_dir=args.output_dir,
        task_name=args.task_name,
        language_instruction=args.language_instruction,
    )

    try:
        camera.open()
        arm.connect()
        hand.connect()
    except Exception as exc:
        camera.release()
        try:
            arm.disconnect()
            hand.disconnect()
        finally:
            cv2.destroyAllWindows()
        print(f"error: startup failed: {exc}", file=sys.stderr)
        return 1

    last_loop_time = time.monotonic()
    fps_estimate = 0.0

    try:
        while True:
            loop_start = time.monotonic()
            result = camera.read()
            if not result.ok or result.frame is None:
                print("warning: failed to read camera frame; continuing", file=sys.stderr)
                state_machine.note_dropped_camera_frame()
                key = cv2.waitKey(1) if args.show_preview else -1
                action = state_machine.process_key(key)
                if action == StateAction.QUIT:
                    state_machine.finalize_for_exit()
                    break
                _sleep_for_target_fps(loop_start, args.target_fps)
                continue

            now = time.monotonic()
            dt = max(now - last_loop_time, 1e-6)
            fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt) if fps_estimate > 0 else 1.0 / dt
            last_loop_time = now

            timestamp = FrameTimestamp.now(
                camera_timestamp_unix=result.timestamp_unix,
                camera_timestamp_msec=result.capture_msec,
            )
            arm_state, arm_action, arm_failed = read_arm_safely(arm, verbose=args.verbose)
            hand_state, hand_action, hand_failed = read_hand_safely(hand, verbose=args.verbose)
            state_machine.record_if_needed(
                frame=result.frame,
                timestamp=timestamp,
                arm_state=arm_state,
                hand_state=hand_state,
                arm_action=arm_action,
                hand_action=hand_action,
                arm_read_failed=arm_failed,
                hand_read_failed=hand_failed,
            )

            if args.show_preview:
                show_frame(window_name, result.frame, state_machine.display_lines(fps_estimate), args)
                key = cv2.waitKey(1)
            else:
                key = -1

            action = state_machine.process_key(key)
            if action in {StateAction.SAVE_REQUESTED, StateAction.QUIT} and state_machine.state.value == "SAVING":
                if args.show_preview:
                    show_frame(window_name, result.frame, state_machine.display_lines(fps_estimate), args)
                    cv2.waitKey(1)
                state_machine.finish_saving()
            if action == StateAction.QUIT:
                break

            _sleep_for_target_fps(loop_start, args.target_fps)
    except KeyboardInterrupt:
        print("Interrupted. Finalizing active episode if needed...", file=sys.stderr)
        state_machine.finalize_for_exit()
    finally:
        state_machine.finalize_for_exit()
        camera.release()
        arm.disconnect()
        hand.disconnect()
        cv2.destroyAllWindows()
    return 0


def _sleep_for_target_fps(loop_start: float, target_fps: float | None) -> None:
    if target_fps is None:
        return
    target_dt = 1.0 / target_fps
    elapsed = time.monotonic() - loop_start
    remaining = target_dt - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
