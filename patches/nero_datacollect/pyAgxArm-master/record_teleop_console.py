#!/usr/bin/env python3
"""Console interactive recorder for Nero + AgxGripper + Dabai (no GUI).

Keys (when running):
  Enter / s  start or stop current segment
  d          delete last saved segment files (asks confirm)
  q          quit

On stop you must label: success / failure(delete) / abort / unreviewed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from platform import system as platform_system

from teleop_quality import (
    DEFAULT_SESSION_PREFIX,
    DEFAULT_TASK,
    OUTCOME_ABORT,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_UNREVIEWED,
    patch_jsonl_metadata,
    sanitize_session_name,
    validate_task,
)


HERE = Path(__file__).resolve().parent
RECORDER = HERE / "record_teleop.py"
IS_WINDOWS = platform_system() == "Windows"
DEFAULT_CAMERA_INDEX = 1


def can_defaults() -> tuple[str, str]:
    if IS_WINDOWS:
        return "gs_usb", "0"
    return "socketcan", "can1"


def popen_kwargs() -> dict:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def interrupt_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if IS_WINDOWS:
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            process.terminate()
            return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except Exception:
        process.terminate()


def force_kill(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def pump_stdout(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        text = line.rstrip()
        if text:
            print(text, flush=True)


def start_segment(
    interface: str,
    channel: str,
    hz: float,
    directory: Path,
    session: str,
    task: str,
    camera_index: int,
) -> tuple[subprocess.Popen[str], Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = directory / f"{session}_{stamp}.jsonl"
    video = output.with_suffix(".rgb.mp4")
    command = [
        sys.executable,
        "-u",
        str(RECORDER),
        "--interface",
        interface,
        "--channel",
        channel,
        "--hz",
        str(hz),
        "--output",
        str(output),
        "--task",
        task,
        "--camera",
        "--camera-backend",
        "opencv",
        "--camera-index",
        str(camera_index),
        "--camera-rotate",
        "180",
        "--camera-video",
        str(video),
        "--no-execute-hand",
        "--keep-unaligned",
    ]
    print(f"\n>>> START {output.name}", flush=True)
    print(f">>> task: {task}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **popen_kwargs(),
    )
    threading.Thread(target=pump_stdout, args=(process,), daemon=True).start()
    return process, output, video


def stop_segment(process: subprocess.Popen[str]) -> int:
    print("\n>>> STOP (waiting for flush)...", flush=True)
    interrupt_process(process)
    try:
        return process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        print(">>> force kill", flush=True)
        force_kill(process)
        return process.wait(timeout=5.0)


def delete_paths(*paths: Path | None) -> None:
    for path in paths:
        if path is not None and path.exists():
            path.unlink()
            print(f"deleted {path.name}", flush=True)


def ask_outcome(output: Path) -> str:
    print(
        f"\nLabel episode {output.name}:\n"
        "  y / success  = keep for training\n"
        "  n / fail     = delete files\n"
        "  a / abort    = keep but skip in converter\n"
        "  u            = unreviewed keep (not recommended)\n",
        flush=True,
    )
    while True:
        ans = input("outcome [y/n/a/u]: ").strip().lower()
        if ans in {"y", "yes", "s", "success"}:
            return OUTCOME_SUCCESS
        if ans in {"n", "no", "f", "fail", "failure"}:
            return OUTCOME_FAILURE
        if ans in {"a", "abort"}:
            return OUTCOME_ABORT
        if ans in {"u", "unreviewed", ""}:
            return OUTCOME_UNREVIEWED
        print("Please enter y / n / a / u", flush=True)


def write_outcome(output: Path, outcome: str) -> None:
    patch_jsonl_metadata(
        output,
        {
            "outcome": outcome,
            "collection": {
                "outcome": outcome,
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        },
    )


def main() -> int:
    interface, channel = can_defaults()
    directory = HERE / "recordings"
    session = DEFAULT_SESSION_PREFIX
    hz = 20.0
    camera_index = DEFAULT_CAMERA_INDEX

    print("Nero console recorder (production)")
    print(f"  CAN={interface}:{channel}")
    print(f"  camera_index={camera_index}")
    print(f"  out_dir={directory}")
    print(f"  default task: {DEFAULT_TASK}")
    raw_task = input(f"task English [{DEFAULT_TASK}]: ").strip()
    try:
        task = validate_task(raw_task or DEFAULT_TASK)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raw_session = input(f"file prefix [{DEFAULT_SESSION_PREFIX}]: ").strip()
    session = sanitize_session_name(raw_session or DEFAULT_SESSION_PREFIX)
    print("Commands: [Enter]=start/stop  d=delete last  q=quit")
    print("-" * 60)

    process: subprocess.Popen[str] | None = None
    last_output: Path | None = None
    last_video: Path | None = None

    while True:
        try:
            cmd = input("nero> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = "q"

        if cmd in {"", "s", "start", "stop"}:
            if process is None:
                process, last_output, last_video = start_segment(
                    interface, channel, hz, directory, session, task, camera_index
                )
            else:
                code = stop_segment(process)
                process = None
                print(f">>> saved exit_code={code}", flush=True)
                if code == 0 and last_output is not None and last_output.exists():
                    outcome = ask_outcome(last_output)
                    if outcome == OUTCOME_FAILURE:
                        try:
                            write_outcome(last_output, OUTCOME_FAILURE)
                        except Exception as exc:
                            print(f"warn: could not write failure mark: {exc}", flush=True)
                        delete_paths(last_output, last_video)
                        last_output = None
                        last_video = None
                    else:
                        try:
                            write_outcome(last_output, outcome)
                            print(f">>> outcome={outcome}", flush=True)
                        except Exception as exc:
                            print(f"ERROR writing outcome: {exc}", flush=True)
                        if last_output is not None:
                            print(f">>> jsonl={last_output}")
                        if last_video is not None:
                            print(f">>> mp4 ={last_video}")
                elif last_output is not None:
                    print(f">>> jsonl={last_output}")
                    if last_video is not None:
                        print(f">>> mp4 ={last_video}")
            continue

        if cmd in {"d", "delete"}:
            if process is not None:
                print("Still recording; stop first.")
                continue
            if last_output is None:
                print("No last segment.")
                continue
            ans = input(f"Delete {last_output.name} (+mp4)? [y/N] ").strip().lower()
            if ans == "y":
                delete_paths(last_output, last_video)
                last_output = None
                last_video = None
            continue

        if cmd in {"q", "quit", "exit"}:
            if process is not None:
                code = stop_segment(process)
                process = None
                if code == 0 and last_output is not None and last_output.exists():
                    outcome = ask_outcome(last_output)
                    if outcome == OUTCOME_FAILURE:
                        delete_paths(last_output, last_video)
                    else:
                        try:
                            write_outcome(last_output, outcome)
                        except Exception as exc:
                            print(f"ERROR writing outcome: {exc}", flush=True)
            print("bye")
            return 0

        if cmd in {"h", "help", "?"}:
            print("Enter/s start|stop · d delete last · q quit")
            continue

        print("Unknown command. Try h for help.")


if __name__ == "__main__":
    raise SystemExit(main())
