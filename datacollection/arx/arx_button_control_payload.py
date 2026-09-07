#!/usr/bin/env python3
"""Run ARX button control with the LifEgo calibrated payload URDF.

This wrapper keeps /home/arx/ARX5_beta unmodified. It imports the existing ARX
button controller, overrides only ARM_CONFIGS to point at the LifEgo URDF with
link6 mass tuned for the handle gripper, then runs the same service loop.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path


ARX_SDK_ROOT = Path("/home/arx/ARX5_beta")
ARX_BUTTON_CONTROL = ARX_SDK_ROOT / "arx_button_control.py"
PAYLOAD_URDF = Path(__file__).resolve().parent / "models" / "X5-2025-gripper-handle-0p65kg.urdf"


def load_arx_button_control():
    if not ARX_BUTTON_CONTROL.exists():
        raise FileNotFoundError(f"ARX button controller not found: {ARX_BUTTON_CONTROL}")
    if not PAYLOAD_URDF.exists():
        raise FileNotFoundError(f"LifEgo payload URDF not found: {PAYLOAD_URDF}")
    if str(ARX_SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(ARX_SDK_ROOT))

    spec = importlib.util.spec_from_file_location("lifego_arx_button_control_base", ARX_BUTTON_CONTROL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {ARX_BUTTON_CONTROL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_arx_button_control()
    urdf_path = str(PAYLOAD_URDF)
    module.ARM_CONFIGS = (
        ("left", {"can_port": "can1", "type": 2, "urdf_path": urdf_path}),
        ("right", {"can_port": "can3", "type": 2, "urdf_path": urdf_path}),
    )
    module.log(f"LifEgo payload URDF active: {urdf_path}")

    lock_handle = None
    state_thread = None
    exit_code = 0
    try:
        lock_handle = module.acquire_process_lock()
        state_thread = threading.Thread(target=module.publish_arm_state, name="arm-state", daemon=True)
        state_thread.start()
        module.log("ARX button controller started; arms remain untouched until button 1")
        exit_code = module.run()
    except Exception as exc:
        module.log(f"Fatal error: {exc}")
        exit_code = 1
    finally:
        if module.arms:
            module.protect_all()
            time.sleep(1.0)
            module.close_all()
        if state_thread is not None:
            state_thread.join(timeout=1.0)
        if lock_handle is not None:
            lock_handle.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
