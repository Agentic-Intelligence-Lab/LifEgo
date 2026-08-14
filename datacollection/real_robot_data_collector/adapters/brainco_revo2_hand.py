from __future__ import annotations

import asyncio
import inspect
import sys
import threading
import time
from typing import Any, Dict

import numpy as np

from real_robot_data_collector.adapters.hand_base import HandAdapter
from real_robot_data_collector.recorder.schema import HAND_DOF, BRAINCO_REVO2_JOINT_ORDER, HandState


REVO2_JOINT_ORDER = BRAINCO_REVO2_JOINT_ORDER


class BrainCoRevo2HandAdapter(HandAdapter):
    """BrainCo Revo 2 adapter based on BrainCo RevoHand SDK / bc-stark-sdk.

    This adapter intentionally uses only documented SDK entry points named in
    the project requirements. CAN FD and EtherCAT setup are left as TODOs until
    the exact official calls are confirmed from BrainCo examples.
    """

    name = "BrainCo Revo 2"
    active_dof = HAND_DOF
    total_dof = 11
    sdk = "bc-stark-sdk"
    state_source = "bc-stark-sdk_polling"
    action_source = "last_commanded_finger_positions_or_nan"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.protocol = str(self.config.get("protocol", "modbus"))
        self.port = self.config.get("port")
        self.baudrate = int(self.config.get("baudrate", 460800))
        self.slave_id = int(self.config.get("slave_id", 126))
        self.auto_detect = bool(self.config.get("auto_detect", True))
        self.poll_hz = float(self.config.get("poll_hz", 100))
        self.hand_side = str(self.config.get("hand_side", "right"))
        self.connect_timeout_sec = float(self.config.get("connect_timeout_sec", 2.0))

        self._libstark = None
        self._device = None
        self._device_handler = None
        self._modbus_handle = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._first_poll_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_state = HandState.nan("BrainCo Revo 2 has not produced a sample yet.")
        self._last_action = np.full((HAND_DOF,), np.nan, dtype=np.float32)

    def connect(self) -> None:
        try:
            import libstark  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "BrainCo Revo 2 requires the official SDK: pip3 install bc-stark-sdk==2.0.2"
            ) from exc

        self._libstark = libstark
        self._device = asyncio.run(self._connect_device())
        if self._device is None:
            raise RuntimeError("BrainCo Revo 2 SDK initialization returned no device.")

        self._stop_event.clear()
        self._first_poll_event.clear()
        self._thread = threading.Thread(target=self._run_poll_thread, name="brainco-revo2-poll", daemon=True)
        self._thread.start()
        self._first_poll_event.wait(timeout=self.connect_timeout_sec)

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_device()
        self._device = None
        self._loop = None

    def get_state(self) -> HandState:
        with self._state_lock:
            state = self._latest_state
        if not state.is_valid:
            raise RuntimeError(state.error_message or "BrainCo Revo 2 state is invalid.")
        return state

    def get_action(self) -> np.ndarray:
        return self._last_action.copy()

    def set_finger_positions(self, positions: np.ndarray) -> None:
        raw = self._to_raw_vector(positions)
        self._run_device_method("set_finger_positions", self.slave_id, raw.tolist())
        self._last_action = (raw / 1000.0).astype(np.float32)

    def set_finger_positions_and_speeds(self, positions: np.ndarray, speeds: np.ndarray) -> None:
        raw_positions = self._to_raw_vector(positions)
        raw_speeds = self._to_raw_vector(speeds, allow_signed=True)
        self._run_device_method("set_finger_positions_and_speeds", self.slave_id, raw_positions.tolist(), raw_speeds.tolist())
        self._last_action = (raw_positions / 1000.0).astype(np.float32)

    def set_finger_speeds(self, speeds: np.ndarray) -> None:
        raw = self._to_raw_vector(speeds, allow_signed=True)
        self._run_device_method("set_finger_speeds", self.slave_id, raw.tolist())

    def set_finger_currents(self, currents: np.ndarray) -> None:
        raw = self._to_raw_vector(currents, allow_signed=True)
        self._run_device_method("set_finger_currents", self.slave_id, raw.tolist())

    def _run_poll_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._poll_forever())
        finally:
            self._loop.close()

    async def _poll_forever(self) -> None:
        interval = 1.0 / max(self.poll_hz, 1.0)
        while not self._stop_event.is_set():
            state = await self._read_state_once()
            with self._state_lock:
                self._latest_state = state
            self._first_poll_event.set()
            await asyncio.sleep(interval)

    async def _connect_device(self) -> Any:
        if self._libstark is None:
            raise RuntimeError("libstark is not loaded")

        if self.auto_detect:
            detected = await self._auto_detect_device()
            device = await self._call_sdk("init_from_detected", detected)
            return device if device is not None else detected

        if self.protocol == "modbus":
            if not self.port:
                raise RuntimeError("BrainCo Revo 2 modbus config requires 'port' when auto_detect=false.")
            self._modbus_handle = await self._call_sdk("modbus_open", self.port, self.baudrate)
            try:
                self._device_handler = await self._call_sdk("init_device_handler", self._modbus_handle)
            except TypeError:
                self._device_handler = await self._call_sdk("init_device_handler")
            return self._device_handler if self._device_handler is not None else self._modbus_handle

        # TODO: Add CAN FD / EtherCAT initialization only after confirming the
        # exact BrainCo official SDK calls and examples.
        raise NotImplementedError("Only BrainCo Revo 2 modbus setup is implemented. CAN FD/EtherCAT are TODO.")

    async def _auto_detect_device(self) -> Any:
        if self.protocol == "modbus" and hasattr(self._libstark, "auto_detect_modbus_revo2"):
            detected = await self._call_sdk("auto_detect_modbus_revo2")
            if detected is not None:
                return detected
        if hasattr(self._libstark, "auto_detect_device"):
            detected = await self._call_sdk("auto_detect_device")
            if detected is not None:
                return detected
        return await self._call_sdk("auto_detect")

    async def _read_state_once(self) -> HandState:
        try:
            raw_positions = self._as_raw_vector(await self._call_device("get_finger_positions", self.slave_id), "positions")
            raw_speeds = self._as_raw_vector(await self._call_device("get_finger_speeds", self.slave_id), "speeds")
            raw_currents = self._as_raw_vector(await self._call_device("get_finger_currents", self.slave_id), "currents")
            motor_states = await self._read_motor_states()
            now_unix = time.time()
            now_mono = time.monotonic()
            return HandState(
                joint_positions=(raw_positions / 1000.0).astype(np.float32),
                joint_velocities=(raw_speeds / 1000.0).astype(np.float32),
                joint_currents_or_forces=(raw_currents / 1000.0).astype(np.float32),
                raw_positions=raw_positions,
                raw_speeds=raw_speeds,
                raw_currents=raw_currents,
                motor_states=motor_states,
                timestamp_unix=now_unix,
                timestamp_monotonic=now_mono,
                is_valid=True,
            )
        except Exception as exc:
            return HandState.nan(error_message=f"BrainCo Revo 2 read failed: {exc}")

    async def _read_motor_states(self) -> list[Any] | None:
        if hasattr(self._device, "get_motor_state"):
            return self._to_list(await self._call_device("get_motor_state", self.slave_id))
        if hasattr(self._device, "get_motor_status"):
            return self._to_list(await self._call_device("get_motor_status", self.slave_id))
        return None

    async def _call_sdk(self, name: str, *args: Any) -> Any:
        func = getattr(self._libstark, name)
        return await self._maybe_await(func(*args))

    async def _call_device(self, name: str, *args: Any) -> Any:
        func = getattr(self._device, name)
        return await self._maybe_await(func(*args))

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _run_device_method(self, name: str, *args: Any) -> Any:
        if self._device is None:
            raise RuntimeError("BrainCo Revo 2 is not connected")
        if self._loop is None or not self._loop.is_running():
            return asyncio.run(self._call_device(name, *args))
        future = asyncio.run_coroutine_threadsafe(self._call_device(name, *args), self._loop)
        return future.result(timeout=2.0)

    def _close_device(self) -> None:
        try:
            if self._device is not None and hasattr(self._device, "close"):
                asyncio.run(self._maybe_await(self._device.close()))
        except Exception as exc:
            print(f"warning: BrainCo Revo 2 device.close() failed: {exc}", file=sys.stderr)
        if self._libstark is None:
            return
        for name, handle in (("modbus_close", self._modbus_handle), ("close_device_handler", self._device_handler)):
            if not hasattr(self._libstark, name):
                continue
            try:
                if handle is not None:
                    asyncio.run(self._call_sdk(name, handle))
                else:
                    asyncio.run(self._call_sdk(name))
            except TypeError:
                try:
                    asyncio.run(self._call_sdk(name))
                except Exception as exc:
                    print(f"warning: libstark.{name}() failed: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"warning: libstark.{name}() failed: {exc}", file=sys.stderr)

    def _as_raw_vector(self, values: Any, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (HAND_DOF,):
            raise ValueError(f"BrainCo Revo 2 {name} must have shape [6], got {array.shape}")
        return array

    def _to_raw_vector(self, values: np.ndarray, allow_signed: bool = False) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (HAND_DOF,):
            raise ValueError(f"BrainCo Revo 2 command must have shape [6], got {array.shape}")
        lower = -1.0 if allow_signed else 0.0
        array = np.clip(array, lower, 1.0)
        return np.rint(array * 1000.0).astype(np.int32)

    def _to_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]
