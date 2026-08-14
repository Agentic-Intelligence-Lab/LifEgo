"""Shared Nero teach-pendant to Revo2 hand mapping."""

from __future__ import annotations

from typing import Optional, Sequence


DEFAULT_OPEN_POSE = (0, 800, 0, 0, 0, 0)
DEFAULT_MAX_RANGE_M = 0.10
DEFAULT_MAX_ANGLE_DEG = 180.0
DEFAULT_MAX_POSITION = 1000


def parse_hand_pose(value: str) -> tuple[int, int, int, int, int, int]:
    try:
        pose = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("pose must contain six comma-separated integers") from exc
    if len(pose) != 6:
        raise ValueError("pose must contain exactly six values")
    if any(position < 0 or position > 1000 for position in pose):
        raise ValueError("pose values must be in 0..1000")
    return pose  # type: ignore[return-value]


class GripperToGraspMapper:
    """Map leader gripper opening to Revo2 normalized finger positions."""

    def __init__(
        self,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
        max_position: int = DEFAULT_MAX_POSITION,
        open_pose: Sequence[int] = DEFAULT_OPEN_POSE,
    ) -> None:
        if max_range_m <= 0 or max_angle_deg <= 0:
            raise ValueError("mapping ranges must be greater than zero")
        if not 0 <= max_position <= 1000:
            raise ValueError("max_position must be in 0..1000")
        if len(open_pose) != 6 or any(not 0 <= value <= 1000 for value in open_pose):
            raise ValueError("open_pose must contain six values in 0..1000")
        self.max_range_m = float(max_range_m)
        self.max_angle_deg = float(max_angle_deg)
        self.max_position = int(max_position)
        self.open_pose = tuple(int(value) for value in open_pose)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def update_max_range(self, max_range_m: Optional[float]) -> None:
        if max_range_m is not None and max_range_m > 0:
            self.max_range_m = float(max_range_m)

    def grasp_from_width(self, width_m: float) -> float:
        return 1.0 - self._clamp01(width_m / self.max_range_m)

    def grasp_from_angle(self, angle_deg: float) -> float:
        return self._clamp01(angle_deg / self.max_angle_deg)

    def grasp(self, mode: str, value: float) -> float:
        return (
            self.grasp_from_angle(value)
            if mode == "angle"
            else self.grasp_from_width(value)
        )

    def positions(self, grasp: float) -> list[int]:
        amount = self._clamp01(grasp)
        return [
            int(round(open_position + amount * (self.max_position - open_position)))
            for open_position in self.open_pose
        ]

    def grasp_from_positions(self, positions: Sequence[int]) -> float:
        """Estimate one robust 0..1 grasp value from six measured fingers."""
        if len(positions) != 6:
            raise ValueError("positions must contain exactly six values")
        candidates = []
        for position, open_position in zip(positions, self.open_pose):
            travel = self.max_position - open_position
            if travel <= 0:
                continue
            candidates.append(
                self._clamp01((float(position) - open_position) / travel)
            )
        if not candidates:
            return 0.0
        candidates.sort()
        middle = len(candidates) // 2
        if len(candidates) % 2:
            return candidates[middle]
        return (candidates[middle - 1] + candidates[middle]) * 0.5


__all__ = [
    "DEFAULT_OPEN_POSE",
    "DEFAULT_MAX_RANGE_M",
    "DEFAULT_MAX_ANGLE_DEG",
    "DEFAULT_MAX_POSITION",
    "parse_hand_pose",
    "GripperToGraspMapper",
]
