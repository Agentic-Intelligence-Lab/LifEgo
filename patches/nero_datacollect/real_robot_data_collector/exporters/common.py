from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from real_robot_data_collector.recorder.manifest import iter_episode_dirs


def resolve_episode_dirs(input_dir: str | Path, episodes: Iterable[str] | None = None) -> List[Path]:
    dirs = iter_episode_dirs(input_dir, selected=episodes)
    missing = [str(path) for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Episode directories not found: {missing}")
    return dirs


def load_arrays(episode_dir: str | Path) -> Dict[str, np.ndarray]:
    episode_dir = Path(episode_dir)
    path = episode_dir / "arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing arrays.npz: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def qpos_from_arrays(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            arrays["arm_joint_positions"].astype(np.float32),
            arrays["hand_joint_positions"].astype(np.float32),
        ],
        axis=1,
    )


def next_qpos_action(qpos: np.ndarray) -> np.ndarray:
    if len(qpos) == 0:
        return np.zeros((0, qpos.shape[1] if qpos.ndim == 2 else 13), dtype=np.float32)
    action = np.empty_like(qpos, dtype=np.float32)
    if len(qpos) == 1:
        action[0] = qpos[0]
    else:
        action[:-1] = qpos[1:]
        action[-1] = action[-2]
    return action


def choose_action(arrays: Dict[str, np.ndarray], qpos: np.ndarray, policy: str) -> Tuple[np.ndarray, str]:
    recorded = arrays["actions"].astype(np.float32)
    if policy == "recorded":
        return recorded, "recorded_adapter_action"
    if policy == "next_qpos":
        return next_qpos_action(qpos), "next_qpos_supervised_action"
    if policy != "auto":
        raise ValueError(f"Unknown action policy: {policy}")
    if recorded.size == 0 or np.isnan(recorded).any():
        return next_qpos_action(qpos), "auto_next_qpos_because_recorded_action_has_nan"
    return recorded, "auto_recorded_adapter_action"


def image_paths_for_episode(episode_dir: str | Path, arrays: Dict[str, np.ndarray]) -> List[Path]:
    episode_dir = Path(episode_dir)
    return [episode_dir / str(path) for path in arrays["image_paths"].tolist()]
