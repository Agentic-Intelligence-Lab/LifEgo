from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from real_robot_data_collector.recorder.schema import DATA_FORMAT_VERSION
from real_robot_data_collector.utils.json_utils import load_json, write_json


EPISODE_RE = re.compile(r"^episode_(\d{6})$")


def episode_name(episode_id: int) -> str:
    return f"episode_{episode_id:06d}"


def parse_episode_id(name: str) -> int | None:
    match = EPISODE_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def scan_episode_ids(output_dir: str | Path) -> List[int]:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    ids: List[int] = []
    for path in output_dir.iterdir():
        if path.is_dir():
            parsed = parse_episode_id(path.name)
            if parsed is not None:
                ids.append(parsed)
    return sorted(ids)


def next_episode_id(output_dir: str | Path) -> int:
    ids = scan_episode_ids(output_dir)
    return max(ids) + 1 if ids else 1


def load_or_create_manifest(output_dir: str | Path, dataset_name: str | None = None) -> Dict[str, object]:
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    manifest = load_json(path)
    if manifest is not None:
        manifest.setdefault("episodes", [])
        return manifest
    return {
        "dataset_name": dataset_name or output_dir.name or "robot_dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_format_version": DATA_FORMAT_VERSION,
        "episodes": [],
    }


def update_manifest(output_dir: str | Path, episode_summary: Dict[str, object], dataset_name: str | None = None) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(output_dir, dataset_name=dataset_name)
    episodes = [ep for ep in manifest.get("episodes", []) if ep.get("episode_id") != episode_summary.get("episode_id")]
    episodes.append(episode_summary)
    episodes.sort(key=lambda item: int(item["episode_id"]))
    manifest["episodes"] = episodes
    manifest["data_format_version"] = DATA_FORMAT_VERSION
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def iter_episode_dirs(output_dir: str | Path, selected: Iterable[str] | None = None) -> List[Path]:
    output_dir = Path(output_dir)
    if selected:
        return [output_dir / name for name in selected]
    return [output_dir / episode_name(ep_id) for ep_id in scan_episode_ids(output_dir)]
