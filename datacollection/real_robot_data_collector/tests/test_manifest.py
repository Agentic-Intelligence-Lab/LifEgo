from real_robot_data_collector.recorder.manifest import next_episode_id, scan_episode_ids, update_manifest


def test_scan_and_next_episode_id(tmp_path):
    (tmp_path / "episode_000001").mkdir()
    (tmp_path / "episode_000003").mkdir()
    (tmp_path / "not_episode").mkdir()

    assert scan_episode_ids(tmp_path) == [1, 3]
    assert next_episode_id(tmp_path) == 4


def test_update_manifest_sorts_and_replaces(tmp_path):
    update_manifest(tmp_path, {"episode_id": 2, "episode_name": "episode_000002", "path": "episode_000002", "num_frames": 5})
    manifest = update_manifest(
        tmp_path,
        {"episode_id": 1, "episode_name": "episode_000001", "path": "episode_000001", "num_frames": 3},
    )

    assert [ep["episode_id"] for ep in manifest["episodes"]] == [1, 2]

    manifest = update_manifest(
        tmp_path,
        {"episode_id": 2, "episode_name": "episode_000002", "path": "episode_000002", "num_frames": 9},
    )
    assert manifest["episodes"][1]["num_frames"] == 9
