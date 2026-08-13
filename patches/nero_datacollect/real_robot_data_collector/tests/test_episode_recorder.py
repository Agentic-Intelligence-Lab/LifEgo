import json

import numpy as np

from real_robot_data_collector.recorder.episode_recorder import EpisodeRecorder
from real_robot_data_collector.recorder.schema import ArmState, HandState
from real_robot_data_collector.utils.time_utils import FrameTimestamp


def test_episode_recorder_writes_required_files(tmp_path):
    recorder = EpisodeRecorder(output_dir=tmp_path, image_format="jpg", camera_id=0)
    recorder.start_episode(1, "pick_cube", "pick up the red cube")

    for i in range(3):
        frame = np.full((32, 48, 3), i * 20, dtype=np.uint8)
        ts = FrameTimestamp(timestamp_unix=1000.0 + i, timestamp_monotonic=50.0 + i * 0.1)
        arm_state = ArmState(np.ones(7) * i, np.zeros(7), np.zeros(7), np.zeros(7))
        hand_state = HandState(
            np.ones(6) * i,
            np.zeros(6),
            np.zeros(6),
            raw_positions=np.ones(6) * i * 1000,
            raw_speeds=np.zeros(6),
            raw_currents=np.zeros(6),
        )
        recorder.record_frame(frame, ts, arm_state, hand_state, np.ones(7) * i, np.ones(6) * i)

    summary = recorder.finish_episode()
    ep_dir = tmp_path / "episode_000001"

    assert summary["episode_id"] == 1
    assert (ep_dir / "images" / "head" / "000000.jpg").exists()
    assert (ep_dir / "timestamps.txt").exists()
    assert (ep_dir / "episode.jsonl").exists()
    assert (ep_dir / "arrays.npz").exists()
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "quality_report.json").exists()
    assert (tmp_path / "manifest.json").exists()

    arrays = np.load(ep_dir / "arrays.npz", allow_pickle=True)
    assert arrays["arm_joint_positions"].shape == (3, 7)
    assert arrays["hand_joint_positions"].shape == (3, 6)
    assert arrays["hand_joint_positions_raw"].shape == (3, 6)
    assert arrays["hand_joint_velocities_raw"].shape == (3, 6)
    assert arrays["hand_joint_currents_raw"].shape == (3, 6)
    assert arrays["actions"].shape == (3, 13)
    assert arrays["image_paths"].shape == (3,)
    np.testing.assert_allclose(arrays["hand_joint_positions"][2], arrays["hand_joint_positions_raw"][2] / 1000.0)

    metadata = json.loads((ep_dir / "metadata.json").read_text())
    assert metadata["task_name"] == "pick_cube"
    assert metadata["language_instruction"] == "pick up the red cube"
    assert metadata["num_frames"] == 3
    assert metadata["robot"]["hand"]["name"] == "BrainCo Revo 2"
    assert metadata["robot"]["hand"]["total_dof"] == 11
    assert metadata["robot"]["hand"]["sdk"] == "bc-stark-sdk"
    assert metadata["robot"]["hand"]["position_normalization"] == "raw / 1000.0"
