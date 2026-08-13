import numpy as np
import pytest

from real_robot_data_collector.recorder.schema import ArmState, HandState, normalize_arm_action, normalize_hand_action


def test_schema_accepts_expected_shapes():
    arm = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7))
    hand = HandState(np.zeros(6), np.zeros(6), np.zeros(6))

    assert arm.joint_positions.shape == (7,)
    assert hand.joint_positions.shape == (6,)
    assert hand.raw_positions.shape == (6,)
    assert hand.raw_speeds.shape == (6,)
    assert hand.raw_currents.shape == (6,)
    assert normalize_arm_action(np.zeros(7)).shape == (7,)
    assert normalize_hand_action(np.zeros(6)).shape == (6,)


def test_schema_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        ArmState(np.zeros(6), np.zeros(7), np.zeros(7), np.zeros(7))
    with pytest.raises(ValueError):
        HandState(np.zeros(6), np.zeros(5), np.zeros(6))


def test_hand_schema_keeps_raw_and_normalized_fields():
    hand = HandState(
        np.arange(6, dtype=np.float32) / 1000.0,
        np.ones(6, dtype=np.float32) * 0.1,
        np.ones(6, dtype=np.float32) * 0.2,
        raw_positions=np.arange(6, dtype=np.float32),
        raw_speeds=np.ones(6, dtype=np.float32) * 100,
        raw_currents=np.ones(6, dtype=np.float32) * 200,
    )
    obs = hand.to_observation_dict()
    assert obs["hand_joint_positions_raw"] == [0, 1, 2, 3, 4, 5]
    assert obs["hand_joint_velocities_raw"] == [100] * 6
    assert obs["hand_joint_currents_raw"] == [200] * 6
