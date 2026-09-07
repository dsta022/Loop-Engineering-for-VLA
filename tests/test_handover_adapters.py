from __future__ import annotations

import unittest

from policy_improvement.intervention import PositionMatchHandoverAdapter


class FakeLeader:
    def __init__(self, readings: list[dict[str, float]]) -> None:
        self.readings = readings
        self.calls = 0

    def get_action(self) -> dict[str, float]:
        reading = self.readings[min(self.calls, len(self.readings) - 1)]
        self.calls += 1
        return reading


class FakeBus:
    def __init__(self, positions: dict[str, float]) -> None:
        self.positions = positions

    def read_present_position(self) -> dict[str, float]:
        return dict(self.positions)


class FakeRobot:
    def __init__(self, positions: dict[str, float]) -> None:
        self.bus = FakeBus(BUS_POSITIONS if positions is FOLLOWER else positions)


FOLLOWER = {"shoulder_pan.pos": 10.0, "gripper.pos": 50.0}
# The Feetech bus reports bare motor names; the adapter adds the ".pos" suffix.
BUS_POSITIONS = {"shoulder_pan": 10.0, "gripper": 50.0}


def adapter(**overrides) -> PositionMatchHandoverAdapter:
    options = {"tolerance": 1.0, "timeout_seconds": 1.0, "poll_interval_seconds": 0.0, "progress_interval_seconds": 0.1}
    options.update(overrides)
    return PositionMatchHandoverAdapter(**options)


class PositionMatchHandoverTests(unittest.TestCase):
    def test_transfers_control_once_every_joint_is_within_tolerance(self) -> None:
        leader = FakeLeader(
            [
                {"shoulder_pan.pos": 40.0, "gripper.pos": 0.0},
                {"shoulder_pan.pos": 12.0, "gripper.pos": 49.0},
                {"shoulder_pan.pos": 10.5, "gripper.pos": 50.4},
            ]
        )
        adapter().prepare_takeover(FakeRobot(FOLLOWER), leader, FOLLOWER)
        self.assertEqual(leader.calls, 3)

    def test_times_out_when_the_leader_never_matches(self) -> None:
        leader = FakeLeader([{"shoulder_pan.pos": 40.0, "gripper.pos": 0.0}])
        with self.assertRaisesRegex(RuntimeError, "not aligned within"):
            adapter(timeout_seconds=0.2).prepare_takeover(FakeRobot(FOLLOWER), leader, FOLLOWER)

    def test_joint_name_mismatch_is_rejected(self) -> None:
        leader = FakeLeader([{"elbow.pos": 10.0, "gripper.pos": 50.0}])
        with self.assertRaisesRegex(RuntimeError, "joint names differ"):
            adapter().prepare_takeover(FakeRobot(FOLLOWER), leader, FOLLOWER)

    def test_follower_pose_is_read_from_the_bus_without_a_last_action(self) -> None:
        leader = FakeLeader([{"shoulder_pan.pos": 10.2, "gripper.pos": 50.0}])
        adapter().prepare_takeover(FakeRobot(FOLLOWER), leader, None)
        self.assertEqual(leader.calls, 1)

    def test_never_touches_torque(self) -> None:
        class Teleop(FakeLeader):
            def disable_torque(self) -> None:
                raise AssertionError("position-match handover must not change leader torque")

        teleop = Teleop([dict(FOLLOWER)])
        handover = adapter()
        handover.prepare_takeover(FakeRobot(FOLLOWER), teleop, FOLLOWER)
        handover.begin_human_control(None, teleop)
        handover.end_human_control(None, teleop)
        handover.resume_autonomous(None, teleop)


if __name__ == "__main__":
    unittest.main()
