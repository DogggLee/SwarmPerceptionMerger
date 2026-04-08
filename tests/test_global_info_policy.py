import unittest

from simulator.global_info import GlobalInfo
from utils.data_utils import MergeOperation, MergeResult, ObjectItem


class TestGlobalInfoPolicy(unittest.TestCase):
    def test_apply_merge_result_create_ops_keep_assigned_global_ids(self) -> None:
        gi = GlobalInfo(valid_observation_count=1, max_unseen_time=4.0, stale_observation_time=10.0)
        merge_result = MergeResult(
            create_ops=[
                MergeOperation(
                    operation="create",
                    target_id=0,
                    payload={
                        "timestamp": 1.0,
                        "position": [10.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "observation": {"timestamp": 1.0, "position": [10.0, 0.0, 0.0], "sensor_type": 2},
                        "class_by_sensor": {"2": 1},
                        "class_votes": {"2": {"1": 1}},
                        "spatial_valid": True,
                    },
                ),
                MergeOperation(
                    operation="create",
                    target_id=1,
                    payload={
                        "timestamp": 1.2,
                        "position": [20.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "observation": {"timestamp": 1.2, "position": [20.0, 0.0, 0.0], "sensor_type": 2},
                        "class_by_sensor": {"2": 2},
                        "class_votes": {"2": {"2": 1}},
                        "spatial_valid": True,
                    },
                ),
            ]
        )

        id_map = gi.apply_merge_result(merge_result)
        self.assertEqual(id_map[0], 0)
        self.assertEqual(id_map[1], 1)
        self.assertIsNotNone(gi.get_item(0))
        self.assertIsNotNone(gi.get_item(1))

    def test_predict_updates_state_and_maintains_items(self) -> None:
        gi = GlobalInfo(valid_observation_count=3, max_unseen_time=4.0, stale_observation_time=2.0)
        item = ObjectItem.from_dict(
            {
                "global_id": 1,
                "position": [0.0, 0.0, 0.0],
                "velocity": [2.0, 0.0, 0.0],
                "timestamp": 1.0,
                "class_by_sensor": {"2": 1},
                "class_votes": {"2": {"1": 1}},
                "trajectory": [[0.0, 0.0, 0.0]],
                "observations": [
                    {"timestamp": 3.5, "position": [5.0, 0.0, 0.0], "sensor_type": 2},
                    {"timestamp": 1.0, "position": [0.0, 0.0, 0.0], "sensor_type": 2}
                ],
            }
        )
        gi._add_item(item)
        gi.predict(5.0)

        updated = gi.get_item(1)
        assert updated is not None
        self.assertAlmostEqual(updated.position[0], 8.0, places=6)
        self.assertEqual(len(updated.observations), 1)

    def test_apply_merge_result_calls_maintain_and_valid_filter(self) -> None:
        gi = GlobalInfo(valid_observation_count=3, max_unseen_time=4.0, stale_observation_time=2.0)
        gi._add_item(
            ObjectItem.from_dict(
                {
                    "global_id": 1,
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "timestamp": 0.0,
                    "class_by_sensor": {"2": 1},
                    "class_votes": {"2": {"1": 1}},
                    "trajectory": [[0.0, 0.0, 0.0]],
                    "observations": [],
                }
            )
        )

        merge_result = MergeResult(
            update_ops=[
                MergeOperation(
                    operation="update",
                    target_id=1,
                    payload={
                        "timestamp": 1.0,
                        "fused_position": [1.0, 0.0, 0.0],
                        "fused_velocity": [0.0, 0.0, 0.0],
                        "observation": {"timestamp": 1.0, "position": [1.0, 0.0, 0.0], "sensor_type": 2},
                        "class_update": {"class_by_sensor": {"2": 1}, "class_votes": {"2": {"1": 1}}},
                    },
                ),
                MergeOperation(
                    operation="update",
                    target_id=1,
                    payload={
                        "timestamp": 1.5,
                        "fused_position": [1.5, 0.0, 0.0],
                        "fused_velocity": [0.0, 0.0, 0.0],
                        "observation": {"timestamp": 1.5, "position": [1.5, 0.0, 0.0], "sensor_type": 2},
                        "class_update": {"class_by_sensor": {"2": 1}, "class_votes": {"2": {"1": 2}}},
                    },
                ),
                MergeOperation(
                    operation="update",
                    target_id=1,
                    payload={
                        "timestamp": 2.0,
                        "fused_position": [2.0, 0.0, 0.0],
                        "fused_velocity": [0.0, 0.0, 0.0],
                        "observation": {"timestamp": 2.0, "position": [2.0, 0.0, 0.0], "sensor_type": 2},
                        "class_update": {"class_by_sensor": {"2": 1}, "class_votes": {"2": {"1": 3}}},
                    },
                ),
            ]
        )
        gi.apply_merge_result(merge_result)
        valid = list(gi.get_valid_items())
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].global_id, 1)

    def test_spatial_invalid_item_is_not_valid_item(self) -> None:
        gi = GlobalInfo(valid_observation_count=1, max_unseen_time=4.0, stale_observation_time=2.0)
        gi._add_item(
            ObjectItem.from_dict(
                {
                    "global_id": 7,
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "timestamp": 1.0,
                    "class_by_sensor": {"2": 1},
                    "class_votes": {"2": {"1": 1}},
                    "trajectory": [[0.0, 0.0, 0.0]],
                    "observations": [{"timestamp": 1.0, "sensor_type": 3}],
                    "spatial_valid": False
                }
            )
        )
        self.assertEqual(len(list(gi.get_valid_items())), 0)


if __name__ == "__main__":
    unittest.main()
