import unittest

from merger.perception_merger import PerceptionMerger
from utils.data_utils import MergeConfig, ObjectItem, PerceptionFrame


def _build_merger() -> PerceptionMerger:
    config = MergeConfig(
        planar_distance_threshold=200.0,
        height_distance_threshold=100.0,
        velocity_angle_threshold_deg=120.0,
        elec_ray_angle_threshold_deg=20.0,
        elec_position_correction_alpha=0.2,
        fusion_alpha=0.6,
    )
    return PerceptionMerger(config=config, class_correlation={})


class TestMergerContract(unittest.TestCase):
    def test_merge_frame_returns_update_and_create_ops(self) -> None:
        merger = _build_merger()
        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 1,
                "sensor_type": 2,
                "sensor_position": [0.0, 0.0, 10.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 5.0,
                "detections": [
                    {
                        "class_id": 1,
                        "position": [10.2, 0.0, 0.0],
                        "velocity": [1.0, 0.0, 0.0],
                        "track_id": 101,
                    },
                    {
                        "class_id": 4,
                        "position": [400.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "track_id": 202,
                    },
                ],
            }
        )
        global_items = [
            ObjectItem.from_dict(
                {
                    "global_id": 10,
                    "position": [10.0, 0.0, 0.0],
                    "velocity": [1.0, 0.0, 0.0],
                    "timestamp": 5.0,
                    "class_by_sensor": {"2": 1},
                    "class_votes": {"2": {"1": 2}},
                    "trajectory": [[9.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                    "observations": [],
                }
            )
        ]

        result = merger.merge_frame(frame, global_items)
        self.assertEqual(len(result.update_ops), 1)
        self.assertEqual(result.update_ops[0].target_id, 10)
        self.assertEqual(result.update_ops[0].operation, "update")
        self.assertIn("fused_position", result.update_ops[0].payload)
        self.assertEqual(len(result.create_ops), 1)
        self.assertEqual(result.create_ops[0].target_id, -1)
        self.assertEqual(result.create_ops[0].operation, "create")

    def test_merge_batch_keeps_processing_multiple_frames(self) -> None:
        merger = _build_merger()
        frames = [
            PerceptionFrame.from_dict(
                {
                    "uav_id": 1,
                    "sensor_type": 2,
                    "sensor_position": [0.0, 0.0, 0.0],
                    "sensor_orientation": [0.0, 0.0, 0.0],
                    "timestamp": 1.0,
                    "detections": [
                        {
                            "class_id": 1,
                            "position": [0.0, 0.0, 0.0],
                            "velocity": [1.0, 0.0, 0.0],
                            "track_id": 1,
                        }
                    ],
                }
            ),
            PerceptionFrame.from_dict(
                {
                    "uav_id": 1,
                    "sensor_type": 2,
                    "sensor_position": [0.0, 0.0, 0.0],
                    "sensor_orientation": [0.0, 0.0, 0.0],
                    "timestamp": 2.0,
                    "detections": [
                        {
                            "class_id": 1,
                            "position": [1.1, 0.0, 0.0],
                            "velocity": [1.0, 0.0, 0.0],
                            "track_id": 1,
                        }
                    ],
                }
            ),
        ]
        global_items = [
            ObjectItem.from_dict(
                {
                    "global_id": 100,
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [1.0, 0.0, 0.0],
                    "timestamp": 1.0,
                    "class_by_sensor": {"2": 1},
                    "class_votes": {"2": {"1": 1}},
                    "trajectory": [[0.0, 0.0, 0.0]],
                    "observations": [],
                }
            )
        ]

        result = merger.merge_batch(frames, global_items)
        self.assertGreaterEqual(len(result.update_ops), 2)
        self.assertEqual(len(result.create_ops), 0)
        self.assertEqual(result.debug_info["num_frames"], 2)

    def test_set_merge_mode_changes_default_mode(self) -> None:
        merger = _build_merger()
        merger.set_merge_mode("kf_reserved")
        self.assertEqual(merger.merge_mode, "kf_reserved")

    def test_elec_unmatched_detection_does_not_create_object(self) -> None:
        merger = _build_merger()
        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 1,
                "sensor_type": 3,
                "sensor_position": [0.0, 0.0, 0.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 1.0,
                "detections": [
                    {
                        "class_id": 1,
                        "position": [50.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "track_id": 1,
                        "bearing_vector": [1.0, 0.0, 0.0],
                    }
                ],
            }
        )
        result = merger.merge_frame(frame, [])
        self.assertEqual(len(result.create_ops), 0)

    def test_elec_matches_only_spatial_valid_objects(self) -> None:
        merger = _build_merger()
        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 1,
                "sensor_type": 3,
                "sensor_position": [0.0, 0.0, 0.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 2.0,
                "detections": [
                    {
                        "class_id": 2,
                        "position": [20.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "track_id": 8,
                        "bearing_vector": [1.0, 0.0, 0.0],
                    }
                ],
            }
        )
        global_items = [
            ObjectItem.from_dict(
                {
                    "global_id": 10,
                    "position": [30.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "timestamp": 2.0,
                    "class_by_sensor": {"3": 2},
                    "class_votes": {"3": {"2": 2}},
                    "trajectory": [[30.0, 0.0, 0.0]],
                    "observations": [],
                    "spatial_valid": False,
                }
            ),
            ObjectItem.from_dict(
                {
                    "global_id": 11,
                    "position": [30.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "timestamp": 2.0,
                    "class_by_sensor": {"3": 2},
                    "class_votes": {"3": {"2": 2}},
                    "trajectory": [[30.0, 0.0, 0.0]],
                    "observations": [],
                    "spatial_valid": True,
                }
            ),
        ]
        result = merger.merge_frame(frame, global_items)
        self.assertEqual(len(result.update_ops), 1)
        self.assertEqual(result.update_ops[0].target_id, 11)

    def test_elec_match_applies_small_position_correction(self) -> None:
        merger = _build_merger()
        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 9,
                "sensor_type": 3,
                "sensor_position": [0.0, 0.0, 0.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 3.0,
                "detections": [
                    {
                        "class_id": 1,
                        "position": [100.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "track_id": 5,
                        "bearing_vector": [1.0, 0.0, 0.0],
                    }
                ],
            }
        )
        global_items = [
            ObjectItem.from_dict(
                {
                    "global_id": 20,
                    "position": [50.0, 10.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "timestamp": 3.0,
                    "class_by_sensor": {"3": 1},
                    "class_votes": {"3": {"1": 1}},
                    "trajectory": [[50.0, 10.0, 0.0]],
                    "observations": [{"timestamp": 2.9, "sensor_type": 2}],
                    "spatial_valid": True,
                }
            )
        ]
        result = merger.merge_frame(frame, global_items)
        self.assertEqual(len(result.update_ops), 1)
        fused_y = result.update_ops[0].payload["fused_position"][1]
        # 初始y=10，alpha=0.2 应向射线(y=0)收敛，变为8
        self.assertAlmostEqual(fused_y, 8.0, places=4)


if __name__ == "__main__":
    unittest.main()
