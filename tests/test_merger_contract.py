import unittest
import logging
import os

from merger.perception_merger import PerceptionMerger
from utils.data_utils import Detection, MergeConfig, ObjectItem, PerceptionFrame, TrackHistory


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
    def test_track_cost_uses_dtw_with_full_history_in_timestamp_window(self) -> None:
        merger = _build_merger()
        key = (1, 2, 7)
        history = TrackHistory()
        history.append((0.0, 0.0, 0.0), 1.0, max_len=20)
        history.append((1.0, 0.0, 0.0), 2.0, max_len=20)
        history.append((2.0, 0.0, 0.0), 3.0, max_len=20)
        history.append((3.0, 0.0, 0.0), 4.0, max_len=20)
        merger.track_memory[key] = history

        frame = PerceptionFrame(
            uav_id=1,
            sensor_type=2,
            sensor_position=(0.0, 0.0, 0.0),
            sensor_orientation=(0.0, 0.0, 0.0),
            timestamp=4.0,
            detections=[Detection(class_id=1, position=(3.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0), track_id=7)],
        )
        det = frame.detections[0]

        obj_good = ObjectItem.from_dict(
            {
                "global_id": 10,
                "position": [3.0, 0.0, 0.0],
                "velocity": [1.0, 0.0, 0.0],
                "timestamp": 4.0,
                "class_by_sensor": {"2": 1},
                "class_votes": {"2": {"1": 2}},
                "trajectory": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                "observations": [
                    {"timestamp": 0.5, "position": [100.0, 100.0, 0.0]},
                    {"timestamp": 1.0, "position": [0.0, 0.0, 0.0]},
                    {"timestamp": 2.0, "position": [1.0, 0.1, 0.0]},
                    {"timestamp": 3.0, "position": [2.0, 0.0, 0.0]},
                    {"timestamp": 4.0, "position": [3.0, 0.0, 0.0]},
                    {"timestamp": 8.0, "position": [-50.0, -50.0, 0.0]},
                ],
            }
        )
        obj_bad = ObjectItem.from_dict(
            {
                "global_id": 11,
                "position": [0.0, 3.0, 0.0],
                "velocity": [0.0, 1.0, 0.0],
                "timestamp": 4.0,
                "class_by_sensor": {"2": 1},
                "class_votes": {"2": {"1": 2}},
                "trajectory": [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                "observations": [
                    {"timestamp": 1.0, "position": [0.0, 0.0, 0.0]},
                    {"timestamp": 2.0, "position": [0.0, 1.0, 0.0]},
                    {"timestamp": 3.0, "position": [0.0, 2.0, 0.0]},
                    {"timestamp": 4.0, "position": [0.0, 3.0, 0.0]},
                ],
            }
        )

        cost_good = merger._track_cost(frame, det, obj_good)
        cost_bad = merger._track_cost(frame, det, obj_bad)
        self.assertLess(cost_good, cost_bad)
        self.assertLess(cost_good, 0.25)
        self.assertGreater(cost_bad, 0.35)

    def test_logger_also_writes_timestamped_file_under_logs(self) -> None:
        merger = _build_merger()
        self.assertGreaterEqual(len(merger._file_logger.handlers), 1)  # type: ignore[attr-defined]
        handler = merger._file_logger.handlers[0]  # type: ignore[attr-defined]
        log_path = getattr(handler, "baseFilename", "")
        self.assertTrue(log_path.endswith(".txt"))
        self.assertIn(f"{os.sep}logs{os.sep}", log_path)
        self.assertRegex(os.path.basename(log_path), r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.txt$")

    def test_logger_records_debug_info_error(self) -> None:
        merger = _build_merger()
        records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("test_merger_logger")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        logger.addHandler(_CaptureHandler())
        logger.propagate = False
        merger.set_logger(logger)

        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 1,
                "sensor_type": 2,
                "sensor_position": [0.0, 0.0, 0.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 1.0,
                "detections": [
                    {"class_id": 1, "position": [5.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "track_id": 1}
                ],
            }
        )
        _ = merger.merge_frame(frame, [])
        with self.assertRaises(Exception):
            merger.merge_frame(frame, None)  # type: ignore[arg-type]

        levels = [record.levelname for record in records]
        self.assertIn("INFO", levels)
        self.assertIn("ERROR", levels)
        messages = [record.getMessage() for record in records]
        self.assertTrue(any("Received " in msg and "Process " in msg and " ms" in msg for msg in messages))

    def test_file_log_keeps_summary_process_and_errors_only(self) -> None:
        merger = _build_merger()
        handler = merger._file_logger.handlers[0]  # type: ignore[attr-defined]
        log_path = getattr(handler, "baseFilename", "")
        frame = PerceptionFrame.from_dict(
            {
                "uav_id": 1,
                "sensor_type": 2,
                "sensor_position": [0.0, 0.0, 0.0],
                "sensor_orientation": [0.0, 0.0, 0.0],
                "timestamp": 1.0,
                "detections": [
                    {"class_id": 1, "position": [5.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "track_id": 1}
                ],
            }
        )
        _ = merger.merge_frame(frame, [])
        with self.assertRaises(Exception):
            merger.merge_frame(frame, None)  # type: ignore[arg-type]
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Received 1 perception results and 0 global objects, merged to 0 updates and 1 creates. Process", content)
        self.assertIn("MERGER_EVENT", content)
        self.assertIn("merger_summary", content)
        self.assertIn("merge_frame failed", content)
        self.assertNotIn("merger_input_meta", content)
        self.assertNotIn("merger_input_perception_frame", content)

    def test_render_returns_status(self) -> None:
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
                    }
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
        render_result = merger.render(frame, global_items, result, output_path=None, show=False)
        self.assertIn(render_result["status"], {"ok", "matplotlib_unavailable"})
        self.assertEqual(render_result["num_detections"], 1)
        self.assertEqual(render_result["num_global_objects"], 1)

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
