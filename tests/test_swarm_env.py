import unittest

from simulator.env import SwarmEnv
from utils.data_utils import MergeOperation, MergeResult


class TestSwarmEnv(unittest.TestCase):
    def test_normalized_coordinates_are_scaled_by_map_size(self) -> None:
        env = SwarmEnv(
            {
                "seed": 1,
                "map_size": [1000, 800],
                "map_altitude": 200,
                "uavs": [
                    {
                        "uav_id": 1,
                        "position": [0.1, 0.2, 0.5],
                        "waypoints": [[0.1, 0.2, 0.5], [0.2, 0.2, 0.5]],
                        "sensor": {"sensor_type": 1},
                    }
                ],
                "targets": [
                    {
                        "target_id": 1,
                        "position": [0.3, 0.4, 0.0],
                        "velocity": [0, 0, 0],
                        "motion_mode": "static",
                        "class_by_sensor": {"1": 1},
                    }
                ],
            }
        )
        uav = env.uavs[0]
        target = env.targets[0]
        self.assertAlmostEqual(uav.position[0], 100.0)
        self.assertAlmostEqual(uav.position[1], 160.0)
        self.assertAlmostEqual(uav.position[2], 100.0)
        self.assertAlmostEqual(target.position[0], 300.0)
        self.assertAlmostEqual(target.position[1], 320.0)

    def test_uav_waypoint_moves_back_and_forth(self) -> None:
        env = SwarmEnv(
            {
                "seed": 1,
                "dt": 1.0,
                "map_size": [500, 500],
                "uavs": [
                    {
                        "uav_id": 1,
                        "position": [100, 100, 50],
                        "waypoints": [[100, 100, 50], [120, 100, 50], [140, 100, 50]],
                        "speed": 40,
                        "sensor": {"sensor_type": 1, "dropout_prob": 1.0},
                    }
                ],
                "targets": [],
            }
        )
        # 多步推进后应进入往返巡航（方向标志发生翻转）
        seen_forward = False
        seen_backward = False
        for _ in range(8):
            env.step()
            if env.uavs[0].patrol_forward:
                seen_forward = True
            else:
                seen_backward = True
        self.assertTrue(seen_forward and seen_backward)

    def test_step_generates_frames_and_render_state(self) -> None:
        env = SwarmEnv(
            {
                "seed": 1,
                "dt": 0.5,
                "map_size": [500, 500],
                "weather": "clear",
                "lighting": "night",
                "uavs": [
                    {
                        "uav_id": 1,
                        "position": [100, 100, 80],
                        "waypoints": [[100, 100, 80], [200, 100, 80]],
                        "sensor": {
                            "sensor_type": 1,
                            "params": {"forward_range": 300.0, "width": 200.0},
                            "dropout_prob": 0.0,
                            "position_noise_std": 0.0,
                            "velocity_noise_std": 0.0,
                        },
                    }
                ],
                "targets": [
                    {
                        "target_id": 11,
                        "position": [130, 100, 0],
                        "velocity": [0, 0, 0],
                        "motion_mode": "static",
                        "class_by_sensor": {"1": 3, "2": 2},
                    }
                ],
            }
        )

        result = env.step()

        self.assertGreaterEqual(result["pending_frames"], 1)
        self.assertEqual(result["render_state"]["weather"], "clear")
        self.assertEqual(len(result["render_state"]["uavs"]), 1)
        self.assertEqual(len(result["render_state"]["targets_truth"]), 1)

        frame = env.pop_next_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.uav_id, 1)
        self.assertGreaterEqual(len(frame.detections), 1)

    def test_record_merge_result_updates_match_edges(self) -> None:
        env = SwarmEnv(
            {
                "seed": 2,
                "uavs": [
                    {
                        "uav_id": 1,
                        "position": [0, 0, 50],
                        "waypoints": [[0, 0, 50], [50, 0, 50]],
                        "sensor": {
                            "sensor_type": 0,
                            "params": {"max_range": 300.0},
                            "dropout_prob": 0.0,
                            "position_noise_std": 0.0,
                            "velocity_noise_std": 0.0,
                        },
                    }
                ],
                "targets": [
                    {
                        "target_id": 1,
                        "position": [10, 0, 0],
                        "velocity": [0, 0, 0],
                        "motion_mode": "static",
                        "class_by_sensor": {"0": 7},
                    }
                ],
            }
        )

        env.step()
        frame = env.pop_next_frame()
        assert frame is not None
        merge_result = MergeResult(
            update_ops=[
                MergeOperation(
                    operation="update",
                    target_id=100,
                    payload={
                        "observation": {
                            "track_id": frame.detections[0].track_id,
                            "position": list(frame.detections[0].position),
                        }
                    },
                    score=1.5,
                )
            ]
        )

        env.record_merge_result(frame, merge_result)
        render_state = env.get_render_state()
        self.assertEqual(len(render_state["match_edges"]), 1)
        self.assertEqual(render_state["match_edges"][0]["target_id"], 100)


if __name__ == "__main__":
    unittest.main()
