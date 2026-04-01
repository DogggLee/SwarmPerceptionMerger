import unittest

from simulator.env import SwarmEnv
from utils.data_utils import MergeOperation, MergeResult


class TestSwarmEnv(unittest.TestCase):
    def test_generated_uavs_same_modality_share_profile(self) -> None:
        env = SwarmEnv(
            {
                "seed": 1,
                "generation_seed": 99,
                "map_size": [1000, 1000],
                "uav_counts": {"RGB": 3},
                "uav_profiles": {
                    "RGB": {
                        "speed": 36.5,
                        "sensor": {
                            "params": {"forward_range": 280.0, "width": 160.0},
                            "position_noise_std": 1.1,
                            "velocity_noise_std": 0.3,
                            "dropout_prob": 0.07,
                        },
                    }
                },
                "patrol_routes": [
                    [[0.1, 0.1, 0.5], [0.9, 0.1, 0.5], [0.9, 0.9, 0.5]],
                    [[0.2, 0.8, 0.4], [0.8, 0.8, 0.4], [0.8, 0.2, 0.4]],
                ],
                "target_count": 0,
            }
        )
        self.assertEqual(len(env.uavs), 3)
        for uav in env.uavs:
            self.assertAlmostEqual(uav.speed, 36.5)
            self.assertAlmostEqual(uav.sensor.position_noise_std, 1.1)
            self.assertAlmostEqual(uav.sensor.velocity_noise_std, 0.3)
            self.assertAlmostEqual(uav.sensor.dropout_prob, 0.07)
            self.assertEqual(uav.sensor.params["forward_range"], 280.0)

    def test_generation_seed_makes_layout_reproducible(self) -> None:
        cfg = {
            "seed": 7,
            "generation_seed": 12345,
            "map_size": [1000, 1000],
            "uav_counts": {"RADAR": 1, "IF": 1, "RGB": 1, "ELEC": 1},
            "uav_profiles": {
                "RADAR": {"speed": 20, "sensor": {"params": {"max_range": 300.0}}},
                "IF": {"speed": 21, "sensor": {"params": {"forward_range": 280.0, "width": 140.0}}},
                "RGB": {"speed": 22, "sensor": {"params": {"forward_range": 290.0, "width": 150.0}}},
                "ELEC": {"speed": 23, "sensor": {"params": {"max_range": 310.0}}},
            },
            "patrol_routes": [
                [[0.1, 0.1, 0.5], [0.9, 0.1, 0.5], [0.9, 0.9, 0.5]],
                [[0.2, 0.2, 0.6], [0.8, 0.2, 0.6], [0.8, 0.8, 0.6]],
            ],
            "target_count": 4,
            "class_correlation": {
                "0": {"1": {"0": [1], "1": [2], "2": [3], "3": [1]}},
                "1": {"2": {"0": [2], "1": [2], "2": [2], "3": [2]}},
            },
        }
        env_a = SwarmEnv(cfg)
        env_b = SwarmEnv(cfg)
        self.assertEqual([u.position for u in env_a.uavs], [u.position for u in env_b.uavs])
        self.assertEqual([u.waypoints for u in env_a.uavs], [u.waypoints for u in env_b.uavs])
        self.assertEqual([t.position for t in env_a.targets], [t.position for t in env_b.targets])
        self.assertEqual([t.class_by_sensor for t in env_a.targets], [t.class_by_sensor for t in env_b.targets])

    def test_generated_targets_sample_classes_from_correlation(self) -> None:
        corr = {
            "0": {
                "5": {"0": [5], "1": [6], "2": [7], "3": [8]},
                "9": {"0": [9], "1": [10], "2": [11], "3": [12]},
            }
        }
        env = SwarmEnv(
            {
                "seed": 3,
                "generation_seed": 4,
                "map_size": [500, 500],
                "uav_counts": {"RGB": 1},
                "uav_profiles": {"RGB": {"speed": 20, "sensor": {"params": {"forward_range": 100.0, "width": 60.0}}}},
                "patrol_routes": [[[0.1, 0.1, 0.5], [0.8, 0.1, 0.5]]],
                "target_count": 6,
                "class_correlation": corr,
            }
        )
        allowed = {
            tuple(sorted({0: 5, 1: 6, 2: 7, 3: 8}.items())),
            tuple(sorted({0: 9, 1: 10, 2: 11, 3: 12}.items())),
        }
        self.assertEqual(len(env.targets), 6)
        for target in env.targets:
            self.assertIn(tuple(sorted(target.class_by_sensor.items())), allowed)

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
