# Merge API Example

## POST /merge request (single frame)
```json
{
  "context": {
    "merge_mode": "simple"
  },
  "global_objects": [
    {
      "global_id": 101,
      "position": [100.0, 100.0, 0.0],
      "velocity": [1.0, 0.0, 0.0],
      "timestamp": 10.0,
      "class_by_sensor": { "2": 1 },
      "class_votes": { "2": { "1": 4 } },
      "trajectory": [[99.0, 100.0, 0.0], [100.0, 100.0, 0.0]],
      "observations": []
    }
  ],
  "perception_frame": {
    "uav_id": 7,
    "sensor_type": 2,
    "sensor_position": [0.0, 0.0, 10.0],
    "sensor_orientation": [0.0, 0.0, 45.0],
    "timestamp": 11.0,
    "sensor_params": {
      "hfov_deg": 120.0,
      "max_range": 1000.0
    },
    "detections": [
      {
        "class_id": 1,
        "position": [101.2, 100.1, 0.0],
        "velocity": [1.1, 0.0, 0.0],
        "track_id": 42,
        "bbox": [50, 60, 120, 140],
        "confidence": 0.97
      },
      {
        "class_id": 3,
        "position": [300.0, 400.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "track_id": 99
      }
    ]
  }
}
```

## Example response
```json
{
  "update_ops": [
    {
      "operation": "update",
      "target_id": 101,
      "payload": {
        "timestamp": 11.0,
        "fused_position": [101.12, 100.06, 0.0],
        "fused_velocity": [1.06, 0.0, 0.0],
        "observation": {
          "uav_id": 7,
          "sensor_type": 2,
          "sensor_position": [0.0, 0.0, 10.0],
          "sensor_orientation": [0.0, 0.0, 45.0],
          "timestamp": 11.0,
          "class_id": 1,
          "position": [101.2, 100.1, 0.0],
          "velocity": [1.1, 0.0, 0.0],
          "track_id": 42,
          "bbox": [50, 60, 120, 140]
        },
        "class_update": {
          "sensor_type": 2,
          "class_id": 1,
          "class_by_sensor": { "2": 1 },
          "class_votes": { "2": { "1": 5 } }
        }
      }
    }
  ],
  "create_ops": [
    {
      "operation": "create",
      "target_id": -1,
      "payload": {
        "timestamp": 11.0,
        "position": [300.0, 400.0, 0.0],
        "velocity": [0.0, 0.0, 0.0]
      }
    }
  ],
  "alerts": [],
  "debug_info": {
    "mode": "simple"
  }
}
```
