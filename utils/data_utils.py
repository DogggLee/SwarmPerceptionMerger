from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple


Vector3 = Tuple[float, float, float]


class SensorType(Enum):
    RADAR = 0
    IF = 1
    RGB = 2
    ELEC = 3

    @classmethod
    def parse(cls, value: Any) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            upper = value.upper()
            if upper in cls.__members__:
                return cls[upper].value
            if value.isdigit():
                return int(value)
        raise ValueError(f"Invalid sensor_type: {value}")


def _vec3(values: Any, name: str) -> Vector3:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must be a list of 3 numbers")
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass
class Detection:
    class_id: int
    position: Vector3
    velocity: Vector3
    track_id: int
    bearing_vector: Optional[Vector3] = None
    bbox: Optional[List[float]] = None
    image: Optional[Any] = None
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(
            class_id=int(data["class_id"]),
            position=_vec3(data["position"], "detection.position"),
            velocity=_vec3(data.get("velocity", [0, 0, 0]), "detection.velocity"),
            track_id=int(data["track_id"]),
            bearing_vector=(
                _vec3(data["bearing_vector"], "detection.bearing_vector")
                if "bearing_vector" in data and data["bearing_vector"] is not None
                else None
            ),
            bbox=data.get("bbox"),
            image=data.get("image"),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "track_id": self.track_id,
            "bearing_vector": list(self.bearing_vector) if self.bearing_vector is not None else None,
            "bbox": self.bbox,
            "image": self.image,
            "confidence": self.confidence,
        }


@dataclass
class Observation:
    uav_id: int
    sensor_type: int
    sensor_position: Vector3
    sensor_orientation: Vector3
    timestamp: float
    detection: Detection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uav_id": self.uav_id,
            "sensor_type": self.sensor_type,
            "sensor_position": list(self.sensor_position),
            "sensor_orientation": list(self.sensor_orientation),
            "timestamp": self.timestamp,
            "detection": self.detection.to_dict(),
        }


@dataclass
class PerceptionFrame:
    uav_id: int
    sensor_type: int
    sensor_position: Vector3
    sensor_orientation: Vector3
    timestamp: float
    detections: List[Detection]
    sensor_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionFrame":
        detections_raw = data.get("detections", [])
        detections = [Detection.from_dict(item) for item in detections_raw]
        return cls(
            uav_id=int(data["uav_id"]),
            sensor_type=SensorType.parse(data["sensor_type"]),
            sensor_position=_vec3(data["sensor_position"], "perception_frame.sensor_position"),
            sensor_orientation=_vec3(
                data.get("sensor_orientation", [0, 0, 0]),
                "perception_frame.sensor_orientation",
            ),
            timestamp=float(data["timestamp"]),
            detections=detections,
            sensor_params=data.get("sensor_params", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uav_id": self.uav_id,
            "sensor_type": self.sensor_type,
            "sensor_position": list(self.sensor_position),
            "sensor_orientation": list(self.sensor_orientation),
            "timestamp": self.timestamp,
            "sensor_params": self.sensor_params,
            "detections": [item.to_dict() for item in self.detections],
        }


@dataclass
class ObjectItem:
    global_id: int
    position: Vector3
    velocity: Vector3
    timestamp: float
    class_by_sensor: Dict[int, int] = field(default_factory=dict)
    class_votes: Dict[int, Dict[int, int]] = field(default_factory=dict)
    trajectory: List[Vector3] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    spatial_valid: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectItem":
        class_by_sensor_raw = data.get("class_by_sensor", {})
        class_votes_raw = data.get("class_votes", {})

        class_by_sensor: Dict[int, int] = {}
        for sensor, class_id in class_by_sensor_raw.items():
            class_by_sensor[int(sensor)] = int(class_id)

        class_votes: Dict[int, Dict[int, int]] = {}
        for sensor, votes in class_votes_raw.items():
            sensor_key = int(sensor)
            class_votes[sensor_key] = {}
            for class_id, count in votes.items():
                class_votes[sensor_key][int(class_id)] = int(count)

        trajectory = [_vec3(p, "object_item.trajectory") for p in data.get("trajectory", [])]
        return cls(
            global_id=int(data["global_id"]),
            position=_vec3(data["position"], "object_item.position"),
            velocity=_vec3(data.get("velocity", [0, 0, 0]), "object_item.velocity"),
            timestamp=float(data["timestamp"]),
            class_by_sensor=class_by_sensor,
            class_votes=class_votes,
            trajectory=trajectory,
            observations=list(data.get("observations", [])),
            spatial_valid=bool(data.get("spatial_valid", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        class_by_sensor = {str(k): v for k, v in self.class_by_sensor.items()}
        class_votes = {str(k): {str(ck): cv for ck, cv in v.items()} for k, v in self.class_votes.items()}
        return {
            "global_id": self.global_id,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "timestamp": self.timestamp,
            "class_by_sensor": class_by_sensor,
            "class_votes": class_votes,
            "trajectory": [list(p) for p in self.trajectory],
            "observations": self.observations,
            "spatial_valid": self.spatial_valid,
        }


@dataclass
class MergeOperation:
    operation: str
    target_id: int
    payload: Dict[str, Any]
    score: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "target_id": self.target_id,
            "payload": self.payload,
            "score": self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeOperation":
        return cls(
            operation=str(data["operation"]),
            target_id=int(data["target_id"]),
            payload=dict(data.get("payload", {})),
            score=float(data["score"]) if data.get("score") is not None else None,
            reason=str(data["reason"]) if data.get("reason") is not None else None,
        )


@dataclass
class MergeResult:
    update_ops: List[MergeOperation] = field(default_factory=list)
    create_ops: List[MergeOperation] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_ops": [op.to_dict() for op in self.update_ops],
            "create_ops": [op.to_dict() for op in self.create_ops],
            "alerts": self.alerts,
            "debug_info": self.debug_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeResult":
        return cls(
            update_ops=[MergeOperation.from_dict(item) for item in data.get("update_ops", [])],
            create_ops=[MergeOperation.from_dict(item) for item in data.get("create_ops", [])],
            alerts=list(data.get("alerts", [])),
            debug_info=dict(data.get("debug_info", {})),
        )


@dataclass
class MergeConfig:
    planar_distance_threshold: float = 80.0
    height_distance_threshold: float = 30.0
    velocity_angle_threshold_deg: float = 80.0
    elec_ray_angle_threshold_deg: float = 12.0
    elec_position_correction_alpha: float = 0.15
    track_window_size: int = 8
    trajectory_window_size: int = 12
    fusion_alpha: float = 0.6
    enable_visibility: bool = False
    default_merge_mode: str = "simple"
    cost_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "class": 2.0,
            "distance": 2.0,
            "velocity": 1.0,
            "track": 1.0,
            "visibility": 0.5,
            "elec_angle": 2.5,
        }
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MergeConfig":
        if not data:
            return cls()
        kwargs = {
            "planar_distance_threshold": float(
                data.get("planar_distance_threshold", cls.planar_distance_threshold)
            ),
            "height_distance_threshold": float(
                data.get("height_distance_threshold", cls.height_distance_threshold)
            ),
            "velocity_angle_threshold_deg": float(
                data.get("velocity_angle_threshold_deg", cls.velocity_angle_threshold_deg)
            ),
            "elec_ray_angle_threshold_deg": float(
                data.get("elec_ray_angle_threshold_deg", cls.elec_ray_angle_threshold_deg)
            ),
            "elec_position_correction_alpha": float(
                data.get("elec_position_correction_alpha", cls.elec_position_correction_alpha)
            ),
            "track_window_size": int(data.get("track_window_size", cls.track_window_size)),
            "trajectory_window_size": int(
                data.get("trajectory_window_size", cls.trajectory_window_size)
            ),
            "fusion_alpha": float(data.get("fusion_alpha", cls.fusion_alpha)),
            "enable_visibility": bool(data.get("enable_visibility", cls.enable_visibility)),
            "default_merge_mode": str(data.get("default_merge_mode", cls.default_merge_mode)),
        }
        config = cls(**kwargs)
        weights = data.get("cost_weights")
        if isinstance(weights, dict):
            for k in config.cost_weights:
                if k in weights:
                    config.cost_weights[k] = float(weights[k])
        return config


@dataclass
class TrackHistory:
    points: Deque[Vector3] = field(default_factory=deque)
    timestamps: Deque[float] = field(default_factory=deque)

    def append(self, position: Vector3, timestamp: float, max_len: int) -> None:
        self.points.append(position)
        self.timestamps.append(timestamp)
        while len(self.points) > max_len:
            self.points.popleft()
            self.timestamps.popleft()
