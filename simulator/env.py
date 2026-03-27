from __future__ import annotations

import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from utils.data_utils import Detection, MergeResult, ObjectItem, PerceptionFrame, SensorType, Vector3


def _vec3(values: Sequence[float], default_z: float = 0.0) -> Vector3:
    """将 2D/3D 数组标准化为三维向量。"""
    if len(values) == 2:
        return (float(values[0]), float(values[1]), float(default_z))
    if len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    raise ValueError("Expected a 2D or 3D vector")


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    """三维向量加法。"""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a: Vector3, b: Vector3) -> Vector3:
    """三维向量减法。"""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_scale(a: Vector3, s: float) -> Vector3:
    """三维向量按标量缩放。"""
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_norm(a: Vector3) -> float:
    """计算三维向量模长。"""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vec_unit(a: Vector3) -> Vector3:
    """计算三维向量单位方向。"""
    norm = _vec_norm(a)
    if norm < 1e-9:
        return (1.0, 0.0, 0.0)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def _yaw_deg_from_vec(vec: Vector3) -> float:
    """根据平面向量计算偏航角（度）。"""
    return math.degrees(math.atan2(vec[1], vec[0]))


@dataclass
class SensorSpec:
    sensor_type: int
    params: Dict[str, Any] = field(default_factory=dict)
    position_noise_std: float = 1.0
    velocity_noise_std: float = 0.5
    dropout_prob: float = 0.0


@dataclass
class UAVState:
    uav_id: int
    sensor: SensorSpec
    position: Vector3
    velocity: Vector3
    yaw_deg: float
    waypoints: List[Vector3]
    waypoint_index: int = 0
    speed: float = 20.0
    patrol_forward: bool = True


@dataclass
class TargetState:
    target_id: int
    class_by_sensor: Dict[int, int]
    position: Vector3
    velocity: Vector3
    motion_mode: str
    speed_range: Tuple[float, float]
    waypoints: List[Vector3] = field(default_factory=list)
    waypoint_index: int = 0
    patrol_forward: bool = True
    random_heading_interval: float = 3.0
    last_heading_change: float = 0.0


class SwarmEnv:
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化集群仿真环境与渲染所需状态缓存。"""
        self.config = config
        self.dt = float(config.get("dt", 0.5))
        self.time = 0.0
        self.map_size = tuple(config.get("map_size", [1000.0, 1000.0]))
        self.map_altitude = float(config.get("map_altitude", 200.0))
        self.global_weather = str(config.get("weather", "clear")).lower()
        self.global_lighting = str(config.get("lighting", "day")).lower()
        self.rng = random.Random(config.get("seed", 0))

        self.uavs: List[UAVState] = []
        self.targets: List[TargetState] = []
        self.perception_queue: Deque[PerceptionFrame] = deque()
        self.global_objects_snapshot: List[Dict[str, Any]] = []
        self.last_merge_feedback: Dict[str, Any] = {
            "match_edges": [],
            "frame_timestamp": None,
        }
        self.render_state: Dict[str, Any] = {}

        self._target_track_ids: Dict[Tuple[int, int, int], int] = {}
        self._next_track_ids: Dict[Tuple[int, int], int] = {}

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境到初始状态并清空缓存池。"""
        if seed is not None:
            self.rng.seed(seed)
        self.time = 0.0
        self.perception_queue.clear()
        self.global_objects_snapshot = []
        self.last_merge_feedback = {"match_edges": [], "frame_timestamp": None}
        self._target_track_ids = {}
        self._next_track_ids = {}
        self.uavs = self._build_uavs()
        self.targets = self._build_targets()
        self._refresh_render_state()
        return self.get_state()

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """推进一个仿真步长并生成感知帧。"""
        step_dt = float(dt if dt is not None else self.dt)
        self.time += step_dt

        self._advance_targets(step_dt)
        self._advance_uavs(step_dt)
        frames = self._generate_perception_frames()
        for frame in frames:
            # 仅缓存有效观测帧，空检测结果不进入融合链路。
            if frame.detections:
                self.perception_queue.append(frame)

        self._refresh_render_state()
        return {
            "time": self.time,
            "generated_frames": [frame.to_dict() for frame in frames],
            "pending_frames": len(self.perception_queue),
            "render_state": deepcopy(self.render_state),
        }

    def pop_next_frame(self) -> Optional[PerceptionFrame]:
        """从感知缓存池按 FIFO 弹出下一帧。"""
        if not self.perception_queue:
            return None
        return self.perception_queue.popleft()

    def get_pending_frames(self) -> List[PerceptionFrame]:
        """获取当前未消费的感知帧列表。"""
        return list(self.perception_queue)

    def set_global_objects(self, global_objects: Sequence[ObjectItem | Dict[str, Any]]) -> None:
        """更新渲染层使用的全局态势快照。"""
        snapshot: List[Dict[str, Any]] = []
        for item in global_objects:
            if isinstance(item, ObjectItem):
                snapshot.append(item.to_dict())
            else:
                snapshot.append(deepcopy(item))
        self.global_objects_snapshot = snapshot
        self._refresh_render_state()

    def sync_global_info(self, global_info: Any) -> None:
        """从 GlobalInfo 同步全局目标记录到渲染缓存。"""
        if hasattr(global_info, "get_all_items"):
            self.set_global_objects(list(global_info.get_all_items()))
            return
        raise TypeError("global_info must provide get_all_items()")

    def record_merge_result(self, frame: PerceptionFrame, merge_result: MergeResult) -> None:
        """记录融合匹配边与分数，用于可视化关联关系。"""
        match_edges: List[Dict[str, Any]] = []
        for op in merge_result.update_ops:
            observation = op.payload.get("observation", {})
            match_edges.append(
                {
                    "type": "update",
                    "frame_timestamp": frame.timestamp,
                    "uav_id": frame.uav_id,
                    "sensor_type": frame.sensor_type,
                    "track_id": observation.get("track_id"),
                    "detection_position": observation.get("position"),
                    "sensor_position": observation.get("sensor_position"),
                    "bearing_vector": observation.get("bearing_vector"),
                    "target_id": op.target_id,
                    "score": op.score,
                }
            )
        for op in merge_result.create_ops:
            observation = op.payload.get("observation", {})
            match_edges.append(
                {
                    "type": "create",
                    "frame_timestamp": frame.timestamp,
                    "uav_id": frame.uav_id,
                    "sensor_type": frame.sensor_type,
                    "track_id": observation.get("track_id"),
                    "detection_position": observation.get("position"),
                    "sensor_position": observation.get("sensor_position"),
                    "bearing_vector": observation.get("bearing_vector"),
                    "target_id": op.target_id,
                    "score": op.score,
                }
            )
        self.last_merge_feedback = {
            "match_edges": match_edges,
            "frame_timestamp": frame.timestamp,
        }
        self._refresh_render_state()

    def get_state(self) -> Dict[str, Any]:
        """返回环境基础状态与当前渲染数据。"""
        return {
            "time": self.time,
            "weather": self.global_weather,
            "lighting": self.global_lighting,
            "pending_frames": len(self.perception_queue),
            "render_state": deepcopy(self.render_state),
        }

    def get_render_state(self) -> Dict[str, Any]:
        """获取可视化层使用的渲染状态快照。"""
        return deepcopy(self.render_state)

    def render(self) -> None:
        """预留渲染接口，当前由外部 UI 消费 render_state 绘制。"""
        raise NotImplementedError("Render is reserved for future visualization integration.")

    def _build_uavs(self) -> List[UAVState]:
        """根据配置构建无人机与传感器状态。"""
        raw_uavs = list(self.config.get("uavs", []))
        shared_route = self.config.get("shared_patrol_route")
        if shared_route and raw_uavs:
            segments = self._split_waypoints(shared_route, len(raw_uavs))
            for idx, uav_cfg in enumerate(raw_uavs):
                uav_cfg = dict(uav_cfg)
                uav_cfg.setdefault("waypoints", segments[idx])
                raw_uavs[idx] = uav_cfg

        uavs: List[UAVState] = []
        for idx, item in enumerate(raw_uavs):
            sensor_cfg = item.get("sensor", {})
            sensor = SensorSpec(
                sensor_type=SensorType.parse(sensor_cfg.get("sensor_type", item.get("sensor_type", 2))),
                params=dict(sensor_cfg.get("params", {})),
                position_noise_std=float(sensor_cfg.get("position_noise_std", 2.0)),
                velocity_noise_std=float(sensor_cfg.get("velocity_noise_std", 0.8)),
                dropout_prob=float(sensor_cfg.get("dropout_prob", 0.05)),
            )
            raw_waypoints = item.get("waypoints", [item.get("position", [0, 0, 50])])
            waypoints = [self._to_world_coord(_vec3(p)) for p in raw_waypoints]
            start_pos = self._to_world_coord(_vec3(item.get("position", waypoints[0])))
            speed = float(item.get("speed", 20.0))
            yaw_deg = float(item.get("yaw_deg", 0.0))
            if len(waypoints) > 1:
                yaw_deg = _yaw_deg_from_vec(_vec_sub(waypoints[1], waypoints[0]))
            uavs.append(
                UAVState(
                    uav_id=int(item.get("uav_id", idx + 1)),
                    sensor=sensor,
                    position=start_pos,
                    velocity=(0.0, 0.0, 0.0),
                    yaw_deg=yaw_deg,
                    waypoints=waypoints,
                    speed=speed,
                    patrol_forward=True,
                )
            )
        return uavs

    def _build_targets(self) -> List[TargetState]:
        """根据配置构建任务目标状态。"""
        targets: List[TargetState] = []
        for idx, item in enumerate(self.config.get("targets", [])):
            position = self._to_world_coord(_vec3(item.get("position", [0, 0, 0])))
            velocity = _vec3(item.get("velocity", [0, 0, 0]))
            waypoints = [self._to_world_coord(_vec3(p)) for p in item.get("waypoints", [])]
            speed_range_raw = item.get("speed_range", [0.0, max(_vec_norm(velocity), 5.0)])
            speed_range = (float(speed_range_raw[0]), float(speed_range_raw[1]))
            targets.append(
                TargetState(
                    target_id=int(item.get("target_id", idx + 1)),
                    class_by_sensor={
                        SensorType.parse(k): int(v)
                        for k, v in item.get("class_by_sensor", {}).items()
                    },
                    position=position,
                    velocity=velocity,
                    motion_mode=str(item.get("motion_mode", "random")).lower(),
                    speed_range=speed_range,
                    waypoints=waypoints,
                    random_heading_interval=float(item.get("random_heading_interval", 3.0)),
                )
            )
        return targets

    def _split_waypoints(self, route: Sequence[Sequence[float]], n_parts: int) -> List[List[Vector3]]:
        """将共享航线按段切分给多架无人机。"""
        if n_parts <= 0:
            return []
        points = [self._to_world_coord(_vec3(p, default_z=50.0)) for p in route]
        if len(points) <= n_parts:
            return [[points[min(i, len(points) - 1)]] for i in range(n_parts)]
        chunk_size = max(2, math.ceil((len(points) - 1) / n_parts) + 1)
        segments: List[List[Vector3]] = []
        start = 0
        for _ in range(n_parts):
            end = min(len(points), start + chunk_size)
            segment = points[start:end]
            if len(segment) == 1 and start > 0:
                segment = [points[start - 1], points[start]]
            segments.append(segment)
            if end == len(points):
                start = len(points) - 1
            else:
                start = end - 1
        return segments

    def _advance_targets(self, dt: float) -> None:
        """推进所有任务目标的运动状态。"""
        for target in self.targets:
            if target.motion_mode == "static":
                target.velocity = (0.0, 0.0, 0.0)
                continue
            if target.motion_mode == "patrol" and len(target.waypoints) >= 2:
                self._move_along_waypoints(target, dt, is_uav=False)
                continue
            self._advance_random_target(target, dt)

    def _advance_random_target(self, target: TargetState, dt: float) -> None:
        """推进随机游走目标。"""
        if self.time - target.last_heading_change >= target.random_heading_interval:
            target.last_heading_change = self.time
            heading = self.rng.uniform(-math.pi, math.pi)
            speed = self.rng.uniform(target.speed_range[0], target.speed_range[1])
            vz = self.rng.uniform(-0.5, 0.5)
            target.velocity = (speed * math.cos(heading), speed * math.sin(heading), vz)

        target.position = self._clip_to_map(_vec_add(target.position, _vec_scale(target.velocity, dt)))

    def _advance_uavs(self, dt: float) -> None:
        """推进所有无人机沿预设航线飞行。"""
        for uav in self.uavs:
            if len(uav.waypoints) < 2:
                continue
            self._move_along_waypoints(uav, dt, is_uav=True)

    def _move_along_waypoints(self, entity: Any, dt: float, is_uav: bool) -> None:
        """按航路点推进 UAV/目标位置与速度。"""
        remaining = dt
        while remaining > 1e-6 and len(entity.waypoints) >= 2:
            current = entity.position
            target_wp = entity.waypoints[entity.waypoint_index]
            direction = _vec_sub(target_wp, current)
            dist = _vec_norm(direction)
            if dist < 1e-6:
                if is_uav:
                    self._advance_uav_waypoint(entity)
                else:
                    self._advance_target_waypoint(entity)
                continue
            speed = entity.speed if is_uav else self.rng.uniform(entity.speed_range[0], entity.speed_range[1])
            step_dist = speed * remaining
            step_vec = _vec_scale(_vec_unit(direction), min(step_dist, dist))
            entity.position = self._clip_to_map(_vec_add(current, step_vec))
            entity.velocity = _vec_scale(_vec_unit(direction), speed)
            if is_uav:
                entity.yaw_deg = _yaw_deg_from_vec(step_vec)
            if step_dist >= dist:
                traveled_time = dist / max(speed, 1e-6)
                remaining -= traveled_time
                if is_uav:
                    self._advance_uav_waypoint(entity)
                else:
                    self._advance_target_waypoint(entity)
            else:
                remaining = 0.0

    def _advance_uav_waypoint(self, uav: UAVState) -> None:
        """更新无人机航路点索引，采用往返巡航。"""
        if uav.patrol_forward:
            uav.waypoint_index += 1
            if uav.waypoint_index >= len(uav.waypoints):
                uav.waypoint_index = max(0, len(uav.waypoints) - 2)
                uav.patrol_forward = False
        else:
            uav.waypoint_index -= 1
            if uav.waypoint_index < 0:
                uav.waypoint_index = 1 if len(uav.waypoints) > 1 else 0
                uav.patrol_forward = True

    def _advance_target_waypoint(self, target: TargetState) -> None:
        """更新巡逻目标下一航路点索引。"""
        if target.patrol_forward:
            target.waypoint_index += 1
            if target.waypoint_index >= len(target.waypoints):
                target.waypoint_index = max(0, len(target.waypoints) - 2)
                target.patrol_forward = False
        else:
            target.waypoint_index -= 1
            if target.waypoint_index < 0:
                target.waypoint_index = 1 if len(target.waypoints) > 1 else 0
                target.patrol_forward = True

    def _generate_perception_frames(self) -> List[PerceptionFrame]:
        """基于可见性、丢帧和噪声生成当前步感知结果。"""
        if not self.targets or not self.uavs:
            return []

        frames: List[PerceptionFrame] = []
        for uav in self.uavs:
            if not self._sensor_available(uav.sensor.sensor_type):
                continue
            detections: List[Detection] = []
            for target in self.targets:
                if not self._target_in_sensor_range(uav, target):
                    self._break_track(uav, target)
                    continue
                if self.rng.random() < uav.sensor.dropout_prob:
                    self._break_track(uav, target)
                    continue
                detection = self._build_detection(uav, target)
                detections.append(detection)
            if detections:
                frames.append(
                    PerceptionFrame(
                        uav_id=uav.uav_id,
                        sensor_type=uav.sensor.sensor_type,
                        sensor_position=uav.position,
                        sensor_orientation=(0.0, 0.0, uav.yaw_deg),
                        timestamp=self.time,
                        detections=detections,
                        sensor_params=deepcopy(uav.sensor.params),
                    )
                )
        return frames

    def _sensor_available(self, sensor_type: int) -> bool:
        """根据天气与昼夜判断传感器是否可用。"""
        if sensor_type == SensorType.RGB.value:
            return self.global_weather not in {"rain", "snow"} and self.global_lighting != "night"
        if sensor_type == SensorType.IF.value:
            return self.global_lighting != "day"
        return True

    def _target_in_sensor_range(self, uav: UAVState, target: TargetState) -> bool:
        """判断目标是否落在当前传感器感知范围。"""
        rel = _vec_sub(target.position, uav.position)
        sensor_type = uav.sensor.sensor_type
        params = uav.sensor.params
        max_range = float(params.get("max_range", 300.0))
        planar_dist = math.hypot(rel[0], rel[1])
        if sensor_type in (SensorType.RADAR.value, SensorType.ELEC.value):
            return planar_dist <= max_range

        forward_range = float(params.get("forward_range", max_range))
        width = float(params.get("width", max_range))
        forward, lateral = self._project_body_frame(rel, uav.yaw_deg)
        return 0.0 <= forward <= forward_range and abs(lateral) <= width / 2.0

    def _build_detection(self, uav: UAVState, target: TargetState) -> Detection:
        """构建单目标观测，包含噪声、track_id 与 bearing 信息。"""
        sensor_type = uav.sensor.sensor_type
        track_id = self._get_track_id(uav, target)
        noisy_position = self._add_gaussian_noise(target.position, uav.sensor.position_noise_std)
        noisy_velocity = self._add_gaussian_noise(target.velocity, uav.sensor.velocity_noise_std)
        class_id = target.class_by_sensor.get(sensor_type, next(iter(target.class_by_sensor.values()), 0))

        bearing_vector = None
        if sensor_type == SensorType.ELEC.value:
            bearing_vector = _vec_unit(_vec_sub(target.position, uav.position))
            rough_range = float(uav.sensor.params.get("bearing_range_estimate", 150.0))
            noisy_position = _vec_add(uav.position, _vec_scale(bearing_vector, rough_range))
            noisy_velocity = (0.0, 0.0, 0.0)

        bbox = None
        if sensor_type in (SensorType.RGB.value, SensorType.IF.value):
            bbox = self._estimate_bbox(uav, target)

        return Detection(
            class_id=class_id,
            position=noisy_position,
            velocity=noisy_velocity,
            track_id=track_id,
            bearing_vector=bearing_vector,
            bbox=bbox,
            confidence=1.0 - uav.sensor.dropout_prob,
        )

    def _to_world_coord(self, point: Vector3) -> Vector3:
        """将0~1归一化坐标映射到当前地图尺度。"""
        x, y, z = point
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x = x * float(self.map_size[0])
            y = y * float(self.map_size[1])
            if 0.0 <= z <= 1.0:
                z = z * self.map_altitude
        return (x, y, z)

    def _get_track_id(self, uav: UAVState, target: TargetState) -> int:
        """分配或复用同 UAV/传感器下的 track_id。"""
        sensor_type = uav.sensor.sensor_type
        if sensor_type == SensorType.RADAR.value:
            return -1
        key = (uav.uav_id, sensor_type, target.target_id)
        if key not in self._target_track_ids:
            counter_key = (uav.uav_id, sensor_type)
            next_id = self._next_track_ids.get(counter_key, 0) + 1
            self._next_track_ids[counter_key] = next_id
            self._target_track_ids[key] = next_id
        return self._target_track_ids[key]

    def _break_track(self, uav: UAVState, target: TargetState) -> None:
        """在目标丢失时打断 track_id 连续性。"""
        key = (uav.uav_id, uav.sensor.sensor_type, target.target_id)
        if key in self._target_track_ids:
            del self._target_track_ids[key]

    def _estimate_bbox(self, uav: UAVState, target: TargetState) -> List[float]:
        """生成用于可视化/融合占位的简化 2D 检测框。"""
        rel = _vec_sub(target.position, uav.position)
        forward, lateral = self._project_body_frame(rel, uav.yaw_deg)
        img_w = float(uav.sensor.params.get("image_width", 1280.0))
        img_h = float(uav.sensor.params.get("image_height", 720.0))
        width = float(uav.sensor.params.get("width", 200.0))
        cx = img_w * (0.5 + lateral / max(width, 1.0))
        cy = img_h * (0.5 - rel[2] / max(float(uav.sensor.params.get("forward_range", 300.0)), 1.0))
        box_w = max(10.0, 1000.0 / max(forward + 1.0, 10.0))
        box_h = box_w
        return [cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0]

    def _project_body_frame(self, rel: Vector3, yaw_deg: float) -> Tuple[float, float]:
        """将世界坐标相对向量投影到机体前向-侧向坐标。"""
        yaw = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = cos_yaw * rel[0] + sin_yaw * rel[1]
        lateral = -sin_yaw * rel[0] + cos_yaw * rel[1]
        return forward, lateral

    def _add_gaussian_noise(self, vec: Vector3, sigma: float) -> Vector3:
        """对向量叠加独立高斯噪声。"""
        if sigma <= 0.0:
            return vec
        return (
            vec[0] + self.rng.gauss(0.0, sigma),
            vec[1] + self.rng.gauss(0.0, sigma),
            vec[2] + self.rng.gauss(0.0, sigma),
        )

    def _clip_to_map(self, position: Vector3) -> Vector3:
        """将位置裁剪到地图边界内。"""
        max_x, max_y = float(self.map_size[0]), float(self.map_size[1])
        return (
            min(max(position[0], 0.0), max_x),
            min(max(position[1], 0.0), max_y),
            position[2],
        )

    def _refresh_render_state(self) -> None:
        """刷新 render_state，供外部界面直接绘制。"""
        self.render_state = {
            "time": self.time,
            "weather": self.global_weather,
            "lighting": self.global_lighting,
            "uavs": [
                {
                    "uav_id": uav.uav_id,
                    "sensor_type": uav.sensor.sensor_type,
                    "position": list(uav.position),
                    "velocity": list(uav.velocity),
                    "yaw_deg": uav.yaw_deg,
                    "waypoints": [list(p) for p in uav.waypoints],
                    "sensor_params": deepcopy(uav.sensor.params),
                }
                for uav in self.uavs
            ],
            "targets_truth": [
                {
                    "target_id": target.target_id,
                    "position": list(target.position),
                    "velocity": list(target.velocity),
                    "motion_mode": target.motion_mode,
                    "class_by_sensor": {str(k): v for k, v in target.class_by_sensor.items()},
                    "trajectory_hint": [list(p) for p in target.waypoints],
                }
                for target in self.targets
            ],
            "global_objects": deepcopy(self.global_objects_snapshot),
            "observations_valid": [frame.to_dict() for frame in self.perception_queue],
            "match_edges": deepcopy(self.last_merge_feedback.get("match_edges", [])),
            "pending_frames": len(self.perception_queue),
        }
