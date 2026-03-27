from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.data_utils import (
    Detection,
    MergeConfig,
    MergeOperation,
    MergeResult,
    ObjectItem,
    PerceptionFrame,
    SensorType,
    TrackHistory,
)

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


class PerceptionMerger:
    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        class_correlation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化融合器配置、类别关联和局部轨迹缓存。"""
        self.config = config or MergeConfig()
        self.class_correlation: Dict[str, Any] = class_correlation or {}
        self.merge_mode = self.config.default_merge_mode
        self.track_memory: Dict[Tuple[int, int, int], TrackHistory] = {}
        self._next_temp_id = -1

    def set_merge_mode(self, merge_mode: str) -> None:
        """设置默认融合模式（可被单次请求覆盖）。"""
        self.merge_mode = merge_mode

    def load_class_correlation(self, class_correlation: Dict[str, Any]) -> None:
        """加载或替换跨模态类别关联表。"""
        self.class_correlation = class_correlation or {}

    def merge_frame(
        self,
        perception_frame: PerceptionFrame,
        global_obj_items: Sequence[ObjectItem],
        merge_mode: Optional[str] = None,
    ) -> MergeResult:
        """执行单帧融合流程并输出 update/create 操作。"""
        mode = merge_mode or self.merge_mode

        # 全局目标记录时空对齐
        aligned_items = self._align_global_objects(global_obj_items, perception_frame.timestamp)

        # 更新局部目标轨迹
        self._update_track_memory(perception_frame)

        cost_matrix, pair_info = self._build_cost_matrix(perception_frame, aligned_items, mode)
        assignments, unmatched_det, _unmatched_obj = self._solve_assignment(cost_matrix)

        result = self._fuse_matches(
            perception_frame=perception_frame,
            aligned_items=aligned_items,
            assignments=assignments,
            unmatched_det=unmatched_det,
            pair_info=pair_info,
            mode=mode,
        )
        result.debug_info.update(
            {
                "mode": mode,
                "num_detections": len(perception_frame.detections),
                "num_global_objects": len(aligned_items),
                "num_assignments": len(assignments),
            }
        )
        return result

    def merge_batch(
        self,
        perception_frames: Sequence[PerceptionFrame],
        global_obj_items: Sequence[ObjectItem],
        merge_mode: Optional[str] = None,
    ) -> MergeResult:
        """按时间顺序融合多帧数据并聚合输出。"""
        mode = merge_mode or self.merge_mode
        shadow_items = [deepcopy(item) for item in global_obj_items]
        aggregate = MergeResult()

        for frame in perception_frames:
            frame_result = self.merge_frame(frame, shadow_items, merge_mode=mode)
            aggregate.update_ops.extend(frame_result.update_ops)
            aggregate.create_ops.extend(frame_result.create_ops)
            aggregate.alerts.extend(frame_result.alerts)
            self._apply_ops_to_shadow(shadow_items, frame_result)

        aggregate.debug_info.update(
            {
                "mode": mode,
                "num_frames": len(perception_frames),
                "num_update_ops": len(aggregate.update_ops),
                "num_create_ops": len(aggregate.create_ops),
            }
        )
        return aggregate

    def _align_global_objects(
        self,
        global_obj_items: Sequence[ObjectItem],
        target_timestamp: float,
    ) -> List[ObjectItem]:
        """根据匀速模型将全局目标对齐到指定时间戳。"""
        aligned: List[ObjectItem] = []
        for item in global_obj_items:
            dt = target_timestamp - item.timestamp
            new_pos = (
                item.position[0] + item.velocity[0] * dt,
                item.position[1] + item.velocity[1] * dt,
                item.position[2] + item.velocity[2] * dt,
            )
            cloned = deepcopy(item)
            cloned.position = new_pos
            cloned.timestamp = target_timestamp
            aligned.append(cloned)
        return aligned

    def _update_track_memory(self, frame: PerceptionFrame) -> None:
        """更新 UAV-传感器-track 维度的局部轨迹缓存。"""
        max_len = self.config.track_window_size
        for det in frame.detections:
            if det.track_id < 0:
                continue
            key = (frame.uav_id, frame.sensor_type, det.track_id)
            history = self.track_memory.setdefault(key, TrackHistory())
            history.append(det.position, frame.timestamp, max_len=max_len)

    def _build_cost_matrix(
        self,
        frame: PerceptionFrame,
        aligned_items: Sequence[ObjectItem],
        mode: str,
    ) -> Tuple[List[List[float]], Dict[Tuple[int, int], Dict[str, float]]]:
        """构建匹配代价矩阵并返回每对候选的分项代价。"""
        n_det = len(frame.detections)
        n_obj = len(aligned_items)
        inf = float("inf")
        matrix: List[List[float]] = [[inf for _ in range(n_obj)] for _ in range(n_det)]
        pair_info: Dict[Tuple[int, int], Dict[str, float]] = {}

        for i, det in enumerate(frame.detections):
            for j, obj in enumerate(aligned_items):
                if frame.sensor_type == SensorType.ELEC.value and not obj.spatial_valid:
                    continue
                if not self._class_compatible(frame.sensor_type, det.class_id, obj):
                    continue
                if frame.sensor_type == SensorType.ELEC.value:
                    if not self._elec_angle_gate(frame, det, obj):
                        continue
                    class_cost = self._class_cost(frame.sensor_type, det.class_id, obj)
                    elec_angle_cost = self._elec_angle_cost(frame, det, obj)
                    total = (
                        self.config.cost_weights["class"] * class_cost
                        + self.config.cost_weights["elec_angle"] * elec_angle_cost
                    )
                    matrix[i][j] = total
                    pair_info[(i, j)] = {
                        "total_cost": total,
                        "class_cost": class_cost,
                        "distance_cost": 0.0,
                        "velocity_cost": 0.0,
                        "track_cost": 0.0,
                        "visibility_cost": 0.0,
                        "elec_angle_cost": elec_angle_cost,
                    }
                    continue

                if not self._distance_gate(det.position, obj.position):
                    continue
                if not self._velocity_gate(det.velocity, obj.velocity):
                    continue

                class_cost = self._class_cost(frame.sensor_type, det.class_id, obj)
                distance_cost = self._distance_cost(det.position, obj.position)
                velocity_cost = self._velocity_cost(det.velocity, obj.velocity)
                track_cost = self._track_cost(frame, det, obj)
                visibility_cost = self._visibility_cost(frame, det, obj, mode)

                total = (
                    self.config.cost_weights["class"] * class_cost
                    + self.config.cost_weights["distance"] * distance_cost
                    + self.config.cost_weights["velocity"] * velocity_cost
                    + self.config.cost_weights["track"] * track_cost
                    + self.config.cost_weights["visibility"] * visibility_cost
                )
                matrix[i][j] = total
                pair_info[(i, j)] = {
                    "total_cost": total,
                    "class_cost": class_cost,
                    "distance_cost": distance_cost,
                    "velocity_cost": velocity_cost,
                    "track_cost": track_cost,
                    "visibility_cost": visibility_cost,
                }

        return matrix, pair_info

    def _solve_assignment(
        self,
        cost_matrix: List[List[float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """求解检测与全局目标的一对一匹配关系。"""
        n_det = len(cost_matrix)
        n_obj = len(cost_matrix[0]) if n_det else 0
        assignments: List[Tuple[int, int]] = []

        if n_det == 0:
            return assignments, [], list(range(n_obj))
        if n_obj == 0:
            return assignments, list(range(n_det)), []

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if math.isfinite(cost_matrix[r][c]):
                    assignments.append((r, c))
        else:
            # Fallback when scipy is unavailable: greedy minimum-cost matching.
            used_det = set()
            used_obj = set()
            candidates: List[Tuple[float, int, int]] = []
            for i in range(n_det):
                for j in range(n_obj):
                    cost = cost_matrix[i][j]
                    if math.isfinite(cost):
                        candidates.append((cost, i, j))
            candidates.sort(key=lambda x: x[0])
            for _cost, i, j in candidates:
                if i in used_det or j in used_obj:
                    continue
                assignments.append((i, j))
                used_det.add(i)
                used_obj.add(j)

        matched_det = {i for i, _ in assignments}
        matched_obj = {j for _, j in assignments}
        unmatched_det = [i for i in range(n_det) if i not in matched_det]
        unmatched_obj = [j for j in range(n_obj) if j not in matched_obj]
        return assignments, unmatched_det, unmatched_obj

    def _fuse_matches(
        self,
        perception_frame: PerceptionFrame,
        aligned_items: Sequence[ObjectItem],
        assignments: Sequence[Tuple[int, int]],
        unmatched_det: Sequence[int],
        pair_info: Dict[Tuple[int, int], Dict[str, float]],
        mode: str,
    ) -> MergeResult:
        """根据匹配结果生成更新操作与新增操作。"""
        result = MergeResult()
        for det_idx, obj_idx in assignments:
            det = perception_frame.detections[det_idx]
            obj = aligned_items[obj_idx]
            is_elec_obs = perception_frame.sensor_type == SensorType.ELEC.value
            fused_pos, fused_vel = self._fuse_state(
                frame=perception_frame,
                det=det,
                obj=obj,
                mode=mode,
                is_elec_obs=is_elec_obs,
            )
            observation_payload = self._build_observation_payload(perception_frame, det)
            class_payload = self._build_class_payload(perception_frame.sensor_type, det.class_id, obj)
            payload = {
                "timestamp": perception_frame.timestamp,
                "fused_position": list(fused_pos),
                "fused_velocity": list(fused_vel),
                "observation": observation_payload,
                "class_update": class_payload,
                "spatial_valid": obj.spatial_valid if is_elec_obs else True,
            }
            score = pair_info.get((det_idx, obj_idx), {}).get("total_cost")
            result.update_ops.append(
                MergeOperation(
                    operation="update",
                    target_id=obj.global_id,
                    payload=payload,
                    score=score,
                )
            )

        for det_idx in unmatched_det:
            det = perception_frame.detections[det_idx]
            if perception_frame.sensor_type == SensorType.ELEC.value:
                # ELEC 为方位主导观测，不允许直接新建目标。
                continue
            temp_id = self._allocate_temp_id()
            payload = {
                "timestamp": perception_frame.timestamp,
                "position": list(det.position),
                "velocity": list(det.velocity),
                "observation": self._build_observation_payload(perception_frame, det),
                "class_by_sensor": {str(perception_frame.sensor_type): det.class_id},
                "class_votes": {str(perception_frame.sensor_type): {str(det.class_id): 1}},
                "spatial_valid": True,
            }
            result.create_ops.append(
                MergeOperation(operation="create", target_id=temp_id, payload=payload)
            )

        if perception_frame.sensor_type == SensorType.ELEC.value and unmatched_det:
            result.alerts.append(
                {
                    "level": "info",
                    "code": "ELEC_UNMATCHED_SKIPPED",
                    "message": "Unmatched ELEC detections are not used to create new objects.",
                    "count": len(unmatched_det),
                }
            )
        if not assignments and perception_frame.detections:
            result.alerts.append(
                {
                    "level": "warning",
                    "code": "NO_MATCH",
                    "message": "No valid match found for this frame.",
                }
            )
        return result

    def _apply_ops_to_shadow(self, shadow_items: List[ObjectItem], result: MergeResult) -> None:
        """在批量模式中将操作结果应用到临时全局态势副本。"""
        id_map = {item.global_id: item for item in shadow_items}
        for op in result.update_ops:
            item = id_map.get(op.target_id)
            if item is None:
                continue
            fused_pos = op.payload["fused_position"]
            fused_vel = op.payload["fused_velocity"]
            item.position = (float(fused_pos[0]), float(fused_pos[1]), float(fused_pos[2]))
            item.velocity = (float(fused_vel[0]), float(fused_vel[1]), float(fused_vel[2]))
            item.timestamp = float(op.payload["timestamp"])
            item.trajectory.append(item.position)
            item.observations.append(op.payload["observation"])
            item.spatial_valid = bool(op.payload.get("spatial_valid", item.spatial_valid))
            class_update = op.payload.get("class_update", {})
            self._apply_class_update(item, class_update)

        for op in result.create_ops:
            payload = op.payload
            item = ObjectItem(
                global_id=int(op.target_id),
                position=(
                    float(payload["position"][0]),
                    float(payload["position"][1]),
                    float(payload["position"][2]),
                ),
                velocity=(
                    float(payload["velocity"][0]),
                    float(payload["velocity"][1]),
                    float(payload["velocity"][2]),
                ),
                timestamp=float(payload["timestamp"]),
                class_by_sensor={int(k): int(v) for k, v in payload["class_by_sensor"].items()},
                class_votes={
                    int(sk): {int(ck): int(cv) for ck, cv in sv.items()}
                    for sk, sv in payload["class_votes"].items()
                },
                trajectory=[tuple(payload["position"])],  # type: ignore[arg-type]
                observations=[payload["observation"]],
                spatial_valid=bool(payload.get("spatial_valid", True)),
            )
            shadow_items.append(item)

    def _class_compatible(self, sensor_type: int, class_id: int, obj: ObjectItem) -> bool:
        """判断 detection 类别与目标类别是否兼容。"""
        if not obj.class_by_sensor:
            return True
        direct = obj.class_by_sensor.get(sensor_type)
        if direct is not None:
            return direct == class_id

        corr_sensor = self.class_correlation.get(str(sensor_type), {})
        corr_classes = corr_sensor.get(str(class_id), {})
        for obj_sensor, obj_class in obj.class_by_sensor.items():
            mapped = corr_classes.get(str(obj_sensor), [])
            if obj_class in mapped:
                return True
        return False

    def _distance_gate(self, det_pos: Tuple[float, float, float], obj_pos: Tuple[float, float, float]) -> bool:
        """距离门限过滤，超阈值的候选直接剔除。"""
        dx = det_pos[0] - obj_pos[0]
        dy = det_pos[1] - obj_pos[1]
        dz = abs(det_pos[2] - obj_pos[2])
        planar = math.hypot(dx, dy)
        return (
            planar <= self.config.planar_distance_threshold
            and dz <= self.config.height_distance_threshold
        )

    def _velocity_gate(self, det_vel: Tuple[float, float, float], obj_vel: Tuple[float, float, float]) -> bool:
        """速度方向门限过滤。"""
        det_norm = self._norm(det_vel)
        obj_norm = self._norm(obj_vel)
        if det_norm < 1e-6 or obj_norm < 1e-6:
            return True
        cos_val = max(-1.0, min(1.0, self._dot(det_vel, obj_vel) / (det_norm * obj_norm)))
        angle = math.degrees(math.acos(cos_val))
        return angle <= self.config.velocity_angle_threshold_deg

    def _visibility_gate(
        self,
        frame: PerceptionFrame,
        det: Detection,
        obj: ObjectItem,
        mode: str,
    ) -> bool:
        """可视性门限过滤，可按模式开关。"""
        if not self.config.enable_visibility and "visibility" not in mode:
            return True
        return self._visibility_in_fov(frame, obj.position)

    def _class_cost(self, sensor_type: int, class_id: int, obj: ObjectItem) -> float:
        """类别一致性代价。"""
        current = obj.class_by_sensor.get(sensor_type)
        if current is None:
            return 0.2
        return 0.0 if current == class_id else 1.0

    def _distance_cost(self, det_pos: Tuple[float, float, float], obj_pos: Tuple[float, float, float]) -> float:
        """空间距离代价（归一化）。"""
        dx = det_pos[0] - obj_pos[0]
        dy = det_pos[1] - obj_pos[1]
        dz = det_pos[2] - obj_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        max_dist = max(self.config.planar_distance_threshold, 1.0)
        return min(1.0, dist / max_dist)

    def _velocity_cost(
        self,
        det_vel: Tuple[float, float, float],
        obj_vel: Tuple[float, float, float],
    ) -> float:
        """速度方向代价。"""
        det_norm = self._norm(det_vel)
        obj_norm = self._norm(obj_vel)
        if det_norm < 1e-6 and obj_norm < 1e-6:
            return 0.0
        if det_norm < 1e-6 or obj_norm < 1e-6:
            return 0.4
        cos_val = max(-1.0, min(1.0, self._dot(det_vel, obj_vel) / (det_norm * obj_norm)))
        return 0.5 * (1 - cos_val)

    def _track_cost(self, frame: PerceptionFrame, det: Detection, obj: ObjectItem) -> float:
        """局部轨迹与全局轨迹方向一致性代价。"""
        if det.track_id < 0:
            return 0.5
        key = (frame.uav_id, frame.sensor_type, det.track_id)
        history = self.track_memory.get(key)
        if history is None or len(history.points) < 2 or len(obj.trajectory) < 2:
            return 0.5
        local_vec = self._vec_sub(history.points[-1], history.points[-2])
        global_vec = self._vec_sub(obj.trajectory[-1], obj.trajectory[-2])
        local_norm = self._norm(local_vec)
        global_norm = self._norm(global_vec)
        if local_norm < 1e-6 or global_norm < 1e-6:
            return 0.5
        cos_val = max(-1.0, min(1.0, self._dot(local_vec, global_vec) / (local_norm * global_norm)))
        return 0.5 * (1 - cos_val)

    def _visibility_cost(
        self,
        frame: PerceptionFrame,
        det: Detection,
        obj: ObjectItem,
        mode: str,
    ) -> float:
        """可视性代价。"""
        if not self.config.enable_visibility and "visibility" not in mode:
            return 0.0
        return 0.0 if self._visibility_in_fov(frame, obj.position) else 1.0

    def _visibility_in_fov(self, frame: PerceptionFrame, target_pos: Tuple[float, float, float]) -> bool:
        """判断目标是否位于当前传感器可视范围内。"""
        params = frame.sensor_params or {}
        hfov = float(params.get("hfov_deg", 360.0))
        max_range = float(params.get("max_range", 1e9))
        rel = (
            target_pos[0] - frame.sensor_position[0],
            target_pos[1] - frame.sensor_position[1],
            target_pos[2] - frame.sensor_position[2],
        )
        planar_dist = math.hypot(rel[0], rel[1])
        if planar_dist > max_range:
            return False
        yaw = math.degrees(math.atan2(rel[1], rel[0]))
        sensor_yaw = frame.sensor_orientation[2]
        delta = abs(((yaw - sensor_yaw + 180) % 360) - 180)
        return delta <= hfov / 2.0

    def _elec_angle_gate(self, frame: PerceptionFrame, det: Detection, obj: ObjectItem) -> bool:
        """ELEC 匹配门限：目标方向与观测射线夹角需小于阈值。"""
        angle = self._elec_angle_deg(frame, det, obj)
        return angle <= self.config.elec_ray_angle_threshold_deg

    def _elec_angle_cost(self, frame: PerceptionFrame, det: Detection, obj: ObjectItem) -> float:
        """ELEC 射线夹角代价（归一化）。"""
        angle = self._elec_angle_deg(frame, det, obj)
        denom = max(self.config.elec_ray_angle_threshold_deg, 1e-6)
        return min(1.0, angle / denom)

    def _elec_angle_deg(self, frame: PerceptionFrame, det: Detection, obj: ObjectItem) -> float:
        """计算观测射线与传感器到目标连线的夹角。"""
        if det.bearing_vector is not None:
            ray = det.bearing_vector
        else:
            ray = (
                det.position[0] - frame.sensor_position[0],
                det.position[1] - frame.sensor_position[1],
                det.position[2] - frame.sensor_position[2],
            )
        to_obj = (
            obj.position[0] - frame.sensor_position[0],
            obj.position[1] - frame.sensor_position[1],
            obj.position[2] - frame.sensor_position[2],
        )
        ray_norm = self._norm(ray)
        obj_norm = self._norm(to_obj)
        if ray_norm < 1e-6 or obj_norm < 1e-6:
            return 180.0
        cos_val = max(-1.0, min(1.0, self._dot(ray, to_obj) / (ray_norm * obj_norm)))
        return math.degrees(math.acos(cos_val))

    def _fuse_state(
        self,
        frame: PerceptionFrame,
        det: Detection,
        obj: ObjectItem,
        mode: str,
        is_elec_obs: bool = False,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """按当前融合模式融合位置与速度状态。"""
        if mode.startswith("elec") or is_elec_obs:
            # ELEC不提供精确位置，仅在已有空间信息基础上做小幅射线校正。
            if not obj.spatial_valid:
                return obj.position, obj.velocity
            if det.bearing_vector is not None:
                ray = det.bearing_vector
            else:
                ray = (
                    det.position[0] - frame.sensor_position[0],
                    det.position[1] - frame.sensor_position[1],
                    det.position[2] - frame.sensor_position[2],
                )
            ray_norm = self._norm(ray)
            if ray_norm < 1e-6:
                return obj.position, obj.velocity
            ray_u = (ray[0] / ray_norm, ray[1] / ray_norm, ray[2] / ray_norm)
            sensor = frame.sensor_position
            to_obj = (
                obj.position[0] - sensor[0],
                obj.position[1] - sensor[1],
                obj.position[2] - sensor[2],
            )
            proj_len = max(0.0, self._dot(to_obj, ray_u))
            proj_point = (
                sensor[0] + ray_u[0] * proj_len,
                sensor[1] + ray_u[1] * proj_len,
                sensor[2] + ray_u[2] * proj_len,
            )
            alpha = max(0.0, min(1.0, self.config.elec_position_correction_alpha))
            corrected_pos = (
                (1.0 - alpha) * obj.position[0] + alpha * proj_point[0],
                (1.0 - alpha) * obj.position[1] + alpha * proj_point[1],
                (1.0 - alpha) * obj.position[2] + alpha * proj_point[2],
            )
            return corrected_pos, obj.velocity
        if mode == "kf_reserved":
            # Reserved mode: keep same interface, currently uses weighted fusion.
            pass
        alpha = self.config.fusion_alpha
        fused_pos = (
            alpha * det.position[0] + (1.0 - alpha) * obj.position[0],
            alpha * det.position[1] + (1.0 - alpha) * obj.position[1],
            alpha * det.position[2] + (1.0 - alpha) * obj.position[2],
        )
        fused_vel = (
            alpha * det.velocity[0] + (1.0 - alpha) * obj.velocity[0],
            alpha * det.velocity[1] + (1.0 - alpha) * obj.velocity[1],
            alpha * det.velocity[2] + (1.0 - alpha) * obj.velocity[2],
        )
        return fused_pos, fused_vel

    def _build_observation_payload(
        self,
        frame: PerceptionFrame,
        det: Detection,
    ) -> Dict[str, Any]:
        """构造写回 GlobalInfo 的观测载荷。"""
        return {
            "uav_id": frame.uav_id,
            "sensor_type": frame.sensor_type,
            "sensor_position": list(frame.sensor_position),
            "sensor_orientation": list(frame.sensor_orientation),
            "timestamp": frame.timestamp,
            "class_id": det.class_id,
            "position": list(det.position),
            "velocity": list(det.velocity),
            "track_id": det.track_id,
            "bearing_vector": list(det.bearing_vector) if det.bearing_vector is not None else None,
            "bbox": det.bbox,
        }

    def _build_class_payload(
        self,
        sensor_type: int,
        class_id: int,
        obj: ObjectItem,
    ) -> Dict[str, Any]:
        """更新类别投票并输出类别写回载荷。"""
        votes = deepcopy(obj.class_votes)
        sensor_votes = votes.setdefault(sensor_type, {})
        sensor_votes[class_id] = sensor_votes.get(class_id, 0) + 1
        winning = max(sensor_votes.items(), key=lambda x: x[1])[0]
        return {
            "sensor_type": sensor_type,
            "class_id": class_id,
            "class_by_sensor": {str(k): v for k, v in obj.class_by_sensor.items()} | {str(sensor_type): winning},
            "class_votes": {
                str(sk): {str(ck): cv for ck, cv in sv.items()} for sk, sv in votes.items()
            },
        }

    def _apply_class_update(self, obj: ObjectItem, class_update: Dict[str, Any]) -> None:
        """将类别更新载荷应用到目标对象。"""
        class_by_sensor = class_update.get("class_by_sensor", {})
        class_votes = class_update.get("class_votes", {})
        if class_by_sensor:
            obj.class_by_sensor = {int(k): int(v) for k, v in class_by_sensor.items()}
        if class_votes:
            obj.class_votes = {
                int(sk): {int(ck): int(cv) for ck, cv in sv.items()}
                for sk, sv in class_votes.items()
            }

    def _allocate_temp_id(self) -> int:
        """分配新增目标使用的临时负 ID。"""
        temp_id = self._next_temp_id
        self._next_temp_id -= 1
        return temp_id

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """三维向量点积。"""
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _norm(a: Tuple[float, float, float]) -> float:
        """三维向量模长。"""
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    @staticmethod
    def _vec_sub(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """三维向量减法。"""
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
