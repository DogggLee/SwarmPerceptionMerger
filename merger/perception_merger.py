from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
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
from utils.DTW import dtw_distance

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None


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
        self.logger = logging.getLogger("PerceptionMerger")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        self._file_logger = self._build_file_logger()
        self._last_frame: Optional[PerceptionFrame] = None
        self._last_global_items: Optional[List[ObjectItem]] = None
        self._last_result: Optional[MergeResult] = None

    def set_merge_mode(self, merge_mode: str) -> None:
        """设置默认融合模式（可被单次请求覆盖）。"""
        self.merge_mode = merge_mode

    def set_logger(self, logger: logging.Logger) -> None:
        """设置融合器日志实例。

        Args:
            logger (logging.Logger): 外部传入的 logger，用于输出 debug/info/error 日志。
        Returns:
            None: 无返回值，直接替换内部 logger。
        """
        self.logger = logger

    def _build_file_logger(self) -> logging.Logger:
        """构建用于落盘的文件日志器。

        Args:
            None: 不需要输入参数。
        Returns:
            logging.Logger: 仅写入 `logs/时间戳.txt` 的 logger 实例。
        """
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        file_path = os.path.join(logs_dir, f"{timestamp_str}.txt")
        logger_name = f"PerceptionMergerFile.{id(self)}"
        file_logger = logging.getLogger(logger_name)
        for old_handler in list(file_logger.handlers):
            file_logger.removeHandler(old_handler)
            old_handler.close()
        file_logger.propagate = False
        file_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        file_logger.addHandler(handler)
        return file_logger

    def _log_debug(self, message: str, *args: Any) -> None:
        """同时向外部 logger 与文件 logger 记录 debug 日志。

        Args:
            message (str): 日志模板字符串。
            *args (Any): 模板参数。
        Returns:
            None: 无返回值，直接写日志。
        """
        self.logger.debug(message, *args)
        self._file_logger.debug(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        """同时向外部 logger 与文件 logger 记录 info 日志。

        Args:
            message (str): 日志模板字符串。
            *args (Any): 模板参数。
        Returns:
            None: 无返回值，直接写日志。
        """
        self.logger.info(message, *args)
        self._file_logger.info(message, *args)

    def _log_info_text(self, message: str, *args: Any) -> None:
        """记录普通 info 文本到终端与文件。

        Args:
            message (str): 日志模板字符串。
            *args (Any): 模板参数。
        Returns:
            None: 无返回值，直接写日志。
        """
        self.logger.info(message, *args)
        self._file_logger.info(message, *args)

    def _log_error(self, message: str, *args: Any) -> None:
        """同时向外部 logger 与文件 logger 记录 error 日志。

        Args:
            message (str): 日志模板字符串。
            *args (Any): 模板参数。
        Returns:
            None: 无返回值，直接写日志。
        """
        self.logger.error(message, *args)
        self._file_logger.error(message, *args)

    def _log_exception(self, message: str, *args: Any) -> None:
        """同时向外部 logger 与文件 logger 记录异常堆栈。

        Args:
            message (str): 日志模板字符串。
            *args (Any): 模板参数。
        Returns:
            None: 无返回值，直接写日志。
        """
        self.logger.exception(message, *args)
        self._file_logger.exception(message, *args)

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
        received_at = datetime.now(timezone.utc).isoformat()
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        try:
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
            self._last_frame = deepcopy(perception_frame)
            self._last_global_items = [deepcopy(item) for item in global_obj_items]
            self._last_result = deepcopy(result)
            
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log_info_text(
                "Received %d perception results and %d global objects, merged to %d updates and %d creates. Process %f ms",
                len(perception_frame.detections),
                len(global_obj_items),
                len(result.update_ops),
                len(result.create_ops),
                round(latency_ms, 3)
            )
            
            self._log_event(
                level="info_file",
                event="merger_summary",
                received_at=received_at,
                request_id=request_id,
                api="merge_frame",
                merge_mode=mode,
                uav_id=perception_frame.uav_id,
                sensor_type=perception_frame.sensor_type,
                frame_timestamp=perception_frame.timestamp,
                num_detections=len(perception_frame.detections),
                num_global_objects=len(global_obj_items),
                num_update_ops=len(result.update_ops),
                num_create_ops=len(result.create_ops),
                latency_ms=round(latency_ms, 3),
            )
            return result
        except Exception:
            self._log_exception("merge_frame failed at %s", received_at)
            raise

    def merge_batch(
        self,
        perception_frames: Sequence[PerceptionFrame],
        global_obj_items: Sequence[ObjectItem],
        merge_mode: Optional[str] = None,
    ) -> MergeResult:
        """按时间顺序融合多帧数据并聚合输出。"""
        received_at = datetime.now(timezone.utc).isoformat()
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        try:
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
            num_det = sum(len(frame.detections) for frame in perception_frames)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log_info_text(
                "Received %d perception results and %d global objects, merged to %d updates and %d creates. Process %f ms",
                num_det,
                len(global_obj_items),
                len(aggregate.update_ops),
                len(aggregate.create_ops),
                round(latency_ms, 3),
            )
            self._log_event(
                level="info_file",
                event="merger_summary",
                received_at=received_at,
                request_id=request_id,
                api="merge_batch",
                merge_mode=mode,
                num_frames=len(perception_frames),
                num_detections=num_det,
                num_global_objects=len(global_obj_items),
                num_update_ops=len(aggregate.update_ops),
                num_create_ops=len(aggregate.create_ops),
                latency_ms=round(latency_ms, 3),
            )
            return aggregate
        except Exception:
            self._log_exception("merge_batch failed at %s", received_at)
            raise

    def render(
        self,
        perception_frame: Optional[PerceptionFrame] = None,
        global_obj_items: Optional[Sequence[ObjectItem]] = None,
        merge_result: Optional[MergeResult] = None,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Dict[str, Any]:
        """绘制一次融合过程的临时可视化。

        Args:
            perception_frame (Optional[PerceptionFrame]): 本次融合输入感知帧；为 None 时使用最近一次 merge_frame 输入。
            global_obj_items (Optional[Sequence[ObjectItem]]): 融合时的全局目标记录；为 None 时使用最近缓存。
            merge_result (Optional[MergeResult]): 融合结果；为 None 时使用最近缓存。
            output_path (Optional[str]): 可选图片输出路径（例如 png）。
            show (bool): 是否弹出窗口显示图像。
        Returns:
            Dict[str, Any]: 渲染元数据，包括状态、输出路径、统计信息。
        """
        frame = perception_frame or self._last_frame
        globals_input = list(global_obj_items) if global_obj_items is not None else self._last_global_items
        result = merge_result or self._last_result
        if frame is None or globals_input is None or result is None:
            raise ValueError("render requires explicit inputs or a previous merge_frame context.")

        aligned_items = self._align_global_objects(globals_input, frame.timestamp)
        global_pos_by_id = {item.global_id: item.position for item in aligned_items}
        render_meta = {
            "num_detections": len(frame.detections),
            "num_global_objects": len(aligned_items),
            "num_update_ops": len(result.update_ops),
            "num_create_ops": len(result.create_ops),
            "output_path": output_path,
        }

        if plt is None:
            self._log_error("render skipped: matplotlib is not available.")
            return {"status": "matplotlib_unavailable", **render_meta}

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("PerceptionMerger Render")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # 全局目标记录：位置 + 历史轨迹
        for item in aligned_items:
            gx, gy = item.position[0], item.position[1]
            ax.scatter([gx], [gy], c="#1f77b4", s=35, marker="o")
            ax.text(gx + 1.0, gy + 1.0, f"G{item.global_id}", color="#1f77b4", fontsize=8)
            if len(item.trajectory) >= 2:
                tx = [p[0] for p in item.trajectory[-30:]]
                ty = [p[1] for p in item.trajectory[-30:]]
                ax.plot(tx, ty, color="#9ecae1", linewidth=1)

        # 即时感知结果：位置 + 类别 + track_id + track_memory轨迹
        for det in frame.detections:
            dx, dy = det.position[0], det.position[1]
            ax.scatter([dx], [dy], c="#ff7f0e", s=35, marker="x")
            ax.text(dx + 1.0, dy - 1.0, f"C{det.class_id}/T{det.track_id}", color="#ff7f0e", fontsize=8)
            if det.track_id >= 0:
                key = (frame.uav_id, frame.sensor_type, det.track_id)
                history = self.track_memory.get(key)
                if history is not None and len(history.points) >= 2:
                    hx = [p[0] for p in history.points]
                    hy = [p[1] for p in history.points]
                    ax.plot(hx, hy, color="#ffbb78", linestyle="--", linewidth=1)

        # 融合关联结果：成功关联的感知结果连向全局目标
        for op in result.update_ops:
            obs = op.payload.get("observation", {})
            det_pos = obs.get("position")
            if not det_pos:
                continue
            target_pos = global_pos_by_id.get(int(op.target_id))
            if target_pos is None:
                continue
            ax.plot(
                [float(det_pos[0]), float(target_pos[0])],
                [float(det_pos[1]), float(target_pos[1])],
                color="#d62728",
                linestyle=":",
                linewidth=1.3,
            )

        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"status": "ok", **render_meta}

    def _log_event(self, level: str, **record: Any) -> None:
        """输出固定字段结构化日志事件。

        Args:
            level (str): 日志等级（debug/info/error）。
            **record (Any): 结构化字段。
        Returns:
            None: 无返回值，直接写日志。
        """
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if level == "debug":
            self._log_debug("MERGER_EVENT %s", line)
            return
        if level == "info":
            self._log_info("MERGER_EVENT %s", line)
            return
        if level == "info_file":
            self._file_logger.info("MERGER_EVENT %s", line)
            return
        self._log_error("MERGER_EVENT %s", line)

    def _log_input_snapshot(
        self,
        api_name: str,
        received_at: str,
        request_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """以 debug 级别记录可复现输入快照。

        Args:
            api_name (str): 调用接口名（merge_frame 或 merge_batch）。
            received_at (str): 接收时间（ISO8601）。
            request_id (str): 请求关联 ID。
            payload (Dict[str, Any]): 输入内容快照。
        Returns:
            None: 无返回值，直接写日志。
        """
        header = {"event": "merger_input_meta", "received_at": received_at, "request_id": request_id, "api": api_name, "merge_mode": payload.get("merge_mode")}
        self._log_event(level="debug", **header)
        if "perception_frame" in payload:
            frame = payload["perception_frame"]
            self._log_event(
                level="debug",
                event="merger_input_perception_frame",
                received_at=received_at,
                request_id=request_id,
                uav_id=frame.get("uav_id"),
                sensor_type=frame.get("sensor_type"),
                frame_timestamp=frame.get("timestamp"),
                perception_frame=frame,
            )
        if "perception_frames" in payload:
            self._log_event(
                level="debug",
                event="merger_input_perception_frames",
                received_at=received_at,
                request_id=request_id,
                perception_frames=payload["perception_frames"],
            )
        if "global_obj_items" in payload:
            self._log_event(
                level="debug",
                event="merger_input_global_obj_items",
                received_at=received_at,
                request_id=request_id,
                global_obj_items=payload["global_obj_items"],
            )

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

        # class_correlation: 
        # 索引顺序：  sensor_type - class_id - sensor_type - List[class_id]
        # cc[sensor_type] 为指定模态传感器下，各个目标类别
        # Dict[class_id] = Dict[sensor_type]

        # 本模态传感器下所有可识别目标类别
        corr_sensor = self.class_correlation.get(str(sensor_type), {})
        
        # 本模态指定目标类型class_id下，在其他模态传感器中的关联类别
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
        """基于全轨迹 DTW 的轨迹相似度代价。"""
        if det.track_id < 0:
            return 0.5
        key = (frame.uav_id, frame.sensor_type, det.track_id)
        history = self.track_memory.get(key)
        if history is None or len(history.points) < 2:
            return 0.5

        local_points = [tuple(p) for p in history.points]
        local_ts = [float(t) for t in history.timestamps]
        if not local_ts:
            return 0.5
        ts_min = min(local_ts)
        ts_max = max(local_ts)

        global_points = self._extract_global_track_points_by_time(obj=obj, ts_min=ts_min, ts_max=ts_max)
        if len(global_points) < 2:
            return 0.5

        local_norm = self._normalize_track_points(local_points)
        global_norm = self._normalize_track_points(global_points)
        dist = dtw_distance(local_norm, global_norm)
        local_len = self._trajectory_path_length(local_norm)
        global_len = self._trajectory_path_length(global_norm)
        scale = max(local_len, global_len, 1.0)
        if not math.isfinite(dist) or scale <= 1e-6:
            return 0.5
        normalized = min(1.0, dist / scale)
        length_penalty = abs(len(local_norm) - len(global_norm)) / max(len(local_norm), len(global_norm), 1)
        return min(1.0, 0.9 * normalized + 0.1 * length_penalty)

    def _extract_global_track_points_by_time(
        self,
        obj: ObjectItem,
        ts_min: float,
        ts_max: float,
    ) -> List[Tuple[float, float, float]]:
        """提取全局目标在指定时间窗内的历史观测轨迹点。"""
        points: List[Tuple[float, float, float]] = []
        observations_sorted = sorted(
            obj.observations,
            key=lambda obs: float(obs.get("timestamp", obj.timestamp)),
        )
        for obs in observations_sorted:
            obs_ts = float(obs.get("timestamp", obj.timestamp))
            if obs_ts < ts_min or obs_ts > ts_max:
                continue
            pos = obs.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                points.append((float(pos[0]), float(pos[1]), float(pos[2])))
        return points

    def _normalize_track_points(
        self,
        points: Sequence[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """将轨迹平移到统一起点，提升轨迹形状相似性评估稳定性。"""
        if not points:
            return []
        origin = points[0]
        return [(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]) for p in points]

    def _trajectory_path_length(self, points: Sequence[Tuple[float, float, float]]) -> float:
        """计算轨迹折线总长度。"""
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(points)):
            total += self._norm(self._vec_sub(points[i], points[i - 1]))
        return total

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
