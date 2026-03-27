from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from utils.data_utils import MergeOperation, MergeResult, ObjectItem, SensorType


class GlobalInfo:
    def __init__(
        self,
        valid_observation_count: int,
        max_unseen_time: float,
        stale_observation_time: float = 2.0,
    ) -> None:
        """初始化全局态势管理器与目标容器。"""
        self.valid_observation_count = valid_observation_count
        self.max_unseen_time = max_unseen_time
        self.stale_observation_time = stale_observation_time
        self.items: Dict[int, ObjectItem] = {}
        self.max_id = 0
        self.current_timestamp = 0.0

    def _assign_id(self) -> int:
        """分配新的全局目标 ID。"""
        self.max_id += 1
        return self.max_id

    def _add_item(self, obj_item: ObjectItem) -> int:
        """向全局态势中新增目标记录并返回最终 ID。"""
        if obj_item.global_id <= 0:
            obj_item.global_id = self._assign_id()
        else:
            self.max_id = max(self.max_id, obj_item.global_id)
        self.items[obj_item.global_id] = obj_item
        return obj_item.global_id

    def get_all_items(self) -> Iterable[ObjectItem]:
        """返回当前全部全局目标记录。"""
        return self.items.values()

    def get_valid_items(self) -> Iterable[ObjectItem]:
        """返回有效观测数量达到阈值且空间信息有效的目标记录。"""
        valid_items: List[ObjectItem] = []
        for item in self.items.values():
            if item.spatial_valid and len(item.observations) >= self.valid_observation_count:
                valid_items.append(item)
        return valid_items

    def get_item(self, global_id: int) -> Optional[ObjectItem]:
        """根据全局 ID 查询目标记录。"""
        return self.items.get(global_id)

    def apply_merge_result(self, merge_result: MergeResult) -> Dict[int, int]:
        """应用融合结果并返回临时 ID 到正式 ID 的映射。"""
        id_map: Dict[int, int] = {}
        latest_ts = self.current_timestamp
        for op in merge_result.update_ops:
            self.apply_update_op(op)
            if "timestamp" in op.payload:
                latest_ts = max(latest_ts, float(op.payload["timestamp"]))
        for op in merge_result.create_ops:
            new_id = self.apply_create_op(op)
            id_map[op.target_id] = new_id
            if "timestamp" in op.payload:
                latest_ts = max(latest_ts, float(op.payload["timestamp"]))

        self.current_timestamp = latest_ts
        self.maintain_items()
        return id_map

    def apply_update_op(self, op: MergeOperation) -> None:
        """将单条更新操作写回已有目标记录。"""
        item = self.items.get(op.target_id)
        if item is None:
            return

        payload = op.payload
        fused_position = payload.get("fused_position")
        fused_velocity = payload.get("fused_velocity")
        if fused_position:
            item.position = (
                float(fused_position[0]),
                float(fused_position[1]),
                float(fused_position[2]),
            )
            item.trajectory.append(item.position)
        if fused_velocity:
            item.velocity = (
                float(fused_velocity[0]),
                float(fused_velocity[1]),
                float(fused_velocity[2]),
            )
        if "timestamp" in payload:
            item.timestamp = float(payload["timestamp"])

        observation = payload.get("observation")
        if observation:
            item.observations.append(observation)

        class_update = payload.get("class_update", {})
        class_by_sensor = class_update.get("class_by_sensor", {})
        class_votes = class_update.get("class_votes", {})
        if class_by_sensor:
            item.class_by_sensor = {int(k): int(v) for k, v in class_by_sensor.items()}
        if class_votes:
            item.class_votes = {
                int(sensor): {int(cid): int(count) for cid, count in votes.items()}
                for sensor, votes in class_votes.items()
            }

    def apply_create_op(self, op: MergeOperation) -> int:
        """将新增操作转换为正式目标记录并分配全局 ID。"""
        payload = op.payload
        item = ObjectItem(
            global_id=self._assign_id(),
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
            class_by_sensor={int(k): int(v) for k, v in payload.get("class_by_sensor", {}).items()},
            class_votes={
                int(sensor): {int(cid): int(count) for cid, count in votes.items()}
                for sensor, votes in payload.get("class_votes", {}).items()
            },
            trajectory=[tuple(payload["position"])],  # type: ignore[arg-type]
            observations=[payload["observation"]] if "observation" in payload else [],
            spatial_valid=bool(payload.get("spatial_valid", True)),
        )
        self.items[item.global_id] = item
        return item.global_id

    def predict(self, timestamp: float) -> None:
        """将当前全局目标状态外推到指定时间戳。"""
        for item in self.items.values():
            dt = float(timestamp) - float(item.timestamp)
            if dt <= 0:
                continue
            item.position = (
                item.position[0] + item.velocity[0] * dt,
                item.position[1] + item.velocity[1] * dt,
                item.position[2] + item.velocity[2] * dt,
            )
            item.timestamp = float(timestamp)
            item.trajectory.append(item.position)

        self.current_timestamp = max(self.current_timestamp, float(timestamp))
        self.maintain_items()

    def maintain_items(self) -> None:
        """剔除过旧观测并删除无有效观测的目标记录。"""
        threshold = self.current_timestamp - self.stale_observation_time
        remove_ids: List[int] = []
        for global_id, item in self.items.items():
            item.observations = [
                obs
                for obs in item.observations
                if float(obs.get("timestamp", self.current_timestamp)) >= threshold
            ]
            if not item.observations:
                remove_ids.append(global_id)
                continue
            # 空间有效性基于“当前有效观测集合”统一决定：
            # 只要存在任一非ELEC观测，即视作空间信息有效；否则无效。
            has_non_elec_obs = any(
                int(obs.get("sensor_type", -1)) != SensorType.ELEC.value
                for obs in item.observations
            )
            item.spatial_valid = has_non_elec_obs

        for global_id in remove_ids:
            del self.items[global_id]
