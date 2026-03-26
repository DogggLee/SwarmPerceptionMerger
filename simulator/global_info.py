from __future__ import annotations

from typing import Dict, Iterable, Optional

from utils.data_utils import MergeOperation, MergeResult, ObjectItem


class GlobalInfo:
    def __init__(self, valid_observation_count: int, max_unseen_time: float) -> None:
        self.valid_observation_count = valid_observation_count
        self.max_unseen_time = max_unseen_time
        self.items: Dict[int, ObjectItem] = {}
        self.max_id = 0

    def _assign_id(self) -> int:
        self.max_id += 1
        return self.max_id

    def _add_item(self, obj_item: ObjectItem) -> int:
        if obj_item.global_id <= 0:
            obj_item.global_id = self._assign_id()
        else:
            self.max_id = max(self.max_id, obj_item.global_id)
        self.items[obj_item.global_id] = obj_item
        return obj_item.global_id

    def get_all_items(self) -> Iterable[ObjectItem]:
        return self.items.values()

    def get_item(self, global_id: int) -> Optional[ObjectItem]:
        return self.items.get(global_id)

    def apply_merge_result(self, merge_result: MergeResult) -> Dict[int, int]:
        id_map: Dict[int, int] = {}
        for op in merge_result.update_ops:
            self.apply_update_op(op)
        for op in merge_result.create_ops:
            new_id = self.apply_create_op(op)
            id_map[op.target_id] = new_id
        return id_map

    def apply_update_op(self, op: MergeOperation) -> None:
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
        )
        self.items[item.global_id] = item
        return item.global_id
