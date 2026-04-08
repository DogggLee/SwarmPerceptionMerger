from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from merger.perception_merger import PerceptionMerger
from utils.data_utils import MergeConfig, ObjectItem, PerceptionFrame


def _load_json_file(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_global_objects(raw_items: List[Dict[str, Any]]) -> List[ObjectItem]:
    return [ObjectItem.from_dict(item) for item in raw_items]


def _build_merger(
    config_path: str | None,
    correlation_path: str | None,
    names_path: str | None,
) -> PerceptionMerger:
    config_dict = _load_json_file(config_path)
    corr_dict = _load_json_file(correlation_path)
    names_dict = _load_json_file(names_path)
    config = MergeConfig.from_dict(config_dict)
    merger = PerceptionMerger(config=config, class_correlation=corr_dict, names_mapping=names_dict)
    merger.set_logger(_build_console_logger())
    return merger


def _build_console_logger() -> logging.Logger:
    """构建用于 merger_server 终端输出的 logger。

    Args:
        None: 不需要输入参数。
    Returns:
        logging.Logger: INFO 及以上级别输出到终端的日志器。
    """
    logger = logging.getLogger("PerceptionMergerServerConsole")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def create_app(merger: PerceptionMerger) -> Flask:
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok", "merge_mode": merger.merge_mode})

    @app.post("/merge")
    def merge() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON body"}), 400

        context = payload.get("context", {})
        if isinstance(context, dict):
            request_mode = context.get("merge_mode")
        else:
            request_mode = None
        request_mode = str(request_mode) if request_mode else None

        raw_global = payload.get("global_objects", [])
        if not isinstance(raw_global, list):
            return jsonify({"error": "global_objects must be a list"}), 400

        global_objects = _parse_global_objects(raw_global)

        try:
            if "perception_frames" in payload:
                raw_frames = payload.get("perception_frames", [])
                if not isinstance(raw_frames, list):
                    return jsonify({"error": "perception_frames must be a list"}), 400
                frames = [PerceptionFrame.from_dict(item) for item in raw_frames]
                result = merger.merge_batch(frames, global_objects, merge_mode=request_mode)
            else:
                raw_frame = payload.get("perception_frame")
                if not isinstance(raw_frame, dict):
                    return jsonify({"error": "perception_frame must be an object"}), 400
                frame = PerceptionFrame.from_dict(raw_frame)
                result = merger.merge_frame(frame, global_objects, merge_mode=request_mode)
                # breakpoint()
                # new_op = False
                # for op in result.create_ops:
                #     print("create: ", op.target_id, op.payload["position"])
                #     new_op = True
                # for op in result.update_ops:
                #     print("update: ", op.target_id, op.payload["fused_position"])
                
        except (KeyError, ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid request payload: {exc}"}), 400

        return jsonify(result.to_dict())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/merger.json",
        help="Path to merger config JSON",
    )
    parser.add_argument(
        "--correlation",
        type=str,
        default="config/class_correlation.json",
        help="Path to class correlation JSON",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="config/names.json",
        help="Path to names.json used for int<->string mapping of sensor_type/class_id",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6801)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--show-access-log",
        action="store_true",
        help="Show Flask/Werkzeug access logs (disabled by default).",
    )
    args = parser.parse_args()

    if not args.show_access_log:
        werkzeug_logger = logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(logging.ERROR)
        werkzeug_logger.propagate = False

    merger = _build_merger(config_path=args.config, correlation_path=args.correlation, names_path=args.names)
    app = create_app(merger)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
