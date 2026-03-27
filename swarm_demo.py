from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

from simulator.env import SwarmEnv
from simulator.global_info import GlobalInfo
from utils.data_utils import MergeResult


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


class DemoUI:
    def __init__(self, env_cfg: Dict[str, Any], demo_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> None:
        self.base_env_cfg = env_cfg
        self.base_demo_cfg = demo_cfg
        self.base_global_cfg = global_cfg

        self.env: Optional[SwarmEnv] = None
        self.global_info: Optional[GlobalInfo] = None
        self.running = False
        self.step_count = 0
        self.latest_status = "Not initialized."

        self.root = tk.Tk()
        self.root.title("Swarm Demo Control")
        self.root.geometry("1300x860")

        self._build_layout()
        self._draw_idle()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.grid(row=0, column=0, sticky="ns")

        vis = ttk.Frame(self.root, padding=10)
        vis.grid(row=0, column=1, sticky="nsew")
        vis.rowconfigure(1, weight=1)
        vis.columnconfigure(0, weight=1)

        ttk.Label(ctrl, text="Progress Control", font=("", 11, "bold")).pack(anchor="w")
        mode_row = ttk.Frame(ctrl)
        mode_row.pack(fill="x", pady=(4, 8))
        ttk.Label(mode_row, text="Mode").pack(side="left")
        self.mode_var = tk.StringVar(value="continue")
        self.mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.mode_var,
            values=["continue", "in-loop", "step"],
            state="readonly",
            width=12,
        )
        self.mode_combo.pack(side="right")

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill="x", pady=(2, 12))
        ttk.Button(btn_row, text="Initialize", command=self.on_initialize).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Start", command=self.on_start).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Pause", command=self.on_pause).pack(side="left", padx=6)

        ttk.Label(ctrl, text="Display Layers", font=("", 11, "bold")).pack(anchor="w", pady=(8, 2))
        self.show_truth_var = tk.BooleanVar(value=True)
        self.show_sensor_var = tk.BooleanVar(value=True)
        self.show_obs_var = tk.BooleanVar(value=True)
        self.show_match_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Target Truth", variable=self.show_truth_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Sensor Range", variable=self.show_sensor_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Observation History", variable=self.show_obs_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Match Edges", variable=self.show_match_var, command=self.redraw).pack(anchor="w")

        ttk.Label(ctrl, text="Parameters", font=("", 11, "bold")).pack(anchor="w", pady=(12, 4))
        notebook = ttk.Notebook(ctrl, width=390, height=420)
        notebook.pack(fill="both", expand=False)
        self.env_tab = ttk.Frame(notebook)
        self.merge_tab = ttk.Frame(notebook)
        self.global_tab = ttk.Frame(notebook)
        notebook.add(self.env_tab, text="Env")
        notebook.add(self.merge_tab, text="Merge")
        notebook.add(self.global_tab, text="GlobalState")

        self._build_env_tab()
        self._build_merge_tab()
        self._build_global_tab()

        self.status_var = tk.StringVar(value=self.latest_status)
        ttk.Label(ctrl, textvariable=self.status_var, wraplength=380).pack(anchor="w", pady=(10, 0))

        ttk.Label(vis, text="Global Situation", font=("", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(vis, width=860, height=780, bg="white")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

    def _build_env_tab(self) -> None:
        self.env_dt = tk.StringVar(value=str(self.base_env_cfg.get("dt", 0.5)))
        self.env_seed = tk.StringVar(value=str(self.base_env_cfg.get("seed", 0)))
        self.env_weather = tk.StringVar(value=str(self.base_env_cfg.get("weather", "clear")))
        self.env_lighting = tk.StringVar(value=str(self.base_env_cfg.get("lighting", "night")))
        self.env_map_w = tk.StringVar(value=str(self.base_env_cfg.get("map_size", [1200, 1200])[0]))
        self.env_map_h = tk.StringVar(value=str(self.base_env_cfg.get("map_size", [1200, 1200])[1]))
        for i, (label, var) in enumerate(
            [
                ("dt", self.env_dt),
                ("seed", self.env_seed),
                ("weather", self.env_weather),
                ("lighting", self.env_lighting),
                ("map_width", self.env_map_w),
                ("map_height", self.env_map_h),
            ]
        ):
            ttk.Label(self.env_tab, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(self.env_tab, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
        self.env_tab.columnconfigure(1, weight=1)

    def _build_merge_tab(self) -> None:
        self.merge_url = tk.StringVar(value=str(self.base_demo_cfg.get("merger_server_url", "http://127.0.0.1:6801")))
        self.merge_timeout = tk.StringVar(value=str(self.base_demo_cfg.get("http_timeout_s", 3.0)))
        self.merge_mode = tk.StringVar(value=str(self.base_demo_cfg.get("merge_mode", "simple")))
        self.step_sleep = tk.StringVar(value=str(self.base_demo_cfg.get("step_sleep_s", 0.0)))
        for i, (label, var) in enumerate(
            [
                ("server_url", self.merge_url),
                ("http_timeout_s", self.merge_timeout),
                ("merge_mode", self.merge_mode),
                ("step_sleep_s", self.step_sleep),
            ]
        ):
            ttk.Label(self.merge_tab, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(self.merge_tab, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
        self.merge_tab.columnconfigure(1, weight=1)

    def _build_global_tab(self) -> None:
        self.global_valid_obs = tk.StringVar(value=str(self.base_global_cfg.get("valid_observation_count", 1)))
        self.global_max_unseen = tk.StringVar(value=str(self.base_global_cfg.get("max_unseen_time", 4.0)))
        self.global_stale_obs = tk.StringVar(value=str(self.base_global_cfg.get("stale_observation_time", 2.0)))
        ttk.Label(self.global_tab, text="valid_observation_count").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.global_tab, textvariable=self.global_valid_obs, width=24).grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Label(self.global_tab, text="max_unseen_time").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.global_tab, textvariable=self.global_max_unseen, width=24).grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Label(self.global_tab, text="stale_observation_time").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.global_tab, textvariable=self.global_stale_obs, width=24).grid(
            row=2, column=1, sticky="ew", padx=6, pady=4
        )
        self.global_tab.columnconfigure(1, weight=1)

    def _collect_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        env_cfg = dict(self.base_env_cfg)
        env_cfg["dt"] = float(self.env_dt.get())
        env_cfg["seed"] = int(self.env_seed.get())
        env_cfg["weather"] = str(self.env_weather.get())
        env_cfg["lighting"] = str(self.env_lighting.get())
        env_cfg["map_size"] = [float(self.env_map_w.get()), float(self.env_map_h.get())]

        demo_cfg = dict(self.base_demo_cfg)
        demo_cfg["merger_server_url"] = str(self.merge_url.get())
        demo_cfg["http_timeout_s"] = float(self.merge_timeout.get())
        demo_cfg["merge_mode"] = str(self.merge_mode.get())
        demo_cfg["step_sleep_s"] = float(self.step_sleep.get())

        global_cfg = dict(self.base_global_cfg)
        global_cfg["valid_observation_count"] = int(self.global_valid_obs.get())
        global_cfg["max_unseen_time"] = float(self.global_max_unseen.get())
        global_cfg["stale_observation_time"] = float(self.global_stale_obs.get())
        return env_cfg, demo_cfg, global_cfg

    def on_initialize(self) -> None:
        env_cfg, demo_cfg, global_cfg = self._collect_configs()
        server_url = str(demo_cfg["merger_server_url"]).rstrip("/")
        health_url = f"{server_url}/healthz"
        try:
            health = _get_json(health_url, timeout=float(demo_cfg["http_timeout_s"]))
        except Exception as exc:
            self._set_status(f"Initialize failed: {exc}")
            return

        self.env = SwarmEnv(env_cfg)
        self.global_info = GlobalInfo(
            valid_observation_count=int(global_cfg["valid_observation_count"]),
            max_unseen_time=float(global_cfg["max_unseen_time"]),
            stale_observation_time=float(global_cfg["stale_observation_time"]),
        )
        self._sync_visual_global_items()
        self.step_count = 0
        self.running = False
        self._set_status(f"Initialized. merger_service={health}")
        self.redraw()

    def on_start(self) -> None:
        if self.env is None or self.global_info is None:
            self._set_status("Please initialize first.")
            return
        self.running = True
        self._run_loop()

    def on_pause(self) -> None:
        self.running = False
        self._set_status("Paused.")

    def _run_loop(self) -> None:
        if not self.running:
            return
        mode = self.mode_var.get()
        try:
            processed, updates, creates = self._run_one_step()
        except Exception as exc:
            self.running = False
            self._set_status(f"Runtime error: {exc}")
            return

        self.step_count += 1
        global_count = 0 if self.global_info is None else len(list(self.global_info.get_valid_items()))
        self._set_status(
            f"step={self.step_count} frames={processed} updates={updates} creates={creates} valid_global_objs={global_count}"
        )
        self.redraw()

        if mode == "continue":
            dt_ms = int(max(20, float(self.env_dt.get()) * 1000))
            self.root.after(dt_ms, self._run_loop)
            return
        if mode == "in-loop":
            # 仅在本轮真实触发融合后暂停；否则继续推进环境直到有帧进入融合链路。
            if processed > 0:
                self.running = False
            else:
                dt_ms = int(max(20, float(self.env_dt.get()) * 1000))
                self.root.after(dt_ms, self._run_loop)
            return
        if mode == "step":
            self.running = False
            return

    def _run_one_step(self) -> Tuple[int, int, int]:
        assert self.env is not None
        assert self.global_info is not None
        demo_cfg = self._collect_configs()[1]

        server_url = str(demo_cfg["merger_server_url"]).rstrip("/")
        merge_url = f"{server_url}/merge"
        timeout_s = float(demo_cfg["http_timeout_s"])
        merge_mode = str(demo_cfg["merge_mode"])
        self.env.step(dt=float(self.env_dt.get()))

        processed = 0
        updates = 0
        creates = 0

        # 缓存池为空时，跳过本轮融合请求。
        if not self.env.get_pending_frames():
            return processed, updates, creates

        while True:
            frame = self.env.pop_next_frame()
            if frame is None:
                break
            payload = {
                "context": {"merge_mode": merge_mode},
                "perception_frame": frame.to_dict(),
                "global_objects": [item.to_dict() for item in self.global_info.get_all_items()],
            }
            resp_json = _post_json(merge_url, payload, timeout=timeout_s)
            merge_result = MergeResult.from_dict(resp_json)
            self.global_info.apply_merge_result(merge_result)
            self.env.record_merge_result(frame, merge_result)
            self._sync_visual_global_items()
            processed += 1
            updates += len(merge_result.update_ops)
            creates += len(merge_result.create_ops)
        return processed, updates, creates

    def _sync_visual_global_items(self) -> None:
        """只将有效目标记录同步到环境可视化层。"""
        assert self.env is not None
        assert self.global_info is not None
        self.env.set_global_objects(list(self.global_info.get_valid_items()))

    def _set_status(self, text: str) -> None:
        self.latest_status = text
        self.status_var.set(text)

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.env is None:
            self._draw_idle()
            return
        rs = self.env.get_render_state()
        map_w, map_h = self._extract_map_size()
        self._draw_axes(map_w, map_h)
        global_pos_by_id = {}

        for item in rs.get("global_objects", []):
            x, y = item["position"][0], item["position"][1]
            cx, cy = self._world_to_canvas(x, y, map_w, map_h)
            global_pos_by_id[int(item["global_id"])] = (cx, cy)
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#0077b6", outline="")
            self.canvas.create_text(cx + 12, cy - 10, text=f"G{item['global_id']}", fill="#0077b6", anchor="w")
            self._draw_velocity_arrow((x, y), item.get("velocity", [0, 0, 0]), map_w, map_h, "#0077b6")
            traj = item.get("trajectory", [])
            if len(traj) >= 2:
                pts = []
                for p in traj[-40:]:
                    tx, ty = self._world_to_canvas(p[0], p[1], map_w, map_h)
                    pts.extend([tx, ty])
                self.canvas.create_line(*pts, fill="#89c2d9", width=1)
            if self.show_obs_var.get():
                for obs in item.get("observations", [])[-40:]:
                    sensor_type = int(obs.get("sensor_type", -1))
                    if sensor_type == 3 and obs.get("bearing_vector") and obs.get("sensor_position"):
                        self._draw_bearing_ray(
                            sensor_position=obs["sensor_position"],
                            bearing_vector=obs["bearing_vector"],
                            map_w=map_w,
                            map_h=map_h,
                            color="#48cae4",
                            length=180.0,
                        )
                    elif "position" in obs:
                        ox, oy = self._world_to_canvas(obs["position"][0], obs["position"][1], map_w, map_h)
                        self.canvas.create_oval(ox - 2, oy - 2, ox + 2, oy + 2, fill="#48cae4", outline="")

        if self.show_truth_var.get():
            for target in rs.get("targets_truth", []):
                tx, ty = self._world_to_canvas(target["position"][0], target["position"][1], map_w, map_h)
                self.canvas.create_rectangle(tx - 5, ty - 5, tx + 5, ty + 5, outline="#f77f00", width=2)
                self.canvas.create_text(tx + 10, ty + 10, text=f"T{target['target_id']}", fill="#f77f00", anchor="w")
                self._draw_velocity_arrow(
                    (target["position"][0], target["position"][1]),
                    target.get("velocity", [0, 0, 0]),
                    map_w,
                    map_h,
                    "#f77f00",
                )

        for uav in rs.get("uavs", []):
            ux, uy = self._world_to_canvas(uav["position"][0], uav["position"][1], map_w, map_h)
            self.canvas.create_polygon(
                ux,
                uy - 7,
                ux - 6,
                uy + 6,
                ux + 6,
                uy + 6,
                fill="#2b9348",
                outline="",
            )
            self.canvas.create_text(ux + 10, uy - 10, text=f"U{uav['uav_id']}", fill="#2b9348", anchor="w")
            if self.show_sensor_var.get():
                self._draw_sensor_range(uav, map_w, map_h)

        if self.show_match_var.get():
            for edge in rs.get("match_edges", []):
                if int(edge.get("sensor_type", -1)) == 3 and edge.get("sensor_position") and edge.get("bearing_vector"):
                    self._draw_bearing_ray(
                        sensor_position=edge["sensor_position"],
                        bearing_vector=edge["bearing_vector"],
                        map_w=map_w,
                        map_h=map_h,
                        color="#d00000",
                        length=220.0,
                    )
                dp = edge.get("detection_position")
                if not dp:
                    continue
                x1, y1 = self._world_to_canvas(dp[0], dp[1], map_w, map_h)
                target_id = edge.get("target_id")
                p2 = global_pos_by_id.get(int(target_id)) if target_id is not None else None
                if p2:
                    x2, y2 = p2
                    self.canvas.create_line(x1, y1, x2, y2, fill="#d00000", dash=(4, 2), width=2)
                    score = edge.get("score")
                    if score is not None:
                        self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=f"{score:.2f}", fill="#d00000")
                else:
                    self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="#d00000", outline="")

        if self.show_obs_var.get():
            # 绘制缓存池中的ELEC观测射线（不绘制伪position点）。
            for frame in rs.get("observations_valid", []):
                if int(frame.get("sensor_type", -1)) != 3:
                    continue
                sensor_pos = frame.get("sensor_position")
                if not sensor_pos:
                    continue
                for det in frame.get("detections", []):
                    bearing = det.get("bearing_vector")
                    if bearing:
                        self._draw_bearing_ray(
                            sensor_position=sensor_pos,
                            bearing_vector=bearing,
                            map_w=map_w,
                            map_h=map_h,
                            color="#ff6b6b",
                            length=220.0,
                        )

    def _draw_idle(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(430, 390, text="Click Initialize to start simulation", fill="gray")

    def _extract_map_size(self) -> Tuple[float, float]:
        if self.env is None:
            return 1200.0, 1200.0
        size = self.env.config.get("map_size", [1200, 1200])
        return float(size[0]), float(size[1])

    def _world_to_canvas(self, x: float, y: float, map_w: float, map_h: float) -> Tuple[float, float]:
        pad = 30.0
        w = float(self.canvas.winfo_width() or 860)
        h = float(self.canvas.winfo_height() or 780)
        sx = (w - 2 * pad) / max(map_w, 1.0)
        sy = (h - 2 * pad) / max(map_h, 1.0)
        cx = pad + x * sx
        cy = h - pad - y * sy
        return cx, cy

    def _draw_axes(self, map_w: float, map_h: float) -> None:
        w = float(self.canvas.winfo_width() or 860)
        h = float(self.canvas.winfo_height() or 780)
        pad = 30.0
        self.canvas.create_rectangle(pad, pad, w - pad, h - pad, outline="#ced4da")
        self.canvas.create_text(pad + 4, h - pad + 14, text="(0,0)", anchor="w", fill="gray")
        self.canvas.create_text(w - pad - 4, pad - 14, text=f"({int(map_w)},{int(map_h)})", anchor="e", fill="gray")

    def _draw_velocity_arrow(
        self,
        pos_xy: Tuple[float, float],
        vel: List[float],
        map_w: float,
        map_h: float,
        color: str,
    ) -> None:
        vx, vy = float(vel[0]), float(vel[1])
        if abs(vx) + abs(vy) < 1e-6:
            return
        x, y = pos_xy
        scale = 3.0
        cx1, cy1 = self._world_to_canvas(x, y, map_w, map_h)
        cx2, cy2 = self._world_to_canvas(x + vx * scale, y + vy * scale, map_w, map_h)
        self.canvas.create_line(cx1, cy1, cx2, cy2, fill=color, arrow=tk.LAST, width=1)

    def _draw_sensor_range(self, uav: Dict[str, Any], map_w: float, map_h: float) -> None:
        sensor_type = int(uav.get("sensor_type", 2))
        ux, uy = float(uav["position"][0]), float(uav["position"][1])
        params = uav.get("sensor_params", {})
        yaw_deg = float(uav.get("yaw_deg", 0.0))

        if sensor_type in (0, 3):
            radius = float(params.get("max_range", 200.0))
            cx1, cy1 = self._world_to_canvas(ux - radius, uy - radius, map_w, map_h)
            cx2, cy2 = self._world_to_canvas(ux + radius, uy + radius, map_w, map_h)
            self.canvas.create_oval(cx1, cy2, cx2, cy1, outline="#95d5b2")
            return

        forward = float(params.get("forward_range", 260.0))
        width = float(params.get("width", 180.0))
        corners = [(0.0, -width / 2), (forward, -width / 2), (forward, width / 2), (0.0, width / 2)]
        yaw = math.radians(yaw_deg)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        pts = []
        for fx, fy in corners:
            wx = ux + fx * cos_yaw - fy * sin_yaw
            wy = uy + fx * sin_yaw + fy * cos_yaw
            cx, cy = self._world_to_canvas(wx, wy, map_w, map_h)
            pts.extend([cx, cy])
        self.canvas.create_polygon(*pts, outline="#95d5b2", fill="", width=1)

    def _draw_bearing_ray(
        self,
        sensor_position: List[float],
        bearing_vector: List[float],
        map_w: float,
        map_h: float,
        color: str,
        length: float,
    ) -> None:
        """按传感器位置与bearing方向绘制ELEC观测射线。"""
        vx = float(bearing_vector[0])
        vy = float(bearing_vector[1])
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return
        ux, uy = vx / norm, vy / norm
        x0, y0 = float(sensor_position[0]), float(sensor_position[1])
        x1, y1 = x0 + ux * length, y0 + uy * length
        cx0, cy0 = self._world_to_canvas(x0, y0, map_w, map_h)
        cx1, cy1 = self._world_to_canvas(x1, y1, map_w, map_h)
        self.canvas.create_line(cx0, cy0, cx1, cy1, fill=color, dash=(3, 2), width=1)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="config/swarm_env.json")
    parser.add_argument("--demo-config", default="config/swarm_demo.json")
    parser.add_argument("--global-config", default="config/global_info.json")
    args = parser.parse_args()

    env_cfg = _load_json(args.env_config)
    demo_cfg = _load_json(args.demo_config)
    global_cfg = _load_json(args.global_config)
    ui = DemoUI(env_cfg=env_cfg, demo_cfg=demo_cfg, global_cfg=global_cfg)
    ui.run()


if __name__ == "__main__":
    main()
