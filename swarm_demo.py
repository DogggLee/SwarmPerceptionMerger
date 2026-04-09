from __future__ import annotations

import argparse
import json
import math
import os
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from simulator.env import SwarmEnv
from simulator.global_info import GlobalInfo
from utils.data_utils import MergeResult, SensorType


SENSOR_LABELS: Dict[int, str] = {
    0: "RADAR",
    1: "IF",
    2: "RGB",
    3: "ELEC",
}
SENSOR_RANGE_COLORS: Dict[int, str] = {
    SensorType.RADAR.value: "#2a9d8f",
    SensorType.IF.value: "#f4a261",
    SensorType.RGB.value: "#457b9d",
    SensorType.ELEC.value: "#9d4edd",
}


def _load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 配置文件。

    Args:
        path (str): 配置文件路径。
    Returns:
        Dict[str, Any]: 解析后的配置字典。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    """发送 GET 请求并返回 JSON 响应。

    Args:
        url (str): 请求地址。
        timeout (float): 超时时间（秒）。
    Returns:
        Dict[str, Any]: 响应 JSON 字典。
    """
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """发送 POST(JSON) 请求并返回 JSON 响应。

    Args:
        url (str): 请求地址。
        payload (Dict[str, Any]): 请求体字典。
        timeout (float): 超时时间（秒）。
    Returns:
        Dict[str, Any]: 响应 JSON 字典。
    """
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
    def __init__(
        self,
        env_cfg: Dict[str, Any],
        demo_cfg: Dict[str, Any],
        global_cfg: Dict[str, Any],
        class_correlation: Dict[str, Any],
        env_config_path: str,
        demo_config_path: str,
        global_config_path: str,
    ) -> None:
        """初始化演示 UI 主窗口、配置缓存与运行状态。

        Args:
            env_cfg (Dict[str, Any]): 仿真环境配置。
            demo_cfg (Dict[str, Any]): 演示流程配置。
            global_cfg (Dict[str, Any]): 全局态势配置。
            class_correlation (Dict[str, Any]): 类别关联配置，用于随机生成目标类别组合。
            env_config_path (str): 环境配置文件路径。
            demo_config_path (str): 融合流程配置文件路径。
            global_config_path (str): 全局态势配置文件路径。
        Returns:
            None: 无返回值，直接构建 UI。
        """
        self.base_env_cfg = env_cfg
        self.base_demo_cfg = demo_cfg
        self.base_global_cfg = global_cfg
        self.class_correlation = class_correlation
        self.env_config_path = env_config_path
        self.demo_config_path = demo_config_path
        self.global_config_path = global_config_path

        self.env: Optional[SwarmEnv] = None
        self.global_info: Optional[GlobalInfo] = None
        self.running = False
        self.step_count = 0
        self.latest_status = "Not initialized."
        self._canvas_resize_job: Optional[str] = None
        self._last_pool_text: Optional[str] = None
        self.target_truth_history: Dict[int, List[Tuple[float, float]]] = {}
        self.target_traj_max_len = 200
        self.pool_text_widget: Optional[tk.Text] = None

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
        self.show_truth_traj_var = tk.BooleanVar(value=True)
        self.show_sensor_var = tk.BooleanVar(value=bool(self.base_demo_cfg.get("show_sensor_range", True)))
        self.show_obs_var = tk.BooleanVar(value=True)
        self.show_match_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Target Truth", variable=self.show_truth_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Target Trajectory", variable=self.show_truth_traj_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Sensor Range", variable=self.show_sensor_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Observation History", variable=self.show_obs_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Show Match Edges", variable=self.show_match_var, command=self.redraw).pack(anchor="w")

        ttk.Label(ctrl, text="Parameters", font=("", 11, "bold")).pack(anchor="w", pady=(12, 4))
        notebook = ttk.Notebook(ctrl, width=390, height=420)
        notebook.pack(fill="both", expand=False)
        self.env_tab = ttk.Frame(notebook)
        self.merge_tab = ttk.Frame(notebook)
        self.global_tab = ttk.Frame(notebook)
        self.sensor_tab = ttk.Frame(notebook)
        notebook.add(self.env_tab, text="Env")
        notebook.add(self.merge_tab, text="Merge")
        notebook.add(self.global_tab, text="GlobalState")
        notebook.add(self.sensor_tab, text="Sensors")

        self._build_env_tab()
        self._build_merge_tab()
        self._build_global_tab()
        self._build_sensor_tab()
        self._build_global_pool_panel(ctrl)

        self.status_var = tk.StringVar(value=self.latest_status)
        ttk.Label(ctrl, textvariable=self.status_var, wraplength=380).pack(anchor="w", pady=(10, 0))

        ttk.Label(vis, text="Global Situation", font=("", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.fig = Figure(figsize=(8.6, 7.8), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=vis)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.plot_widget.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """在画布尺寸变化时触发重绘，保持内容居中且自适应窗口大小。

        Args:
            event (tk.Event): Tkinter 画布尺寸变化事件。
        Returns:
            None: 无返回值，延迟触发重绘。
        """
        if event.widget is not self.plot_widget:
            return
        if self._canvas_resize_job is not None:
            self.root.after_cancel(self._canvas_resize_job)
        # 防抖处理，避免拖动窗口时高频重绘造成卡顿。
        self._canvas_resize_job = self.root.after(60, self._redraw_after_resize)

    def _redraw_after_resize(self) -> None:
        """执行尺寸变化后的重绘动作。

        Args:
            None: 无输入参数。
        Returns:
            None: 无返回值，直接调用重绘流程。
        """
        self._canvas_resize_job = None
        self._sync_figure_size_to_widget()
        self.redraw()

    def _sync_figure_size_to_widget(self) -> None:
        """将 Matplotlib Figure 尺寸同步到当前 Tk 绘图控件尺寸。

        Args:
            None: 无输入参数。
        Returns:
            None: 无返回值，直接更新 Figure 尺寸。
        """
        w = max(1, int(self.plot_widget.winfo_width() or 1))
        h = max(1, int(self.plot_widget.winfo_height() or 1))
        dpi = float(self.fig.get_dpi() or 100.0)
        self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

    def _build_env_tab(self) -> None:
        uav_counts = self._extract_uav_counts(self.base_env_cfg)
        self.env_dt = tk.StringVar(value=str(self.base_env_cfg.get("dt", 0.5)))
        self.env_seed = tk.StringVar(value=str(self.base_env_cfg.get("seed", 0)))
        self.env_generation_seed = tk.StringVar(value=str(self.base_env_cfg.get("generation_seed", self.base_env_cfg.get("seed", 0))))
        self.env_weather = tk.StringVar(value=str(self.base_env_cfg.get("weather", "clear")))
        self.env_lighting = tk.StringVar(value=str(self.base_env_cfg.get("lighting", "night")))
        self.env_map_w = tk.StringVar(value=str(self.base_env_cfg.get("map_size", [1200, 1200])[0]))
        self.env_map_h = tk.StringVar(value=str(self.base_env_cfg.get("map_size", [1200, 1200])[1]))
        self.env_target_count = tk.StringVar(value=str(self.base_env_cfg.get("target_count", len(self.base_env_cfg.get("targets", [])))))
        self.env_count_radar = tk.StringVar(value=str(uav_counts["RADAR"]))
        self.env_count_if = tk.StringVar(value=str(uav_counts["IF"]))
        self.env_count_rgb = tk.StringVar(value=str(uav_counts["RGB"]))
        self.env_count_elec = tk.StringVar(value=str(uav_counts["ELEC"]))
        self.env_weather.trace_add("write", self._on_live_env_condition_change)
        self.env_lighting.trace_add("write", self._on_live_env_condition_change)
        for i, (label, var) in enumerate(
            [
                ("dt", self.env_dt),
                ("seed", self.env_seed),
                ("generation_seed", self.env_generation_seed),
                ("weather", self.env_weather),
                ("lighting", self.env_lighting),
                ("map_width", self.env_map_w),
                ("map_height", self.env_map_h),
                ("target_count", self.env_target_count),
                ("uav_count_radar", self.env_count_radar),
                ("uav_count_if", self.env_count_if),
                ("uav_count_rgb", self.env_count_rgb),
                ("uav_count_elec", self.env_count_elec),
            ]
        ):
            ttk.Label(self.env_tab, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            if label == "weather":
                ttk.Combobox(
                    self.env_tab,
                    textvariable=var,
                    values=["clear", "rain", "snow", "fog"],
                    state="readonly",
                    width=21,
                ).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
            elif label == "lighting":
                ttk.Combobox(
                    self.env_tab,
                    textvariable=var,
                    values=["day", "night", "dusk", "dawn"],
                    state="readonly",
                    width=21,
                ).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
            else:
                ttk.Entry(self.env_tab, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
        btn_row = len(
            [
                "dt",
                "seed",
                "generation_seed",
                "weather",
                "lighting",
                "map_width",
                "map_height",
                "target_count",
                "uav_count_radar",
                "uav_count_if",
                "uav_count_rgb",
                "uav_count_elec",
            ]
        )
        btn_bar = ttk.Frame(self.env_tab)
        btn_bar.grid(row=btn_row, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 4))
        ttk.Button(btn_bar, text="Load", command=self._on_load_env_config).pack(side="left")
        ttk.Button(btn_bar, text="Save", command=self._on_save_env_config).pack(side="left", padx=(8, 0))
        self.env_tab.columnconfigure(1, weight=1)

    def _on_live_env_condition_change(self, *_args: Any) -> None:
        """实时将天气与光照改动应用到已初始化环境。

        Args:
            *_args (Any): Tkinter trace 回调附带参数，未使用。
        Returns:
            None: 无返回值，直接更新环境并触发重绘。
        """
        if self.env is None:
            return
        self.env.global_weather = str(self.env_weather.get()).lower()
        self.env.global_lighting = str(self.env_lighting.get()).lower()
        self.env.config["weather"] = self.env.global_weather
        self.env.config["lighting"] = self.env.global_lighting
        self.env._refresh_render_state()
        self.redraw()

    def _extract_uav_counts(self, env_cfg: Dict[str, Any]) -> Dict[str, int]:
        """提取各模态 UAV 数量配置。

        Args:
            env_cfg (Dict[str, Any]): 环境配置。
        Returns:
            Dict[str, int]: `RADAR/IF/RGB/ELEC` 四类数量。
        """
        default_counts = {"RADAR": 0, "IF": 0, "RGB": 0, "ELEC": 0}
        raw_counts = env_cfg.get("uav_counts")
        if isinstance(raw_counts, dict):
            for key in default_counts:
                default_counts[key] = int(raw_counts.get(key, default_counts[key]))
            return default_counts
        for item in env_cfg.get("uavs", []):
            sensor_cfg = item.get("sensor", {})
            sensor_type = SensorType.parse(sensor_cfg.get("sensor_type", item.get("sensor_type", 2)))
            label = SENSOR_LABELS.get(sensor_type)
            if label is not None:
                default_counts[label] += 1
        return default_counts

    def _build_merge_tab(self) -> None:
        """构建 Merge 参数页签，配置服务地址、超时、模式和循环频率。

        Args:
            None: 无输入参数。
        Returns:
            None: 无返回值，直接构建并绑定界面控件。
        """
        self.merge_url = tk.StringVar(value=str(self.base_demo_cfg.get("merger_server_url", "http://127.0.0.1:6801")))
        self.merge_timeout = tk.StringVar(value=str(self.base_demo_cfg.get("http_timeout_s", 3.0)))
        self.merge_mode = tk.StringVar(value=str(self.base_demo_cfg.get("merge_mode", "simple")))
        self.step_sleep = tk.StringVar(value=str(self.base_demo_cfg.get("step_sleep_s", 0.0)))
        default_loop_rate = 1.0 / max(float(self.base_env_cfg.get("dt", 0.5)), 1e-6)
        self.loop_rate = tk.StringVar(value=str(self.base_demo_cfg.get("loop_rate", default_loop_rate)))
        for i, (label, var) in enumerate(
            [
                ("server_url", self.merge_url),
                ("http_timeout_s", self.merge_timeout),
                ("merge_mode", self.merge_mode),
                ("loop_rate", self.loop_rate),
                ("step_sleep_s", self.step_sleep),
            ]
        ):
            ttk.Label(self.merge_tab, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(self.merge_tab, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", padx=6, pady=4)
        btn_bar = ttk.Frame(self.merge_tab)
        btn_bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 4))
        ttk.Button(btn_bar, text="Load", command=self._on_load_demo_config).pack(side="left")
        ttk.Button(btn_bar, text="Save", command=self._on_save_demo_config).pack(side="left", padx=(8, 0))
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
        btn_bar = ttk.Frame(self.global_tab)
        btn_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 4))
        ttk.Button(btn_bar, text="Load", command=self._on_load_global_config).pack(side="left")
        ttk.Button(btn_bar, text="Save", command=self._on_save_global_config).pack(side="left", padx=(8, 0))
        self.global_tab.columnconfigure(1, weight=1)

    def _build_sensor_tab(self) -> None:
        """构建 Sensors 参数页签，配置各模态感知范围尺寸。

        Args:
            None: 无输入参数。
        Returns:
            None: 无返回值，直接构建并绑定界面控件。
        """
        self.sensor_radar_max_range = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "RADAR", "max_range", 200.0)))
        self.sensor_if_forward_range = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "IF", "forward_range", 300.0)))
        self.sensor_if_width = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "IF", "width", 200.0)))
        self.sensor_rgb_forward_range = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "RGB", "forward_range", 320.0)))
        self.sensor_rgb_width = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "RGB", "width", 220.0)))
        self.sensor_elec_max_range = tk.StringVar(value=str(self._get_sensor_param(self.base_env_cfg, "ELEC", "max_range", 320.0)))

        sensor_rows = [
            ("RADAR.max_range", self.sensor_radar_max_range),
            ("IF.forward_range", self.sensor_if_forward_range),
            ("IF.width", self.sensor_if_width),
            ("RGB.forward_range", self.sensor_rgb_forward_range),
            ("RGB.width", self.sensor_rgb_width),
            ("ELEC.max_range", self.sensor_elec_max_range),
        ]
        for i, (label, var) in enumerate(sensor_rows):
            ttk.Label(self.sensor_tab, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(self.sensor_tab, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", padx=6, pady=4)

        btn_bar = ttk.Frame(self.sensor_tab)
        btn_bar.grid(row=len(sensor_rows), column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 4))
        ttk.Button(btn_bar, text="Load", command=self._on_load_env_config).pack(side="left")
        ttk.Button(btn_bar, text="Save", command=self._on_save_env_config).pack(side="left", padx=(8, 0))
        self.sensor_tab.columnconfigure(1, weight=1)

    def _get_sensor_param(self, env_cfg: Dict[str, Any], profile_name: str, key: str, default: float) -> float:
        """读取环境配置中指定模态传感器参数。

        Args:
            env_cfg (Dict[str, Any]): 环境配置字典。
            profile_name (str): 模态名称（如 RADAR/IF/RGB/ELEC）。
            key (str): 参数键名（如 max_range/forward_range/width）。
            default (float): 缺省值。
        Returns:
            float: 解析后的参数值。
        """
        profiles = env_cfg.get("uav_profiles", {})
        profile = profiles.get(profile_name, {})
        sensor_cfg = profile.get("sensor", {})
        params = sensor_cfg.get("params", {})
        return float(params.get(key, default))

    def _collect_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        env_cfg = dict(self.base_env_cfg)
        env_cfg["dt"] = float(self.env_dt.get())
        env_cfg["seed"] = int(self.env_seed.get())
        env_cfg["generation_seed"] = int(self.env_generation_seed.get())
        env_cfg["weather"] = str(self.env_weather.get())
        env_cfg["lighting"] = str(self.env_lighting.get())
        env_cfg["map_size"] = [float(self.env_map_w.get()), float(self.env_map_h.get())]
        env_cfg["target_count"] = int(self.env_target_count.get())
        env_cfg["uav_counts"] = {
            "RADAR": int(self.env_count_radar.get()),
            "IF": int(self.env_count_if.get()),
            "RGB": int(self.env_count_rgb.get()),
            "ELEC": int(self.env_count_elec.get()),
        }
        profiles = env_cfg.setdefault("uav_profiles", {})
        for sensor_name in ("RADAR", "IF", "RGB", "ELEC"):
            profile = profiles.setdefault(sensor_name, {})
            sensor_cfg = profile.setdefault("sensor", {})
            sensor_cfg.setdefault("params", {})
        profiles["RADAR"]["sensor"]["params"]["max_range"] = float(self.sensor_radar_max_range.get())
        profiles["IF"]["sensor"]["params"]["forward_range"] = float(self.sensor_if_forward_range.get())
        profiles["IF"]["sensor"]["params"]["width"] = float(self.sensor_if_width.get())
        profiles["RGB"]["sensor"]["params"]["forward_range"] = float(self.sensor_rgb_forward_range.get())
        profiles["RGB"]["sensor"]["params"]["width"] = float(self.sensor_rgb_width.get())
        profiles["ELEC"]["sensor"]["params"]["max_range"] = float(self.sensor_elec_max_range.get())
        env_cfg["class_correlation"] = self.class_correlation

        demo_cfg = dict(self.base_demo_cfg)
        demo_cfg["merger_server_url"] = str(self.merge_url.get())
        demo_cfg["http_timeout_s"] = float(self.merge_timeout.get())
        demo_cfg["merge_mode"] = str(self.merge_mode.get())
        demo_cfg["loop_rate"] = float(self.loop_rate.get())
        demo_cfg["step_sleep_s"] = float(self.step_sleep.get())
        demo_cfg["show_sensor_range"] = bool(self.show_sensor_var.get())

        global_cfg = dict(self.base_global_cfg)
        global_cfg["valid_observation_count"] = int(self.global_valid_obs.get())
        global_cfg["max_unseen_time"] = float(self.global_max_unseen.get())
        global_cfg["stale_observation_time"] = float(self.global_stale_obs.get())
        return env_cfg, demo_cfg, global_cfg

    def _select_config_to_load(self, initial_path: str) -> Optional[str]:
        """弹出文件选择器并返回待加载的配置路径。

        Args:
            initial_path (str): 对话框初始路径。
        Returns:
            Optional[str]: 用户选择的文件路径；取消时返回 None。
        """
        path = filedialog.askopenfilename(
            title="Load Config",
            initialfile=os.path.basename(initial_path),
            initialdir=os.path.dirname(initial_path) or ".",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        return path or None

    def _select_config_to_save(self, initial_path: str) -> Optional[str]:
        """弹出文件保存对话框并返回目标路径。

        Args:
            initial_path (str): 对话框初始路径。
        Returns:
            Optional[str]: 用户确认的保存路径；取消时返回 None。
        """
        path = filedialog.asksaveasfilename(
            title="Save Config",
            initialfile=os.path.basename(initial_path),
            initialdir=os.path.dirname(initial_path) or ".",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        return path or None

    def _save_json_to_path(self, path: str, payload: Dict[str, Any]) -> None:
        """将配置字典写入 JSON 文件。

        Args:
            path (str): 文件路径。
            payload (Dict[str, Any]): 要保存的配置内容。
        Returns:
            None: 无返回值，直接写文件。
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _on_load_env_config(self) -> None:
        """加载 Env 选项卡配置并回填参数输入框。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接更新界面状态。
        """
        path = self._select_config_to_load(self.env_config_path)
        if path is None:
            return
        cfg = _load_json(path)
        self.base_env_cfg = cfg
        self.env_config_path = path
        self._apply_env_config_to_vars(cfg)
        self._set_status(f"Env config loaded: {path}")

    def _on_save_env_config(self) -> None:
        """保存当前 Env 选项卡参数到 JSON 文件。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接写入配置文件。
        """
        path = self._select_config_to_save(self.env_config_path)
        if path is None:
            return
        env_cfg, _, _ = self._collect_configs()
        self._save_json_to_path(path, env_cfg)
        self.base_env_cfg = env_cfg
        self.env_config_path = path
        self._set_status(f"Env config saved: {path}")

    def _on_load_demo_config(self) -> None:
        """加载 Merge 选项卡配置并回填参数输入框。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接更新界面状态。
        """
        path = self._select_config_to_load(self.demo_config_path)
        if path is None:
            return
        cfg = _load_json(path)
        self.base_demo_cfg = cfg
        self.demo_config_path = path
        self._apply_demo_config_to_vars(cfg)
        self._set_status(f"Merge config loaded: {path}")

    def _on_save_demo_config(self) -> None:
        """保存当前 Merge 选项卡参数到 JSON 文件。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接写入配置文件。
        """
        path = self._select_config_to_save(self.demo_config_path)
        if path is None:
            return
        _, demo_cfg, _ = self._collect_configs()
        self._save_json_to_path(path, demo_cfg)
        self.base_demo_cfg = demo_cfg
        self.demo_config_path = path
        self._set_status(f"Merge config saved: {path}")

    def _on_load_global_config(self) -> None:
        """加载 GlobalState 选项卡配置并回填参数输入框。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接更新界面状态。
        """
        path = self._select_config_to_load(self.global_config_path)
        if path is None:
            return
        cfg = _load_json(path)
        self.base_global_cfg = cfg
        self.global_config_path = path
        self._apply_global_config_to_vars(cfg)
        self._set_status(f"Global config loaded: {path}")

    def _on_save_global_config(self) -> None:
        """保存当前 GlobalState 选项卡参数到 JSON 文件。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接写入配置文件。
        """
        path = self._select_config_to_save(self.global_config_path)
        if path is None:
            return
        _, _, global_cfg = self._collect_configs()
        self._save_json_to_path(path, global_cfg)
        self.base_global_cfg = global_cfg
        self.global_config_path = path
        self._set_status(f"Global config saved: {path}")

    def _apply_env_config_to_vars(self, env_cfg: Dict[str, Any]) -> None:
        """将 Env 配置字典写回界面输入变量。

        Args:
            env_cfg (Dict[str, Any]): 环境配置。
        Returns:
            None: 无返回值，直接更新变量。
        """
        counts = self._extract_uav_counts(env_cfg)
        self.env_dt.set(str(env_cfg.get("dt", 0.5)))
        self.env_seed.set(str(env_cfg.get("seed", 0)))
        self.env_generation_seed.set(str(env_cfg.get("generation_seed", env_cfg.get("seed", 0))))
        self.env_weather.set(str(env_cfg.get("weather", "clear")))
        self.env_lighting.set(str(env_cfg.get("lighting", "night")))
        self.env_map_w.set(str(env_cfg.get("map_size", [1200, 1200])[0]))
        self.env_map_h.set(str(env_cfg.get("map_size", [1200, 1200])[1]))
        self.env_target_count.set(str(env_cfg.get("target_count", len(env_cfg.get("targets", [])))))
        self.env_count_radar.set(str(counts["RADAR"]))
        self.env_count_if.set(str(counts["IF"]))
        self.env_count_rgb.set(str(counts["RGB"]))
        self.env_count_elec.set(str(counts["ELEC"]))
        if hasattr(self, "sensor_radar_max_range"):
            self.sensor_radar_max_range.set(str(self._get_sensor_param(env_cfg, "RADAR", "max_range", 200.0)))
            self.sensor_if_forward_range.set(str(self._get_sensor_param(env_cfg, "IF", "forward_range", 300.0)))
            self.sensor_if_width.set(str(self._get_sensor_param(env_cfg, "IF", "width", 200.0)))
            self.sensor_rgb_forward_range.set(str(self._get_sensor_param(env_cfg, "RGB", "forward_range", 320.0)))
            self.sensor_rgb_width.set(str(self._get_sensor_param(env_cfg, "RGB", "width", 220.0)))
            self.sensor_elec_max_range.set(str(self._get_sensor_param(env_cfg, "ELEC", "max_range", 320.0)))

    def _apply_demo_config_to_vars(self, demo_cfg: Dict[str, Any]) -> None:
        """将 Merge 配置字典写回界面输入变量。

        Args:
            demo_cfg (Dict[str, Any]): 融合流程配置。
        Returns:
            None: 无返回值，直接更新变量。
        """
        self.merge_url.set(str(demo_cfg.get("merger_server_url", "http://127.0.0.1:6801")))
        self.merge_timeout.set(str(demo_cfg.get("http_timeout_s", 3.0)))
        self.merge_mode.set(str(demo_cfg.get("merge_mode", "simple")))
        default_loop_rate = 1.0 / max(float(self.env_dt.get()), 1e-6)
        self.loop_rate.set(str(demo_cfg.get("loop_rate", default_loop_rate)))
        self.step_sleep.set(str(demo_cfg.get("step_sleep_s", 0.0)))
        self.show_sensor_var.set(bool(demo_cfg.get("show_sensor_range", True)))

    def _apply_global_config_to_vars(self, global_cfg: Dict[str, Any]) -> None:
        """将 GlobalState 配置字典写回界面输入变量。

        Args:
            global_cfg (Dict[str, Any]): 全局态势配置。
        Returns:
            None: 无返回值，直接更新变量。
        """
        self.global_valid_obs.set(str(global_cfg.get("valid_observation_count", 1)))
        self.global_max_unseen.set(str(global_cfg.get("max_unseen_time", 4.0)))
        self.global_stale_obs.set(str(global_cfg.get("stale_observation_time", 2.0)))

    def _build_global_pool_panel(self, parent: ttk.Frame) -> None:
        """在主界面中构建嵌入式全局态势池面板。

        Args:
            parent (ttk.Frame): 主界面左侧控制区父容器。
        Returns:
            None: 无返回值，内部构建面板和布局。
        """
        wrapper = ttk.LabelFrame(parent, text="Global Object Pool", padding=8)
        wrapper.pack(fill="both", expand=False, pady=(10, 0))

        header = ttk.Frame(wrapper)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Global Object Info Logger", font=("", 10, "bold")).pack(side="left")

        body = ttk.Frame(wrapper)
        body.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.pool_text_widget = tk.Text(
            body,
            wrap="none",
            height=12,
            bg="white",
            borderwidth=0,
            highlightthickness=0,
            font=("Courier New", 10),
        )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.pool_text_widget.yview)
        self.pool_text_widget.configure(yscrollcommand=scrollbar.set, state="disabled")
        self.pool_text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

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
        self.target_truth_history = {}
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
        """按模式执行单轮仿真与融合，并根据 loop_rate 调度下一轮。

        Args:
            None: 无输入参数。
        Returns:
            None: 无返回值，通过 Tk after 机制继续或停止循环。
        """
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
            self.root.after(self._compute_loop_interval_ms(), self._run_loop)
            return
        if mode == "in-loop":
            # 仅在本轮真实触发融合后暂停；否则继续推进环境直到有帧进入融合链路。
            if processed > 0:
                self.running = False
            else:
                self.root.after(self._compute_loop_interval_ms(), self._run_loop)
            return
        if mode == "step":
            self.running = False
            return

    def _compute_loop_interval_ms(self) -> int:
        """根据 loop_rate(HZ) 计算 run_loop 的真实时间调度间隔。

        Args:
            None: 无输入参数。
        Returns:
            int: 下一次循环调度间隔（毫秒），最小 20ms。
        """
        loop_rate_hz = max(1e-6, float(self.loop_rate.get()))
        return int(max(20.0, 1000.0 / loop_rate_hz))

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
        self._sync_figure_size_to_widget()
        self.ax.clear()
        self._refresh_global_pool_panel()
        if self.env is None:
            self._draw_idle()
            self.plot_canvas.draw_idle()
            return
        rs = self.env.get_render_state()
        self._update_target_truth_history(rs)
        map_w, map_h = self._extract_map_size()
        self._draw_axes(map_w, map_h)
        self._draw_runtime_info(rs)
        global_pos_by_id = {}

        for item in rs.get("global_objects", []):
            x, y = item["position"][0], item["position"][1]
            cx, cy = self._world_to_canvas(x, y, map_w, map_h)
            global_pos_by_id[int(item["global_id"])] = (cx, cy)
            self.ax.scatter([cx], [cy], c="#0077b6", s=36, marker="o", zorder=4)
            self.ax.text(cx + map_w * 0.008, cy + map_h * 0.008, f"G{item['global_id']}", color="#0077b6", fontsize=9, zorder=5)
            self._draw_velocity_arrow((x, y), item.get("velocity", [0, 0, 0]), map_w, map_h, "#0077b6")
            traj = item.get("trajectory", [])
            if len(traj) >= 2:
                xs = [float(p[0]) for p in traj[-40:]]
                ys = [float(p[1]) for p in traj[-40:]]
                self.ax.plot(xs, ys, color="#89c2d9", linewidth=1.0, zorder=2)
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
                        self.ax.scatter([ox], [oy], c="#48cae4", s=10, marker="o", zorder=3)

        if self.show_truth_var.get():
            if self.show_truth_traj_var.get():
                for target_id, hist in self.target_truth_history.items():
                    if len(hist) < 2:
                        continue
                    xs = [p[0] for p in hist]
                    ys = [p[1] for p in hist]
                    self.ax.plot(xs, ys, color="#ffb703", linewidth=1.2, alpha=0.9, zorder=2)
                    self.ax.text(xs[-1], ys[-1], f"Tr{target_id}", color="#ffb703", fontsize=7, zorder=3)
            for target in rs.get("targets_truth", []):
                tx, ty = self._world_to_canvas(target["position"][0], target["position"][1], map_w, map_h)
                self.ax.add_patch(Rectangle((tx - 5.0, ty - 5.0), 10.0, 10.0, fill=False, edgecolor="#f77f00", linewidth=1.6, zorder=4))
                self.ax.text(tx + map_w * 0.008, ty + map_h * 0.008, f"T{target['target_id']}", color="#f77f00", fontsize=9, zorder=5)
                self._draw_velocity_arrow(
                    (target["position"][0], target["position"][1]),
                    target.get("velocity", [0, 0, 0]),
                    map_w,
                    map_h,
                    "#f77f00",
                )

        for uav in rs.get("uavs", []):
            ux, uy = self._world_to_canvas(uav["position"][0], uav["position"][1], map_w, map_h)
            self.ax.add_patch(
                Polygon(
                    [[ux, uy + 7.0], [ux - 6.0, uy - 6.0], [ux + 6.0, uy - 6.0]],
                    closed=True,
                    facecolor="#2b9348",
                    edgecolor="none",
                    zorder=6,
                )
            )
            sensor_label = SENSOR_LABELS.get(int(uav.get("sensor_type", -1)), f"S{uav.get('sensor_type', '?')}")
            self.ax.text(ux + map_w * 0.008, uy + map_h * 0.008, f"U{uav['uav_id']} [{sensor_label}]", color="#2b9348", fontsize=9, zorder=7)
            if self.show_sensor_var.get():
                sensor_available = self._sensor_available_for_draw(
                    sensor_type=int(uav.get("sensor_type", -1)),
                    weather=str(rs.get("weather", "clear")),
                    lighting=str(rs.get("lighting", "day")),
                )
                self._draw_sensor_range(uav, map_w, map_h, sensor_available=sensor_available)

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
                    self.ax.plot([x1, x2], [y1, y2], color="#d00000", linestyle=(0, (4, 2)), linewidth=1.5, zorder=3)
                    score = edge.get("score")
                    if score is not None:
                        self.ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0, f"{score:.2f}", color="#d00000", fontsize=8, zorder=4)
                else:
                    self.ax.scatter([x1], [y1], c="#d00000", s=16, zorder=4)

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
        self.plot_canvas.draw_idle()

    def _update_target_truth_history(self, render_state: Dict[str, Any]) -> None:
        """维护任务目标真实轨迹历史，用于可选轨迹显示。

        Args:
            render_state (Dict[str, Any]): 环境渲染状态字典，需包含 `targets_truth` 列表。
        Returns:
            None: 无返回值，更新内部目标轨迹缓存。
        """
        targets = render_state.get("targets_truth", [])
        active_ids: set[int] = set()
        for target in targets:
            target_id = int(target.get("target_id", -1))
            pos = target.get("position")
            if target_id < 0 or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            active_ids.add(target_id)
            history = self.target_truth_history.setdefault(target_id, [])
            point = (float(pos[0]), float(pos[1]))
            if not history or history[-1] != point:
                history.append(point)
                if len(history) > self.target_traj_max_len:
                    del history[: len(history) - self.target_traj_max_len]

        stale_ids = [tid for tid in self.target_truth_history.keys() if tid not in active_ids]
        for tid in stale_ids:
            del self.target_truth_history[tid]

    def _draw_idle(self) -> None:
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "Click Initialize to start simulation", transform=self.ax.transAxes, ha="center", va="center", color="gray", fontsize=12)
        self.plot_canvas.draw_idle()

    def _draw_runtime_info(self, render_state: Dict[str, Any]) -> None:
        """在主画布左上角绘制时间、天气、光照状态。

        Args:
            render_state (Dict[str, Any]): 渲染状态字典。
        Returns:
            None: 无返回值，仅负责绘制文本。
        """
        current_time = float(render_state.get("time", 0.0))
        weather = str(render_state.get("weather", "unknown"))
        lighting = str(render_state.get("lighting", "unknown"))
        info_text = f"time={current_time:.2f}s  weather={weather}  lighting={lighting}"
        self.ax.text(0.015, 0.985, info_text, transform=self.ax.transAxes, ha="left", va="top", color="#495057", fontsize=10, fontweight="bold")

    def _refresh_global_pool_panel(self) -> None:
        """刷新嵌入式全局态势池面板内容。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接更新窗口控件。
        """
        if self.pool_text_widget is None:
            return
        text_lines: List[str] = []
        if self.global_info is None:
            text_lines.append("GlobalInfo 未初始化。")
        else:
            items = sorted(list(self.global_info.get_all_items()), key=lambda x: x.global_id)
            if not items:
                text_lines.append("当前目标记录池为空。")
            else:
                current_ts = float(self.global_info.current_timestamp)
                stale_time = max(float(self.global_info.stale_observation_time), 1e-6)
                text_lines.append(f"{'全局ID':<10}{'有效观测数量':<16}{'失效进度':<20}")
                text_lines.append("-" * 46)
                for item in items:
                    valid_obs_count = len(item.observations)
                    last_obs_ts = float(item.timestamp)
                    if item.observations:
                        last_obs_ts = max(float(obs.get('timestamp', item.timestamp)) for obs in item.observations)
                    age = max(0.0, current_ts - last_obs_ts)
                    text_lines.append(
                        f"{str(item.global_id):<10}{str(valid_obs_count):<16}{age:.2f}/{stale_time:.2f}"
                    )
        text_content = "\n".join(text_lines)
        if text_content == self._last_pool_text:
            return
        self._last_pool_text = text_content
        self.pool_text_widget.configure(state="normal")
        self.pool_text_widget.delete("1.0", tk.END)
        self.pool_text_widget.insert(tk.END, text_content)
        self.pool_text_widget.configure(state="disabled")

    def _extract_map_size(self) -> Tuple[float, float]:
        if self.env is None:
            return 1200.0, 1200.0
        size = self.env.config.get("map_size", [1200, 1200])
        return float(size[0]), float(size[1])

    def _world_to_canvas(self, x: float, y: float, map_w: float, map_h: float) -> Tuple[float, float]:
        """将世界坐标映射到绘图坐标（Matplotlib 下与世界坐标一致）。

        Args:
            x (float): 世界坐标系下的 x 坐标。
            y (float): 世界坐标系下的 y 坐标。
            map_w (float): 地图世界宽度。
            map_h (float): 地图世界高度。
        Returns:
            Tuple[float, float]: 对应画布坐标 (cx, cy)。
        """
        return float(x), float(y)

    def _draw_axes(self, map_w: float, map_h: float) -> None:
        """配置地图坐标轴样式，保持等比例显示。

        Args:
            map_w (float): 地图世界宽度。
            map_h (float): 地图世界高度。
        Returns:
            None: 无返回值，直接在画布绘制坐标边框与标签。
        """
        safe_map_w = max(map_w, 1.0)
        safe_map_h = max(map_h, 1.0)
        self._apply_centered_axes_layout(safe_map_w, safe_map_h, occupy_ratio=0.85)
        self.ax.set_xlim(0.0, safe_map_w)
        self.ax.set_ylim(0.0, safe_map_h)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_anchor("C")
        self.ax.grid(color="#ced4da", linestyle="--", linewidth=0.6, alpha=0.6)
        self.ax.set_facecolor("white")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

    def _apply_centered_axes_layout(self, map_w: float, map_h: float, occupy_ratio: float = 0.85) -> None:
        """按目标占比与地图宽高比计算居中绘图区域。

        Args:
            map_w (float): 地图世界宽度。
            map_h (float): 地图世界高度。
            occupy_ratio (float): 目标绘图区占 Figure 的比例（0~1）。
        Returns:
            None: 无返回值，直接更新 Axes 在 Figure 中的位置。
        """
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        fig_w_px = max(fig_w_in * self.fig.get_dpi(), 1.0)
        fig_h_px = max(fig_h_in * self.fig.get_dpi(), 1.0)
        map_ratio = max(map_w, 1e-6) / max(map_h, 1e-6)

        max_w = max(0.01, min(1.0, occupy_ratio))
        max_h = max(0.01, min(1.0, occupy_ratio))
        ax_w = max_w
        ax_h = ax_w * (fig_w_px / fig_h_px) / map_ratio
        if ax_h > max_h:
            ax_h = max_h
            ax_w = ax_h * map_ratio * (fig_h_px / fig_w_px)

        left = (1.0 - ax_w) / 2.0
        bottom = (1.0 - ax_h) / 2.0
        self.ax.set_position([left, bottom, ax_w, ax_h])

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
        self.ax.annotate(
            "",
            xy=(cx2, cy2),
            xytext=(cx1, cy1),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            zorder=4,
        )

    def _sensor_available_for_draw(self, sensor_type: int, weather: str, lighting: str) -> bool:
        """在当前天气和光照条件下评估传感器模态是否可用。

        Args:
            sensor_type (int): 传感器模态编号。
            weather (str): 当前天气状态。
            lighting (str): 当前光照状态。
        Returns:
            bool: True 表示可用，False 表示当前条件下失效。
        """
        weather_l = str(weather).lower()
        lighting_l = str(lighting).lower()
        if sensor_type == SensorType.RGB.value:
            return weather_l not in {"rain", "snow"} and lighting_l != "night"
        if sensor_type == SensorType.IF.value:
            return lighting_l != "day"
        return True

    def _draw_sensor_range(self, uav: Dict[str, Any], map_w: float, map_h: float, sensor_available: bool) -> None:
        """绘制单个 UAV 的传感器感知范围（Matplotlib）。

        Args:
            uav (Dict[str, Any]): UAV 状态字典，需包含位置、传感器类型与参数。
            map_w (float): 地图世界坐标宽度。
            map_h (float): 地图世界坐标高度。
            sensor_available (bool): 当前环境下该传感器是否可用。
        Returns:
            None: 无返回值，直接在画布上绘制。
        """
        sensor_type = int(uav.get("sensor_type", 2))
        ux, uy = float(uav["position"][0]), float(uav["position"][1])
        params = uav.get("sensor_params", {})
        yaw_deg = float(uav.get("yaw_deg", 0.0))
        sensor_color = SENSOR_RANGE_COLORS.get(sensor_type, "#457b9d")
        outline_color = sensor_color if sensor_available else "#6c757d"
        dash_style = (4, 3) if not sensor_available else None

        if sensor_type in (0, 3):
            radius = float(params.get("max_range", 200.0))
            if sensor_available:
                patch = Circle((ux, uy), radius=radius, facecolor=sensor_color, edgecolor=outline_color, linewidth=1.5, alpha=0.2, zorder=1)
            else:
                patch = Circle((ux, uy), radius=radius, facecolor="none", edgecolor=outline_color, linewidth=1.5, linestyle=(0, dash_style), zorder=1)
            self.ax.add_patch(patch)
            return

        forward = float(params.get("forward_range", 260.0))
        width = float(params.get("width", 180.0))
        corners = [(0.0, -width / 2), (forward, -width / 2), (forward, width / 2), (0.0, width / 2)]
        yaw = math.radians(yaw_deg)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        pts: List[List[float]] = []
        for fx, fy in corners:
            wx = ux + fx * cos_yaw - fy * sin_yaw
            wy = uy + fx * sin_yaw + fy * cos_yaw
            cx, cy = self._world_to_canvas(wx, wy, map_w, map_h)
            pts.append([cx, cy])
        if sensor_available:
            patch = Polygon(pts, closed=True, facecolor=sensor_color, edgecolor=outline_color, linewidth=1.5, alpha=0.2, zorder=1)
        else:
            patch = Polygon(pts, closed=True, facecolor="none", edgecolor=outline_color, linewidth=1.5, linestyle=(0, dash_style), zorder=1)
        self.ax.add_patch(patch)

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
        self.ax.plot([cx0, cx1], [cy0, cy1], color=color, linestyle=(0, (3, 2)), linewidth=1.0, zorder=2)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="config/swarm_env.json")
    parser.add_argument("--demo-config", default="config/swarm_demo.json")
    parser.add_argument("--global-config", default="config/global_info.json")
    parser.add_argument("--class-correlation", default="config/class_correlation.json")
    args = parser.parse_args()

    env_cfg = _load_json(args.env_config)
    demo_cfg = _load_json(args.demo_config)
    global_cfg = _load_json(args.global_config)
    class_corr = _load_json(args.class_correlation)
    ui = DemoUI(
        env_cfg=env_cfg,
        demo_cfg=demo_cfg,
        global_cfg=global_cfg,
        class_correlation=class_corr,
        env_config_path=args.env_config,
        demo_config_path=args.demo_config,
        global_config_path=args.global_config,
    )
    ui.run()


if __name__ == "__main__":
    main()
