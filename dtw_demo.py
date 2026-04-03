from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from utils.DTW import dtw_distance

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("matplotlib is required for dtw_demo.py") from exc

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
MAP_MIN = -100.0
MAP_MAX = 100.0


class TrajectoryDrawer:
    """手动绘制两条轨迹并触发 DTW 计算的交互控制器。"""

    def __init__(self) -> None:
        """初始化交互画布、轨迹缓存和事件绑定。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，直接创建图形对象。
        """
        self.tracks_2d: List[List[Point2D]] = [[], []]
        self.active_index = 0
        self.finished = False
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.set_title(
            "Draw Trajectory A (Left Click). Press 'n' for Trajectory B, Enter to finish."
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(MAP_MIN, MAP_MAX)
        self.ax.set_ylim(MAP_MIN, MAP_MAX)
        self.ax.grid(alpha=0.25)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()

    def _on_click(self, event: object) -> None:
        """响应鼠标点击，向当前轨迹追加一个点。

        Args:
            event (object): matplotlib 点击事件对象。
        Returns:
            None: 无返回值，更新轨迹并重绘。
        """
        if self.finished:
            return
        x = getattr(event, "xdata", None)
        y = getattr(event, "ydata", None)
        button = getattr(event, "button", None)
        if x is None or y is None or button != 1:
            return
        self.tracks_2d[self.active_index].append((float(x), float(y)))
        self._redraw()

    def _on_key(self, event: object) -> None:
        """响应键盘控制轨迹绘制流程。

        Args:
            event (object): matplotlib 键盘事件对象。
        Returns:
            None: 无返回值，按键操作后更新界面状态。
        """
        key = str(getattr(event, "key", "")).lower()
        if key == "n":
            self.active_index = min(1, self.active_index + 1)
            self._redraw()
            return
        if key == "u":
            if self.tracks_2d[self.active_index]:
                self.tracks_2d[self.active_index].pop()
                self._redraw()
            return
        if key == "c":
            self.tracks_2d[self.active_index] = []
            self._redraw()
            return
        if key in {"enter", "return"}:
            if len(self.tracks_2d[0]) >= 2 and len(self.tracks_2d[1]) >= 2:
                self.finished = True
                plt.close(self.fig)
            return
        if key == "q":
            self.finished = False
            plt.close(self.fig)

    def _redraw(self) -> None:
        """重绘当前轨迹和交互提示文本。

        Args:
            None: 不需要输入参数。
        Returns:
            None: 无返回值，刷新画布显示。
        """
        self.ax.clear()
        self.ax.grid(alpha=0.25)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(MAP_MIN, MAP_MAX)
        self.ax.set_ylim(MAP_MIN, MAP_MAX)
        track_a = self.tracks_2d[0]
        track_b = self.tracks_2d[1]
        if track_a:
            xa = [p[0] for p in track_a]
            ya = [p[1] for p in track_a]
            self.ax.plot(xa, ya, "o-", color="#1f77b4", label=f"Trajectory A ({len(track_a)} pts)")
        if track_b:
            xb = [p[0] for p in track_b]
            yb = [p[1] for p in track_b]
            self.ax.plot(xb, yb, "o-", color="#d62728", label=f"Trajectory B ({len(track_b)} pts)")

        active_name = "A" if self.active_index == 0 else "B"
        self.ax.set_title(
            f"Active: {active_name} | LeftClick:add  n:next  u:undo  c:clear  Enter:finish  q:quit"
        )
        if track_a or track_b:
            self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    def run(self) -> Tuple[bool, List[Point2D], List[Point2D]]:
        """进入交互式绘制流程并返回两条轨迹。

        Args:
            None: 不需要输入参数。
        Returns:
            Tuple[bool, List[Point2D], List[Point2D]]:
            是否成功完成绘制、轨迹A点集、轨迹B点集。
        """
        plt.show()
        return self.finished, self.tracks_2d[0], self.tracks_2d[1]


def _to_3d(points: Sequence[Point2D]) -> List[Point3D]:
    """将二维轨迹点转换为三维轨迹点（z=0）。

    Args:
        points (Sequence[Point2D]): 二维点序列。
    Returns:
        List[Point3D]: 三维点序列。
    """
    return [(float(p[0]), float(p[1]), 0.0) for p in points]


def _similarity_from_dtw(distance: float) -> float:
    """将 DTW 距离映射为 0~1 相似度分数（越大越相似）。

    Args:
        distance (float): DTW 距离。
    Returns:
        float: 相似度分值。
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def _plot_result(track_a: Sequence[Point2D], track_b: Sequence[Point2D], dtw_dist: float, sim: float) -> bool:
    """绘制最终轨迹对比图并标注 DTW 结果，支持重置继续绘制。

    Args:
        track_a (Sequence[Point2D]): 轨迹A。
        track_b (Sequence[Point2D]): 轨迹B。
        dtw_dist (float): DTW 距离。
        sim (float): 相似度分数。
    Returns:
        bool: True 表示用户选择重新初始化，False 表示退出。
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    xa = [p[0] for p in track_a]
    ya = [p[1] for p in track_a]
    xb = [p[0] for p in track_b]
    yb = [p[1] for p in track_b]
    ax.plot(xa, ya, "o-", color="#1f77b4", label="Trajectory A")
    ax.plot(xb, yb, "o-", color="#d62728", label="Trajectory B")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(MAP_MIN, MAP_MAX)
    ax.set_ylim(MAP_MIN, MAP_MAX)
    ax.set_title(f"DTW Distance={dtw_dist:.4f} | Similarity={sim:.4f} | Press r:restart q:quit")

    decision = {"restart": False}

    def _on_key(event: object) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "r":
            decision["restart"] = True
            plt.close(fig)
        elif key == "q":
            decision["restart"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return bool(decision["restart"])


def main() -> None:
    """运行手动绘制 DTW demo 的入口函数。

    Args:
        None: 不需要输入参数。
    Returns:
        None: 无返回值，执行交互和输出结果。
    """
    parser = argparse.ArgumentParser(description="Manual DTW trajectory demo")
    parser.parse_args()

    while True:
        drawer = TrajectoryDrawer()
        ok, track_a_2d, track_b_2d = drawer.run()
        if not ok:
            print("Cancelled. No DTW result is computed.")
            return

        track_a_3d = _to_3d(track_a_2d)
        track_b_3d = _to_3d(track_b_2d)
        dist = dtw_distance(track_a_3d, track_b_3d)
        sim = _similarity_from_dtw(dist)
        print(f"DTW distance: {dist:.6f}")
        print(f"Similarity  : {sim:.6f}")
        restart = _plot_result(track_a_2d, track_b_2d, dist, sim)
        if not restart:
            return


if __name__ == "__main__":
    main()
