from __future__ import annotations

import math
from typing import Sequence, Tuple

Vector3 = Tuple[float, float, float]


def dtw_distance(seq_a: Sequence[Vector3], seq_b: Sequence[Vector3]) -> float:
    """计算两条三维轨迹序列的 DTW 距离。

    Args:
        seq_a (Sequence[Vector3]): 第一条轨迹序列。
        seq_b (Sequence[Vector3]): 第二条轨迹序列。
    Returns:
        float: DTW 累积距离。若任一序列为空，返回无穷大。
    """
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _euclidean(seq_a[i - 1], seq_b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def _euclidean(a: Vector3, b: Vector3) -> float:
    """计算三维点之间欧式距离。

    Args:
        a (Vector3): 点 a。
        b (Vector3): 点 b。
    Returns:
        float: 欧式距离。
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

