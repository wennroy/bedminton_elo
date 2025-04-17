"""
DoublesScheduler 模块：

需求总结：
1. 给定 n 名选手及其标准化后的 Elo 分数，安排 m 场双打比赛。
2. 目标：保证每人上场期望为 4m/n，同时最大程度平衡队伍实力差异。
3. 通过决策变量 p_j 表示第 j 种配置出现的概率：
   - 线性项：最小化 \sum_j c_j p_j，其中 c_j 为该配置的队伍 Elo 差成本。
   - 二次正则：加入 \alpha_{orig} \sum_j p_j^2，防止某些配置过度集中。
4. 为便于选取 \alpha 在 [0,1] 范围，可归一化：
   \[ \alpha' = \frac{\alpha_{orig}}{\bar c \, M}, \quad \alpha_{orig} = \alpha' (\bar c \, M). \]
   其中 \bar c = \frac1M\sum_j c_j，M 为配置总数。

模型数学形式：
\begin{aligned}
&\min_{p\in\mathbb{R}^M} \sum_j c_j p_j + \alpha_{orig} \sum_j p_j^2,\\
&\text{s.t. } \sum_j p_j = 1, \quad \sum_j A_{i,j} p_j = \frac{4}{n},\ i=1,\dots,n, \quad p_j\ge0.
\end{aligned}
"""

import itertools
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt
from collections import Counter


def generate_double_configurations(players: List[str]) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    枚举所有 4 人双打配置，并分成两个队。
    返回列表：[((p1,p2),(p3,p4)), ...]
    """
    configs = []
    for quartet in itertools.combinations(players, 4):
        a, b, c, d = quartet
        pairings = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
        configs.extend(pairings)
    return configs


def standardized_elo(players_dict: Dict[str, float]) -> Dict[str, float]:
    if not players_dict:
        return {}
    values = np.array(list(players_dict.values()))
    mean = values.mean()
    std_dev = values.std()

    if std_dev == 0:
        standardized = np.zeros_like(values)
    else:
        standardized = (values - mean) / std_dev

    return {k: v for k, v in zip(players_dict.keys(), standardized)}


class DoublesScheduler:
    def __init__(self,
                 player_elo: Dict[str, float],
                 total_games: int,
                 use_squared: bool = False):
        self.player_elo = player_elo
        self.players = list(player_elo.keys())
        self.n = len(self.players)
        self.m = total_games
        self.use_squared = use_squared

        self.configs = generate_double_configurations(self.players)
        self.M = len(self.configs)
        self.costs = np.array(self._compute_costs())
        self.A = self._build_incidence()
        self.probs: Optional[np.ndarray] = None

    def _compute_costs(self) -> List[float]:
        costs = []
        for team1, team2 in self.configs:
            e1 = sum(self.player_elo[p] for p in team1)
            e2 = sum(self.player_elo[p] for p in team2)
            diff = e1 - e2
            costs.append(diff*diff if self.use_squared else abs(diff))
        return costs

    def _build_incidence(self) -> np.ndarray:
        A = np.zeros((self.n, self.M))
        for j, (t1, t2) in enumerate(self.configs):
            for p in (*t1, *t2):
                i = self.players.index(p)
                A[i, j] = 1
        return A

    def solve(self, alpha_orig: float = 0.0) -> np.ndarray:
        """
        alpha_orig: 原始正则参数 (二次项系数)
        返回长度 M 的概率向量 p。
        """
        # 构造等式约束 A_eq p = b_eq
        b_players = np.full(self.n, 4/self.n)
        A_eq = np.vstack([self.A, np.ones(self.M)])
        b_eq = np.concatenate([b_players, [1.0]])

        if alpha_orig == 0.0:
            res = linprog(c=self.costs, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(0,1)]*self.M, method='highs')
            if not res.success:
                raise RuntimeError("LP 失败: " + res.message)
            p = res.x
        else:
            def obj(p): return self.costs.dot(p) + alpha_orig * p.dot(p)
            constraints = [{'type':'eq', 'fun': lambda p, Arow=A_eq[i], be=b_eq[i]: Arow.dot(p)-be}
                           for i in range(A_eq.shape[0])]
            x0 = np.full(self.M, 1/self.M)
            res = minimize(obj, x0, bounds=[(0,1)]*self.M,
                           constraints=constraints, method='SLSQP',
                           options={'maxiter':1000, 'ftol':1e-9})
            if not res.success:
                raise RuntimeError("QP 失败: " + res.message)
            p = res.x
        self.probs = p
        return p

    def sample_schedule(self) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if self.probs is None:
            raise ValueError("请先调用 solve() 获取概率分布")
        return random.choices(self.configs, weights=self.probs, k=self.m)

    def visualize_alpha_effect(self, alpha_primes: List[float]):
        c_bar = np.mean(self.costs)
        M = self.M

        max_probs, entropies = [], []
        alpha_orig_values = []

        for alpha_prime in alpha_primes:
            alpha_orig = alpha_prime * c_bar * M
            try:
                p = self.solve(alpha_orig)
                max_probs.append(np.max(p))
                entropies.append(-np.sum(p * np.log(p + 1e-12)))
                alpha_orig_values.append(alpha_orig)
            except RuntimeError:
                max_probs.append(np.nan)
                entropies.append(np.nan)
                alpha_orig_values.append(alpha_orig)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(alpha_primes, max_probs, label="Max p_j")
        plt.xscale("log")
        plt.xlabel("alpha'")
        plt.ylabel("Max probability")
        plt.title("Max p_j vs alpha'")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(alpha_primes, entropies, label="Entropy")
        plt.xscale("log")
        plt.xlabel("alpha'")
        plt.ylabel("Entropy")
        plt.title("Entropy vs alpha'")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_player_counts(self, schedule: List[Tuple[Tuple[str, str], Tuple[str, str]]]):
        count = Counter()
        for team1, team2 in schedule:
            for p in team1 + team2:
                count[p] += 1

        ordered_players = sorted(self.players)
        counts = [count[p] for p in ordered_players]

        plt.figure(figsize=(8, 4))
        plt.bar(ordered_players, counts, color='skyblue')
        plt.axhline(y=4 * self.m / self.n, color='red', linestyle='--', label='Theoretical Expectation')
        plt.title("Player Participation Count")
        plt.xlabel("Player")
        plt.ylabel("Number of Games Played")
        plt.legend()
        plt.tight_layout()
        plt.show()

# --- 测试与示例 ---
if __name__ == "__main__":
    player_elo = {'1': -0.926, '2': -0.799, '3': 1.641,
                  '4': 0.078, '5': 0.953, '6': -0.946}
    total_games = 20

    sched = DoublesScheduler(player_elo, total_games)
    # 计算归一化常量
    c_bar = np.mean(sched.costs)
    M = sched.M
    # 设定归一化参数 alpha' = 0.1，映射到原始 alpha
    alpha_prime = 0.1
    alpha_orig = alpha_prime * c_bar * M

    print(f"归一化 alpha'={alpha_prime}, 对应 alpha_orig={alpha_orig:.4f}")

    # 求解并采样
    p = sched.solve(alpha_orig)
    print(f"分布 p_j 最大值: {np.max(p):.4f}, 熵: {-np.sum(p*np.log(p+1e-12)):.4f}")
    schedule = sched.sample_schedule()

    print("生成的对局安排:")
    for idx, match in enumerate(schedule, 1):
        print(f"Game {idx}: Team1 {match[0]} vs Team2 {match[1]}")

    alpha_grid = np.geomspace(1e-4, 1e2, 3000)  # 更细的对数间隔
    sched.visualize_alpha_effect(alpha_grid)

    sched.visualize_player_counts(schedule)