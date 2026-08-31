import itertools
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linprog, minimize


def generate_double_configurations(players: List[str]) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
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
                 total_games: int):
        self.player_elo = player_elo
        self.players = list(player_elo.keys())
        self.n = len(self.players)
        self.m = total_games

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
            costs.append(abs(diff))
        return costs

    def _build_incidence(self) -> np.ndarray:
        A = np.zeros((self.n, self.M))
        for j, (t1, t2) in enumerate(self.configs):
            for p in (*t1, *t2):
                i = self.players.index(p)
                A[i, j] = 1
        return A

    def solve(self, lambda_weight: float = 0.0) -> np.ndarray:
        b_players = np.full(self.n, 4 / self.n)
        A_eq = np.vstack([self.A, np.ones(self.M)])
        b_eq = np.concatenate([b_players, [1.0]])

        if lambda_weight == 0.0:
            res = linprog(c=self.costs, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(0, 1)] * self.M, method='highs')
            if not res.success:
                raise RuntimeError("LP 失败: " + res.message)
            p = res.x
        else:
            def obj(p):
                cost_term = (1 - lambda_weight) * self.costs.dot(p)
                q = 1.5
                tsallis = (1 - np.sum(p ** q)) / (q - 1)
                reg_term = lambda_weight * tsallis
                return cost_term + reg_term

            constraints = [{'type': 'eq', 'fun': lambda p, Arow=A_eq[i], be=b_eq[i]: Arow.dot(p) - be}
                           for i in range(A_eq.shape[0])]
            x0 = np.full(self.M, 1 / self.M)
            res = minimize(obj, x0, bounds=[(0, 1)] * self.M,
                           constraints=constraints, method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-9})
            if not res.success:
                raise RuntimeError("QP 失败: " + res.message)
            p = res.x

        self.probs = p
        return p

    def sample_schedule(self, seed: Optional[int] = None) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if self.probs is None:
            raise ValueError("请先调用 solve() 获取概率分布")
        if seed is not None:
            random.seed(seed)
        return random.choices(self.configs, weights=self.probs, k=self.m)


# --- 测试与可视分析 ---
if __name__ == "__main__":
    player_elo = {'1': -0.926, '2': -0.799, '3': 1.641,
                  '4': 0.078, '5': 0.953, '6': -0.946}
    player_elo = {'1': -0.926, '2': -0.799, '3': 1.641,
                  '4': 0.078, '5': 0.953, '6': -0.946, '7': 0.3, '8': -0.38, '9': 0.358}
    player_elo = standardized_elo(player_elo)
    total_games = 12

    lambdas = np.linspace(0, 1, 100)
    max_ps, entropies, costs = [], [], []

    # 打印一个 lambda=0.3 的结果
    print("\n最终对局安排 (lambda=0.3):")
    final_sched = DoublesScheduler(player_elo, total_games)
    final_sched.solve(lambda_weight=0.3)
    schedule = final_sched.sample_schedule()
    for idx, match in enumerate(schedule, 1):
        print(f"Game {idx}: Team1 {match[0]} vs Team2 {match[1]}")

    import matplotlib.pyplot as plt

    for lam in lambdas:
        sched = DoublesScheduler(player_elo, total_games)
        p = sched.solve(lambda_weight=lam)
        max_ps.append(np.max(p))
        entropies.append(-np.sum(p * np.log(p + 1e-12)))
        costs.append(sched.costs.dot(p))

    print("lambda\tmax_p\tentropy\tcost")
    for i in range(len(lambdas)):
        print(f"{lambdas[i]:.2f}\t{max_ps[i]:.4f}\t{entropies[i]:.4f}\t{costs[i]:.4f}")
    # 绘图：max_p, entropy, cost vs lambda
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(lambdas, max_ps, marker='o', color='blue')
    axs[0].set_ylabel("max(p_j)")
    axs[0].set_title("max_p vs lambda")

    axs[1].plot(lambdas, entropies, marker='o', color='green')
    axs[1].set_ylabel("Entropy")
    axs[1].set_title("Entropy vs lambda")

    axs[2].plot(lambdas, costs, marker='o', color='red')
    axs[2].set_ylabel("Elo")
    axs[2].set_title("ELO vs lambda")
    axs[2].set_xlabel("lambda")

    plt.tight_layout()
    plt.show()
