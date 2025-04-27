import math
from scipy.special import erfinv

# 定义一个 Player 类表示一个玩家的评分状态
class Player:
    def __init__(self, mu=25.0, sigma=8.333):
        """
        初始化玩家评分，mu表示均值，sigma表示不确定性。
        """
        self.mu = mu
        self.sigma = sigma

    def __str__(self):
        return f"Rating(mu={self.mu:.3f}, sigma={self.sigma:.3f})"


# 定义一个 TrueSkill 系统类
class TrueSkill:
    def __init__(self, mu=25.0, sigma=8.333, beta=None, tau=0.0, draw_probability=0.0):
        """
        mu: 初始均值
        sigma: 初始不确定性
        beta: 表现波动参数，若未指定默认为 sigma/2
        tau: 可用于控制评级系统随时间的漂移（此示例中未作详细实现）
        draw_probability: 平局的概率（此示例中用于预测胜平负概率）
        """
        self.mu = mu
        self.sigma = sigma
        self.beta = beta if beta is not None else sigma / 2.0
        self.tau = tau
        self.draw_probability = draw_probability

    # 标准正态分布的概率密度函数
    def pdf(self, x):
        return math.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)

    # 标准正态分布的累积分布函数
    def cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # 标准正态分布的逆累积分布函数（quantile函数）
    def phi_inv(self, p):
        # 利用反误差函数erfinv (Python3.8+ 支持)
        return math.sqrt(2) * erfinv(2 * p - 1)

    def rate_1v1(self, winner, loser):
        """
        进行一场 1v1 比赛的评分更新。
        参数:
          winner: 获胜方 Player 实例
          loser:  失利方 Player 实例
        使用简化的更新公式：
          c = sqrt(2 * beta^2 + sigma_winner^2 + sigma_loser^2)
          t = (mu_winner - mu_loser) / c
          定义辅助函数：
              v(t) = pdf(t) / cdf(t)
              w(t) = v(t) * (v(t) + t)
          更新公式:
              mu_new = mu + (sigma^2 / c) * v(t)
              sigma_new = sigma * sqrt(1 - (sigma^2 / c^2) * w(t))
          对于胜者采用正向更新，失利者取相反方向。
        """
        # 计算综合不确定性
        c = math.sqrt(2 * self.beta ** 2 + winner.sigma ** 2 + loser.sigma ** 2)
        t = (winner.mu - loser.mu) / c

        # 为避免除 0，这里做下小值判断
        cdf_t = self.cdf(t)
        if cdf_t < 1e-10:
            v = 0.0
        else:
            v = self.pdf(t) / cdf_t
        w = v * (v + t)

        # 更新获胜方
        new_winner_mu = winner.mu + (winner.sigma ** 2 / c) * v
        new_winner_sigma = winner.sigma * math.sqrt(max(1 - (winner.sigma ** 2 / c ** 2) * w, 1e-10))

        # 更新失利方
        new_loser_mu = loser.mu - (loser.sigma ** 2 / c) * v
        new_loser_sigma = loser.sigma * math.sqrt(max(1 - (loser.sigma ** 2 / c ** 2) * w, 1e-10))

        # 写回更新结果
        winner.mu, winner.sigma = new_winner_mu, new_winner_sigma
        loser.mu, loser.sigma = new_loser_mu, new_loser_sigma

        return winner, loser

    def rate_team(self, teamA, teamB, result):
        """
        进行团队（1v1 或 2v2）比赛的评分更新。
        参数:
          teamA: list，队伍 A 中的 Player 实例列表
          teamB: list，队伍 B 中的 Player 实例列表
          result: 比赛结果：
                   1  —— 表示队伍 A 获胜，
                  -1  —— 表示队伍 B 获胜，
                   0  —— 表示平局（此示例中暂不更新）。
        更新思路：
          ① 计算两队的“团队均值”和“团队不确定性”，其中：
              team_mu = ∑(player.mu)
              team_sigma_sq = ∑(player.sigma^2) + n * beta^2  （n为队内球员数）
          ② 定义：
              c = sqrt(teamA_sigma_sq + teamB_sigma_sq)
              t = (teamA_mu - teamB_mu) / c
          ③ 根据获胜或失利计算 v, w，此处平局则不更新
          ④ 将更新量按照每位球员的比例 (player.sigma^2 + beta^2)/c 分摊。
        注意：此实现为简化模型，并非完整的 TrueSkill 消息传递算法。
        """
        # 计算队伍 A 的整体参数
        teamA_mu = sum(p.mu for p in teamA)
        teamA_sigma_sq = sum(p.sigma ** 2 for p in teamA) + len(teamA) * self.beta ** 2

        # 计算队伍 B 的整体参数
        teamB_mu = sum(p.mu for p in teamB)
        teamB_sigma_sq = sum(p.sigma ** 2 for p in teamB) + len(teamB) * self.beta ** 2

        c = math.sqrt(teamA_sigma_sq + teamB_sigma_sq)
        t = (teamA_mu - teamB_mu) / c

        if result == 1:  # 队伍 A 获胜
            cdf_t = self.cdf(t)
            v = self.pdf(t) / cdf_t if cdf_t > 1e-10 else 0.0
            w = v * (v + t)
            teamA_update = v
            teamB_update = -v
        elif result == -1:  # 队伍 B 获胜
            cdf_t_neg = self.cdf(-t)
            v = self.pdf(-t) / cdf_t_neg if cdf_t_neg > 1e-10 else 0.0
            w = v * (v - t)
            teamA_update = -v
            teamB_update = v
        else:
            # 平局暂时不做更新（实际系统中可能会有更精细的处理）
            return teamA, teamB

        # 更新队伍 A 中的每个玩家
        for p in teamA:
            factor = (p.sigma ** 2 + self.beta ** 2) / c
            p.mu = p.mu + factor * teamA_update
            p.sigma = p.sigma * math.sqrt(max(1 - ((p.sigma ** 2 + self.beta ** 2) / c ** 2) * w, 1e-10))

        # 更新队伍 B 中的每个玩家
        for p in teamB:
            factor = (p.sigma ** 2 + self.beta ** 2) / c
            p.mu = p.mu + factor * teamB_update
            p.sigma = p.sigma * math.sqrt(max(1 - ((p.sigma ** 2 + self.beta ** 2) / c ** 2) * w, 1e-10))

        return teamA, teamB

    def predict_team_outcome(self, teamA, teamB):
        """
        计算队伍A（相对于队伍B）的预测获胜、平局、失败的概率。
        步骤如下：
          ① 对于每个队伍，计算团队均值和团队不确定性：
                team_mu = ∑(player.mu)
                team_sigma_sq = ∑(player.sigma^2) + n * beta^2
          ② 比赛表现差 d 服从：
                d ~ N(Δμ, Δσ²)
                Δμ = teamA_mu - teamB_mu
                Δσ² = teamA_sigma_sq + teamB_sigma_sq
          ③ 若 draw_probability > 0，由 draw_probability 得到平局边界：
                draw_margin = norm.ppf((draw_probability+1)/2) * Δσ
          ④ 则：
                p_win  = Φ((Δμ - draw_margin)/Δσ)
                p_draw = Φ((Δμ + draw_margin)/Δσ) - Φ((Δμ - draw_margin)/Δσ)
                p_loss = 1 - Φ((Δμ + draw_margin)/Δσ)
          ⑤ 如果 draw_probability == 0，则 draw_margin = 0，且 p_draw = 0。
        返回一个字典：{'win': ..., 'draw': ..., 'loss': ...}
        """
        # 计算队伍 A 的整体参数
        teamA_mu = sum(p.mu for p in teamA)
        teamA_sigma_sq = sum(p.sigma ** 2 for p in teamA) + len(teamA) * self.beta ** 2

        # 计算队伍 B 的整体参数
        teamB_mu = sum(p.mu for p in teamB)
        teamB_sigma_sq = sum(p.sigma ** 2 for p in teamB) + len(teamB) * self.beta ** 2

        delta_mu = teamA_mu - teamB_mu
        delta_sigma_sq = teamA_sigma_sq + teamB_sigma_sq
        delta_sigma = math.sqrt(delta_sigma_sq)

        # 计算平局边界 draw_margin
        if self.draw_probability > 0:
            # norm.ppf((draw_probability+1)/2) = phi_inv((draw_probability+1)/2)
            draw_margin = self.phi_inv((self.draw_probability + 1) / 2.0) * delta_sigma
        else:
            draw_margin = 0.0

        # 计算各概率
        # p_win = P(d > draw_margin)
        p_win = self.cdf((delta_mu - draw_margin) / delta_sigma)
        # p_loss = P(d < -draw_margin)
        p_loss = 1 - self.cdf((delta_mu + draw_margin) / delta_sigma)
        # p_draw = 1 - p_win - p_loss
        p_draw = max(0.0, 1 - p_win - p_loss)

        return {'win': p_win, 'draw': p_draw, 'loss': p_loss}

    def predict_team_outcome_win(self, teamA, teamB):
        return self.predict_team_outcome(teamA, teamB)["win"]


# 示例：运行一些简单测试
if __name__ == "__main__":
    # 1v1 的例子
    player1 = Player()
    player2 = Player()
    ts = TrueSkill(draw_probability=0.1)  # 设置平局概率为 10%
    print("1v1 比赛前：")
    print("Player1:", player1)
    print("Player2:", player2)
    # 假如 player1 获胜
    ts.rate_1v1(winner=player1, loser=player2)
    print("\n1v1 比赛后（player1 获胜）：")
    print("Player1:", player1)
    print("Player2:", player2)

    # 2v2 的例子
    teamA = [Player(), Player()]
    teamB = [Player(), Player()]
    print("\n2v2 比赛前：")
    print("TeamA:", [str(p) for p in teamA])
    print("TeamB:", [str(p) for p in teamB])
    # 计算比赛 outcome 概率
    probs = ts.predict_team_outcome(teamA, teamB)
    print("\n预测结果 (TeamA vs TeamB):", probs)
    # 假如 teamA 获胜（result = 1）
    ts.rate_team(teamA, teamB, result=1)
    print("\n2v2 比赛后（TeamA 获胜）：")
    print("TeamA:", [str(p) for p in teamA])
    print("TeamB:", [str(p) for p in teamB])

    print("test")