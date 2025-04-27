from trueskill_utils import Player, TrueSkill
import random
import math
from collections import Counter


def chaos_by_entropy(lst):
    """
    基于香农熵的混乱度，返回 0–1 之间的浮点数。
    列表长度 n=0 或 1 时，定义为 1。
    """
    n = len(lst)
    if n <= 1:
        return 1.0
    freqs = Counter(lst)
    H = -sum((cnt/n) * math.log(cnt/n) for cnt in freqs.values())
    return 1 - (H / math.log(n))


def compute_loss(schedule, players, ts, lambda_weight):
    """
    计算给定赛程的损失和指标。
    schedule: list of matches，每场为 (team1, team2)，team1/team2为玩家索引列表。
    players: 玩家对象列表（TrueSkill 可识别的对象）。
    ts: TrueSkill 对象，具有 predict_team_outcome_win 方法。
    lambda_weight: lambda 权重。
    返回 (total_loss, alpha_variance, closeness_sum)。
    """
    n = len(players)
    m = len(schedule)
    counts = [0]*n
    closeness_sum = 0.0
    closeness_list = []
    from collections import defaultdict
    schedule_team = []

    # 遍历每场比赛，统计出场次数并累加胜率偏差
    for team1, team2 in schedule:
        # 累加出场次数
        schedule_team.append(tuple(team1))
        schedule_team.append(tuple(team2))
        for i in team1 + team2:
            counts[i] += 1
        # 预测胜率
        res = ts.predict_team_outcome_win(
            [players[i] for i in team1],
            [players[j] for j in team2]
        )
        p = res  # 假设取team1的胜率
        closeness = abs(p - 0.5)
        closeness_list.append(closeness)
        closeness_sum += closeness


    # 计算出场次数方差
    mean_count = sum(counts) / n
    alpha = sum((c - mean_count)**2 for c in counts) / n

    total_loss = alpha + lambda_weight * chaos_by_entropy(schedule_team) + (1-lambda_weight) * closeness_sum / len(closeness_list) * 2
    return total_loss, alpha, closeness_sum, closeness_list

def random_initial_schedule(n, m):
    """
    随机生成初始赛程：n名玩家，m场比赛。
    每场随机选4个不同玩家并随机分为两队。
    返回 schedule: [(team1, team2), ...]，其中team1/team2为索引列表。
    """
    schedule = []
    players = list(range(n))
    for _ in range(m):
        # 随机选择4人
        team = random.sample(players, 4)
        # 从4人中随机选2人作为team1，其余为team2
        team1 = random.sample(team, 2)
        team2 = [p for p in team if p not in team1]
        schedule.append((team1, team2))
    return schedule

def get_neighbor(schedule, n):
    """
    生成当前赛程的邻居：通过随机扰动产生一个新赛程。
    可能的操作：交换两场比赛的选手、换场比赛的一个选手等。
    """
    new_schedule = [ (list(team1), list(team2)) for team1, team2 in schedule ]
    m = len(new_schedule)
    # 随机选择一种扰动方式
    choice = random.choice(['swap_match', 'swap_player', 'reshuffle_team'])
    if choice == 'swap_match' and m >= 2:
        # 从两场比赛各随机换一人
        i, j = random.sample(range(m), 2)
        t1a, t2a = new_schedule[i]
        t1b, t2b = new_schedule[j]
        # 从比赛i随机选一人，从比赛j随机选一人，交换他们（保留场内4人不同）
        pa = random.choice(t1a + t2a)
        pb = random.choice(t1b + t2b)
        if pa not in t1a+t2a or pb not in t1b+t2b:
            return new_schedule
        # 交换玩家
        t1a = [pb if x == pa else x for x in t1a]
        t2a = [pb if x == pa else x for x in t2a]
        t1b = [pa if x == pb else x for x in t1b]
        t2b = [pa if x == pb else x for x in t2b]
        # 检查是否出现重复，若重复则放弃这次交换
        if len(set(t1a+t2a)) == 4 and len(set(t1b+t2b)) == 4:
            new_schedule[i] = (t1a, t2a)
            new_schedule[j] = (t1b, t2b)

    elif choice == 'swap_player':
        # 从一场比赛中更换一个玩家到同一场的另一位置
        i = random.randrange(m)
        team1, team2 = new_schedule[i]
        # 随机从team1/team2中选一人替换为新的玩家（未在该比赛中出现）
        if random.random() < 0.5 and len(team1) > 0:
            # 替换team1中的一人
            old = random.choice(team1)
            # 从场外玩家中选择新玩家
            outside = [p for p in range(n) if p not in team1+team2]
            if outside:
                new = random.choice(outside)
                idx = team1.index(old)
                team1[idx] = new
        else:
            # 替换team2中的一人
            old = random.choice(team2)
            outside = [p for p in range(n) if p not in team1+team2]
            if outside:
                new = random.choice(outside)
                idx = team2.index(old)
                team2[idx] = new
        new_schedule[i] = (team1, team2)

    elif choice == 'reshuffle_team':
        # 随机一场比赛中重新分队
        i = random.randrange(m)
        team1, team2 = new_schedule[i]
        all4 = team1 + team2
        # 随机分为两队
        if len(all4) == 4:
            new_team1 = random.sample(all4, 2)
            new_team2 = [p for p in all4 if p not in new_team1]
            new_schedule[i] = (new_team1, new_team2)

    return new_schedule

def optimize_schedule(n, m, ts, players, lambda_weight=0.5, iters=1000, start_temp=1.0, alpha_decay=0.995, max_play_gap=2, seed=None):
    if seed is not None:
        random.seed(seed)  # ✅ 加上这句，控制随机性！
    final_closeness = []
    best_schedule = random_initial_schedule(n, m)
    best_loss, best_alpha, best_closeness, closeness_list = compute_loss(best_schedule, players, ts, lambda_weight)
    current_schedule = best_schedule
    current_loss = best_loss
    T = start_temp

    for k in range(iters):
        # 生成邻域新赛程
        neighbor = get_neighbor(current_schedule, n)
        loss, alpha_var, closeness_sum, closeness_list = compute_loss(neighbor, players, ts, lambda_weight)
        # 如果新解更优则接受，或按概率接受较差解
        if loss < current_loss or random.random() < math.exp((current_loss - loss) / T):
            current_schedule = neighbor
            current_loss = loss
            # 更新最优解
            if loss < best_loss:
                best_schedule = neighbor
                best_loss = loss
                best_alpha = alpha_var
                best_closeness = closeness_sum
                final_closeness = closeness_list
        # 降低温度
        T *= alpha_decay
        if T < 1e-4:
            break

    # ✅ 赛程结束后统计胜率偏差情况
    mean_closeness = sum(final_closeness) / len(final_closeness)
    max_closeness = max(final_closeness)
    return best_schedule, best_alpha, best_loss, mean_closeness, max_closeness



if __name__ == "__main__":
    players = [
        Player(mu=25.0, sigma=8.333),  # 基准玩家
        Player(mu=30.5, sigma=8.333),  # 强一点
        Player(mu=20.0, sigma=8.333),  # 弱一点
        Player(mu=35.0, sigma=8.0),  # 小幅波动
        Player(mu=15.5, sigma=8.5),  # 小幅波动
        Player(mu=32.0, sigma=7.5),  # 比较强且稳定
        Player(mu=16.5, sigma=9.0),  # 比较弱且不稳定
    ]
    ts = TrueSkill()
    n = len(players)  # 7个玩家
    m = 10  # 10场比赛
    lambda_weight = 0  # 控制胜率平衡的重要性

    # 调用 optimize_schedule （假设你已经有了上次那段代码）
    schedule, alpha_var, best_loss, mean_closeness, max_closeness = optimize_schedule(
        n=7,
        m=10,
        ts=ts,
        players=players,
        lambda_weight=0.5,
        iters=5000,
        max_play_gap=2,
        seed=42
    )
    print(schedule)
    # 输出结果
    print("最终安排：")
    for idx, (team1, team2) in enumerate(schedule):
        team1_names = [f"Player{p + 1}" for p in team1]
        team2_names = [f"Player{p + 1}" for p in team2]
        print(f"第{idx + 1}场: {team1_names} VS {team2_names}")

    print(f"\n最终出场次数方差 α = {alpha_var:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Mean closeness: {mean_closeness}")
    print(f"max closeness: {max_closeness}")
    print("test")