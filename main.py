import streamlit as st
import sqlite3
import pandas as pd
import itertools
import numpy as np
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import math
from trueskill_utils import TrueSkill, Player
from random_utils import optimize_schedule

from utils import generate_double_configurations, standardized_elo, DoublesScheduler


# 参数配置
INITIAL_RATING = 1000
K_SINGLES = 32
K_DOUBLES = 16


# 初始化数据库
def init_db():
    conn = sqlite3.connect('badminton.db')
    c = conn.cursor()

    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE)''')

    # 比赛记录表
    c.execute('''CREATE TABLE IF NOT EXISTS matches
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  match_type TEXT,
                  player_a1 INTEGER,
                  player_a2 INTEGER,
                  player_b1 INTEGER,
                  player_b2 INTEGER,
                  score_a INTEGER,
                  score_b INTEGER,
                  date TEXT,
                  FOREIGN KEY(player_a1) REFERENCES users(id),
                  FOREIGN KEY(player_a2) REFERENCES users(id),
                  FOREIGN KEY(player_b1) REFERENCES users(id),
                  FOREIGN KEY(player_b2) REFERENCES users(id))''')

    # 选手Elo表
    c.execute('''CREATE TABLE IF NOT EXISTS players
                 (user_id INTEGER PRIMARY KEY,
                  elo REAL,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()


# 获取所有用户
def get_users():
    conn = sqlite3.connect('badminton.db')
    users = pd.read_sql('SELECT id, name FROM users', conn).set_index('id')['name'].to_dict()
    conn.close()
    return {v: k for k, v in users.items()}  # 返回{name: id}字典


# 用户管理页面
def user_management():
    st.header("👥 用户管理")

    # 添加用户
    with st.form("添加用户"):
        new_user = st.text_input("新用户姓名")
        if st.form_submit_button("添加用户"):
            if new_user:
                try:
                    conn = sqlite3.connect('badminton.db')
                    conn.execute('INSERT INTO users (name) VALUES (?)', (new_user,))
                    conn.commit()
                    st.success(f"用户 {new_user} 添加成功！")
                except sqlite3.IntegrityError:
                    st.error("用户名已存在")
                finally:
                    conn.close()

    # 删除用户
    st.subheader("现有用户")
    users = get_users()
    df_users = pd.DataFrame({'姓名': list(users.keys())})

    edited_users = st.data_editor(df_users, num_rows="dynamic", use_container_width=True,
                                  column_config={"删除": st.column_config.CheckboxColumn(required=True)},
                                  key="user_editor")

    if st.button("确认删除"):
        to_delete = df_users[~df_users['姓名'].isin(edited_users['姓名'])]['姓名'].tolist()
        if to_delete:
            conn = sqlite3.connect('badminton.db')
            conn.execute(f'DELETE FROM users WHERE name IN ({",".join(["?"] * len(to_delete))})', to_delete)
            conn.commit()
            conn.close()
            st.success(f"已删除 {len(to_delete)} 个用户")
        else:
            st.warning("没有选择要删除的用户")


# 单打Elo计算
def calculate_singles_elo(winner_rating, loser_rating):
    expected = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    new_winner = winner_rating + K_SINGLES * (1 - expected)
    new_loser = loser_rating + K_SINGLES * (expected - 1)
    return new_winner, new_loser


# 添加比赛记录
def add_match(match_type, a1, a2, b1, b2, score_a, score_b, date):
    score_a = int(score_a)
    score_b = int(score_b)

    conn = sqlite3.connect('badminton.db')

    # 插入比赛记录
    conn.execute('''INSERT INTO matches 
                    (match_type, player_a1, player_a2, player_b1, player_b2, score_a, score_b, date)
                    VALUES (?,?,?,?,?,?,?,?)''', (match_type, a1, a2, b1, b2, score_a, score_b, date))

    conn.commit()
    conn.close()

# 主页面
def main_page():
    st.header("🏸 比分记录")
    users = get_users()
    match_type = st.selectbox("比赛类型", ["双打", "单打"])
    with st.form("match_form"):
        if match_type == "单打":
            col1, col2 = st.columns(2)
            with col1:
                a1 = st.selectbox("选手A", users)
            with col2:
                b1 = st.selectbox("选手B", users)
            a2 = b2 = None
        else:
            col1, col2 = st.columns(2)
            with col1:
                a1 = st.selectbox("队伍A选手1", users)
                a2 = st.selectbox("队伍A选手2", users)
            with col2:
                b1 = st.selectbox("队伍B选手1", users)
                b2 = st.selectbox("队伍B选手2", users)

        col3, col4 = st.columns(2)
        with col3:
            score_a = st.number_input("A队得分", min_value=0, step=1, value=21)
        with col4:
            score_b = st.number_input("B队得分", min_value=0, step=1, value=21)
        match_date = st.date_input("比赛日期")

        if st.form_submit_button("保存记录"):
            if (match_type == "双打" and not all([a1, a2, b1, b2])) or (match_type == "单打" and not all([a1, b1])):
                st.error("请填写所有选手信息")
            elif score_a == score_b:
                st.error("比分不能相同")
            else:
                add_match(match_type, a1, a2, b1, b2, score_a, score_b, str(match_date))
                if match_type == "双打":
                    st.success(f"比赛 {a1}/{a2} vs {b1}/{b2} 记录保存成功！")
                else:
                    st.success(f"比赛 {a1} vs {b1} 记录保存成功！")


def calculate_elo():
    conn = sqlite3.connect('badminton.db')

    # 获取所有比赛记录
    matches = conn.execute('SELECT * FROM matches ORDER BY date').fetchall()

    # 初始化选手Elo
    players = conn.execute('SELECT user_id, elo FROM players').fetchall()
    elo_dict = {}
    last_date = None
    elo_date_player_list = []
    for match in matches:
        match_type, a1, a2, b1, b2, score_a, score_b, date = match[1:]
        if last_date is None or last_date != date:
            for player, value in elo_dict.items():
                elo_date_player_list.append({
                    "Date": last_date,
                    "id": player,
                    "Elo": value
                })
            last_date = date
        a1 = conn.execute(f'SELECT id FROM users WHERE name = "{a1}"').fetchall()[0][0]
        b1 = conn.execute(f'SELECT id FROM users WHERE name = "{b1}"').fetchall()[0][0]
        if match_type == '单打':
            players = [a1, b1]
            # 获取当前Elo
            ratings = {a1: elo_dict.get(a1, float(INITIAL_RATING)), b1: elo_dict.get(b1, float(INITIAL_RATING))}

            # 计算新Elo
            if score_a > score_b:
                w_elo, l_elo = calculate_singles_elo(ratings[a1], ratings[b1])
                elo_dict[a1] = w_elo
                elo_dict[b1] = l_elo
            else:
                w_elo, l_elo = calculate_singles_elo(ratings[b1], ratings[a1])
                elo_dict[b1] = w_elo
                elo_dict[a1] = l_elo

        else:  # 双打处理
            a2 = conn.execute(f'SELECT id FROM users WHERE name = "{a2}"').fetchall()[0][0]
            b2 = conn.execute(f'SELECT id FROM users WHERE name = "{b2}"').fetchall()[0][0]
            all_players = [a1, a2, b1, b2]
            # 初始化缺失选手
            for p in all_players:
                if p not in elo_dict:
                    elo_dict[p] = float(INITIAL_RATING)

            # 计算队伍平均分
            team_a_avg = (elo_dict[a1] + elo_dict[a2]) / 2
            team_b_avg = (elo_dict[b1] + elo_dict[b2]) / 2

            # 计算每个选手的变化
            for player in [a1, a2]:
                expected = 1 / (1 + 10 ** ((team_b_avg - elo_dict[player]) / 400))
                change = K_DOUBLES * (1 if score_a > score_b else 0 - expected)
                elo_dict[player] += change

            for player in [b1, b2]:
                expected = 1 / (1 + 10 ** ((team_a_avg - elo_dict[player]) / 400))
                change = K_DOUBLES * (1 if score_b > score_a else 0 - expected)
                elo_dict[player] += change

    for player, value in elo_dict.items():
        elo_date_player_list.append({
            "Date": last_date,
            "id": player,
            "Elo": value
        })

    # 更新数据库中的Elo
    for player_id, elo in elo_dict.items():
        # 确保 player_id 是整数，elo 是浮点数
        conn.execute('INSERT OR IGNORE INTO players (user_id, elo) VALUES (?,?)', (int(player_id), float(elo)))
        conn.execute('UPDATE players SET elo=? WHERE user_id=?', (float(elo), int(player_id)))

    conn.commit()
    conn.close()
    elo_df = pd.DataFrame(elo_date_player_list)
    return elo_df


def calculate_trueskill():
    conn = sqlite3.connect('badminton.db')

    # 获取所有比赛记录，假设表 matches 字段顺序为：
    # id, match_type, a1, a2, b1, b2, score_a, score_b, date
    matches = conn.execute('SELECT * FROM matches ORDER BY date').fetchall()

    # 初始化所有选手的 TrueSkill 状态字典
    # key: 用户ID, value: Player 实例（初始值默认为 mu=25, sigma=8.333）
    ts_dict = {}

    # 构造 TrueSkill 工具类，平局概率设为 0
    ts_util = TrueSkill(draw_probability=0.0)

    # 存储每天选手评分记录的列表（用于 DataFrame 输出）
    ts_date_player_list = []
    last_date = None

    for match in matches:
        # match 元组中依次为 (id, match_type, a1, a2, b1, b2, score_a, score_b, date)
        match_type, a1, a2, b1, b2, score_a, score_b, date = match[1:]

        # 当日期发生变化时，将上一天所有选手的状态记录下来
        if last_date is None or date != last_date:
            if last_date is not None:  # 记录上一天的状态
                for user_id, player in ts_dict.items():
                    ts_date_player_list.append({
                        "Date": last_date,
                        "id": user_id,
                        "mu": player.mu,
                        "sigma": player.sigma
                    })
            last_date = date

        if match_type == '单打':
            # 单打比赛中只涉及 a1 和 b1 两位选手
            row = conn.execute(f'SELECT id FROM users WHERE name = "{a1}"').fetchone()
            a1_id = row[0] if row else None
            row = conn.execute(f'SELECT id FROM users WHERE name = "{b1}"').fetchone()
            b1_id = row[0] if row else None

            if a1_id is None or b1_id is None:
                continue  # 基础数据有误，跳过此场比赛

            # 若选手未初始化，则创建新的 Player 实例
            if a1_id not in ts_dict:
                ts_dict[a1_id] = Player()
            if b1_id not in ts_dict:
                ts_dict[b1_id] = Player()

            # 获取选手实例
            p1 = ts_dict[a1_id]
            p2 = ts_dict[b1_id]

            # 根据比分确定胜负，平局不计
            if score_a > score_b:
                ts_util.rate_1v1(winner=p1, loser=p2)
            else:
                ts_util.rate_1v1(winner=p2, loser=p1)

        else:
            # 双打比赛：涉及 4 名选手 a1, a2, b1, b2
            row = conn.execute(f'SELECT id FROM users WHERE name = "{a1}"').fetchone()
            a1_id = row[0] if row else None
            row = conn.execute(f'SELECT id FROM users WHERE name = "{a2}"').fetchone()
            a2_id = row[0] if row else None
            row = conn.execute(f'SELECT id FROM users WHERE name = "{b1}"').fetchone()
            b1_id = row[0] if row else None
            row = conn.execute(f'SELECT id FROM users WHERE name = "{b2}"').fetchone()
            b2_id = row[0] if row else None

            if None in [a1_id, a2_id, b1_id, b2_id]:
                continue  # 数据有误，跳过此场比赛

            # 初始化所有选手（若未曾出现过）
            for uid in [a1_id, a2_id, b1_id, b2_id]:
                if uid not in ts_dict:
                    ts_dict[uid] = Player()

            # 构造双打队伍
            teamA = [ts_dict[a1_id], ts_dict[a2_id]]
            teamB = [ts_dict[b1_id], ts_dict[b2_id]]

            # 根据比分决定哪支队伍获胜
            result = 1 if score_a > score_b else -1
            ts_util.rate_team(teamA, teamB, result)

    # 结束所有比赛后，记录最后一天的状态
    for user_id, player in ts_dict.items():
        ts_date_player_list.append({
            "Date": last_date,
            "id": user_id,
            "mu": player.mu,
            "sigma": player.sigma
        })

    # 创建/更新 players_trueskill 表存储最终的 TrueSkill 得分
    conn.execute('CREATE TABLE IF NOT EXISTS players_trueskill (user_id INTEGER PRIMARY KEY, mu REAL, sigma REAL)')
    for user_id, player in ts_dict.items():
        conn.execute('INSERT OR REPLACE INTO players_trueskill (user_id, mu, sigma) VALUES (?, ?, ?)',
                     (user_id, player.mu, player.sigma))
    conn.commit()
    conn.close()

    ts_df = pd.DataFrame(ts_date_player_list)
    return ts_df

@st.cache_data
def load_players_data():
    conn = sqlite3.connect('badminton.db')
    users_df = pd.read_sql("SELECT id, name FROM users", conn)
    elo_df = pd.read_sql("SELECT user_id, elo FROM players", conn)
    ts_df = pd.read_sql("SELECT user_id, mu, sigma FROM players_trueskill", conn)
    conn.close()
    # 合并信息，方便后续选项展示
    players_df = users_df.merge(elo_df, left_on="id", right_on="user_id", how="left")
    players_df = players_df.merge(ts_df, on="user_id", how="left")
    # 若某些字段缺失则赋默认值（Elo 初始1000, TrueSkill默认 mu=25, sigma=8.33）
    players_df["elo"] = players_df["elo"].fillna(1000)
    players_df["mu"] = players_df["mu"].fillna(25)
    players_df["sigma"] = players_df["sigma"].fillna(8.333)
    return players_df


def predict_elo(teamA_ids, teamB_ids, players_df = None):
    """
    根据 Elo 评分计算双打比赛预测胜率。双方队伍 Elo 为队内选手 Elo 平均值。
    返回:
       teamA_win_prob, teamB_win_prob
    """
    if players_df is None:
        st.error("Something went into error!")
        return 0, 0
    # 从 players_df 中查找 Elo 分数
    teamA_elo = players_df[players_df["id"].isin(teamA_ids)]["elo"].mean()
    teamB_elo = players_df[players_df["id"].isin(teamB_ids)]["elo"].mean()
    # Elo 预测公式（1v1 也适用于取平均值的双打）
    teamA_win_prob = 1 / (1 + 10 ** ((teamB_elo - teamA_elo) / 400))
    teamB_win_prob = 1 - teamA_win_prob
    return teamA_win_prob, teamB_win_prob


def predict_trueskill(teamA_ids, teamB_ids, players_df = None):
    """
    根据 TrueSkill 预测双打结果，构造双方 Player 对象（使用 mu、sigma）。
    调用 TrueSkill 类中的 predict_team_outcome 方法计算胜率等。
    返回一个字典，包含 'win', 'draw', 'loss' 其中：
         win: 队伍 A 获胜概率
         loss: 队伍 A 失败（队伍 B 获胜）概率
         draw: 平局概率（此处设置为 0）
    """
    if players_df is None:
        st.error("Something went into error!")
        return {
            "win": 0,
            "draw": 0,
            "loss": 0
        }
    # 创建 TrueSkill 工具实例，设置 draw_probability 为 0
    ts_util = TrueSkill(draw_probability=0.0)

    def make_player(user_id):
        row = players_df[players_df["id"] == user_id].iloc[0]
        # 构造 Player 对象，参数来自数据库；可调整初始化值的尺度
        return Player(mu=row["mu"], sigma=row["sigma"])

    teamA = [make_player(pid) for pid in teamA_ids]
    teamB = [make_player(pid) for pid in teamB_ids]

    # 利用 ts_util 对团队进行预测
    outcome = ts_util.predict_team_outcome(teamA, teamB)
    # 返回的 outcome 包含 'win', 'draw', 'loss'
    return outcome


# Elo页面
def elo_page():
    select_rating_system = st.selectbox(label="比赛得分排名系统: ",
                                        options=["TrueSkill (Based on distribution)", "ELO (Based on score)"])
    if select_rating_system.startswith("ELO"):
        st.header("🏅 Elo排名")
        st.markdown("起始分1000分，默认**K factor**: 32 (双打为16)。")

        elo_df = calculate_elo()

        conn = sqlite3.connect('badminton.db')
        user_df = pd.read_sql("""
            SELECT id, name
            from users
        """, conn)
        elo_df = elo_df.merge(user_df, "left", on=["id"])
        df = pd.read_sql('''
            SELECT u.name, round(p.elo, 1) as elo 
            FROM players p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.elo DESC
        ''', conn)
        conn.close()

        st.dataframe(df, use_container_width=True, column_order=["name", "elo"], hide_index=True)

        elo_df["Date"] = pd.to_datetime(elo_df["Date"])
        fig = px.line(elo_df, x="Date", y="Elo", color="name", title="Elo Trends")
        st.markdown("---")
        st.markdown("#### Elo 趋势图")
        st.plotly_chart(fig)
    elif select_rating_system.startswith("TrueSkill"):
        st.header("🏅 TrueSkill排名")
        st.markdown(r"玩家的初始分布设置为$\mathcal{N}\left(25, \left(\frac{25}{3}\right)^2\right)$")
        # 先计算 TrueSkill 的历史记录（需要按日期顺序更新评分，返回 DataFrame）
        ts_history_df = calculate_trueskill()
        # ts_history_df 的结构：["Date", "id", "mu", "sigma"]

        # 连接数据库，获取用户信息以及最终 TrueSkill 得分
        conn = sqlite3.connect('badminton.db')
        user_df = pd.read_sql("""
            SELECT id, name
            FROM users
        """, conn)

        # 将 TrueSkill 历史记录与用户姓名关联（按用户id合并）
        ts_history_df = ts_history_df.merge(user_df, how="left", on="id")

        # 获取最终 TrueSkill 得分（存储在 players_trueskill 表中），并按 mu 排序
        final_ts_df = pd.read_sql('''
            SELECT u.name, round(p.mu, 1) as mean, round(p.sigma, 2) as std_deviation
            FROM players_trueskill p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.mu DESC
        ''', conn)
        conn.close()
        # st.write(final_ts_df)
        # 显示最终 TrueSkill 得分表
        st.dataframe(final_ts_df, use_container_width=True, column_order=["name", "mean", "std_deviation"], hide_index=True)

        # 合并用户姓名
        ts_history_df["Date"] = pd.to_datetime(ts_history_df["Date"])

        # Streamlit 选项：是否错开同一天内的点（在 x 轴上添加微小的时间偏移）
        offset_option = st.checkbox("是否错开显示用户坐标？", value=False)

        if offset_option:
            # 针对每周最多 10 个人，为了使展示更清晰，
            # 采用对称的两个小时偏移：假设所有用户均匀分布在 -1 小时到 +1 小时之间
            unique_names = sorted(ts_history_df["name"].dropna().unique())
            n = len(unique_names)
            if n > 1:
                # 间隔 = 总时长 2 小时除以 (n-1)
                spacing = pd.Timedelta(hours=48) / (n - 1)
            else:
                spacing = pd.Timedelta(0)

            # 每个用户的偏移： (index - (n-1)/2) * spacing
            offset_map = {name: (pd.Timedelta(0) + (index - (n - 1) / 2) * spacing)
                          for index, name in enumerate(unique_names)}
            # 新增 Date_offset 列：原始日期加上用户对应的偏移
            ts_history_df["Date_offset"] = ts_history_df.apply(
                lambda row: row["Date"] + offset_map.get(row["name"], pd.Timedelta(0)),
                axis=1
            )
            x_column = "Date_offset"
        else:
            x_column = "Date"

        # 根据 sigma 计算 95% 置信区间半宽度（1.96 * sigma）
        ts_history_df["error"] = 1.96 * ts_history_df["sigma"]

        # 绘制趋势图：每个数据点显示 mu 值和对应的误差条
        fig = px.line(
            ts_history_df,
            x=x_column,
            y="mu",
            color="name",
            error_y="error",
            title="Trueskill Trends (with 95% CI)"
        )
        st.markdown("#### Trueskill 趋势图")
        st.plotly_chart(fig)

    st.markdown("---")

    players_df = load_players_data()

    # 提供双方队伍的选择
    st.markdown("### 请选择双打双方的选手")
    # 生成选项列表，格式："name (id)"；确保选手不会重复选择
    player_options = players_df.apply(lambda row: f'{row["name"]}', axis=1).tolist()
    player2id = {option: row["id"] for option, row in zip(player_options, players_df.to_dict("records"))}

    st.markdown("#### 队伍 A")
    teamA_player1 = st.selectbox("队伍 A – 选手 1", player_options, key="A1")
    teamA_player2 = st.selectbox("队伍 A – 选手 2", player_options, key="A2")

    st.markdown("#### 队伍 B")
    teamB_player1 = st.selectbox("队伍 B – 选手 1", player_options, key="B1")
    teamB_player2 = st.selectbox("队伍 B – 选手 2", player_options, key="B2")

    # 简单检查：若同一个选手出现在两个队中，可给出警告
    selected_ids = [player2id[teamA_player1], player2id[teamA_player2],
                    player2id[teamB_player1], player2id[teamB_player2]]
    if len(set(selected_ids)) < 4:
        st.warning("请确保四位选手均不重复！")

    if st.button("Predict"):
        # 队伍的选手 ID 列表
        teamA_ids = [player2id[teamA_player1], player2id[teamA_player2]]
        teamB_ids = [player2id[teamB_player1], player2id[teamB_player2]]

        if select_rating_system.startswith("ELO"):
            # Elo 预测
            elo_teamA_prob, elo_teamB_prob = predict_elo(teamA_ids, teamB_ids, players_df)
            st.write(f"队伍 A 预测胜率 (基于elo算法)：{elo_teamA_prob * 100:.1f}%")
            st.write(f"队伍 B 预测胜率 (基于elo算法)：{elo_teamB_prob * 100:.1f}%")
        elif select_rating_system.startswith("TrueSkill"):
            # TrueSkill 预测
            ts_outcome = predict_trueskill(teamA_ids, teamB_ids, players_df)
            st.write(f"队伍 A 预测胜率 (基于TrueSkill算法)：{ts_outcome['win'] * 100:.1f}%")
            # st.write(f"平局概率：{ts_outcome['draw'] * 100:.1f}%")
            st.write(f"队伍 B 预测胜率 (基于TrueSkill算法)：{ts_outcome['loss'] * 100:.1f}%")


# 管理页面
def manage_page():
    st.header("📝 数据管理")

    conn = sqlite3.connect('badminton.db')

    # 从数据库读取原始字段（包含 player_a1、player_a2 等）
    df = pd.read_sql('''
        SELECT 
            id, 
            match_type,
            player_a1,
            player_a2,
            player_b1,
            player_b2,
            score_a,
            score_b,
            date
        FROM matches
    ''', conn, parse_dates=['date'])  # 直接解析日期为 datetime 类型

    # 动态生成展示用的 match_info 列（无需保存到数据库）
    df['match_info'] = df.apply(lambda row: (f"{row['player_a1']} vs {row['player_b1']}" if row[
                                                                                                'match_type'] == '单打' else f"{row['player_a1']}/{row['player_a2']} vs {row['player_b1']}/{row['player_b2']}"),
                                axis=1)

    # 配置列属性：隐藏 id，指定日期格式，设置中文列名
    edited_df = st.data_editor(df,
                               column_config={"id": None, "date": st.column_config.DateColumn("日期"),
                                              "match_info": "比赛信息（自动生成）",
                                              "player_a1": "选手A1", "player_a2": "选手A2", "player_b1": "选手B1",
                                              "player_b2": "选手B2",
                                              "score_a": "比分A", "score_b": "比分B", "match_type": "比赛类型"},
                               use_container_width=True)

    if st.button("保存修改"):
        # 将日期转换回字符串格式（确保与数据库兼容）
        edited_df['date'] = edited_df['date'].dt.strftime('%Y-%m-%d')

        # 清空旧数据并写入新数据（排除自动生成的 match_info 列）
        conn.execute("DELETE FROM matches")
        edited_df[
            ['match_type', 'player_a1', 'player_a2', 'player_b1', 'player_b2', 'score_a', 'score_b', 'date']].to_sql(
            'matches', conn, if_exists='append', index=False)

        st.success("数据已更新！需要重新计算Elo分数请手动处理历史数据。")

    conn.close()

# 新增比赛分配页面
def match_scheduler_page():
    st.header("🎯 双打比赛分配")

    # 初始化数据库
    conn = sqlite3.connect('badminton.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pending_matches
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  player_a1 INTEGER,
                  player_a2 INTEGER,
                  player_b1 INTEGER,
                  player_b2 INTEGER,
                  score_a INTEGER,
                  score_b INTEGER,
                  submitted BOOLEAN DEFAULT FALSE,
                  FOREIGN KEY(player_a1) REFERENCES users(id),
                  FOREIGN KEY(player_a2) REFERENCES users(id),
                  FOREIGN KEY(player_b1) REFERENCES users(id),
                  FOREIGN KEY(player_b2) REFERENCES users(id))''')
    conn.commit()
    conn.close()

    # 获取用户列表
    users = get_users()
    id_to_name = {v: k for k, v in users.items()}

    # 参数输入表单
    with st.form("match_params"):
        st.subheader("比赛参数设置")
        total_matches = st.number_input("总比赛场次", min_value=1, value=4)
        selected_players = st.multiselect("选择参赛选手", list(users.keys()))
        seed = st.number_input("随机种子（可选）", min_value=0, format="%d", value=None)
        temperature = st.slider("温度：温度越低平衡性越好", min_value=0.0, max_value=1.0, step=0.05, value=0.5,
                                      format="%0.1f")
        # max_player_gap = st.slider("最大比赛数量差", min_value=1, max_value=3, value=1, step=1)
        generate_btn = st.form_submit_button("生成比赛")

    if generate_btn:
        # 输入验证
        if len(selected_players) < 4:
            st.error("至少需要选择4名选手")
            return

        # 设置随机种子
        if seed is None:
            seed = random.randint(0, 1000000)
            st.info(f"自动生成随机种子: {seed}")
        random.seed(seed)
        np.random.seed(seed)

        # 获取选手trueskill数据
        ts_history_df = calculate_trueskill()
        conn = sqlite3.connect('badminton.db')
        ts_df = pd.read_sql('''
            SELECT u.name, u.id, round(p.mu, 1) as mean, round(p.sigma, 2) as std_deviation
            FROM players_trueskill p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.mu DESC
        ''', conn)
        conn.close()

        # 初始化参赛次数计数器
        player_list = []
        player_id_list = []
        for index, row in ts_df.iterrows():
            player_id_list.append((row["name"], row["id"]))
            player_list.append(Player(row["mean"], row["std_deviation"]))

        schedule, alpha_var, best_loss, mean_closeness, max_closeness = optimize_schedule(
            n=len(player_list),
            m=total_matches,
            ts=TrueSkill(draw_probability=0.0),
            players=player_list,
            lambda_weight=temperature,
            iters=5000,
            max_play_gap=2,
            seed=seed
        )
        st.write(f"生成比赛最大胜率差为{round(max_closeness, 2)}, 胜率差均值为{round(mean_closeness, 2)},比赛场次数方差为{round(alpha_var, 2)}")
        matches = []
        for idx, (team1, team2) in enumerate(schedule):
            team1_id = [player_id_list[p][1] for p in team1]
            team2_id = [player_id_list[p][1] for p in team2]
            matches.append((team1_id, team2_id))
        # 保存到数据库
        conn = sqlite3.connect('badminton.db')
        c = conn.cursor()
        c.execute("DELETE FROM pending_matches")  # 清除旧数据

        # 插入比赛
        for match in matches:
            c.execute('''INSERT INTO pending_matches 
                      (player_a1, player_a2, player_b1, player_b2) 
                      VALUES (?,?,?,?)''',
                      (match[0][0], match[0][1], match[1][0], match[1][1]))
        conn.commit()
        conn.close()
        st.success(f"成功生成 {len(matches)} 场比赛！")
    # 显示待处理比赛
    st.subheader("待处理比赛列表")
    conn = sqlite3.connect('badminton.db')
    pending_matches = pd.read_sql("SELECT * FROM pending_matches", conn)
    conn.close()

    if pending_matches.empty:
        st.info("当前没有待处理的比赛")
    else:
        for count, (_, match) in enumerate(pending_matches.iterrows()):
            with st.expander(f"比赛 {count+1}"):
                with st.form(key=f"match_{match['id']}"):
                    # 显示选手姓名
                    a1 = id_to_name[match['player_a1']]
                    a2 = id_to_name[match['player_a2']]
                    b1 = id_to_name[match['player_b1']]
                    b2 = id_to_name[match['player_b2']]
                    st.markdown(f"**{a1} / {a2}** vs **{b1} / {b2}**")

                    # 比分输入
                    col1, col2 = st.columns(2)
                    with col1:
                        score_a = st.number_input("A队得分", min_value=0, value=int(match['score_a']) if match['score_a'] and not math.isnan(match['score_a']) else 0,
                                                  key=f"a_{match['id']}")
                    with col2:
                        score_b = st.number_input("B队得分", min_value=0, value=int(match['score_b']) if match['score_b'] and not math.isnan(match['score_b']) else 0,
                                                  key=f"b_{match['id']}")

                    # 提交按钮
                    if st.form_submit_button("提交比分"):
                        if score_a == score_b:
                            st.error("比分不能相同！")
                        else:
                            conn = sqlite3.connect('badminton.db')
                            c = conn.cursor()
                            c.execute('''UPDATE pending_matches 
                                      SET score_a=?, score_b=?, submitted=?
                                      WHERE id=?''',
                                      (score_a, score_b, True, match['id']))
                            conn.commit()
                            conn.close()
                            st.success("比分已更新！")

        # 最终操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存所有已提交比赛到主记录"):
                conn = sqlite3.connect('badminton.db')
                submitted = pd.read_sql("SELECT * FROM pending_matches WHERE submitted=1", conn)
                for _, match in submitted.iterrows():
                    add_match(
                        match_type="双打",
                        a1=id_to_name[match['player_a1']],
                        a2=id_to_name[match['player_a2']],
                        b1=id_to_name[match['player_b1']],
                        b2=id_to_name[match['player_b2']],
                        score_a=match['score_a'],
                        score_b=match['score_b'],
                        date=datetime.now().strftime("%Y-%m-%d")
                    )
                conn.execute("DELETE FROM pending_matches WHERE submitted=1")
                conn.commit()
                conn.close()
                st.success(f"已保存 {len(submitted)} 场比赛到主记录！")
                st.rerun()

        with col2:
            if st.button("⚠️ 重置所有比赛"):
                conn = sqlite3.connect('badminton.db')
                conn.execute("DELETE FROM pending_matches")
                conn.commit()
                conn.close()
                st.success("已重置所有待处理比赛！")
                st.rerun()


# 主程序
def main():
    st.set_page_config("卷技术小分队🏸")
    init_db()
    st.sidebar.title("导航")
    pages = {
        "用户管理": user_management,
        "比赛记录": main_page,
        "Ranking": elo_page,
        "数据管理": manage_page,
        "比赛分配": match_scheduler_page
    }
    page = st.sidebar.radio("页面", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
