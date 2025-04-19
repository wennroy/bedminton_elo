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


# Elo页面
def elo_page():
    select_rating_system = st.selectbox(label="比赛得分排名系统: ",
                                        options=["ELO (Based on score)", "TrueSkill (Based on distribution)"])
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
        # alpha = st.slider("随机比赛比例", 0.0, 1.0, 0.2)
        alpha = 0
        selected_players = st.multiselect("选择参赛选手", list(users.keys()))
        seed = st.number_input("随机种子（可选）", min_value=0, format="%d", value=None)
        temperature = st.number_input("温度：温度越低平衡性越好", min_value=0.0, max_value=1.0, step=0.000001, value=0.5,
                                      format="%0.6f")
        generate_btn = st.form_submit_button("生成比赛")

    if generate_btn:
        # 输入验证
        if len(selected_players) < 4:
            st.error("至少需要选择4名选手")
            return
        # if len(selected_players) % 2 != 0:
        #     st.error("选手数量必须为偶数")
        #     return

        # 设置随机种子
        if seed is None:
            seed = random.randint(0, 1000000)
            st.info(f"自动生成随机种子: {seed}")
        random.seed(seed)
        np.random.seed(seed)

        # 获取选手ELO数据
        conn = sqlite3.connect('badminton.db')
        query = '''SELECT u.id, u.name, COALESCE(p.elo, ?) as elo 
                   FROM users u LEFT JOIN players p ON u.id = p.user_id
                   WHERE u.name IN ({})'''.format(','.join(['?'] * len(selected_players)))
        players = pd.read_sql(query, conn, params=[INITIAL_RATING] + selected_players).to_dict('records')
        conn.close()

        # 初始化参赛次数计数器
        player_ids = [p['id'] for p in players]
        play_count = {pid: 0 for pid in player_ids}
        players_dict = {player["id"]: player["elo"] for player in players}
        # st.write(players_dict)

        # 计算各类型比赛数量
        non_random_num = int(np.ceil(total_matches * (1 - alpha)))
        random_num = total_matches - non_random_num

        # 生成非随机比赛（改进后的算法）
        non_random_matches = []
        player_elo = standardized_elo(players_dict)

        sched = DoublesScheduler(player_elo, non_random_num)
        # 计算归一化常量
        c_bar = np.mean(sched.costs)
        M = sched.M
        # 设定归一化参数 alpha' = 0.1，映射到原始 alpha
        alpha_prime = 5000
        alpha_orig = alpha_prime * c_bar * M

        # print(f"归一化 alpha'={alpha_prime}, 对应 alpha_orig={alpha_orig:.4f}")

        # 求解并采样
        p = sched.solve(alpha_orig)
        st.write(alpha_orig)
        st.write(f"分布 p_j 最大值: {np.max(p):.4f}, 熵: {-np.sum(p * np.log(p + 1e-12)):.4f}")
        schedule = sched.sample_schedule()

        for idx, match in enumerate(schedule, 1):
            # 更新参赛次数
            play_count[match[0][0]] += 1
            play_count[match[0][1]] += 1
            play_count[match[1][0]] += 1
            play_count[match[1][1]] += 1
            non_random_matches.append((match[0], match[1]))

        # 生成随机比赛（改进后的算法）
        random_matches = []
        for _ in range(random_num):
            # 根据参赛次数加权随机选择
            weights = [1 / (play_count[pid] + 0.1) for pid in player_ids]  # 参赛次数越少权重越高
            selected = random.choices(player_ids, weights=weights, k=4)

            # 随机分成两队
            random.shuffle(selected)
            team_a = sorted(selected[:2])
            team_b = sorted(selected[2:])

            random_matches.append((team_a, team_b))

            # 更新参赛次数
            for pid in selected:
                play_count[pid] += 1

        # 保存到数据库
        conn = sqlite3.connect('badminton.db')
        c = conn.cursor()
        c.execute("DELETE FROM pending_matches")  # 清除旧数据

        # 插入非随机比赛
        for match in non_random_matches:
            c.execute('''INSERT INTO pending_matches 
                      (player_a1, player_a2, player_b1, player_b2) 
                      VALUES (?,?,?,?)''',
                      (match[0][0], match[0][1], match[1][0], match[1][1]))

        # 插入随机比赛
        for match in random_matches:
            c.execute('''INSERT INTO pending_matches 
                      (player_a1, player_a2, player_b1, player_b2) 
                      VALUES (?,?,?,?)''',
                      (match[0][0], match[0][1], match[1][0], match[1][1]))

        conn.commit()
        conn.close()
        st.success(f"成功生成 {len(non_random_matches) + len(random_matches)} 场比赛！")

        # 显示参赛次数统计
        st.subheader("选手参赛次数统计")
        count_data = []
        for pid, cnt in play_count.items():
            count_data.append({
                "选手": id_to_name[pid],
                "参赛次数": cnt,
                "ELO": players_dict[pid]
            })
        df_counts = pd.DataFrame(count_data).sort_values("参赛次数")
        st.dataframe(df_counts, use_container_width=True)
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
        # "比赛分配": match_scheduler_page
    }
    page = st.sidebar.radio("页面", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
