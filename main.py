import streamlit as st
import sqlite3
import pandas as pd
import itertools
import numpy as np
import random
from datetime import datetime
import plotly.express as px

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

# Elo页面
def elo_page():
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
        alpha = st.slider("随机比赛比例", 0.0, 1.0, 0.2)
        selected_players = st.multiselect("选择参赛选手", list(users.keys()))
        seed = st.number_input("随机种子（可选）", min_value=0, format="%d", value=None)
        generate_btn = st.form_submit_button("生成比赛")

    if generate_btn:
        # 输入验证
        if len(selected_players) < 4:
            st.error("至少需要选择4名选手")
            return
        if len(selected_players) % 2 != 0:
            st.error("选手数量必须为偶数")
            return

        # 设置随机种子
        if seed is None:
            seed = random.randint(0, 1000000)
            st.info(f"自动生成随机种子: {seed}")
        random.seed(seed)
        np.random.seed(seed)

        # 获取选手ELO数据
        conn = sqlite3.connect('badminton.db')
        query = '''SELECT u.id, COALESCE(p.elo, ?) as elo 
                   FROM users u LEFT JOIN players p ON u.id = p.user_id
                   WHERE u.name IN ({})'''.format(','.join(['?'] * len(selected_players)))
        players = pd.read_sql(query, conn, params=[INITIAL_RATING] + selected_players).to_dict('records')
        conn.close()

        # 按ELO排序
        players_sorted = sorted(players, key=lambda x: x['elo'], reverse=True)
        player_ids = [p['id'] for p in players_sorted]

        # 计算各类型比赛数量
        non_random_num = int(np.ceil(total_matches * (1 - alpha)))
        random_num = total_matches - non_random_num

        # 生成非随机比赛
        non_random_matches = []
        used_players = []
        remaining_players = player_ids.copy()

        for _ in range(non_random_num):
            if len(remaining_players) < 4:
                st.warning("选手不足，无法生成更多非随机比赛")
                break

            # 取前4名选手
            candidates = remaining_players[:4]
            remaining_players = remaining_players[4:]

            # 计算最佳组合
            elo_values = [next(p['elo'] for p in players_sorted if p['id'] == pid) for pid in candidates]
            best_diff = float('inf')
            best_pair = None

            # 尝试所有可能的分组组合
            for team_a in itertools.combinations(candidates, 2):
                team_b = [p for p in candidates if p not in team_a]
                avg_a = (elo_values[candidates.index(team_a[0])] + elo_values[candidates.index(team_a[1])]) / 2
                avg_b = (elo_values[candidates.index(team_b[0])] + elo_values[candidates.index(team_b[1])]) / 2
                diff = abs(avg_a - avg_b)

                if diff < best_diff:
                    best_diff = diff
                    best_pair = (sorted(team_a), sorted(team_b))

            non_random_matches.append(best_pair)
            used_players.extend(candidates)

        # 生成随机比赛
        all_combinations = []
        available_players = [pid for pid in player_ids if pid not in used_players]

        # 生成所有可能的四人组合
        for quad in itertools.combinations(available_players, 4):
            # 生成所有可能的分组方式
            for team_a in itertools.combinations(quad, 2):
                team_b = tuple(p for p in quad if p not in team_a)
                sorted_teams = sorted([sorted(team_a), sorted(team_b)])
                all_combinations.append((sorted_teams[0], sorted_teams[1]))

        # 移除与非随机比赛重复的组合
        unique_combinations = [c for c in all_combinations if c not in non_random_matches]
        random.shuffle(unique_combinations)
        random_matches = unique_combinations[:random_num]

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

    # 显示待处理比赛
    st.subheader("待处理比赛列表")
    conn = sqlite3.connect('badminton.db')
    pending_matches = pd.read_sql("SELECT * FROM pending_matches", conn)
    conn.close()

    if pending_matches.empty:
        st.info("当前没有待处理的比赛")
    else:
        for _, match in pending_matches.iterrows():
            with st.expander(f"比赛 {match['id']}"):
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
                        score_a = st.number_input("A队得分", min_value=0, value=match['score_a'] or 0,
                                                  key=f"a_{match['id']}")
                    with col2:
                        score_b = st.number_input("B队得分", min_value=0, value=match['score_b'] or 0,
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
                        a1=match['player_a1'],
                        a2=match['player_a2'],
                        b1=match['player_b1'],
                        b2=match['player_b2'],
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
        "Elo排名": elo_page,
        "数据管理": manage_page,
        "比赛分配": match_scheduler_page  # 新增页面
    }
    page = st.sidebar.radio("页面", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
