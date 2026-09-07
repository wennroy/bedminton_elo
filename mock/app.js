const icons = {
  court: '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M12 4v16M3 12h18M6 4v16M18 4v16"/>',
  home: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  chart: '<path d="M4 3v17h17M8 15l4-5 4 2 5-7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  back: '<path d="M20 12H4m6-6-6 6 6 6"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/>',
  moon: '<path d="M20.8 13.1A9 9 0 0 1 10.9 3.2 9 9 0 1 0 20.8 13.1Z"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/>',
  users: '<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 5v1"/>',
  trophy: '<path d="M8 3h8v7a4 4 0 0 1-8 0V3ZM8 5H4v3a4 4 0 0 0 4 4m8-7h4v3a4 4 0 0 1-4 4M12 14v5M8 21h8M9 19h6"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  swap: '<path d="M4 7h16m-4-4 4 4-4 4M20 17H4m4-4-4 4 4 4"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  x: '<path d="m6 6 12 12M6 18 18 6"/>',
  reset: '<path d="M3 11a9 9 0 1 1 2 7M3 4v7h7"/>',
  calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.01"/>',
};
const icon = name => `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.court}</svg>`;
const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const sign = value => `${value > 0 ? '+' : ''}${value}`;
const tone = value => value >= 0 ? 'positive' : 'negative';
function readStorage(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch { return fallback; } }
function saveStorage(key, value) { try { localStorage.setItem(key, value); } catch {} }
const state = {
  me: Number(readStorage('courtside-identity', '1')), range: '4', filter: 'all', historyLimit: 5,
  search: '', sort: 'elo', slots: [null, null, null, null], scores: [21, 0], slotIndex: 0,
  pickerMode: 'slot', pickerSearch: '', submitted: false, resultIds: [],
};
if (!player(state.me)) state.me = 1;
const avatar = (id, size = '') => { const p = player(id); return `<span class="avatar ${p.color} ${size}" aria-hidden="true">${/^[小阿]/.test(p.name) ? p.name.slice(-1) : p.name[0]}</span>`; };
const person = (id, caption = '', size = '') => `<div class="person">${avatar(id, size)}<div><div class="person-name">${player(id).name}${id === state.me ? '<small>我</small>' : ''}</div>${caption ? `<div class="person-caption">${caption}</div>` : ''}</div></div>`;
const ranks = () => [...PLAYERS].sort((a, b) => DATA.ratings[b.id] - DATA.ratings[a.id]);
const rankOf = id => ranks().findIndex(p => p.id === id) + 1;
const myName = () => player(state.me).name;
function route() {
  const raw = location.hash.slice(1) || 'overview';
  if (/^player\/\d+$/.test(raw) && player(raw.split('/')[1])) return { page: 'player', id: Number(raw.split('/')[1]) };
  return { page: ['overview', 'players', 'record', 'signup', 'trends'].includes(raw) ? raw : 'overview' };
}
function brand() { return `<a href="#overview" class="brand" aria-label="卷技术小分队首页"><span class="brand-mark">${icon('court')}</span><div><div class="brand-name">卷技术小分队</div><div class="brand-sub">COURTSIDE CLUB</div></div></a>`; }
function navigation(page) {
  const items = [['overview', 'home', '俱乐部总览'], ['players', 'users', '球员分析'], ['trends', 'chart', '全员趋势'], ['signup', 'calendar', '每周报名'], ['record', 'plus', '记一场比赛']];
  return items.map(([key, glyph, label]) => `<a class="nav-item ${page === key || (key === 'players' && page === 'player') ? 'active' : ''}" href="#${key}" title="${label}" ${page === key || (key === 'players' && page === 'player') ? 'aria-current="page"' : ''}>${icon(glyph)}<span>${label}</span>${key === 'players' ? '<span class="count">08</span>' : ''}</a>`).join('');
}
function render(preserveScroll = false) {
  const y = window.scrollY;
  const current = route();
  const titles = { overview: '俱乐部总览', players: '球员分析', player: '球员分析', record: '记一场比赛', signup: '每周报名', trends: '全员趋势' };
  const theme = document.documentElement.dataset.theme;
  document.title = `${current.page === 'player' ? player(current.id).name + ' · 球员分析' : titles[current.page]} · 卷技术小分队原型`;
  document.getElementById('app').innerHTML = `
    <aside class="sidebar">${brand()}<div class="club-label">YOUR BADMINTON CLUB</div><nav class="side-nav" aria-label="主导航">${navigation(current.page)}</nav>
      <div class="side-note"><div class="court-mini" aria-hidden="true"></div><p>卷技术小分队<br>羽毛球双打数据</p></div>
      <button class="side-user" data-action="identity" aria-label="切换当前身份，${myName()}">${avatar(state.me)}<div><h3>${myName()}</h3><div class="person-caption">当前身份</div></div>${icon('chevron')}</button>
    </aside>
    <div class="shell"><header class="topbar"><div class="mobile-brand">${brand()}</div><div class="breadcrumb">俱乐部 <span>/</span> <strong>${titles[current.page]}</strong>${current.page === 'player' ? `<span>/</span> ${player(current.id).name}` : ''}</div>
      <div class="top-actions"><span class="top-date">2026 年 9 月 7 日 · 周一</span><span class="prototype-label">DESIGN 02</span><button class="icon-btn" data-action="theme" aria-label="${theme === 'dark' ? '切换浅色模式' : '切换深色模式'}" title="${theme === 'dark' ? '切换浅色模式' : '切换深色模式'}">${icon(theme === 'dark' ? 'sun' : 'moon')}</button><button data-action="identity" aria-label="切换身份">${avatar(state.me, 'small')}</button></div>
    </header><main id="main" class="${current.page === 'record' ? 'record-page' : ''}"><div class="${preserveScroll ? '' : 'page-enter'}">${current.page === 'overview' ? overview() : current.page === 'players' ? playersPage() : current.page === 'player' ? profile(current.id) : current.page === 'signup' ? signupPage() : current.page === 'trends' ? trendsPage() : recordPage()}</div><footer class="footnote"><span>设计预览 · 所有人物与比赛均为虚构示例</span><span class="wordmark">COURTSIDE · UI PROTOTYPE</span></footer></main></div>
    <nav class="bottom-nav" aria-label="手机主导航"><a class="bottom-item ${current.page === 'overview' ? 'active' : ''}" href="#overview" ${current.page === 'overview' ? 'aria-current="page"' : ''}>${icon('home')}<span>总览</span></a><a class="bottom-item ${['players', 'player', 'trends'].includes(current.page) ? 'active' : ''}" href="#players" ${['players', 'player', 'trends'].includes(current.page) ? 'aria-current="page"' : ''}>${icon('chart')}<span>球员</span></a><a class="bottom-item record ${current.page === 'record' ? 'active' : ''}" href="#record" ${current.page === 'record' ? 'aria-current="page"' : ''}><span class="nav-icon">${icon('plus')}</span><span>记一场</span></a><a class="bottom-item ${current.page === 'signup' ? 'active' : ''}" href="#signup" ${current.page === 'signup' ? 'aria-current="page"' : ''}>${icon('calendar')}<span>报名</span></a></nav>`;
  if (preserveScroll) window.scrollTo({ top: y, behavior: 'instant' });
  else window.scrollTo({ top: 0, behavior: 'instant' });
}
function overview() { return communityOverview(); }
function matchMini(m) {
  return `<div class="match-mini"><div class="flex between"><span class="small muted">${m.date.slice(5).replace('-', '.')} · 双打</span><span class="pill ${m.won ? 'positive' : 'negative'}">${m.won ? '胜' : '负'} · ELO ${sign(m.delta)}</span></div><div class="teams"><span class="team-names">${m.team.map(id => player(id).name).join(' / ')}</span><div class="num">${m.score}<span>:</span>${m.against}</div><span class="team-names muted">${m.opponents.map(id => player(id).name).join(' / ')}</span></div></div>`;
}
function playersPage() {
  return `<div class="page-heading"><div><div class="eyebrow">PLAYER DIRECTORY</div><h1>球员档案</h1><p></p></div><span class="pill">${PLAYERS.length} 位球友</span></div><div class="players-toolbar"><label class="search-field">${icon('search')}<input id="player-search" data-input="player-search" placeholder="搜索球员姓名" aria-label="搜索球员姓名" value="${esc(state.search)}"></label><div class="segmented" aria-label="球员排序"><button data-action="sort" data-value="elo" class="${state.sort === 'elo' ? 'active' : ''}" aria-pressed="${state.sort === 'elo'}">按 ELO</button><button data-action="sort" data-value="winRate" class="${state.sort === 'winRate' ? 'active' : ''}" aria-pressed="${state.sort === 'winRate'}">按胜率</button><button data-action="sort" data-value="total" class="${state.sort === 'total' ? 'active' : ''}" aria-pressed="${state.sort === 'total'}">按场次</button></div></div><div id="player-cards" class="player-grid">${playerCards()}</div>`;
}
function playerCards() {
  const entries = [...PLAYERS].filter(p => p.name.includes(state.search.trim())).sort((a, b) => summary(b.id)[state.sort] - summary(a.id)[state.sort]);
  if (!entries.length) return '<div class="empty-state">没有找到这位球员，试试其他名字。</div>';
  return entries.map(p => { const s = summary(p.id); return `<a class="panel player-card" href="#player/${p.id}" aria-label="查看${p.name}的详细分析">${avatar(p.id)}<span class="card-rank">#${String(rankOf(p.id)).padStart(2, '0')}</span><h2>${p.name}${p.id === state.me ? ' <span class="pill">我</span>' : ''}</h2><div class="small muted">${s.total} 场比赛 · ${s.winRate}% 胜率</div><div class="card-bottom"><div><div class="eyebrow">ELO RATING</div><div class="num">${s.elo}</div></div>${sparkline(p.id)}</div></a>`; }).join('');
}
function sparkline(id) {
  const points = DATA.history[id].map(p => p.elo).slice(-9);
  const min = Math.min(...points) - 5, max = Math.max(...points) + 5;
  return `<svg class="sparkline" viewBox="0 0 95 35" role="img" aria-label="${player(id).name}最近积分走势"><polyline points="${points.map((n, i) => `${i / (points.length - 1) * 95},${32 - (n - min) / (max - min) * 29}`).join(' ')}" fill="none" stroke="var(--chart)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}
function profile(id) {
  const p = player(id), s = summary(id);
  return `<a class="text-link back-link" href="#players">${icon('back')} 所有球员</a><div class="profile-heading"><div class="identity">${avatar(id, 'large')}<div><div class="eyebrow">PLAYER ${String(id).padStart(2, '0')} / ${p.en}</div><h1>${p.name} ${id === state.me ? '<span class="pill">我</span>' : ''}</h1><div class="rank-badge">俱乐部排名 <strong>#${rankOf(id)}</strong> <span style="padding:0 8px">·</span> 双打球员</div></div></div><div class="profile-actions"><button class="btn btn-secondary" data-action="choose-profile">${icon('users')} 切换球员</button></div></div>
    <section class="metrics" aria-label="球员核心数据"><div class="metric"><div class="metric-label">当前 ELO <button data-action="elo-info" aria-label="了解 ELO">${icon('info')}</button></div><div class="metric-value"><span class="num">${s.elo}</span><span class="pill ${tone(s.delta)}">${sign(s.delta)}</span></div><div class="metric-sub">较 08.31 前 · 最高 ${s.peak}</div></div><div class="metric"><div class="metric-label">生涯胜率</div><div class="metric-value"><span class="num">${s.winRate}<span class="unit">%</span></span></div><div class="metric-sub">${s.wins} 胜 / ${s.losses} 负</div></div><div class="metric"><div class="metric-label">累计出场</div><div class="metric-value"><span class="num">${s.total}<span class="unit">场</span></span></div><div class="metric-sub">生涯双打比赛</div></div><div class="metric"><div class="metric-label">当前状态</div><div class="metric-value"><span class="num">${s.streak}<span class="unit">连${s.streakWin ? '胜' : '负'}</span></span></div><div class="metric-sub">最近一场 · ${s.matches[0].date.slice(5).replace('-', '.')}</div></div></section>
    <div class="analysis-grid"><section class="panel panel-pad chart-panel"><div class="panel-heading"><div><h2>ELO 趋势</h2><div class="eyebrow">ELO PROGRESSION</div></div><div class="segmented" aria-label="积分趋势周期">${[['4', '近 4 周'], ['12', '近 12 周'], ['all', '全部']].map(([value, label]) => `<button data-action="range" data-value="${value}" class="${state.range === value ? 'active' : ''}" aria-pressed="${state.range === value}">${label}</button>`).join('')}</div></div><div id="chart-content">${chart(id)}</div></section>
      <section class="panel form-panel"><div class="panel-heading"><div><h2>近期战绩</h2><div class="eyebrow">RECENT FORM</div></div><span class="pill">近 8 场</span></div><div class="recent-form" aria-label="最近八场，从左到右由早到晚">${s.matches.slice(0, 8).reverse().map(m => `<span class="result-dot ${m.won ? '' : 'loss'}" title="${m.date} · ${m.score}:${m.against}">${m.won ? '胜' : '负'}</span>`).join('')}</div><div class="form-stats"><div class="form-stat"><span>近 8 场胜率</span><span class="num">${Math.round(s.matches.slice(0, 8).filter(m => m.won).length / Math.min(s.total, 8) * 100)}<small>%</small></span></div><div class="form-stat"><span>生涯场均净胜分</span><span class="num">${Number(s.avgDiff) > 0 ? '+' : ''}${s.avgDiff}</span></div></div><div class="insight">目前距离生涯最高积分 <strong>${s.peak - s.elo} 分</strong>。</div></section></div>
    <section class="relationships"><div class="section-label"><h2>搭档与对手</h2><small>生涯数据 · 至少搭档 / 交手 3 场</small></div><div class="relation-grid">${relationPanel(id, 'partners')}${relationPanel(id, 'opponents')}</div></section>
    <section class="panel history"><div class="panel-heading"><div><h2>比赛记录</h2><div class="eyebrow">MATCH HISTORY</div></div><div class="segmented" aria-label="比赛胜负筛选">${[['all', '全部'], ['wins', '获胜'], ['losses', '失利']].map(([value, label]) => `<button data-action="filter" data-value="${value}" class="${state.filter === value ? 'active' : ''}" aria-pressed="${state.filter === value}">${label}</button>`).join('')}</div></div><div id="match-history">${historyRows(id)}</div></section>`;
}
function chart(id) {
  const cutoff = state.range === 'all' ? '2000-01-01' : new Date(Date.parse(DEMO_TODAY + 'T00:00:00Z') - Number(state.range) * 7 * 86400000).toISOString().slice(0, 10);
  let points = DATA.history[id].filter(p => p.date >= cutoff);
  const previous = DATA.history[id].filter(p => p.date < cutoff).at(-1);
  if (previous) points = [{ date: cutoff, elo: previous.elo }, ...points];
  const viewport = window.innerWidth;
  const width = viewport <= 760 ? viewport - 84 : Math.max(300, Math.min(800, (viewport - (viewport <= 1000 ? 76 : viewport <= 1190 ? 192 : 222) - 80) * .65 - 60));
  const height = 215, left = 33, right = width - 12, top = 20, bottom = height - 30;
  const min = Math.floor((Math.min(...points.map(p => p.elo)) - 10) / 20) * 20;
  const max = Math.ceil((Math.max(...points.map(p => p.elo)) + 10) / 20) * 20;
  const start = Date.parse(points[0].date), end = Date.parse(points.at(-1).date);
  const positions = points.map(p => ({ ...p, x: left + (Date.parse(p.date) - start) / Math.max(1, end - start) * (right - left), y: bottom - (p.elo - min) / (max - min) * (bottom - top) }));
  const path = positions.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const diff = Math.round(points.at(-1).elo) - Math.round(points[0].elo);
  return `<div class="chart-meta"><span class="chart-legend">${player(id).name}</span><span id="chart-readout" aria-live="polite">${points.at(-1).date.slice(5).replace('-', '.')} · ELO ${Math.round(points.at(-1).elo)}</span></div>
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="group" aria-label="${player(id).name}的 ELO 趋势，可聚焦数据点查看日期和积分"><defs><linearGradient id="chart-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="var(--chart-fill)" stop-opacity=".24"/><stop offset="100%" stop-color="var(--chart-fill)" stop-opacity="0"/></linearGradient></defs>${[0, 1, 2, 3].map(i => { const y = top + (bottom - top) * i / 3; return `<line class="chart-grid" x1="${left}" y1="${y}" x2="${right}" y2="${y}"/><text x="0" y="${y + 3}">${Math.round(max - (max - min) * i / 3)}</text>`; }).join('')}
    <path d="${path} L${right},${bottom} L${left},${bottom} Z" fill="url(#chart-area)"/><path d="${path}" stroke="var(--chart)" stroke-width="2.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/>
    ${positions.map((p, i) => `<circle class="chart-point" cx="${p.x}" cy="${p.y}" r="${i === positions.length - 1 ? 4 : 2.5}"/><circle class="chart-hit" tabindex="0" role="img" aria-label="${p.date}，ELO ${Math.round(p.elo)}" data-chart-date="${p.date}" data-chart-elo="${Math.round(p.elo)}" cx="${p.x}" cy="${p.y}" r="11"><title>${p.date} · ${Math.round(p.elo)}</title></circle>`).join('')}
    ${[0, Math.floor((points.length - 1) / 2), points.length - 1].map((idx, i) => `<text x="${positions[idx].x}" y="${height - 4}" text-anchor="${i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}">${points[idx].date.slice(5).replace('-', '.')}</text>`).join('')}</svg><div class="chart-caption"><span>区间变化 <strong class="${tone(diff)}">${sign(diff)} ELO</strong></span><span>按比赛日汇总 · 触碰数据点查看</span></div>`;
}
function relationPanel(id, kind) {
  const entries = relations(id, kind).filter(p => p.total >= 3);
  const best = kind === 'partners' ? entries[0] : [...entries].sort((a, b) => a.rate - b.rate || b.total - a.total)[0];
  const isPartner = kind === 'partners';
  if (!best) return `<section class="panel panel-pad"><h2>${isPartner ? '黄金搭档' : '值得研究的对手'}</h2><div class="empty-state">${isPartner ? '搭档' : '交手'}满 3 场后解锁</div></section>`;
  return `<section class="panel panel-pad"><div class="panel-heading"><h2>${isPartner ? '黄金搭档' : '值得研究的对手'}</h2><button class="text-link" data-action="relations" data-kind="${kind}">全部 ${icon('arrow')}</button></div><a class="relation-hero ${isPartner ? '' : 'opponent'}" href="#player/${best.id}">${person(best.id, isPartner ? '搭档胜率最高' : '对阵胜率最低')}<div class="right"><div class="num">${best.rate}<span>%</span></div><div class="small muted">${isPartner ? '搭档' : '对阵'}胜率 · ${best.total} 场</div></div></a><div class="${isPartner ? '' : 'opponent-bars'}">${entries.slice(0, 3).map(p => `<div class="relation-row"><a href="#player/${p.id}" class="person">${avatar(p.id, 'small')}${p.name}</a><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${p.rate}%"></div></div><span class="num">${p.rate}%</span><span class="muted" style="text-align:right">${p.wins}胜 ${p.losses}负</span></div>`).join('')}</div></section>`;
}
function historyRows(id) {
  const matches = personalMatches(id).filter(m => state.filter === 'all' || (state.filter === 'wins' ? m.won : !m.won));
  if (!matches.length) return '<div class="empty-state">这个筛选下还没有比赛记录。</div>';
  return matches.slice(0, state.historyLimit).map(m => `<div class="match-row"><span class="date">${m.date.replaceAll('-', '.')}</span><span class="result-dot ${m.won ? '' : 'loss'}">${m.won ? '胜' : '负'}</span><div class="team">${m.team.map(id => player(id).name).join(' / ')}<small>我方阵容</small></div><span class="score num">${m.score}<span>:</span>${m.against}</span><div class="team">${m.opponents.map(id => player(id).name).join(' / ')}<small>对方阵容</small></div><span class="delta num ${tone(m.delta)}">${sign(m.delta)}</span></div>`).join('') + `<div class="table-foot">${matches.length > state.historyLimit ? `<button class="text-link" data-action="load-more">再看 5 场 · 共 ${matches.length} 场 ${icon('chevron')}</button>` : `<span class="text-link">已展示全部 ${matches.length} 场比赛</span>`}</div>`;
}
function recordPage() {
  return `<div class="page-heading"><div><div class="eyebrow">MATCH RECORD</div><h1>录入比赛</h1><p>选择双方阵容，输入最终比分。</p></div></div><div class="record-layout"><div class="record-main"><div class="record-meta"><span class="flex">${icon('calendar')} 2026.09.07 · 今天</span><button class="text-link" data-action="identity">录入人：${myName()} ${icon('chevron')}</button></div><section class="court-board" aria-label="双打比赛记分板"><div class="board-header"><span class="eyebrow">DOUBLES / 双打</span><small>FINAL SCORE</small></div><div class="court-teams">${[0, 1].map(team => `<div class="team-column ${team ? 'team-b' : ''}"><span class="team-label">TEAM ${team ? 'B' : 'A'} / ${team ? 'B' : 'A'} 队</span><div class="team-slots">${[team * 2, team * 2 + 1].map(index => `<button class="slot ${state.slots[index] ? '' : 'empty'}" data-action="slot" data-index="${index}" aria-label="选择${team ? 'B' : 'A'}队${index % 2 + 1}号球员${state.slots[index] ? '，当前' + player(state.slots[index]).name : ''}">${state.slots[index] ? avatar(state.slots[index]) : icon('plus')}<span>${state.slots[index] ? player(state.slots[index]).name : '选择球员'}</span></button>`).join('')}</div><input class="score-input" type="number" inputmode="numeric" min="0" max="99" step="1" value="${state.scores[team]}" data-input="score" data-team="${team}" aria-label="${team ? 'B' : 'A'}队比分"><div class="score-controls"><button data-action="score" data-team="${team}" data-delta="-1" aria-label="${team ? 'B' : 'A'}队减一分" ${state.scores[team] <= 0 ? 'disabled' : ''}>−</button><button data-action="score" data-team="${team}" data-delta="1" aria-label="${team ? 'B' : 'A'}队加一分" ${state.scores[team] >= 99 ? 'disabled' : ''}>+</button></div></div>`).join('')}</div><div class="board-footer"><button data-action="swap">${icon('swap')} 交换两边</button><button data-action="reset">${icon('reset')} 清空阵容</button><small>点击数字可直接输入</small></div></section><div id="record-validation" class="error-message" role="status"></div><button id="submit-record" class="btn btn-primary record-submit" data-action="review-record" ${canSubmit() ? '' : 'disabled'}>${icon('check')} 确认比分 ${icon('arrow')}</button><p class="record-note" id="record-note">${recordHint()}</p></div>
      <aside class="record-side"><section class="panel panel-pad"><div class="eyebrow">MATCH ENTRY</div><h2 style="margin-top:9px">录入说明</h2><div class="instruction-step"><span class="step">01</span><div><h3>按实际阵容选人</h3><p>左右各一队，每队两位球员。</p></div></div><div class="instruction-step"><span class="step">02</span><div><h3>输入最终比分</h3><p>支持直接输入，也可用 + / − 调整。</p></div></div><div class="instruction-step"><span class="step">03</span><div><h3>确认后查看积分变化</h3><p>查看四位球员的积分变化。</p></div></div></section><section class="panel panel-pad"><div class="flex between"><h3>你的最近一场</h3><a class="text-link" href="#player/${state.me}">${icon('arrow')}</a></div>${matchMini(personalMatches(state.me)[0])}</section></aside></div>`;
}
function canSubmit() { return !state.submitted && state.slots.every(Boolean) && new Set(state.slots).size === 4 && state.scores.every(n => Number.isInteger(n) && n >= 0 && n <= 99) && state.scores[0] !== state.scores[1]; }
function recordHint() {
  if (state.submitted) return '这场演示比赛已记录，继续记分可开始下一场。';
  const count = state.slots.filter(Boolean).length;
  if (count < 4) return `已选 ${count} / 4 人 · 选好双方球员后确认比分`;
  if (state.scores[0] === state.scores[1]) return '比赛不能以平局结束，请填写最终比分。';
  if (!state.scores.every(n => Number.isInteger(n) && n >= 0 && n <= 99)) return '请输入 0–99 的整数比分。';
  return '确认后展示四位球员的 ELO 变化';
}

const dialog = document.getElementById('dialog');
let dialogTrigger = null;
function openDialog(content) {
  if (!dialog.open) dialogTrigger = document.activeElement;
  dialog.innerHTML = content;
  if (!dialog.open) dialog.showModal();
}
function dialogHead(title) { return `<div class="dialog-heading"><h2 id="dialog-title">${title}</h2><button class="icon-btn" data-action="close-dialog" aria-label="关闭弹窗">${icon('x')}</button></div>`; }
function closeDialog() {
  dialog.close();
  if (dialogTrigger?.isConnected) dialogTrigger.focus({ preventScroll: true });
}
function picker() {
  const title = state.pickerMode === 'identity' ? '选择当前身份' : state.pickerMode === 'profile' ? '选择球员' : `选择 ${state.slotIndex < 2 ? 'A' : 'B'} 队 · ${state.slotIndex % 2 + 1} 号球员`;
  const description = state.pickerMode === 'identity' ? '选择用于报名和录入比赛的身份。' : state.pickerMode === 'profile' ? '选择球员以查看详细数据。' : '已在其他位置的球员不能重复选择。';
  openDialog(`${dialogHead(title)}<p class="dialog-desc">${description}</p><label class="search-field">${icon('search')}<input autofocus data-input="picker-search" placeholder="搜索球员姓名" aria-label="搜索可选球员" value="${esc(state.pickerSearch)}"></label><div class="picker-list" id="picker-list">${pickerPeople()}</div>${state.pickerMode === 'slot' && state.slots[state.slotIndex] ? '<div class="dialog-actions"><button class="btn btn-secondary" data-action="remove-slot">移除此位置球员</button></div>' : ''}`);
}
function pickerPeople() {
  const entries = PLAYERS.filter(p => p.name.includes(state.pickerSearch.trim()));
  if (!entries.length) return '<div class="empty-state">没有匹配的球员</div>';
  return entries.map(p => {
    const occupied = state.pickerMode === 'slot' && state.slots.includes(p.id) && state.slots[state.slotIndex] !== p.id;
    const selected = state.pickerMode === 'identity' ? state.me === p.id : state.pickerMode === 'slot' ? state.slots[state.slotIndex] === p.id : route().id === p.id;
    return `<button class="picker-person ${selected ? 'selected' : ''}" data-action="pick" data-id="${p.id}" ${occupied ? 'disabled' : ''} aria-label="${p.name}${occupied ? '，已在阵容中' : ''}">${avatar(p.id)}<span>${p.name}<small>${occupied ? '已在阵容中' : selected ? '当前选择' : `ELO ${summary(p.id).elo}`}</small></span></button>`;
  }).join('');
}
function showRelations(kind) {
  const id = route().id;
  const entries = relations(id, kind);
  openDialog(`${dialogHead(kind === 'partners' ? '搭档表现' : '对手交锋')}<p class="dialog-desc">${player(id).name}的生涯数据 · 胜率均为${player(id).name}一方的胜率。</p>${entries.map(p => `<a href="#player/${p.id}" class="delta-row">${person(p.id, `${p.wins} 胜 ${p.losses} 负 · ${p.total} 场${p.total < 3 ? ' · 样本较少' : ''}`)}<span class="num">${p.rate}%</span></a>`).join('')}`);
}
function reviewRecord() {
  if (!canSubmit()) return;
  openDialog(`${dialogHead('确认这场比赛')}<p class="dialog-desc">检查双方阵容和最终比分。当前为原型，确认仅记录一场本地演示比赛。</p><div class="confirm-score"><div>${state.slots.slice(0, 2).map(id => `<div>${player(id).name}</div>`).join('')}<div class="small" style="margin-top:8px">A 队</div></div><span class="num">${state.scores[0]} : ${state.scores[1]}</span><div>${state.slots.slice(2).map(id => `<div>${player(id).name}</div>`).join('')}<div class="small" style="margin-top:8px">B 队</div></div></div><div class="dialog-actions"><button class="btn btn-secondary" data-action="close-dialog">返回修改</button><button class="btn btn-primary" data-action="submit-demo">确认 · 演示记分</button></div>`);
}
function submitDemo() {
  if (!canSubmit()) return;
  const m = { id: MATCHES.length + 1, date: DEMO_TODAY, a: state.slots.slice(0, 2), b: state.slots.slice(2), scoreA: state.scores[0], scoreB: state.scores[1] };
  state.submitted = true;
  MATCHES.push(m);
  DATA = replay();
  state.resultIds = [...state.slots];
  render(true);
  const submit = document.getElementById('submit-record');
  if (submit) {
    submit.disabled = false;
    submit.dataset.action = 'next-record';
    submit.innerHTML = `${icon('plus')} 继续记下一场 ${icon('arrow')}`;
  }
  openDialog(`<div class="success-icon">${icon('check')}</div><h2 id="dialog-title" class="success-title">比赛已记录</h2><p class="dialog-desc success-title" style="margin-top:7px">演示比赛已更新到球员分析 · 刷新后还原</p><div class="delta-list">${state.resultIds.map(id => { const delta = Math.round(m.after[id]) - Math.round(m.before[id]); return `<div class="delta-row">${person(id, `${Math.round(m.before[id])} → ${Math.round(m.after[id])} ELO`)}<span class="num ${tone(delta)}">${sign(delta)}</span></div>`; }).join('')}</div><div class="dialog-actions"><button class="btn btn-secondary" data-action="result-profile">查看我的数据</button><button class="btn btn-primary" data-action="next-record">再记一场 ${icon('arrow')}</button></div>`);
}
function nextRecord() {
  state.submitted = false;
  state.scores = [21, 0];
  closeDialog();
  if (route().page !== 'record') location.hash = 'record';
  else render(true);
  toast('已保留双方阵容，可以继续记分或更换球员。');
}
let toastTimer;
function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 3500);
}
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#141815' : '#f4f5f0';
  document.querySelectorAll('[data-action="theme"]').forEach(button => {
    button.innerHTML = icon(theme === 'dark' ? 'sun' : 'moon');
    button.setAttribute('aria-label', theme === 'dark' ? '切换浅色模式' : '切换深色模式');
    button.title = button.getAttribute('aria-label');
  });
}
function updateRecordControls() {
  const submit = document.getElementById('submit-record');
  if (submit) {
    submit.disabled = state.submitted ? false : !canSubmit();
    submit.dataset.action = state.submitted ? 'next-record' : 'review-record';
    submit.innerHTML = state.submitted ? `${icon('plus')} 继续记下一场 ${icon('arrow')}` : `${icon('check')} 确认比分 ${icon('arrow')}`;
  }
  const note = document.getElementById('record-note');
  if (note) note.textContent = recordHint();
  for (const button of document.querySelectorAll('[data-action="score"]')) {
    const score = state.scores[Number(button.dataset.team)];
    button.disabled = Number(button.dataset.delta) < 0 ? score <= 0 : score >= 99;
  }
}
function setSegment(button) {
  for (const sibling of button.parentElement.querySelectorAll('button')) {
    const active = sibling === button;
    sibling.classList.toggle('active', active);
    sibling.setAttribute('aria-pressed', String(active));
  }
}
document.addEventListener('click', event => {
  if (dialog.open && event.target.closest('a[href^="#"]')) closeDialog();
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  const action = button.dataset.action;
  if (action === 'theme') {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    saveStorage('courtside-theme', theme);
    setTheme(theme);
  } else if (action === 'identity' || action === 'choose-profile' || action === 'slot') {
    state.pickerMode = action === 'identity' ? 'identity' : action === 'choose-profile' ? 'profile' : 'slot';
    state.slotIndex = Number(button.dataset.index || 0);
    state.pickerSearch = '';
    picker();
  } else if (action === 'close-dialog') closeDialog();
  else if (action === 'pick') {
    const id = Number(button.dataset.id);
    if (state.pickerMode === 'identity') {
      state.me = id;
      community.partySize = activeSignup()?.size || 1;
      saveStorage('courtside-identity', String(id));
      closeDialog();
      render(true);
      updateRecordControls();
      toast(`当前身份：${myName()}`);
    } else if (state.pickerMode === 'profile') {
      closeDialog();
      location.hash = `player/${id}`;
    } else {
      state.slots[state.slotIndex] = id;
      closeDialog();
      render(true);
      updateRecordControls();
      document.querySelector(`[data-action="slot"][data-index="${state.slotIndex}"]`)?.focus({ preventScroll: true });
    }
  } else if (action === 'remove-slot') {
    state.slots[state.slotIndex] = null;
    closeDialog(); render(true); updateRecordControls();
  } else if (action === 'range') {
    state.range = button.dataset.value;
    setSegment(button);
    document.getElementById('chart-content').innerHTML = chart(route().id);
  } else if (action === 'filter') {
    state.filter = button.dataset.value;
    state.historyLimit = 5;
    setSegment(button);
    document.getElementById('match-history').innerHTML = historyRows(route().id);
  } else if (action === 'load-more') {
    state.historyLimit += 5;
    document.getElementById('match-history').innerHTML = historyRows(route().id);
  } else if (action === 'sort') {
    state.sort = button.dataset.value;
    setSegment(button);
    document.getElementById('player-cards').innerHTML = playerCards();
  } else if (action === 'relations') showRelations(button.dataset.kind);
  else if (action === 'elo-info') openDialog(`${dialogHead('ELO 是什么？')}<p class="dialog-desc">ELO 用比赛结果估计球员的相对水平。赢下比赛通常会加分，输球通常会减分；对手越强，获胜时通常加得越多。</p><p class="small muted">本原型按现有应用的双打规则，从 1000 分起算。头像旁的数字是俱乐部排名，绿色 / 红色变化值均同时带有正负号。</p><div class="dialog-actions"><button class="btn btn-primary" data-action="close-dialog">知道了</button></div>`);
  else if (action === 'score') {
    const team = Number(button.dataset.team);
    state.scores[team] = Math.max(0, Math.min(99, (Number.isFinite(state.scores[team]) ? state.scores[team] : 0) + Number(button.dataset.delta)));
    document.querySelector(`[data-input="score"][data-team="${team}"]`).value = state.scores[team];
    updateRecordControls();
  } else if (action === 'swap') {
    state.slots = [...state.slots.slice(2), ...state.slots.slice(0, 2)];
    state.scores.reverse();
    render(true); updateRecordControls(); toast('双方阵容和比分已一起交换。');
  } else if (action === 'reset') {
    state.slots = [null, null, null, null];
    state.submitted = false;
    render(true); toast('已清空阵容，比分已保留。');
  } else if (action === 'review-record') reviewRecord();
  else if (action === 'submit-demo') submitDemo();
  else if (action === 'next-record') nextRecord();
  else if (action === 'result-profile') { closeDialog(); location.hash = `player/${state.me}`; }
});
document.addEventListener('input', event => {
  const input = event.target;
  if (input.dataset.input === 'player-search') {
    state.search = input.value;
    document.getElementById('player-cards').innerHTML = playerCards();
  } else if (input.dataset.input === 'picker-search') {
    state.pickerSearch = input.value;
    document.getElementById('picker-list').innerHTML = pickerPeople();
  } else if (input.dataset.input === 'score') {
    state.scores[Number(input.dataset.team)] = input.value === '' ? NaN : Number(input.value);
    updateRecordControls();
  }
});
function chartReadout(event) {
  const point = event.target.closest('[data-chart-date]');
  if (point) document.getElementById('chart-readout').textContent = `${point.dataset.chartDate.slice(5).replace('-', '.')} · ELO ${point.dataset.chartElo}`;
}
document.addEventListener('pointerover', chartReadout);
document.addEventListener('focusin', chartReadout);
document.addEventListener('click', chartReadout);
window.addEventListener('hashchange', () => {
  state.filter = 'all'; state.historyLimit = 5;
  if (dialog.open) closeDialog();
  render(); updateRecordControls();
});
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (route().page === 'player') document.getElementById('chart-content').innerHTML = chart(route().id); }, 160);
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
  if (!readStorage('courtside-theme', '')) setTheme(event.matches ? 'dark' : 'light');
});
render();
setTheme(document.documentElement.dataset.theme);
