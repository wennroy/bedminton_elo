/* Community views share the same fictional matches as the player profiles. */
const community = {
  range: '12', mode: 'elo', selected: new Set(PLAYERS.map(p => p.id)), inspectedDate: null,
  quoteOffset: 0, partySize: 1,
  signups: [{ id: 2, size: 1 }, { id: 3, size: 2 }, { id: 4, size: 1 }, { id: 5, size: 1 }, { id: 6, size: 2 }],
};
const ORIGINAL_QUOTES = [
  '把注意力留给下一拍。',
  '球还没落地，就还有下一种可能。',
  '最好的配合，是有人接住你的下一拍。',
  '比分会归零，练习不会。',
  '先站稳，再出手。',
  '有些答案，要多打一场才知道。',
];
function mondayOf(date) {
  const shanghai = new Date(date.getTime() + 8 * 3600000);
  const day = shanghai.getUTCDay();
  return new Date(Date.UTC(shanghai.getUTCFullYear(), shanghai.getUTCMonth(), shanghai.getUTCDate() - (day + 6) % 7));
}
function weeklyQuote(date = new Date(), offset = community.quoteOffset) {
  const start = mondayOf(date);
  start.setUTCDate(start.getUTCDate() + offset * 7);
  const end = new Date(start.getTime() + 6 * 86400000);
  const index = Math.floor(start.getTime() / (7 * 86400000));
  return { key: start.toISOString().slice(0, 10), text: ORIGINAL_QUOTES[((index % ORIGINAL_QUOTES.length) + ORIGINAL_QUOTES.length) % ORIGINAL_QUOTES.length], dateLabel: `${start.toISOString().slice(0, 10).replaceAll('-', '.')} — ${end.toISOString().slice(5, 10).replace('-', '.')}` };
}
function quoteHeading() {
  const q = weeklyQuote();
  return `<div class="quote-heading"><div><div class="eyebrow">每周名句 <span class="quote-date">${q.dateLabel}</span></div><h1 id="weekly-quote" data-week="${q.key}">${q.text}</h1><div class="quote-credit">原创 · 每周一更新</div></div><div class="quote-controls"><button class="icon-btn" data-community="quote-prev" aria-label="查看上周名句">${icon('back')}</button><button class="icon-btn" data-community="quote-next" aria-label="查看下一周名句" ${community.quoteOffset >= 0 ? 'disabled' : ''}>${icon('arrow')}</button></div></div>`;
}
function signupSummaryDemo() {
  const count = community.signups.length;
  const total = community.signups.reduce((sum, entry) => sum + entry.size, 0);
  return { count, total, guests: total - count };
}
function activeSignup() { return community.signups.find(entry => entry.id === state.me); }
function signupPreview() {
  const sums = signupSummaryDemo(), mine = activeSignup();
  return `<section class="panel signup-preview"><div class="flex between"><h2>本周报名</h2><span class="pill ${mine ? 'positive' : ''}">${mine ? '已报名' : '报名中'}</span></div><div class="signup-preview-event"><div class="date-tile"><span>9 月</span><strong class="num">09</strong><span>周三</span></div><div><h3>周三羽毛球局</h3><p class="muted small">18:00–20:00</p><div class="signup-tally"><strong class="num" data-signup-total>${sums.total}</strong> <span class="muted small">人参加 · 含 ${sums.guests} 位小伙伴</span></div></div></div><a class="btn btn-secondary" href="#signup">${mine ? '查看我的报名' : '查看名单 · 去报名'} ${icon('arrow')}</a></section>`;
}
function communityOverview() {
  const s = summary(state.me);
  const recentCount = MATCHES.filter(m => m.date >= '2026-08-31').length;
  return `<div id="quote-heading">${quoteHeading()}</div><div class="overview-top"><section class="welcome-card overview-summary"><div class="court-art" aria-hidden="true"><span></span></div><div class="flex between" style="width:100%;position:relative"><h2>${myName()}</h2><span class="summary-rank">俱乐部 #${rankOf(state.me)}</span></div><div class="overview-personal-stats"><div><span>当前 ELO</span><strong class="num">${s.elo}</strong></div><div><span>生涯胜率</span><strong class="num">${s.winRate}<small>%</small></strong></div><div><span>近一周比赛 · 全员</span><strong class="num">${recentCount}<small>场</small></strong></div></div><div class="flex wrap"><a class="btn btn-primary" href="#record">${icon('plus')} 记一场比赛</a><a class="summary-link" href="#player/${state.me}">我的数据 ${icon('arrow')}</a></div></section>${signupPreview()}</div>
    ${clubTrendPanel(true)}
    <div class="overview-bottom"><section class="panel panel-pad"><div class="panel-heading"><h2>球员排行榜</h2><a class="text-link" href="#players">全部球员 ${icon('arrow')}</a></div><table class="ranking-table"><thead><tr><th>#</th><th>球员</th><th class="optional">胜率</th><th>ELO</th><th>近一周</th></tr></thead><tbody>${ranks().slice(0, 6).map((p, i) => { const row = summary(p.id); return `<tr class="${p.id === state.me ? 'is-me' : ''}"><td class="num">${String(i + 1).padStart(2, '0')}</td><td><a href="#player/${p.id}">${person(p.id)}</a></td><td class="optional muted">${row.winRate}%</td><td><span class="num">${row.elo}</span></td><td><span class="${tone(row.delta)}">${sign(row.delta)}</span></td></tr>`; }).join('')}</tbody></table></section><section class="panel panel-pad"><div class="panel-heading"><h2>我的最近比赛</h2><a class="text-link" href="#player/${state.me}">全部战绩 ${icon('arrow')}</a></div>${s.matches.slice(0, 4).map(matchMini).join('')}</section></div>`;
}
function signupPage() {
  const sums = signupSummaryDemo(), mine = activeSignup();
  const size = mine?.size || community.partySize;
  return `<div class="page-heading"><div><div class="eyebrow">WEEKLY SESSION</div><h1>每周报名</h1></div><span class="pill positive">报名中</span></div><section class="panel session-banner"><div class="session-title"><div class="date-tile large-date"><span>2026 / 09</span><strong class="num">09</strong><span>WED · 周三</span></div><div><h2>周三羽毛球局</h2><div class="flex muted small">${icon('calendar')} 9 月 9 日 · 18:00–20:00</div><p class="small muted">每周三固定场次</p></div></div><div class="session-counts"><div><strong class="num" data-signup-total>${sums.total}</strong><span>参加人数</span></div><div><strong class="num">${sums.count}</strong><span>报名成员</span></div><div><strong class="num">${sums.guests}</strong><span>随行小伙伴</span></div></div></section>
    <div class="signup-layout"><section class="panel roster-panel"><div class="panel-heading"><h2>报名名单</h2><span class="muted small">${sums.count} 位成员 · 共 ${sums.total} 人</span></div><div class="roster-labels"><span>成员</span><span>参加人数</span></div>${community.signups.length ? community.signups.map((entry, index) => `<div class="roster-row"><span class="roster-index num">${String(index + 1).padStart(2, '0')}</span><a href="#player/${entry.id}">${person(entry.id, entry.size === 2 ? '携带 1 位小伙伴' : '本人参加')}</a><span class="roster-size num">${entry.size}<small> 人</small></span></div>`).join('') : '<div class="empty-state">本期还没有人报名。</div>'}</section>
    <aside class="signup-action panel panel-pad"><div class="panel-heading"><h2>我的报名</h2>${mine ? '<span class="pill positive">已报名</span>' : '<span class="pill">未报名</span>'}</div><div class="signup-identity">${person(state.me, '当前报名身份')}<button class="text-link" data-action="identity">切换 ${icon('chevron')}</button></div><div class="signup-field-label">参加人数</div><div class="party-options">${[1, 2].map(n => `<button data-community="party-size" data-value="${n}" class="party-option ${size === n ? 'selected' : ''}" aria-pressed="${size === n}"><span class="option-check">${size === n ? icon('check') : ''}</span><div><strong>${n === 1 ? '自己来' : '带一位小伙伴'}</strong><small>${n === 1 ? '共 1 人' : '共 2 人'}</small></div></button>`).join('')}</div>${mine ? `<div class="signup-confirmed">${icon('check')} 已报名 · ${mine.size} 人参加</div><button class="btn btn-secondary signup-submit" data-community="cancel-signup">取消报名</button><p class="record-note">切换人数会自动更新报名</p>` : '<button class="btn btn-primary signup-submit" data-community="join-signup">确认报名 '+icon('arrow')+'</button>'}<div class="signup-notes"><h3>报名说明</h3><p>每位成员最多带 1 位小伙伴。<br>本期报名于周三 20:00 切换至下周。<br>演示报名仅在本页会话内有效。</p></div></aside></div>`;
}
function trendsPage() {
  return `<div class="page-heading"><div><div class="eyebrow">CLUB STATISTICS</div><h1>全员 ELO 趋势</h1><p>按比赛日汇总 · 可选择成员对比</p></div><a href="#players" class="btn btn-secondary">球员档案 ${icon('arrow')}</a></div>${clubTrendPanel(false)}`;
}
function clubTrendPanel(compact) {
  return `<section class="panel club-trend ${compact ? 'on-overview' : ''}"><div class="panel-heading"><div><h2>${compact ? '全员 ELO 趋势' : '积分与排名'}</h2><p class="muted small">${PLAYERS.length} 位成员 · <span id="trend-selection-count">${community.selected.size}</span> 位已选</p></div>${compact ? `<a href="#trends" class="text-link">展开大图 ${icon('arrow')}</a>` : '<span class="pill">初始 ELO 1000</span>'}</div><div class="club-chart-toolbar"><div class="segmented" aria-label="全员趋势类型">${[['elo', 'ELO 积分'], ['rank', '排名']].map(([key, title]) => `<button data-community="trend-mode" data-value="${key}" aria-pressed="${community.mode === key}" class="${community.mode === key ? 'active' : ''}">${title}</button>`).join('')}</div><div class="segmented" aria-label="全员趋势周期">${[['4', '近 4 周'], ['12', '近 12 周'], ['all', '全部']].map(([key, title]) => `<button data-community="trend-range" data-value="${key}" aria-pressed="${community.range === key}" class="${community.range === key ? 'active' : ''}">${title}</button>`).join('')}</div></div><div class="club-chart-layout"><div class="club-chart-plot" id="club-chart-plot">${clubChart()}</div><aside class="club-readout"><div class="readout-heading"><span id="club-readout-date">${inspectedClubDate()}</span><span>${community.mode === 'rank' ? '名次 / ELO' : 'ELO'}</span></div><div id="club-values">${clubValues()}</div></aside></div><div class="club-chart-footer"><span>点击日期查看当日排名与积分</span><span>颜色与球员固定对应</span></div><div class="club-selector"><div class="flex between wrap"><h3>选择成员</h3><div class="flex"><button class="text-link" data-community="select-all">全选</button><button class="text-link" data-community="select-me">只看自己</button><button class="text-link" data-community="select-none">清空</button></div></div><div class="player-chips">${PLAYERS.map((p, index) => `<button class="player-chip ${community.selected.has(p.id) ? 'selected' : ''}" style="--series:var(--series-${index + 1})" data-community="toggle-player" data-id="${p.id}" aria-pressed="${community.selected.has(p.id)}"><span class="series-dot"></span>${p.name}${p.id === state.me ? '<small>我</small>' : ''}${icon('check')}</button>`).join('')}</div></div></section>`;
}
function clubDates() {
  const all = DATA.history[PLAYERS[0].id].map(p => p.date);
  if (community.range === 'all') return all;
  const cutoff = new Date(Date.parse(DEMO_TODAY + 'T00:00:00Z') - Number(community.range) * 7 * 86400000).toISOString().slice(0, 10);
  return [cutoff, ...all.filter(date => date > cutoff)];
}
function ratingAt(id, date) { return Math.round(DATA.history[id].filter(p => p.date <= date).at(-1)?.elo ?? 1000); }
function rankingAt(date) { return [...PLAYERS].sort((a, b) => ratingAt(b.id, date) - ratingAt(a.id, date) || a.id - b.id); }
function inspectedClubDate() { const dates = clubDates(); return dates.includes(community.inspectedDate) ? community.inspectedDate : dates.at(-1); }
function clubValues() {
  const date = inspectedClubDate();
  return rankingAt(date).map((p, index) => ({ ...p, rank: index + 1 })).filter(p => community.selected.has(p.id)).map(p => `<a href="#player/${p.id}" class="club-value" style="--series:var(--series-${PLAYERS.findIndex(entry => entry.id === p.id) + 1})"><span class="value-rank num">${String(p.rank).padStart(2, '0')}</span><span class="series-dot"></span><span>${p.name}</span><strong class="num">${ratingAt(p.id, date)}</strong></a>`).join('') || '<p class="small muted" style="padding:12px 0">尚未选择成员</p>';
}
function clubChart() {
  if (!community.selected.size) return '<div class="empty-trend">选择下方成员，查看 ELO 趋势。</div>';
  const dates = clubDates();
  const width = window.innerWidth <= 760 ? Math.max(280, window.innerWidth - 82) : 720;
  const height = window.innerWidth <= 760 ? 265 : 290;
  const left = 37, right = width - 16, top = 18, bottom = height - 32;
  const rankMode = community.mode === 'rank';
  const selected = PLAYERS.filter(p => community.selected.has(p.id));
  const values = selected.flatMap(p => dates.map(date => ratingAt(p.id, date)));
  const min = rankMode ? 1 : Math.floor((Math.min(...values) - 15) / 50) * 50;
  const max = rankMode ? PLAYERS.length : Math.ceil((Math.max(...values) + 15) / 50) * 50;
  const x = date => left + (Date.parse(date) - Date.parse(dates[0])) / Math.max(1, Date.parse(dates.at(-1)) - Date.parse(dates[0])) * (right - left);
  const y = (id, date) => {
    const value = rankMode ? rankingAt(date).findIndex(p => p.id === id) + 1 : ratingAt(id, date);
    return rankMode ? top + (value - 1) / Math.max(1, max - 1) * (bottom - top) : bottom - (value - min) / (max - min) * (bottom - top);
  };
  const ticks = rankMode ? [1, 3, 5, 8] : [max, max - (max - min) / 3, max - (max - min) * 2 / 3, min];
  const cursorDate = dates.includes(community.inspectedDate) ? community.inspectedDate : dates.at(-1);
  return `<svg id="club-chart-svg" viewBox="0 0 ${width} ${height}" role="group" aria-label="所有选中成员的${rankMode ? '排名' : 'ELO'}趋势，聚焦或触碰日期查看明细">${ticks.map(value => { const pos = rankMode ? top + (value - 1) / Math.max(1, max - 1) * (bottom - top) : bottom - (value - min) / (max - min) * (bottom - top); return `<line class="chart-grid" x1="${left}" x2="${right}" y1="${pos}" y2="${pos}"/><text x="0" y="${pos + 3}">${rankMode ? '#' : ''}${Math.round(value)}</text>`; }).join('')}
    ${selected.map(p => { const color = `var(--series-${PLAYERS.findIndex(entry => entry.id === p.id) + 1})`; return `<path class="club-line" data-player="${p.id}" d="${dates.map((date, index) => `${index ? 'L' : 'M'}${x(date).toFixed(1)},${y(p.id, date).toFixed(1)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${selected.length === 1 ? 3 : 2}" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${x(dates.at(-1))}" cy="${y(p.id, dates.at(-1))}" r="3" fill="${color}"/>`; }).join('')}
    <line id="club-cursor" x1="${x(cursorDate)}" x2="${x(cursorDate)}" y1="${top}" y2="${bottom}" stroke="var(--muted)" stroke-dasharray="3 4" opacity=".55"/>
    ${dates.map((date, index) => { const hitLeft = index === 0 ? left : (x(dates[index - 1]) + x(date)) / 2; const hitRight = index === dates.length - 1 ? right : (x(dates[index + 1]) + x(date)) / 2; return `<rect class="club-date-hit" tabindex="0" role="button" aria-label="查看 ${date} 全员积分" data-club-date="${date}" data-x="${x(date)}" x="${hitLeft}" y="${top}" width="${Math.max(1, hitRight - hitLeft)}" height="${bottom - top}" fill="transparent"><title>${date}</title></rect>`; }).join('')}
    ${[0, Math.floor((dates.length - 1) / 2), dates.length - 1].map((index, pos) => `<text x="${x(dates[index])}" y="${height - 5}" text-anchor="${pos === 0 ? 'start' : pos === 2 ? 'end' : 'middle'}">${dates[index].slice(5).replace('-', '.')}</text>`).join('')}</svg>`;
}
function refreshClubTrend() {
  const panel = document.querySelector('.club-trend');
  if (!panel) return;
  const focused = document.activeElement?.dataset;
  const action = focused?.community, value = focused?.value, id = focused?.id;
  panel.outerHTML = clubTrendPanel(route().page === 'overview');
  if (action) document.querySelector(`[data-community="${action}"]${value ? `[data-value="${value}"]` : ''}${id ? `[data-id="${id}"]` : ''}`)?.focus({ preventScroll: true });
}
function inspectClubDate(event) {
  const target = event.target.closest('[data-club-date]');
  if (!target) return;
  community.inspectedDate = target.dataset.clubDate;
  document.getElementById('club-readout-date').textContent = community.inspectedDate;
  document.getElementById('club-values').innerHTML = clubValues();
  document.getElementById('club-cursor').setAttribute('x1', target.dataset.x);
  document.getElementById('club-cursor').setAttribute('x2', target.dataset.x);
}
document.addEventListener('click', event => {
  const button = event.target.closest('[data-community]');
  if (!button || button.disabled) return;
  const action = button.dataset.community;
  if (action.startsWith('quote-')) {
    community.quoteOffset += action === 'quote-prev' ? -1 : 1;
    community.quoteOffset = Math.min(0, community.quoteOffset);
    document.getElementById('quote-heading').innerHTML = quoteHeading();
    document.querySelector(`[data-community="${action}"]:not(:disabled)`)?.focus({ preventScroll: true });
  } else if (action === 'party-size') {
    const size = Number(button.dataset.value);
    community.partySize = size;
    const mine = activeSignup();
    if (mine) mine.size = size;
    render(true);
    document.querySelector(`[data-community="party-size"][data-value="${size}"]`)?.focus({ preventScroll: true });
    if (mine) toast(`已更新为 ${size} 人参加（演示）`);
  } else if (action === 'join-signup') {
    if (!activeSignup()) community.signups.push({ id: state.me, size: community.partySize });
    render(true); toast(`已报名 · ${activeSignup().size} 人参加（演示）`);
  } else if (action === 'cancel-signup') {
    openDialog(`${dialogHead('取消本周报名？')}<p class="dialog-desc">将移除${myName()}及随行小伙伴的演示报名。之后可以重新报名。</p><div class="dialog-actions"><button class="btn btn-secondary" data-action="close-dialog">保留报名</button><button class="btn btn-primary" data-community="confirm-cancel">确认取消</button></div>`);
  } else if (action === 'confirm-cancel') {
    community.signups = community.signups.filter(entry => entry.id !== state.me);
    closeDialog(); render(true); toast('已取消演示报名');
  } else {
    if (action === 'trend-range') { community.range = button.dataset.value; community.inspectedDate = null; }
    else if (action === 'trend-mode') community.mode = button.dataset.value;
    else if (action === 'select-all') community.selected = new Set(PLAYERS.map(p => p.id));
    else if (action === 'select-me') community.selected = new Set([state.me]);
    else if (action === 'select-none') community.selected.clear();
    else if (action === 'toggle-player') { const id = Number(button.dataset.id); if (community.selected.has(id)) community.selected.delete(id); else community.selected.add(id); }
    refreshClubTrend();
  }
});
document.addEventListener('pointerover', inspectClubDate);
document.addEventListener('focusin', inspectClubDate);
document.addEventListener('click', inspectClubDate);
document.addEventListener('keydown', event => {
  if (event.target.matches('[data-club-date]') && ['Enter', ' '].includes(event.key)) { event.preventDefault(); inspectClubDate(event); }
});
let clubResizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(clubResizeTimer);
  clubResizeTimer = setTimeout(() => { if (document.getElementById('club-chart-plot')) document.getElementById('club-chart-plot').innerHTML = clubChart(); }, 160);
});
