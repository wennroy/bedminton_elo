/* Fictional, deterministic fixtures. No network or production database access. */
const PLAYERS = [
  { id: 1, name: '林一', en: 'LIN YI', color: 'lime', skill: 0.9 },
  { id: 2, name: '阿哲', en: 'A ZHE', color: 'blue', skill: 0.76 },
  { id: 3, name: '小雨', en: 'XIAO YU', color: 'peach', skill: 0.7 },
  { id: 4, name: '嘉宁', en: 'JIA NING', color: 'lilac', skill: 0.63 },
  { id: 5, name: '大可', en: 'DA KE', color: 'blue', skill: 0.6 },
  { id: 6, name: '小满', en: 'XIAO MAN', color: 'lime', skill: 0.5 },
  { id: 7, name: '阿森', en: 'A SEN', color: 'lilac', skill: 0.44 },
  { id: 8, name: '乐乐', en: 'LE LE', color: 'peach', skill: 0.38 },
];
const DEMO_TODAY = '2026-09-07';
let fixtureSeed = 37;
function random() {
  fixtureSeed = (fixtureSeed * 1664525 + 1013904223) >>> 0;
  return fixtureSeed / 4294967296;
}
const MATCHES = [];
for (let week = 0; week < 12; week++) {
  for (let round = 0; round < 8; round++) {
    const pool = [...PLAYERS];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const four = pool.slice(0, 4);
    const won = random() < 0.5 + (four[0].skill + four[1].skill - four[2].skill - four[3].skill) * 0.42;
    const date = new Date(Date.UTC(2026, 5, 21 + week * 7));
    const losing = 10 + Math.floor(random() * 10);
    MATCHES.push({ id: MATCHES.length + 1, date: date.toISOString().slice(0, 10), a: four.slice(0, 2).map(p => p.id), b: four.slice(2).map(p => p.id), scoreA: won ? 21 : losing, scoreB: won ? losing : 21 });
  }
}
function player(id) { return PLAYERS.find(p => p.id === Number(id)); }
function replay() {
  const ratings = Object.fromEntries(PLAYERS.map(p => [p.id, 1000]));
  const history = Object.fromEntries(PLAYERS.map(p => [p.id, [{ date: '2026-06-20', elo: 1000 }]]));
  for (const m of MATCHES) {
    m.before = { ...ratings };
    const avgA = (ratings[m.a[0]] + ratings[m.a[1]]) / 2;
    const avgB = (ratings[m.b[0]] + ratings[m.b[1]]) / 2;
    const won = m.scoreA > m.scoreB;
    for (const id of m.a) ratings[id] += 16 * (Number(won) - 1 / (1 + 10 ** ((avgB - ratings[id]) / 400)));
    for (const id of m.b) ratings[id] += 16 * (Number(!won) - 1 / (1 + 10 ** ((avgA - ratings[id]) / 400)));
    m.after = { ...ratings };
    for (const p of PLAYERS) {
      const points = history[p.id];
      if (points.at(-1).date === m.date) points.at(-1).elo = ratings[p.id];
      else points.push({ date: m.date, elo: ratings[p.id] });
    }
  }
  return { ratings, history };
}
let DATA = replay();
function personalMatches(id) {
  return MATCHES.filter(m => [...m.a, ...m.b].includes(id)).map(m => {
    const inA = m.a.includes(id);
    return { ...m, team: inA ? m.a : m.b, opponents: inA ? m.b : m.a, score: inA ? m.scoreA : m.scoreB, against: inA ? m.scoreB : m.scoreA, won: inA ? m.scoreA > m.scoreB : m.scoreB > m.scoreA, delta: Math.round(m.after[id]) - Math.round(m.before[id]) };
  }).reverse();
}
function summary(id) {
  const matches = personalMatches(id);
  const wins = matches.filter(m => m.won).length;
  const history = DATA.history[id];
  const elo = Math.round(DATA.ratings[id]);
  const previous = history.filter(p => p.date < '2026-08-31').at(-1)?.elo ?? 1000;
  let streak = 0;
  for (const m of matches) { if (m.won !== matches[0].won) break; streak++; }
  return { elo, matches, wins, losses: matches.length - wins, total: matches.length, winRate: matches.length ? Math.round(wins / matches.length * 100) : 0, delta: elo - Math.round(previous), peak: Math.round(Math.max(...history.map(p => p.elo))), streak, streakWin: matches[0]?.won, avgDiff: matches.length ? (matches.reduce((sum, m) => sum + m.score - m.against, 0) / matches.length).toFixed(1) : '0.0' };
}
function relations(id, kind) {
  return PLAYERS.filter(p => p.id !== id).map(p => {
    const matches = personalMatches(id).filter(m => kind === 'partners' ? m.team.includes(p.id) : m.opponents.includes(p.id));
    const wins = matches.filter(m => m.won).length;
    return { ...p, total: matches.length, wins, losses: matches.length - wins, rate: matches.length ? Math.round(wins / matches.length * 100) : 0 };
  }).filter(p => p.total > 0).sort((a, b) => kind === 'partners' ? b.rate - a.rate || b.total - a.total : b.total - a.total);
}
