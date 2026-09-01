/**
 * sqlite datetime('now') 存的是 UTC "YYYY-MM-DD HH:MM:SS"（无时区标记），
 * 直接 new Date() 会被浏览器当本地时间解析（Safari 甚至不认空格格式），
 * 所以统一补 Z 按 UTC 处理；已带 Z 或 ±offset 的 ISO 串原样解析。
 */
function parseDbDateTime(value: string): Date {
  const iso = value.includes("T") ? value : value.replace(" ", "T");
  const withZone = /(Z|[+-]\d{2}:?\d{2})$/.test(iso) ? iso : `${iso}Z`;
  return new Date(withZone);
}

/** 本周战绩底部的元信息行,如 "8月30日 对局 · 凯哥 录入于 21:43"。 */
export function formatMatchMeta(
  playedAt: string,
  createdAt: string,
  enteredByName: string
): string {
  const [, month, day] = playedAt.split("-").map(Number);
  const created = parseDbDateTime(createdAt);
  const hh = String(created.getHours()).padStart(2, "0");
  const mm = String(created.getMinutes()).padStart(2, "0");
  return `${month}月${day}日 对局 · ${enteredByName} 录入于 ${hh}:${mm}`;
}
