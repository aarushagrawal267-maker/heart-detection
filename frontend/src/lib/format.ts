export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function int(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString("en-US");
}

/** Format seconds as m:ss.s (ECG time-of-record). */
export function clock(seconds: number, decimals = 1): string {
  const sign = seconds < 0 ? "-" : "";
  const s = Math.abs(seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const restStr = rest.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, "0");
  return `${sign}${m}:${restStr}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  if (m < 60) return `${m} min ${s.toString().padStart(2, "0")} s`;
  const h = Math.floor(m / 60);
  return `${h} h ${(m - h * 60).toString().padStart(2, "0")} min`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function relative(iso: string | null | undefined): string {
  if (!iso) return "—";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return `${Math.floor(diff / 86400)} d ago`;
}

/** Min/max decimation used for client-side downsampling (mirrors the backend). */
export function minMaxDecimate(t: number[], y: number[], buckets: number): [number[], number[]] {
  const n = y.length;
  if (n <= buckets * 2) return [t, y];
  const outT: number[] = [];
  const outY: number[] = [];
  for (let b = 0; b < buckets; b++) {
    const a = Math.floor((b * n) / buckets);
    const e = Math.floor(((b + 1) * n) / buckets);
    let iMin = a;
    let iMax = a;
    for (let i = a; i < e; i++) {
      if (y[i] < y[iMin]) iMin = i;
      if (y[i] > y[iMax]) iMax = i;
    }
    const [i1, i2] = iMin < iMax ? [iMin, iMax] : [iMax, iMin];
    outT.push(t[i1], t[i2]);
    outY.push(y[i1], y[i2]);
  }
  return [outT, outY];
}

/** Find the index of the item whose `time` is closest to `t` (items sorted by time). */
export function nearestByTime<T extends { time_s: number }>(items: T[], t: number): number {
  if (!items.length) return -1;
  let lo = 0;
  let hi = items.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (items[mid].time_s < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(items[lo].time_s - t) <= Math.abs(items[hi].time_s - t) ? lo : hi;
}
