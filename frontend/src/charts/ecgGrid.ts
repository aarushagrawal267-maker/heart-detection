import type uPlot from "uplot";

/**
 * ECG paper grid drawn inside the plotting area.
 * Standard paper: minor 0.04 s × 0.1 mV, major 0.2 s × 0.5 mV (25 mm/s, 10 mm/mV).
 * When zoomed out, line levels that would be denser than a few pixels are skipped and
 * replaced by coarser 1 s / 5 s / 30 s lines.
 */
export function drawEcgGrid(u: uPlot, opts: { xUnit?: "s" | "ms" } = {}) {
  const ctx = u.ctx;
  const { left, top, width, height } = u.bbox;
  if (width <= 0 || height <= 0) return;
  const xs = u.scales.x;
  const ys = u.scales.y;
  if (xs.min == null || xs.max == null || ys.min == null || ys.max == null) return;
  const toSec = opts.xUnit === "ms" ? 1000 : 1;
  const pxPerX = width / (xs.max - xs.min);
  const pxPerY = height / (ys.max - ys.min);

  const xLevels: { step: number; color: string; w: number }[] = [
    { step: 0.04 * toSec, color: "rgba(242,85,90,0.07)", w: 1 },
    { step: 0.2 * toSec, color: "rgba(242,85,90,0.17)", w: 1 },
    { step: 1 * toSec, color: "rgba(242,85,90,0.22)", w: 1 },
    { step: 5 * toSec, color: "rgba(242,85,90,0.26)", w: 1 },
    { step: 30 * toSec, color: "rgba(242,85,90,0.3)", w: 1 },
  ];
  const yLevels: { step: number; color: string; w: number }[] = [
    { step: 0.1, color: "rgba(242,85,90,0.07)", w: 1 },
    { step: 0.5, color: "rgba(242,85,90,0.17)", w: 1 },
    { step: 1.0, color: "rgba(242,85,90,0.2)", w: 1 },
  ];

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, width, height);
  ctx.clip();
  const dpr = devicePixelRatio || 1;
  const minPx = 4 * dpr;

  for (const lvl of xLevels) {
    if (lvl.step * pxPerX < minPx) continue;
    ctx.strokeStyle = lvl.color;
    ctx.lineWidth = lvl.w;
    ctx.beginPath();
    const start = Math.ceil(xs.min / lvl.step) * lvl.step;
    for (let v = start; v <= xs.max; v += lvl.step) {
      const x = Math.round(left + (v - xs.min) * pxPerX) + 0.5;
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + height);
    }
    ctx.stroke();
  }
  for (const lvl of yLevels) {
    if (lvl.step * pxPerY < minPx) continue;
    ctx.strokeStyle = lvl.color;
    ctx.lineWidth = lvl.w;
    ctx.beginPath();
    const start = Math.ceil(ys.min / lvl.step) * lvl.step;
    for (let v = start; v <= ys.max; v += lvl.step) {
      const y = Math.round(top + (ys.max - v) * pxPerY) + 0.5;
      ctx.moveTo(left, y);
      ctx.lineTo(left + width, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

export const AXIS_STYLE = {
  stroke: "#5F6A7A",
  grid: { show: false },
  ticks: { show: true, stroke: "#273241", width: 1, size: 4 },
  font: `10px "JetBrains Mono", ui-monospace, monospace`,
  labelFont: `10px "JetBrains Mono", ui-monospace, monospace`,
};
