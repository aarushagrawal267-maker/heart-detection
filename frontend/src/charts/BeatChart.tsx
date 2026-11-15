import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { AXIS_STYLE, drawEcgGrid } from "./ecgGrid";

interface Props {
  timeMs: number[];
  series: { values: number[]; color: string; width?: number; dash?: number[] }[];
  attribution?: number[] | null;
  height?: number;
  yLabel?: string;
  grid?: boolean;
  testId?: string;
}

/** Single-heartbeat waveform with optional attribution heat underlay. */
export default function BeatChart({ timeMs, series, attribution, height = 260, yLabel = "mV", grid = true, testId }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const attrRef = useRef(attribution);
  attrRef.current = attribution;

  useEffect(() => {
    if (!wrap.current || !timeMs.length) return;
    const el = wrap.current;
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      padding: [10, 10, 0, 0],
      scales: {
        x: { time: false },
        y: {
          range: (_u, min, max) => {
            const pad = Math.max((max - min) * 0.15, 0.1);
            return [min - pad, max + pad];
          },
        },
      },
      legend: { show: false },
      cursor: { points: { show: false }, drag: { x: false, y: false } },
      axes: [
        { ...AXIS_STYLE, values: (_u, v) => v.map((x) => `${Math.round(x)}`), label: "ms from reference beat", labelSize: 16, size: 30 },
        { ...AXIS_STYLE, values: (_u, v) => v.map((x) => x.toFixed(1)), label: yLabel, labelSize: 14, size: 40 },
      ],
      series: [
        {},
        ...series.map((s) => ({ stroke: s.color, width: s.width ?? 1.6, dash: s.dash, points: { show: false } })),
      ],
      hooks: {
        drawClear: [
          (u) => {
            const a = attrRef.current;
            if (a && a.length === timeMs.length) {
              const ctx = u.ctx;
              const { top, height: h } = u.bbox;
              for (let i = 0; i < a.length - 1; i++) {
                const x0 = u.valToPos(timeMs[i], "x", true);
                const x1 = u.valToPos(timeMs[i + 1], "x", true);
                const v = a[i];
                const alpha = Math.min(Math.abs(v), 1) * 0.55;
                ctx.fillStyle = v >= 0 ? `rgba(43,179,163,${alpha})` : `rgba(232,163,61,${alpha})`;
                ctx.fillRect(x0, top, x1 - x0 + 0.6, h);
              }
            }
            if (grid) drawEcgGrid(u, { xUnit: "ms" });
          },
        ],
        draw: [
          (u) => {
            const ctx = u.ctx;
            const x = u.valToPos(0, "x", true);
            ctx.save();
            ctx.strokeStyle = "rgba(139,149,165,0.5)";
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(x, u.bbox.top);
            ctx.lineTo(x, u.bbox.top + u.bbox.height);
            ctx.stroke();
            ctx.restore();
          },
        ],
      },
    };
    const u = new uPlot(opts, [timeMs, ...series.map((s) => s.values)], el);
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      u.destroy();
    };
  }, [timeMs, series, attribution, height, yLabel, grid]);

  return <div ref={wrap} className="w-full" data-testid={testId ?? "beat-chart"} />;
}
