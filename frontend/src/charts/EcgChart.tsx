import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { classColor } from "../lib/classes";
import { clock } from "../lib/format";
import { AXIS_STYLE, drawEcgGrid } from "./ecgGrid";

export interface EcgMarker {
  time_s: number;
  label: string; // AAMI class (or symbol) shown in the lane
  cls: string | null; // AAMI class for colour
  key?: number | null; // beat index used for selection
  title?: string;
}

export interface PredMarker extends EcgMarker {
  confidence: number;
  mismatch: boolean;
}

interface Props {
  t: number[];
  y: number[];
  range: [number, number];
  height?: number;
  reference?: EcgMarker[];
  predictions?: PredMarker[];
  selected?: number | null; // time of the selected beat
  shading?: { start_s: number; end_s: number; color: string }[];
  onSelectTime?: (t: number) => void;
  onZoom?: (start: number, end: number) => void;
  yRange?: [number, number];
  traceColor?: string;
  testId?: string;
}

const LANE_H = 20;

/**
 * Canvas ECG viewer built on uPlot: ECG paper grid, min/max decimated trace,
 * a reference-annotation lane (top) and a model-prediction lane (bottom).
 */
export default function EcgChart({
  t,
  y,
  range,
  height = 320,
  reference = [],
  predictions,
  selected = null,
  shading = [],
  onSelectTime,
  onZoom,
  yRange,
  traceColor = "#F2555A",
  testId,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const state = useRef({ reference, predictions, selected, shading, onSelectTime, onZoom, range });
  state.current = { reference, predictions, selected, shading, onSelectTime, onZoom, range };

  // create once
  useEffect(() => {
    if (!wrap.current) return;
    const el = wrap.current;
    const hasPred = predictions !== undefined;
    const opts: uPlot.Options = {
      width: el.clientWidth || 800,
      height,
      padding: [LANE_H + 6, 8, 4, 0],
      scales: {
        // read the live range from a ref: a closure here would freeze the initial window
        x: { time: false, auto: false, range: () => [state.current.range[0], state.current.range[1]] },
        y: {
          auto: !yRange,
          range: yRange
            ? () => yRange
            : (_u, min, max) => {
                const pad = Math.max((max - min) * 0.12, 0.2);
                return [min - pad, max + pad];
              },
        },
      },
      cursor: { drag: { x: !!onZoom, y: false, setScale: false }, points: { show: false } },
      select: { show: true, left: 0, top: 0, width: 0, height: 0 },
      legend: { show: false },
      axes: [
        {
          ...AXIS_STYLE,
          values: (_u, vals) => vals.map((v) => clock(v, state.current.range[1] - state.current.range[0] < 20 ? 1 : 0)),
          // leave room for the prediction lane between the plot and the time labels
          size: hasPred ? 28 + LANE_H : 28,
          gap: hasPred ? LANE_H + 2 : 5,
        },
        { ...AXIS_STYLE, values: (_u, vals) => vals.map((v) => v.toFixed(1)), size: 42, label: "mV", labelSize: 14 },
      ],
      series: [{}, { stroke: traceColor, width: 1.25, points: { show: false } }],
      hooks: {
        drawClear: [
          (u) => {
            const ctx = u.ctx;
            const { left, top, width, height: h } = u.bbox;
            for (const s of state.current.shading) {
              const x0 = u.valToPos(s.start_s, "x", true);
              const x1 = u.valToPos(s.end_s, "x", true);
              ctx.fillStyle = s.color;
              ctx.fillRect(Math.max(x0, left), top, Math.min(x1, left + width) - Math.max(x0, left), h);
            }
            drawEcgGrid(u);
          },
        ],
        draw: [(u) => drawLanes(u, state.current.reference, state.current.predictions, state.current.selected)],
        setSelect: [
          (u) => {
            const sel = u.select;
            if (sel.width > 6 && state.current.onZoom) {
              const a = u.posToVal(sel.left, "x");
              const b = u.posToVal(sel.left + sel.width, "x");
              state.current.onZoom(a, b);
            }
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          },
        ],
      },
    };
    const u = new uPlot(opts, [t, y], el);
    plot.current = u;
    let downX = 0;
    const over = u.over;
    const onDown = (e: MouseEvent) => (downX = e.clientX);
    const onUp = (e: MouseEvent) => {
      if (Math.abs(e.clientX - downX) > 4 || !state.current.onSelectTime) return;
      const rect = over.getBoundingClientRect();
      state.current.onSelectTime(u.posToVal(e.clientX - rect.left, "x"));
    };
    over.addEventListener("mousedown", onDown);
    over.addEventListener("mouseup", onUp);
    over.style.cursor = onSelectTime ? "crosshair" : "default";
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      over.removeEventListener("mousedown", onDown);
      over.removeEventListener("mouseup", onUp);
      u.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, predictions === undefined, traceColor, yRange?.[0], yRange?.[1]]);

  // update data / range
  useEffect(() => {
    const u = plot.current;
    if (!u) return;
    u.batch(() => {
      u.setData([t, y], false);
      u.setScale("x", { min: range[0], max: range[1] });
      if (!yRange) u.setScale("y", { min: Math.min(...(y.length ? y : [0])), max: Math.max(...(y.length ? y : [1])) });
    });
  }, [t, y, range, yRange]);

  // redraw markers
  useEffect(() => {
    plot.current?.redraw(false, false);
  }, [reference, predictions, selected, shading]);

  return <div ref={wrap} className="w-full select-none" data-testid={testId ?? "ecg-chart"} />;
}

function drawLanes(u: uPlot, reference: EcgMarker[], predictions: PredMarker[] | undefined, selected: number | null) {
  const ctx = u.ctx;
  const dpr = devicePixelRatio || 1;
  const { left, top, width, height } = u.bbox;
  const xs = u.scales.x;
  if (xs.min == null || xs.max == null) return;
  const visible = (m: EcgMarker) => m.time_s >= xs.min! && m.time_s <= xs.max!;
  const refs = reference.filter(visible);
  const preds = (predictions ?? []).filter(visible);
  const dense = refs.length > width / (dpr * 14);
  ctx.save();
  ctx.font = `${10 * dpr}px "JetBrains Mono", ui-monospace, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // lane labels
  const laneY = top - (LANE_H / 2) * dpr - 2 * dpr;
  ctx.fillStyle = "#5F6A7A";
  ctx.textAlign = "right";
  ctx.fillText("REF", left - 6 * dpr, laneY);
  if (predictions) ctx.fillText("CNN", left - 6 * dpr, top + height + (LANE_H / 2 + 5) * dpr);
  ctx.textAlign = "center";

  // selected beat highlight
  if (selected !== null && selected >= xs.min && selected <= xs.max) {
    const x = u.valToPos(selected, "x", true);
    ctx.fillStyle = "rgba(43,179,163,0.10)";
    const w = Math.max(u.valToPos(selected + 0.3, "x", true) - u.valToPos(selected - 0.3, "x", true), 2 * dpr);
    ctx.fillRect(x - w / 2, top, w, height);
    ctx.strokeStyle = "rgba(43,179,163,0.8)";
    ctx.lineWidth = 1 * dpr;
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // reference lane
  for (const m of refs) {
    const x = u.valToPos(m.time_s, "x", true);
    const c = classColor(m.cls);
    ctx.strokeStyle = c + "55";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + 6 * dpr);
    ctx.stroke();
    if (dense) {
      ctx.fillStyle = c;
      ctx.fillRect(x - 1 * dpr, laneY - 4 * dpr, 2 * dpr, 8 * dpr);
    } else {
      ctx.fillStyle = c;
      ctx.fillText(m.label, x, laneY);
    }
  }

  // prediction lane
  if (predictions) {
    const py = top + height + (LANE_H / 2 + 5) * dpr;
    for (const m of preds) {
      const x = u.valToPos(m.time_s, "x", true);
      const c = classColor(m.cls);
      ctx.globalAlpha = 0.35 + 0.65 * m.confidence;
      if (dense) {
        ctx.fillStyle = c;
        ctx.fillRect(x - 1 * dpr, py - 4 * dpr, 2 * dpr, 8 * dpr);
      } else {
        ctx.fillStyle = c;
        ctx.fillText(m.label, x, py);
      }
      ctx.globalAlpha = 1;
      if (m.mismatch) {
        ctx.strokeStyle = "#F2555A";
        ctx.lineWidth = 1.25 * dpr;
        ctx.beginPath();
        ctx.arc(x, py, (dense ? 4 : 7) * dpr, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = "rgba(242,85,90,0.35)";
        ctx.setLineDash([2 * dpr, 3 * dpr]);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }
  ctx.restore();
}
