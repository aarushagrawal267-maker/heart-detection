import { useEffect, useRef } from "react";
import { clock } from "../lib/format";

interface Props {
  t: number[];
  min: number[];
  max: number[];
  duration: number;
  range: [number, number];
  onJump: (center: number) => void;
  marks?: { time_s: number; color: string }[];
  height?: number;
}

/** Whole-recording envelope with the current viewport; click or drag to navigate. */
export default function Minimap({ t, min, max, duration, range, onJump, marks = [], height = 56 }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const draw = () => {
      const dpr = devicePixelRatio || 1;
      const w = c.clientWidth;
      c.width = w * dpr;
      c.height = height * dpr;
      const ctx = c.getContext("2d")!;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, height);
      if (!t.length) return;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < min.length; i++) {
        lo = Math.min(lo, min[i]);
        hi = Math.max(hi, max[i]);
      }
      // robust vertical range (ignore extreme artefacts)
      const sortedLo = [...min].sort((a, b) => a - b);
      const sortedHi = [...max].sort((a, b) => a - b);
      lo = sortedLo[Math.floor(sortedLo.length * 0.01)];
      hi = sortedHi[Math.floor(sortedHi.length * 0.99)];
      const sy = (v: number) => height - 14 - ((v - lo) / (hi - lo || 1)) * (height - 20);
      const sx = (v: number) => (v / duration) * w;
      for (const m of marks) {
        ctx.fillStyle = m.color;
        ctx.fillRect(sx(m.time_s), height - 12, 1, 4);
      }
      ctx.strokeStyle = "rgba(242,85,90,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < t.length; i++) {
        const x = sx(t[i]) + 0.5;
        ctx.moveTo(x, sy(Math.min(max[i], hi)));
        ctx.lineTo(x, sy(Math.max(min[i], lo)));
      }
      ctx.stroke();
      const x0 = sx(range[0]);
      const x1 = Math.max(sx(range[1]), x0 + 2);
      ctx.fillStyle = "rgba(43,179,163,0.14)";
      ctx.fillRect(x0, 0, x1 - x0, height - 8);
      ctx.strokeStyle = "rgba(43,179,163,0.9)";
      ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, height - 9);
      ctx.fillStyle = "#7E8899";
      ctx.font = `11px "JetBrains Mono", monospace`;
      ctx.textBaseline = "bottom";
      for (let s = 0; s <= duration; s += 300) {
        const x = sx(s);
        ctx.fillText(clock(s, 0), Math.min(x + 2, w - 28), height);
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [t, min, max, duration, range, marks, height]);

  const handle = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    onJump(Math.max(0, Math.min(1, frac)) * duration);
  };

  return (
    <canvas
      ref={canvas}
      className="block w-full cursor-pointer rounded-md"
      style={{ height }}
      onMouseDown={(e) => {
        dragging.current = true;
        handle(e);
      }}
      onMouseMove={(e) => dragging.current && handle(e)}
      onMouseUp={() => (dragging.current = false)}
      onMouseLeave={() => (dragging.current = false)}
      data-testid="minimap"
    />
  );
}
