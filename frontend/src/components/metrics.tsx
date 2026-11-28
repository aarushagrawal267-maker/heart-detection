import { useState } from "react";
import type { Metrics } from "../api/client";
import { classColor, CLASS_NAMES } from "../lib/classes";
import { int, num, pct } from "../lib/format";
import { Seg } from "./ui";

export function ConfusionMatrix({ metrics, testId }: { metrics: Metrics; testId?: string }) {
  const [mode, setMode] = useState<"count" | "row">("row");
  const cm = metrics.confusion_matrix;
  const classes = metrics.classes;
  return (
    <div data-testid={testId ?? "confusion-matrix"}>
      <div className="mb-3 flex items-center justify-between">
        <span className="text-2xs text-fg-dim">Rows: reference (expert) · Columns: CNN prediction</span>
        <Seg
          value={mode}
          onChange={setMode}
          options={[
            { value: "row", label: "Row %" },
            { value: "count", label: "Counts" },
          ]}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="mx-auto border-separate border-spacing-1">
          <thead>
            <tr>
              <th />
              {classes.map((c) => (
                <th key={c} className="px-1 pb-1 text-center font-mono text-2xs font-medium" style={{ color: classColor(c) }}>
                  {c}
                </th>
              ))}
              <th className="px-2 pb-1 text-right text-2xs font-medium text-fg-dim">n</th>
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={i}>
                  <th className="pr-2 text-right font-mono text-2xs font-medium" style={{ color: classColor(classes[i]) }}>
                    {classes[i]}
                  </th>
                  {row.map((v, j) => {
                    const frac = total ? v / total : 0;
                    const diag = i === j;
                    const bg = diag ? `rgba(43,179,163,${0.08 + frac * 0.7})` : `rgba(242,85,90,${frac > 0 ? 0.06 + frac * 0.8 : 0})`;
                    return (
                      <td
                        key={j}
                        className="num h-12 w-16 rounded-md border border-line-soft text-center text-xs sm:w-20"
                        style={{ background: bg, color: frac > 0.45 ? "#E4E8EE" : "#9AA3B2" }}
                        title={`reference ${classes[i]} → predicted ${classes[j]}: ${v} (${pct(frac)})`}
                      >
                        {mode === "row" ? (total ? pct(frac, frac < 0.1 ? 1 : 0) : "—") : int(v)}
                      </td>
                    );
                  })}
                  <td className="num pl-2 text-right text-2xs text-fg-dim">{int(total)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PerClassTable({ metrics, ci }: { metrics: Metrics; ci?: Record<string, unknown> | null }) {
  const interval = (key: string) => {
    const v = ci?.[key];
    return Array.isArray(v) ? `${num(v[0] as number, 2)}–${num(v[1] as number, 2)}` : null;
  };
  return (
    <div className="overflow-x-auto">
      <table className="table" data-testid="per-class-table">
        <thead>
          <tr>
            <th>Class</th>
            <th className="text-right">Support</th>
            <th className="text-right">Sensitivity</th>
            <th className="text-right">+P</th>
            <th className="text-right">Specificity</th>
            <th className="text-right">F1</th>
            {ci && <th className="whitespace-nowrap text-right">F1 95% CI*</th>}
            {metrics.roc && <th className="text-right">ROC-AUC</th>}
            {metrics.pr && <th className="text-right">PR-AP</th>}
          </tr>
        </thead>
        <tbody>
          {metrics.classes.map((c) => {
            const p = metrics.per_class[c];
            return (
              <tr key={c}>
                <td>
                  <span className="font-mono font-medium" style={{ color: classColor(c) }} title={CLASS_NAMES[c]}>
                    {c}
                  </span>
                </td>
                <td className="num text-right">{int(p.support)}</td>
                <td className="num text-right text-fg">{pct(p.sensitivity)}</td>
                <td className="num text-right text-fg">{pct(p.precision)}</td>
                <td className="num text-right">{pct(p.specificity)}</td>
                <td className="num text-right font-medium text-fg">{num(p.f1)}</td>
                {ci && <td className="num text-right">{interval(`${c}_f1`) ?? "—"}</td>}
                {metrics.roc && <td className="num text-right">{num(metrics.roc[c]?.auc)}</td>}
                {metrics.pr && <td className="num text-right">{num(metrics.pr[c]?.ap)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {ci && <p className="mt-2 text-2xs text-fg-dim">* Cluster bootstrap over subjects (patients resampled, {String(ci.n_boot)} replicates).</p>}
    </div>
  );
}

export function ScoreBars({ scores, highlight }: { scores: Record<string, number>; highlight?: string }) {
  return (
    <div className="space-y-2.5" data-testid="score-bars">
      {Object.entries(scores).map(([c, v]) => (
        <div key={c}>
          <div className="mb-1 flex items-center justify-between text-2xs">
            <span className="flex items-center gap-2">
              <span className="font-mono font-medium" style={{ color: classColor(c) }}>
                {c}
              </span>
              <span className="text-fg-dim">{CLASS_NAMES[c]}</span>
            </span>
            <span className={`num ${c === highlight ? "text-fg" : "text-fg-muted"}`}>{pct(v, 1)}</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-ink-600">
            <div
              className="h-full w-full origin-left rounded-full transition-transform duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]"
              style={{ transform: `scaleX(${Math.max(0, Math.min(1, v))})`, background: classColor(c), opacity: c === highlight ? 1 : 0.55 }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function Headline({ metrics }: { metrics: Metrics }) {
  const items = [
    ["Macro F1", num(metrics.macro_f1)],
    ["Weighted F1", num(metrics.weighted_f1)],
    ["Macro recall", pct(metrics.macro_recall)],
    ["Accuracy", pct(metrics.accuracy)],
    ["Macro ROC-AUC", num(metrics.macro_roc_auc ?? null)],
    ["Beats", int(metrics.n)],
  ];
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
      {items.map(([k, v]) => (
        <div key={k} className="bg-ink-800 px-3 py-2.5">
          <div className="label">{k}</div>
          <div className="num mt-1 text-xl text-fg">{v}</div>
        </div>
      ))}
    </div>
  );
}
