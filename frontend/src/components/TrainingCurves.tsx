import type { EpochRow } from "../api/client";
import Plot from "../charts/Plot";

export function LossCurves({ epochs, height = 240 }: { epochs: EpochRow[]; height?: number }) {
  const rows = epochs.filter((e) => e.fold === -1);
  const x = rows.map((e) => e.epoch);
  return (
    <Plot
      testId="loss-curves"
      height={height}
      data={[
        { type: "scatter", mode: "lines+markers", name: "Training loss", x, y: rows.map((e) => e.train_loss), line: { color: "#2BB3A3", width: 2 }, marker: { size: 5 } },
        { type: "scatter", mode: "lines+markers", name: "Validation loss", x, y: rows.map((e) => e.val_loss), line: { color: "#F2555A", width: 2 }, marker: { size: 5 } },
      ]}
      layout={{ xaxis: { title: { text: "epoch" }, dtick: 1 }, yaxis: { title: { text: "loss" } } }}
    />
  );
}

export function F1Curves({ epochs, height = 240 }: { epochs: EpochRow[]; height?: number }) {
  const rows = epochs.filter((e) => e.fold === -1);
  const x = rows.map((e) => e.epoch);
  const classes = Object.keys(rows[0]?.val_per_class_f1 ?? {});
  const colors: Record<string, string> = { N: "#94A3B8", S: "#E8A33D", V: "#A78BFA", F: "#4EA8DE", Q: "#6B7280" };
  return (
    <Plot
      testId="f1-curves"
      height={height}
      data={[
        { type: "scatter", mode: "lines", name: "Train macro F1", x, y: rows.map((e) => e.train_macro_f1), line: { color: "#2BB3A3", width: 2, dash: "dot" } },
        { type: "scatter", mode: "lines+markers", name: "Val macro F1", x, y: rows.map((e) => e.val_macro_f1), line: { color: "#E4E8EE", width: 2 }, marker: { size: 5 } },
        ...classes.map((c) => ({
          type: "scatter" as const,
          mode: "lines" as const,
          name: `Val F1 ${c}`,
          x,
          y: rows.map((e) => e.val_per_class_f1?.[c] ?? null),
          line: { color: colors[c], width: 1.2 },
        })),
      ]}
      layout={{ xaxis: { title: { text: "epoch" }, dtick: 1 }, yaxis: { title: { text: "F1" }, range: [0, 1] } }}
    />
  );
}
