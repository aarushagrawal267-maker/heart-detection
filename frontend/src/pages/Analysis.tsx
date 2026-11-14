import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type RecordAnalysis } from "../api/client";
import EcgChart from "../charts/EcgChart";
import Minimap from "../charts/Minimap";
import Plot from "../charts/Plot";
import { ConfusionMatrix, Headline } from "../components/metrics";
import RecordPicker from "../components/RecordPicker";
import { ClassChip, Empty, ErrorState, Loading, Note, PageHeader, Panel, PartitionChip, ResearchNotice, Seg, Stat } from "../components/ui";
import { classColor, CLASS_COLORS } from "../lib/classes";
import { clock, int, num, pct } from "../lib/format";
import { useUrlState } from "../lib/useUrlState";

export default function Analysis() {
  const [record, setRecord] = useUrlState("record", "213");
  const [mode, setMode] = useState<"reference" | "detected">("reference");
  const [range, setRange] = useState<[number, number]>([0, 15]);
  const [result, setResult] = useState<RecordAnalysis | null>(null);
  const [onlyMismatch, setOnlyMismatch] = useState(false);

  const run = useMutation({ mutationFn: () => api.analyzeRecord(record, mode), onSuccess: setResult });
  useEffect(() => {
    setResult(null);
    setRange([0, 15]);
  }, [record, mode]);

  const sig = useQuery({
    queryKey: ["signal", record, "analysis", range[0].toFixed(3), range[1].toFixed(3)],
    queryFn: () => api.signal(record, range[0], range[1], undefined, true, 5000),
    placeholderData: keepPreviousData,
  });
  const env = useQuery({ queryKey: ["envelope", record, undefined], queryFn: () => api.overviewEnvelope(record) });
  const quality = useQuery({ queryKey: ["quality", record, undefined], queryFn: () => api.quality(record) });
  const duration = env.data?.duration_s ?? 1805.55;
  const span = range[1] - range[0];
  const clamp = (a: number, b: number): [number, number] => {
    const w = Math.min(Math.max(b - a, 1), duration);
    let s = Math.max(0, a);
    if (s + w > duration) s = duration - w;
    return [s, s + w];
  };

  const { refs, preds, mismatches } = useMemo(() => {
    const refs: { time_s: number; label: string; cls: string | null }[] = [];
    const preds: { time_s: number; label: string; cls: string; confidence: number; mismatch: boolean }[] = [];
    const mismatches: RecordAnalysis["beats"] = [];
    for (const b of result?.beats ?? []) {
      const mm = b.reference !== null && result!.classes.includes(b.reference) && b.reference !== b.predicted;
      if (mm) mismatches.push(b);
      if (b.time_s < range[0] - 1 || b.time_s > range[1] + 1) continue;
      if (b.reference) refs.push({ time_s: b.time_s, label: b.reference, cls: b.reference });
      preds.push({ time_s: b.time_s, label: b.predicted, cls: b.predicted, confidence: b.confidence, mismatch: mm });
    }
    return { refs, preds, mismatches };
  }, [result, range]);

  const jumpMismatch = (d: number) => {
    if (!mismatches.length) return;
    const c = (range[0] + range[1]) / 2;
    const next = d > 0 ? mismatches.find((m) => m.time_s > c + 0.05) ?? mismatches[0] : [...mismatches].reverse().find((m) => m.time_s < c - 0.05) ?? mismatches[mismatches.length - 1];
    setRange(clamp(next.time_s - span / 2, next.time_s + span / 2));
  };

  const timeline = useMemo(() => {
    if (!result) return null;
    const bin = 30;
    const nb = Math.ceil(result.duration_s / bin);
    const series = result.classes.map((c) => ({ c, counts: new Array(nb).fill(0) as number[] }));
    for (const b of result.beats) {
      const k = Math.min(Math.floor(b.time_s / bin), nb - 1);
      const s = series.find((x) => x.c === b.predicted);
      if (s) s.counts[k]++;
    }
    const x = Array.from({ length: nb }, (_, i) => (i * bin) / 60);
    return series
      .filter((s) => s.c !== "N")
      .map((s) => ({ type: "bar" as const, name: s.c, x, y: s.counts, marker: { color: CLASS_COLORS[s.c] } }));
  }, [result]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Full ECG Analysis"
        sub="Classify every heartbeat of a recording with the CNN. Expert reference annotations (top lane) and model predictions (bottom lane) are always shown separately; disagreements are circled."
        actions={
          <>
            <RecordPicker value={record} onChange={setRecord} />
            <Seg
              value={mode}
              onChange={setMode}
              options={[
                { value: "reference", label: "Reference beat positions" },
                { value: "detected", label: "Automatic R-peak detection" },
              ]}
              testId="analysis-mode"
            />
            <button className="btn-primary" onClick={() => run.mutate()} disabled={run.isPending} data-testid="run-analysis">
              <Play className="h-3.5 w-3.5" /> {run.isPending ? "Analysing…" : "Analyse recording"}
            </button>
          </>
        }
      />
      <ResearchNotice />
      {mode === "detected" && (
        <Note>
          Automatic mode uses the project's own Pan-Tompkins-style detector instead of expert positions. Each detection is matched to a
          reference beat within 150 ms (AAMI EC57). Detector errors propagate into classification; the principal evaluation uses
          reference positions.
        </Note>
      )}
      {run.error && <ErrorState error={run.error} />}

      {result && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5" data-testid="analysis-stats">
          <Stat label="Beats classified" value={int(result.n_beats)} sub={`${result.mode === "reference" ? "reference" : "detected"} positions`} />
          <Stat label="Agreement" value={pct(result.agreement?.accuracy)} sub="with expert labels (model classes)" />
          <Stat label="Macro F1 (this record)" value={num(result.agreement?.macro_f1 ?? null)} sub="present classes only" />
          <Stat label="Disagreements" value={int(mismatches.length)} accent={mismatches.length ? "#F2555A" : undefined} sub="circled in red" />
          <Stat label="Inference" value={`${num(result.inference_s, 2)} s`} sub="CPU, whole record" />
        </div>
      )}
      {result && (
        <div className="flex flex-wrap items-center gap-2 text-2xs text-fg-dim">
          <PartitionChip partition={result.partition} /> {result.partition_note}
        </div>
      )}

      <Panel
        title={`Record ${record} · preprocessed lead ${sig.data?.lead ?? ""}`}
        sub={<span className="num">{clock(range[0])} – {clock(range[1])}</span>}
        actions={
          <div className="flex flex-wrap items-center gap-1.5">
            <button className="btn px-2" onClick={() => setRange(clamp(range[0] - span * 0.8, range[1] - span * 0.8))} aria-label="Pan left">
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <button className="btn px-2" onClick={() => setRange(clamp(range[0] + span * 0.8, range[1] + span * 0.8))} aria-label="Pan right">
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
            <Seg value={span} onChange={(w) => setRange(clamp(range[0], range[0] + w))} options={[5, 15, 30, 60].map((s) => ({ value: s, label: `${s}s` }))} />
            {result && (
              <>
                <button className="btn" onClick={() => jumpMismatch(-1)} disabled={!mismatches.length}>
                  ← disagreement
                </button>
                <button className="btn" onClick={() => jumpMismatch(1)} disabled={!mismatches.length} data-testid="next-mismatch">
                  disagreement →
                </button>
              </>
            )}
          </div>
        }
        bodyClass="p-3 sm:p-4"
      >
        {sig.data ? (
          <EcgChart
            t={sig.data.t}
            y={sig.data.y}
            range={range}
            height={320}
            reference={refs}
            predictions={result ? preds : undefined}
            onZoom={(a, b) => setRange(clamp(a, b))}
            testId="analysis-ecg"
          />
        ) : (
          <Loading />
        )}
        {env.data && (
          <div className="mt-3">
            <Minimap
              t={env.data.t}
              min={env.data.min}
              max={env.data.max}
              duration={duration}
              range={range}
              onJump={(c) => setRange(clamp(c - span / 2, c + span / 2))}
              marks={(onlyMismatch ? mismatches : result?.beats.filter((b) => b.predicted !== "N") ?? []).map((b) => ({
                time_s: b.time_s,
                color: onlyMismatch ? "#F2555A" : classColor(b.predicted),
              }))}
            />
            {result && (
              <label className="mt-2 flex items-center gap-2 text-2xs text-fg-dim">
                <input type="checkbox" checked={onlyMismatch} onChange={(e) => setOnlyMismatch(e.target.checked)} /> Minimap: show
                disagreements instead of non-N predictions
              </label>
            )}
          </div>
        )}
      </Panel>

      {!result ? (
        <Panel>
          <Empty title="Run the analysis to classify every beat">
            The waveform above is the preprocessed signal. Press “Analyse recording” to run the active CNN over all beats.
          </Empty>
        </Panel>
      ) : (
        <>
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Class distribution" sub="Expert reference vs. CNN prediction" testId="analysis-distribution">
              <table className="table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th className="text-right">Reference</th>
                    <th className="text-right">CNN</th>
                  </tr>
                </thead>
                <tbody>
                  {[...result.classes, ...Object.keys(result.reference_distribution).filter((c) => !result.classes.includes(c))].map((c) => (
                    <tr key={c}>
                      <td>{c === "unmatched" ? <span className="text-fg-dim">No matching reference beat</span> : <ClassChip cls={c} />}</td>
                      <td className="num text-right">{int(result.reference_distribution[c] ?? 0)}</td>
                      <td className="num text-right text-fg">{result.classes.includes(c) ? int(result.predicted_distribution[c] ?? 0) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.detection && (
                <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line-soft pt-4 text-xs sm:grid-cols-4" data-testid="detection-stats">
                  <div>
                    <div className="label">Detector Se</div>
                    <div className="num mt-1 text-fg">{pct(result.detection.sensitivity, 2)}</div>
                  </div>
                  <div>
                    <div className="label">Detector +P</div>
                    <div className="num mt-1 text-fg">{pct(result.detection.positive_predictivity, 2)}</div>
                  </div>
                  <div>
                    <div className="label">Missed / extra</div>
                    <div className="num mt-1 text-fg">
                      {result.detection.fn} / {result.detection.fp}
                    </div>
                  </div>
                  <div>
                    <div className="label">Mean |error|</div>
                    <div className="num mt-1 text-fg">{num(result.detection.mean_abs_error_ms, 1)} ms</div>
                  </div>
                </div>
              )}
            </Panel>
            <Panel title="Agreement with expert annotations" sub="Beats whose reference class is a model class">
              {result.agreement ? <ConfusionMatrix metrics={result.agreement} /> : <Empty title="No comparable beats" />}
            </Panel>
          </div>
          {result.agreement && <Headline metrics={result.agreement} />}
          <div className="grid gap-5 lg:grid-cols-3">
            <Panel className="lg:col-span-2" title="Prediction timeline" sub="Non-N CNN predictions per 30-second bin">
              {timeline && (
                <Plot
                  data={timeline}
                  layout={{ barmode: "stack", xaxis: { title: { text: "minutes" } }, yaxis: { title: { text: "beats / 30 s" } } }}
                  height={240}
                  testId="timeline"
                />
              )}
            </Panel>
            <Panel title="Signal quality" sub="Raw analysis lead, 10 s windows">
              {quality.data ? (
                <div className="space-y-2 text-xs">
                  {[
                    ["Mean quality score", num(quality.data.summary.mean_score, 2)],
                    ["Good windows", pct(quality.data.summary.good_fraction)],
                    ["Fair windows", pct(quality.data.summary.fair_fraction)],
                    ["Poor windows", pct(quality.data.summary.poor_fraction)],
                    ["Baseline wander power", pct(quality.data.summary.mean_baseline_ratio)],
                    ["60 Hz power", pct(quality.data.summary.mean_powerline_ratio, 3)],
                    ["Expert noise annotations", String(quality.data.noise_annotations.length)],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between border-b border-line-soft pb-1.5">
                      <span className="text-fg-dim">{k}</span>
                      <span className="num text-fg">{v}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <Loading />
              )}
            </Panel>
          </div>
          <Panel title="Disagreements" sub={`${mismatches.length} beats where the CNN differs from the expert`} bodyClass="p-0">
            <div className="max-h-[360px] overflow-y-auto">
              <table className="table">
                <thead className="sticky top-0 bg-ink-800">
                  <tr>
                    <th>Time</th>
                    <th>Beat</th>
                    <th>Reference</th>
                    <th>CNN</th>
                    <th className="text-right">Score</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {mismatches.slice(0, 400).map((b) => (
                    <tr key={b.sample}>
                      <td className="num">{clock(b.time_s, 2)}</td>
                      <td className="num">{b.beat_index ?? "—"}</td>
                      <td>
                        <ClassChip cls={b.reference} label={`${b.reference} ${b.symbol ?? ""}`} />
                      </td>
                      <td>
                        <ClassChip cls={b.predicted} />
                      </td>
                      <td className="num text-right">{pct(b.confidence)}</td>
                      <td className="text-right">
                        <button className="btn py-0.5" onClick={() => setRange(clamp(b.time_s - span / 2, b.time_s + span / 2))}>
                          view
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
