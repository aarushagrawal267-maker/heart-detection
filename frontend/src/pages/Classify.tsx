import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, ChevronLeft, ChevronRight, Play, ShieldQuestion, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type BeatPrediction } from "../api/client";
import BeatChart from "../charts/BeatChart";
import { ScoreBars } from "../components/metrics";
import RecordPicker from "../components/RecordPicker";
import { ClassChip, Empty, ErrorState, Loading, Note, PageHeader, Panel, PartitionChip, ResearchNotice, Seg } from "../components/ui";
import { classColor, CLASS_NAMES } from "../lib/classes";
import { clock, num, pct } from "../lib/format";
import { useUrlState } from "../lib/useUrlState";

const METHODS = [
  { value: "integrated_gradients", label: "Integrated Gradients" },
  { value: "grad_cam", label: "Grad-CAM" },
  { value: "saliency", label: "Saliency" },
  { value: "occlusion", label: "Occlusion" },
];

export default function Classify() {
  const [record, setRecord] = useUrlState("record", "100");
  const [beatStr, setBeatStr] = useUrlState("beat", "");
  const [filter, setFilter] = useState<string>("all");
  const [method, setMethod] = useState("integrated_gradients");
  const [result, setResult] = useState<BeatPrediction | null>(null);
  const model = useQuery({ queryKey: ["active-model"], queryFn: api.activeModel });
  const beats = useQuery({ queryKey: ["beats", record], queryFn: () => api.beats(record) });

  const list = useMemo(() => {
    const all = beats.data?.beats ?? [];
    return filter === "all" ? all : all.filter((b) => b.aami === filter);
  }, [beats.data, filter]);
  const beatIndex = beatStr === "" ? list[0]?.beat_index ?? null : Number(beatStr);
  const current = (beats.data?.beats ?? []).find((b) => b.beat_index === beatIndex) ?? null;
  const posInList = list.findIndex((b) => b.beat_index === beatIndex);

  const classify = useMutation({
    mutationFn: () => api.classifyBeat(record, beatIndex!, method),
    onSuccess: setResult,
  });
  const reexplain = useMutation({
    mutationFn: (m: string) => api.explain(record, beatIndex!, m),
    onSuccess: (exp) => setResult((r) => (r ? { ...r, explanation: exp } : r)),
  });
  const sanity = useMutation({ mutationFn: () => api.sanity(record, beatIndex!) });

  useEffect(() => {
    setResult(null);
    sanity.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record, beatIndex]);
  useEffect(() => {
    setBeatStr("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const b of beats.data?.beats ?? []) c[b.aami] = (c[b.aami] ?? 0) + 1;
    return c;
  }, [beats.data]);

  const step = (d: number) => {
    if (!list.length) return;
    const i = Math.min(Math.max((posInList < 0 ? 0 : posInList) + d, 0), list.length - 1);
    setBeatStr(String(list[i].beat_index));
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Heartbeat Classification Centre"
        sub="Run the project's trained 1D CNN on one expert-annotated beat, compare its output with the reference label and inspect which parts of the waveform influenced the score."
        actions={<RecordPicker value={record} onChange={setRecord} />}
      />
      <ResearchNotice />
      {model.isSuccess && !model.data && (
        <Note tone="warn">
          No trained CNN is available. <Link to="/lab" className="underline">Train one in the Model Laboratory</Link> first.
        </Note>
      )}

      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <Panel title="Select a heartbeat" sub={`${beats.data?.n ?? "…"} reference beats in record ${record}`} bodyClass="p-0" testId="beat-selector">
          <div className="border-b border-line-soft p-3">
            <div className="flex flex-wrap gap-1.5">
              {["all", "N", "S", "V", "F", "Q"].map((c) => (
                <button
                  key={c}
                  onClick={() => {
                    setFilter(c);
                    setBeatStr("");
                  }}
                  className="rounded-md border px-2 py-1 font-mono text-2xs transition-colors"
                  style={{
                    borderColor: filter === c ? (c === "all" ? "#2BB3A3" : classColor(c)) : "#212A36",
                    color: c === "all" ? (filter === c ? "#2BB3A3" : "#8B95A5") : classColor(c),
                    background: filter === c ? "#171E28" : "transparent",
                  }}
                  disabled={c !== "all" && !counts[c]}
                >
                  {c === "all" ? "All" : c} {c !== "all" && <span className="text-fg-dim">{counts[c] ?? 0}</span>}
                </button>
              ))}
            </div>
          </div>
          <div className="max-h-[520px] overflow-y-auto">
            {beats.isLoading && <Loading />}
            {list.slice(0, 600).map((b) => (
              <button
                key={b.beat_index}
                onClick={() => setBeatStr(String(b.beat_index))}
                className={`flex w-full items-center justify-between border-b border-line-soft px-3 py-1.5 text-left text-2xs transition-colors hover:bg-ink-750 ${
                  b.beat_index === beatIndex ? "bg-ink-700" : ""
                }`}
              >
                <span className="num text-fg-muted">
                  #{b.beat_index} <span className="text-fg-dim">· {clock(b.time_s, 1)}</span>
                </span>
                <span className="flex items-center gap-2">
                  <span className="num text-fg-dim">RR {num(b.rr_prev_ratio, 2)}×</span>
                  <ClassChip cls={b.aami} label={`${b.aami} ${b.symbol}`} />
                </span>
              </button>
            ))}
            {list.length > 600 && <div className="px-3 py-2 text-2xs text-fg-dim">Showing first 600 of {list.length}. Filter by class to narrow.</div>}
          </div>
        </Panel>

        <div className="min-w-0 space-y-5">
          <Panel
            title={current ? `Record ${record} · beat #${current.beat_index} · ${clock(current.time_s, 2)}` : "No beat selected"}
            sub={current ? `Reference: "${current.symbol}" → AAMI ${current.aami} (${CLASS_NAMES[current.aami]})` : undefined}
            actions={
              <div className="flex items-center gap-1.5">
                <button className="btn px-2" onClick={() => step(-1)} aria-label="Previous beat">
                  <ChevronLeft className="h-3.5 w-3.5" />
                </button>
                <button className="btn px-2" onClick={() => step(1)} aria-label="Next beat">
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
                <button
                  className="btn-primary"
                  onClick={() => classify.mutate()}
                  disabled={beatIndex === null || classify.isPending || !model.data}
                  data-testid="run-classify"
                >
                  <Play className="h-3.5 w-3.5" /> {classify.isPending ? "Running…" : "Run 1D CNN"}
                </button>
              </div>
            }
            testId="classify-panel"
          >
            {classify.error && <ErrorState error={classify.error} />}
            {!result ? (
              current ? (
                <PreviewBeat record={record} beatIndex={current.beat_index} />
              ) : (
                <Empty title="Choose a beat from the list" />
              )
            ) : (
              <div className="space-y-4">
                <BeatChart
                  timeMs={result.time_ms}
                  series={[{ values: result.waveform_mv, color: "#F2555A", width: 1.8 }]}
                  attribution={result.explanation?.attribution}
                  height={260}
                  testId="classified-beat"
                />
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <Seg
                    value={result.explanation?.method ?? method}
                    onChange={(m) => {
                      setMethod(m);
                      reexplain.mutate(m);
                    }}
                    options={METHODS}
                    testId="explain-method"
                  />
                  <div className="flex items-center gap-3 text-2xs text-fg-dim">
                    <span className="flex items-center gap-1.5">
                      <span className="h-2.5 w-3 rounded-sm bg-teal/50" /> raises score for {result.explanation?.target_class}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="h-2.5 w-3 rounded-sm bg-cls-S/50" /> lowers score
                    </span>
                  </div>
                </div>
                {reexplain.isPending && <Loading label="Computing attribution" />}
                <Note>
                  {result.explanation?.description}. {result.explanation?.caveat}
                  {result.explanation?.completeness_gap !== undefined &&
                    ` Completeness check: attributions sum to the score change within ${num(result.explanation.completeness_gap, 4)}.`}
                </Note>
              </div>
            )}
          </Panel>

          {result && (
            <div className="grid gap-5 lg:grid-cols-2">
              <Panel title="Model output vs. expert reference" testId="prediction-result">
                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded-lg border border-line bg-ink-850 p-3">
                    <div className="label">Expert reference</div>
                    <div className="mt-2">
                      <ClassChip cls={result.reference.aami} size="lg" />
                    </div>
                    <div className="mt-2 text-2xs text-fg-dim">
                      "{result.reference.symbol}" · {result.reference.description}
                    </div>
                  </div>
                  <div className="rounded-lg border border-line bg-ink-850 p-3">
                    <div className="label">CNN prediction</div>
                    <div className="mt-2 flex items-center gap-2">
                      <ClassChip cls={result.prediction.class} size="lg" />
                      <span className="num text-sm text-fg">{pct(result.prediction.confidence)}</span>
                    </div>
                    <div className="mt-2 text-2xs text-fg-dim">{CLASS_NAMES[result.prediction.class]}</div>
                  </div>
                </div>
                <div className="mt-4 flex items-center gap-2 text-xs" data-testid="agreement">
                  {result.agrees_with_reference === null ? (
                    <>
                      <ShieldQuestion className="h-4 w-4 text-fg-dim" />
                      <span className="text-fg-muted">Reference class {result.reference.aami} is not one of the model's classes.</span>
                    </>
                  ) : result.agrees_with_reference ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-teal" />
                      <span className="text-fg-muted">Prediction agrees with the expert annotation.</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-4 w-4 text-trace" />
                      <span className="text-fg-muted">Prediction differs from the expert annotation.</span>
                    </>
                  )}
                </div>
                <div className="mt-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <PartitionChip partition={result.partition} />
                  </div>
                  <p className="text-2xs leading-relaxed text-fg-dim">{result.partition_note}</p>
                  <p className="text-2xs text-fg-dim">
                    Scores are softmax outputs of a research model, not calibrated probabilities and not a diagnosis.
                    Inference {num(result.latency_ms, 1)} ms on CPU.
                  </p>
                </div>
              </Panel>
              <Panel title="Class scores" sub="Softmax output of the CNN for each AAMI class">
                <ScoreBars scores={result.prediction.scores} highlight={result.prediction.class} />
                <div className="mt-5 border-t border-line-soft pt-4">
                  <div className="label mb-2">RR-interval inputs</div>
                  <div className="grid grid-cols-2 gap-2 text-2xs">
                    {Object.entries(result.rr).map(([k, v], i) => (
                      <div key={k} className="flex items-center justify-between rounded-md bg-ink-850 px-2 py-1.5">
                        <span className="text-fg-dim">{k.replace(/_/g, " ")}</span>
                        <span className="num text-fg">
                          {num(v, 3)}
                          {k.includes("ratio") ? "×" : " s"}
                          {result.explanation?.rr_attribution && (
                            <span className={`ml-1.5 ${result.explanation.rr_attribution[i] >= 0 ? "text-teal" : "text-cls-S"}`}>
                              {result.explanation.rr_attribution[i] >= 0 ? "+" : ""}
                              {num(result.explanation.rr_attribution[i], 2)}
                            </span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                  {result.explanation?.rr_attribution && (
                    <p className="mt-2 text-2xs text-fg-dim">Coloured numbers: Integrated-Gradients attribution of each RR feature to the predicted class score.</p>
                  )}
                </div>
                <div className="mt-4 border-t border-line-soft pt-4">
                  <button className="btn" onClick={() => sanity.mutate()} disabled={sanity.isPending} data-testid="run-sanity">
                    {sanity.isPending ? "Running…" : "Run explanation sanity check"}
                  </button>
                  {sanity.data && (
                    <div className="mt-2 text-2xs text-fg-muted">
                      Correlation with a randomly-initialised network:{" "}
                      {Object.entries(sanity.data.correlation_trained_vs_random).map(([k, v]) => (
                        <span key={k} className="num mr-3">
                          {k.replace(/_/g, " ")} {v === null ? "—" : num(v, 2)}
                        </span>
                      ))}
                      <div className="mt-1 text-fg-dim">{sanity.data.interpretation}</div>
                    </div>
                  )}
                </div>
              </Panel>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PreviewBeat({ record, beatIndex }: { record: string; beatIndex: number }) {
  const q = useQuery({ queryKey: ["beat", record, beatIndex], queryFn: () => api.beat(record, beatIndex) });
  if (!q.data) return <Loading />;
  return (
    <div>
      <BeatChart timeMs={q.data.time_ms} series={[{ values: q.data.filtered_mv, color: "#F2555A", width: 1.8 }]} height={260} />
      <p className="mt-2 text-2xs text-fg-dim">Preprocessed segment (90 samples before, 166 after the reference position). Press “Run 1D CNN” to classify.</p>
    </div>
  );
}
