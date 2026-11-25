import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Lock, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { api, type EvaluationEntry } from "../api/client";
import Plot from "../charts/Plot";
import { ConfusionMatrix, Headline, PerClassTable } from "../components/metrics";
import { F1Curves, LossCurves } from "../components/TrainingCurves";
import { Empty, ErrorState, Loading, Note, PageHeader, Panel, Seg } from "../components/ui";
import { CLASS_COLORS } from "../lib/classes";
import { dateTime, int, num, pct } from "../lib/format";
import { useUrlState } from "../lib/useUrlState";

type Part = "test" | "validation" | "train";

export default function Evaluation() {
  const qc = useQueryClient();
  const exps = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const active = useQuery({ queryKey: ["active-model"], queryFn: api.activeModel });
  const completed = (exps.data ?? []).filter((e) => e.status === "completed");
  const [expParam, setExp] = useUrlState("exp", "");
  const expId = expParam || active.data?.id || completed[0]?.id || "";
  const exp = useQuery({ queryKey: ["experiment", expId], queryFn: () => api.experiment(expId), enabled: !!expId });
  const evals = useQuery({ queryKey: ["evaluations", expId], queryFn: () => api.evaluations(expId), enabled: !!expId });
  const epochs = useQuery({ queryKey: ["epochs", expId], queryFn: () => api.epochs(expId), enabled: !!expId });
  const splits = useQuery({ queryKey: ["splits"], queryFn: api.splits });
  const log = useQuery({ queryKey: ["test-log"], queryFn: api.testAccessLog });
  const [part, setPart] = useState<Part | null>(null);
  const [purpose, setPurpose] = useState("");
  const runTest = useMutation({
    mutationFn: () => api.evaluateTest(expId, purpose || "final held-out evaluation"),
    onSuccess: () => {
      qc.invalidateQueries();
      setPart("test");
    },
  });

  const available = Object.keys(evals.data ?? {}) as Part[];
  const shown: Part | null = part && available.includes(part) ? part : available.includes("test") ? "test" : available.includes("validation") ? "validation" : null;
  const ev: EvaluationEntry | undefined = shown ? evals.data?.[shown] : undefined;

  const baselines = useMemo(() => completed.filter((e) => e.kind === "baseline"), [completed]);
  const cnns = useMemo(() => completed.filter((e) => e.kind === "cnn"), [completed]);

  if (exps.isLoading) return <Loading />;
  if (!completed.length)
    return (
      <div>
        <PageHeader title="Model Evaluation" />
        <Empty title="No completed experiments">Train a model in the Model Laboratory first.</Empty>
      </div>
    );

  const m = ev?.metrics;
  return (
    <div className="space-y-5">
      <PageHeader
        title="Model Evaluation"
        sub="Metrics computed from stored predictions on real beats. Training, validation and held-out test results are kept strictly separate; test subjects are never used for model selection."
        actions={
          <select className="input w-auto max-w-[360px]" value={expId} onChange={(e) => setExp(e.target.value)} data-testid="eval-experiment">
            <optgroup label="1D CNN">
              {cnns.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} · val F1 {num(e.summary.val_macro_f1 ?? null, 3)}
                  {e.active ? " · active" : ""}
                </option>
              ))}
            </optgroup>
            <optgroup label="Baselines">
              {baselines.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} · val F1 {num(e.summary.val_macro_f1 ?? null, 3)}
                </option>
              ))}
            </optgroup>
          </select>
        }
      />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <Seg
          value={shown ?? "validation"}
          onChange={(v) => setPart(v)}
          options={(["test", "validation", "train"] as Part[]).filter((p) => available.includes(p)).map((p) => ({
            value: p,
            label: p === "test" ? "Held-out test subjects" : p === "validation" ? "Validation subjects" : "Training subjects",
          }))}
          testId="partition-picker"
        />
        {ev && (
          <span className="text-2xs text-fg-dim">
            {ev.records.length} recordings · evaluated {dateTime(ev.created_at)}
          </span>
        )}
      </div>
      {shown === "train" && <Note tone="warn">Training-subject metrics show how well the model fits data it learned from. They say nothing about new patients.</Note>}
      {shown === "validation" && <Note>Validation subjects were held out from training and used for early stopping and model selection, so these numbers are slightly optimistic.</Note>}
      {shown === "test" && (
        <Note tone="safety">
          Held-out test subjects: never used for training, early stopping or model selection. These are the numbers that estimate generalisation to new
          patients — within the limits of MIT-BIH (1970s Holter recordings, 21 test subjects). Split: inter-patient (subject-corrected DS1/DS2), so these
          numbers are not comparable with papers that split beats from the same patients between training and test.
        </Note>
      )}

      {!available.includes("test") && exp.data && (
        <Panel title="Held-out test evaluation is locked" testId="test-lock">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1 space-y-2">
              <p className="flex items-start gap-2 text-xs text-fg-muted">
                <Lock className="mt-0.5 h-3.5 w-3.5 text-cls-S" />
                Running the test evaluation is logged. Do it once per final model. Comparing many models on the test set and keeping the best turns it into a
                validation set.
              </p>
              <input className="input" placeholder="Purpose (logged)" value={purpose} onChange={(e) => setPurpose(e.target.value)} />
            </div>
            <button className="btn-primary" onClick={() => runTest.mutate()} disabled={runTest.isPending} data-testid="run-test-eval">
              <ShieldCheck className="h-3.5 w-3.5" /> {runTest.isPending ? "Evaluating…" : "Evaluate on held-out subjects"}
            </button>
          </div>
          {runTest.error && <ErrorState error={runTest.error} />}
        </Panel>
      )}

      {!ev ? (
        <Loading />
      ) : (
        <>
          <Headline metrics={m!} />
          <div className="grid gap-5 xl:grid-cols-5">
            <Panel className="xl:col-span-2" title="Confusion matrix" testId="eval-confusion">
              <ConfusionMatrix metrics={m!} />
            </Panel>
            <Panel className="xl:col-span-3" title="Per-class performance" sub="Sensitivity = recall; +P = positive predictivity (precision)">
              <PerClassTable metrics={m!} ci={ev.bootstrap} />
              {ev.bootstrap && (
                <p className="mt-2 text-2xs text-fg-dim">
                  Macro-F1 95% CI (subject bootstrap): {Array.isArray(ev.bootstrap.macro_f1) ? `${num(ev.bootstrap.macro_f1[0] as number)} – ${num(ev.bootstrap.macro_f1[1] as number)}` : "—"}
                </p>
              )}
            </Panel>
          </div>

          {m!.roc && m!.pr && (
            <div className="grid gap-5 lg:grid-cols-2">
              <Panel title="ROC curves (one-vs-rest)">
                <Plot
                  testId="roc-curves"
                  height={300}
                  data={[
                    ...Object.entries(m!.roc).map(([c, r]) => ({
                      type: "scatter" as const,
                      mode: "lines" as const,
                      name: `${c} · AUC ${num(r.auc, 3)}`,
                      x: (r.points ?? []).map((p) => p[0]),
                      y: (r.points ?? []).map((p) => p[1]),
                      line: { color: CLASS_COLORS[c], width: 2 },
                    })),
                    { type: "scatter", mode: "lines", name: "chance", x: [0, 1], y: [0, 1], line: { color: "#3A4555", dash: "dot", width: 1 }, showlegend: false },
                  ]}
                  layout={{ xaxis: { title: { text: "false positive rate" }, range: [0, 1] }, yaxis: { title: { text: "sensitivity" }, range: [0, 1.02] } }}
                />
              </Panel>
              <Panel title="Precision–recall curves" sub="More informative than ROC for rare classes; dotted lines = class prevalence">
                <Plot
                  testId="pr-curves"
                  height={300}
                  data={[
                    ...Object.entries(m!.pr).map(([c, r]) => ({
                      type: "scatter" as const,
                      mode: "lines" as const,
                      name: `${c} · AP ${num(r.ap, 3)}`,
                      x: (r.points ?? []).map((p) => p[0]),
                      y: (r.points ?? []).map((p) => p[1]),
                      line: { color: CLASS_COLORS[c], width: 2 },
                    })),
                    ...Object.entries(m!.pr).map(([c, r]) => ({
                      type: "scatter" as const,
                      mode: "lines" as const,
                      x: [0, 1],
                      y: [r.prevalence, r.prevalence],
                      line: { color: CLASS_COLORS[c], dash: "dot" as const, width: 1 },
                      showlegend: false,
                      hoverinfo: "skip" as const,
                    })),
                  ]}
                  layout={{ xaxis: { title: { text: "recall" }, range: [0, 1] }, yaxis: { title: { text: "precision" }, range: [0, 1.02] } }}
                />
              </Panel>
            </div>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Per-patient breakdown" sub={`${ev.per_subject.length} subjects in this partition`} bodyClass="p-0" testId="per-subject">
              <div className="max-h-[360px] overflow-auto">
                <table className="table">
                  <thead className="sticky top-0 bg-ink-800">
                    <tr>
                      <th>Subject</th>
                      <th className="text-right">Beats</th>
                      <th className="text-right">Accuracy</th>
                      {m!.classes.map((c) => (
                        <th key={c} className="text-right" style={{ color: CLASS_COLORS[c] }}>
                          {c} recall
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ev.per_subject.map((r) => (
                      <tr key={String(r.subject)}>
                        <td className="num text-fg">{String(r.subject)}</td>
                        <td className="num text-right">{int(r.n as number)}</td>
                        <td className="num text-right">{pct(r.accuracy as number)}</td>
                        {m!.classes.map((c) => (
                          <td key={c} className="num text-right">
                            {r[`${c}_recall`] === null ? <span className="text-fg-dim">—</span> : `${pct(r[`${c}_recall`] as number, 0)}`}
                            <span className="ml-1 text-fg-dim">({r[`${c}_support`]})</span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
            <Panel title="Inference cost">
              <dl className="grid grid-cols-2 gap-3 text-xs">
                {Object.entries(ev.inference).map(([k, v]) => (
                  <div key={k} className="rounded-md bg-ink-850 px-3 py-2">
                    <dt className="text-2xs text-fg-dim">{k.replace(/_/g, " ")}</dt>
                    <dd className="num mt-0.5 text-fg">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </Panel>
          </div>
        </>
      )}

      {exp.data?.kind === "cnn" && epochs.data && epochs.data.length > 0 && (
        <Panel title="Training and validation curves" sub={`Best epoch ${exp.data.summary.best_epoch ?? "—"} of ${exp.data.summary.epochs_run ?? "—"} (early stopping on validation macro F1)`}>
          <div className="grid gap-4 lg:grid-cols-2">
            <LossCurves epochs={epochs.data} />
            <F1Curves epochs={epochs.data} />
          </div>
        </Panel>
      )}

      <Panel title="Model comparison" sub="Validation subjects (identical for all runs with the same validation fold); test only where explicitly evaluated" bodyClass="p-0" testId="model-comparison">
        <div className="overflow-x-auto">
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th className="text-right">Val macro F1</th>
                {["N", "S", "V", "F"].map((c) => (
                  <th key={c} className="text-right" style={{ color: CLASS_COLORS[c] }}>
                    Val F1 {c}
                  </th>
                ))}
                <th className="text-right">Train macro F1</th>
                <th className="text-right">Test evaluated</th>
              </tr>
            </thead>
            <tbody>
              {[...completed]
                .sort((a, b) => (b.summary.val_macro_f1 ?? 0) - (a.summary.val_macro_f1 ?? 0))
                .map((e) => (
                  <tr key={e.id} className={e.id === expId ? "bg-ink-750" : ""}>
                    <td>
                      <button className="text-left text-fg hover:text-teal" onClick={() => setExp(e.id)}>
                        {e.name}
                      </button>
                    </td>
                    <td>{e.kind === "cnn" ? "1D CNN" : "baseline"}</td>
                    <td className="num text-right font-medium text-fg">{num(e.summary.val_macro_f1 ?? null)}</td>
                    {["N", "S", "V", "F"].map((c) => (
                      <td key={c} className="num text-right">
                        {num(e.summary.val_per_class_f1?.[c] ?? null, 2)}
                      </td>
                    ))}
                    <td className="num text-right">{num(e.summary.train_macro_f1 ?? null)}</td>
                    <td className="text-right">{e.has_test_evaluation ? "yes" : "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {splits.data && exp.data?.split && (
        <Panel title="Exact patient-level split used" testId="exact-split">
          <div className="grid gap-4 text-xs lg:grid-cols-3">
            {[
              ["Training", exp.data.split.train_records, exp.data.split.n_train_beats],
              ["Validation", exp.data.split.validation_records, exp.data.split.n_validation_beats],
              ["Held-out test", exp.data.split.test_records, undefined],
            ].map(([label, recs, n]) => (
              <div key={label as string}>
                <div className="label mb-2">
                  {label as string} {n !== undefined && <span className="num normal-case">· {int(n as number)} beats</span>}
                </div>
                <div className="flex flex-wrap gap-1">
                  {(recs as string[] | undefined)?.map((r) => (
                    <span key={r} className="num rounded border border-line px-1.5 py-0.5 text-2xs text-fg-muted">
                      {r}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-2xs text-fg-dim">
            Excluded: {exp.data.split.excluded_records?.join(", ")}. {splits.data.difference_from_original} All leakage checks{" "}
            {splits.data.leakage.passed ? "pass" : "DO NOT pass"}.
          </p>
        </Panel>
      )}

      <Panel title="Test-set access log" sub="Every evaluation on held-out subjects is recorded" bodyClass="p-0">
        {log.data?.length ? (
          <table className="table">
            <thead>
              <tr>
                <th>When</th>
                <th>Experiment</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              {log.data.map((l, i) => (
                <tr key={i}>
                  <td className="num">{dateTime(l.timestamp)}</td>
                  <td className="num">{l.experiment_id}</td>
                  <td>{l.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="p-4 text-xs text-fg-dim">No test-set evaluations yet.</div>
        )}
      </Panel>
    </div>
  );
}
