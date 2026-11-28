import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type EpochRow, type Experiment } from "../api/client";
import { F1Curves, LossCurves } from "../components/TrainingCurves";
import { Bar, ErrorState, Field, Loading, Note, PageHeader, Panel, Seg, StatusDot } from "../components/ui";
import { duration, int, num } from "../lib/format";

type Cfg = Record<string, unknown>;

function useLiveExperiment(id: string | null) {
  const [exp, setExp] = useState<Experiment | null>(null);
  const [epochs, setEpochs] = useState<EpochRow[]>([]);
  const qc = useQueryClient();
  useEffect(() => {
    if (!id) return;
    setExp(null);
    setEpochs([]);
    const es = new EventSource(`/api/experiments/${id}/stream`);
    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      setExp(d.experiment);
      if (d.epochs) setEpochs(d.epochs);
      if (["completed", "failed", "cancelled"].includes(d.experiment.status)) {
        es.close();
        qc.invalidateQueries();
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [id, qc]);
  return { exp, epochs };
}

export default function Lab() {
  const qc = useQueryClient();
  const defaults = useQuery({ queryKey: ["training-defaults"], queryFn: api.trainingDefaults, staleTime: Infinity });
  const current = useQuery({ queryKey: ["current-job"], queryFn: api.currentJob, refetchInterval: 5000 });
  const [kind, setKind] = useState<"cnn" | "baseline">("cnn");
  const [name, setName] = useState("");
  const [cfg, setCfg] = useState<Cfg | null>(null);
  const [bcfg, setBcfg] = useState<Cfg | null>(null);
  const [watchId, setWatchId] = useState<string | null>(null);

  useEffect(() => {
    if (defaults.data && !cfg) {
      setCfg({ ...defaults.data.cnn });
      setBcfg({ ...defaults.data.baseline });
    }
  }, [defaults.data, cfg]);
  useEffect(() => {
    if (current.data && !watchId) setWatchId(current.data.id);
  }, [current.data, watchId]);

  const live = useLiveExperiment(watchId);
  const create = useMutation({
    mutationFn: () =>
      api.createExperiment({
        name: name || (kind === "cnn" ? "ECGNet experiment" : `${String(bcfg?.model)} baseline`),
        kind,
        config: kind === "cnn" ? cfg! : bcfg!,
      }),
    onSuccess: (e) => {
      setWatchId(e.id);
      qc.invalidateQueries({ queryKey: ["current-job"] });
    },
  });
  const cancel = useMutation({ mutationFn: (id: string) => api.cancelExperiment(id) });

  if (defaults.isLoading || !cfg || !bcfg) return <Loading />;
  if (defaults.error) return <ErrorState error={defaults.error} />;
  const d = defaults.data!;
  const set = (k: string, v: unknown) => setCfg((c) => ({ ...c!, [k]: v }));
  const busy = !!current.data;
  const running = live.exp && ["running", "queued", "cancelling"].includes(live.exp.status);
  const len = Number(cfg.window_pre) + Number(cfg.window_post);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Model Laboratory"
        sub="Train the project's own 1D CNN (random initialisation, no pretrained weights) or a baseline on real training-subject beats. Model selection uses validation subjects only; held-out test subjects are never touched here."
      />
      <Note>
        Training runs in a separate process on <strong className="text-fg">{d.device.toUpperCase()}</strong>. One job at a time. Nothing trains
        automatically. A default ECGNet run takes roughly 5–10 minutes on a 4-core laptop CPU.
      </Note>

      <div className="grid gap-5 xl:grid-cols-[400px_1fr]">
        <Panel title="Configuration" actions={<Seg value={kind} onChange={setKind} options={[{ value: "cnn", label: "1D CNN" }, { value: "baseline", label: "Baseline" }]} testId="kind-picker" />} testId="lab-config">
          <div className="space-y-4">
            <Field label="Experiment name">
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. ECGNet focal γ=2" data-testid="exp-name" />
            </Field>
            {kind === "cnn" ? (
              <>
                <Field label="Heartbeat classes" hint="Q has ≈15 beats outside paced records; the 5-class mode is for exploration.">
                  <Seg
                    value={(cfg.classes as string[]).join("")}
                    onChange={(v) => set("classes", v.split(""))}
                    options={d.options.classes.map((c) => ({ value: c.join(""), label: c.join(" · ") }))}
                  />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Epochs">
                    <input type="number" min={1} max={200} className="input" value={Number(cfg.epochs)} onChange={(e) => set("epochs", Number(e.target.value))} data-testid="cfg-epochs" />
                  </Field>
                  <Field label="Batch size">
                    <input type="number" min={16} max={2048} className="input" value={Number(cfg.batch_size)} onChange={(e) => set("batch_size", Number(e.target.value))} />
                  </Field>
                  <Field label="Learning rate">
                    <input type="number" step="0.0001" className="input" value={Number(cfg.lr)} onChange={(e) => set("lr", Number(e.target.value))} />
                  </Field>
                  <Field label="Weight decay">
                    <input type="number" step="0.0001" className="input" value={Number(cfg.weight_decay)} onChange={(e) => set("weight_decay", Number(e.target.value))} />
                  </Field>
                  <Field label="Dropout">
                    <input type="number" step="0.05" min={0} max={0.9} className="input" value={Number(cfg.dropout)} onChange={(e) => set("dropout", Number(e.target.value))} />
                  </Field>
                  <Field label="Early-stop patience">
                    <input type="number" min={1} className="input" value={Number(cfg.early_stopping_patience)} onChange={(e) => set("early_stopping_patience", Number(e.target.value))} />
                  </Field>
                </div>
                <Field label="Loss" hint="Class weights are computed from training-subject counts only.">
                  <Seg value={String(cfg.loss)} onChange={(v) => set("loss", v)} options={d.options.loss.map((l) => ({ value: l, label: l }))} />
                </Field>
                <Field label="Class-weight exponent" hint="0 = unweighted, 0.5 = inverse square-root frequency, 1 = inverse frequency.">
                  <input type="number" step="0.25" min={0} max={2} className="input" value={Number(cfg.class_weight_power)} onChange={(e) => set("class_weight_power", Number(e.target.value))} />
                </Field>
                <Field label="Sampler" hint="'balanced' oversamples minority beats inside the training partition only (never across partitions).">
                  <Seg value={String(cfg.sampler)} onChange={(v) => set("sampler", v)} options={d.options.sampler.map((l) => ({ value: l, label: l }))} />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Window before (samples)">
                    <input type="number" min={16} max={d.options.window_limits.max_pre} className="input" value={Number(cfg.window_pre)} onChange={(e) => set("window_pre", Number(e.target.value))} />
                  </Field>
                  <Field label="Window after (samples)">
                    <input type="number" min={16} max={d.options.window_limits.max_post} className="input" value={Number(cfg.window_post)} onChange={(e) => set("window_post", Number(e.target.value))} />
                  </Field>
                </div>
                <p className="-mt-2 text-2xs text-fg-dim">
                  {len} samples = {num((len / 360) * 1000, 0)} ms at 360 Hz.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Validation fold">
                    <select className="input" value={Number(cfg.val_fold)} onChange={(e) => set("val_fold", Number(e.target.value))}>
                      {d.options.val_fold.map((f) => (
                        <option key={f} value={f}>
                          Fold {f}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Seed">
                    <input type="number" className="input" value={Number(cfg.seed)} onChange={(e) => set("seed", Number(e.target.value))} />
                  </Field>
                </div>
                <div className="space-y-2">
                  {[
                    ["use_rr", "Use RR-interval features (prematurity)"],
                    ["augment", "Training augmentation (shift, scale, noise)"],
                    ["cross_validate", "Also run 5-fold subject-grouped cross-validation (≈6× longer)"],
                  ].map(([k, l]) => (
                    <label key={k} className="flex items-center gap-2 text-xs text-fg-muted">
                      <input type="checkbox" checked={Boolean(cfg[k])} onChange={(e) => set(k, e.target.checked)} /> {l}
                    </label>
                  ))}
                </div>
              </>
            ) : (
              <Field label="Baseline model">
                <div className="space-y-1.5">
                  {Object.entries(d.baselines).map(([k, v]) => (
                    <label key={k} className="flex items-start gap-2 text-xs text-fg-muted">
                      <input type="radio" name="baseline" checked={bcfg.model === k} onChange={() => setBcfg({ ...bcfg, model: k })} className="mt-0.5" />
                      <span>
                        <span className="font-mono text-fg">{k}</span> — {v}
                      </span>
                    </label>
                  ))}
                </div>
              </Field>
            )}
            {create.error && <ErrorState error={create.error} />}
            <button className="btn-primary w-full py-2" onClick={() => create.mutate()} disabled={busy || create.isPending} data-testid="start-training">
              <Play className="h-3.5 w-3.5" /> {busy ? "A job is already running" : "Start training on real data"}
            </button>
          </div>
        </Panel>

        <div className="min-w-0 space-y-5">
          <Panel
            title={live.exp ? live.exp.name : "Training monitor"}
            sub={live.exp ? `${live.exp.id} · ${live.exp.model_name}` : "Start a job to see live progress"}
            actions={
              live.exp && (
                <div className="flex items-center gap-2">
                  <StatusDot status={live.exp.status} />
                  {running && (
                    <button className="btn-danger" onClick={() => cancel.mutate(live.exp!.id)} data-testid="cancel-training">
                      <Square className="h-3 w-3" /> Cancel
                    </button>
                  )}
                </div>
              )
            }
            testId="training-monitor"
          >
            {!live.exp ? (
              <div className="py-8 text-center text-xs text-fg-dim">No job selected. Completed runs are in the Experiment History.</div>
            ) : (
              <div className="space-y-4">
                <div>
                  <div className="mb-1.5 flex justify-between text-2xs text-fg-muted">
                    <span data-testid="training-message">{live.exp.message}</span>
                    <span className="num">{Math.round(live.exp.progress * 100)}%</span>
                  </div>
                  <Bar value={live.exp.progress} color="#9AA3B2" />
                </div>
                {live.exp.error && <pre className="max-h-40 overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-trace">{live.exp.error}</pre>}
                {live.exp.kind === "cnn" && live.epochs.length > 0 && (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <LossCurves epochs={live.epochs} />
                    <F1Curves epochs={live.epochs} />
                  </div>
                )}
                {live.exp.status === "completed" && (
                  <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line-strong bg-ink-750 p-3 text-xs">
                    <span className="text-fg">
                      Completed in {duration(live.exp.duration_s)} · validation macro F1{" "}
                      <span className="num text-fg">{num(live.exp.summary.val_macro_f1 ?? null)}</span>
                    </span>
                    <Link to={`/evaluation?exp=${live.exp.id}`} className="btn">
                      View evaluation
                    </Link>
                    {live.exp.checkpoint_path && <span className="num text-2xs text-fg-dim">checkpoint: {live.exp.checkpoint_path}</span>}
                  </div>
                )}
              </div>
            )}
          </Panel>

          <Panel title="ECGNet architecture" sub={`${int(d.n_params)} trainable parameters (default configuration)`} bodyClass="p-0" testId="architecture">
            <table className="table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Output (channels × length)</th>
                  <th className="text-right">Parameters</th>
                </tr>
              </thead>
              <tbody>
                {d.architecture.map((l, i) => (
                  <tr key={i}>
                    <td className="font-mono text-fg">{l.layer}</td>
                    <td className="num">{l.output_shape ? l.output_shape.join(" × ") : "—"}</td>
                    <td className="num text-right">{int(l.params)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>
    </div>
  );
}
