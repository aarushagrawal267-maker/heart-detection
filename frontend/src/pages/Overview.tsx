import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Database, Download, FlaskConical } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import EcgChart from "../charts/EcgChart";
import { ClassChip, Empty, ErrorState, Loading, PageHeader, Panel, ResearchNotice, Stat, StatusDot } from "../components/ui";
import { CLASS_COLORS, CLASS_NAMES } from "../lib/classes";
import { int, num, pct, relative } from "../lib/format";

function HeroStrip() {
  const sig = useQuery({ queryKey: ["hero-signal"], queryFn: () => api.signal("208", 312, 320, undefined, false, 4000) });
  const ann = useQuery({ queryKey: ["hero-ann"], queryFn: () => api.annotations("208", 312, 320) });
  const refs = useMemo(
    () =>
      (ann.data?.annotations ?? [])
        .filter((a) => a.is_beat)
        .map((a) => ({ time_s: a.time_s, label: a.aami ?? a.symbol, cls: a.aami })),
    [ann.data],
  );
  if (!sig.data) return <div className="h-[200px]" />;
  return (
    <div>
      <EcgChart t={sig.data.t} y={sig.data.y} range={[312, 320]} height={200} reference={refs} testId="hero-ecg" />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-2xs text-fg-dim">
        <span>
          Record 208 · lead {sig.data.lead} · 5:12–5:20 · raw signal with expert reference annotations (AAMI classes)
        </span>
        <span className="num">360 Hz · 25 mm/s grid · 0.1 mV minor</span>
      </div>
    </div>
  );
}

export default function Overview() {
  const q = useQuery({ queryKey: ["overview"], queryFn: api.overview, refetchInterval: 10000 });
  if (q.isLoading) return <Loading label="Loading overview" />;
  if (q.error) return <ErrorState error={q.error} />;
  const d = q.data!;
  const counts = d.dataset.class_counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const am = d.active_model;

  return (
    <div className="space-y-5">
      <PageHeader
        title="ECG Intelligence Overview"
        sub="Heartbeat classification on real MIT-BIH recordings with a 1D convolutional network trained from scratch in this project, evaluated on patients it never saw during training."
        actions={
          <>
            <Link to="/explorer" className="btn">
              Open Signal Explorer <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
            <Link to="/classify" className="btn-primary">
              Classify a heartbeat <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </>
        }
      />
      <ResearchNotice />

      {!d.dataset.present ? (
        <Panel title="Dataset not downloaded">
          <Empty title="MIT-BIH Arrhythmia Database is not on this machine yet">
            Go to the <Link to="/dataset" className="text-teal">Dataset Explorer</Link> to download (≈77 MB) and verify it
            against PhysioNet's published SHA-256 checksums.
          </Empty>
        </Panel>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4" data-testid="overview-stats">
            <Stat label="Recordings" value={d.dataset.n_records} sub={`${d.dataset.n_subjects} subjects · 201 & 202 share one`} />
            <Stat label="Segmented beats" value={int(d.dataset.n_segments)} sub="at expert reference positions" />
            <Stat
              label="Trained experiments"
              value={d.experiments.completed}
              sub={d.experiments.running ? `${d.experiments.running} running now` : `${d.experiments.total} total`}
            />
            <Stat
              label="Held-out macro F1"
              value={am?.test ? num(am.test.macro_f1) : "—"}
              accent={am?.test ? "#2BB3A3" : undefined}
              sub={am?.test ? `${int(am.test.n)} beats from unseen patients` : "not evaluated yet"}
            />
          </div>

          <Panel title="Live waveform" sub="A real excerpt rendered by the platform's ECG viewer" bodyClass="p-3 sm:p-4">
            <HeroStrip />
          </Panel>

          <div className="grid gap-5 lg:grid-cols-3">
            <Panel title="Heartbeat classes" sub="AAMI EC57 classes across all 48 recordings" testId="class-distribution">
              <div className="space-y-3">
                {["N", "S", "V", "F", "Q"].map((c) => (
                  <div key={c}>
                    <div className="mb-1 flex items-center justify-between text-2xs">
                      <span className="flex items-center gap-2">
                        <ClassChip cls={c} />
                        <span className="text-fg-muted">{CLASS_NAMES[c]}</span>
                      </span>
                      <span className="num text-fg-muted">
                        {int(counts[c] ?? 0)} <span className="text-fg-dim">· {pct((counts[c] ?? 0) / (total || 1), 2)}</span>
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-600">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${Math.max(((counts[c] ?? 0) / (total || 1)) * 100, 0.4)}%`, background: CLASS_COLORS[c] }}
                      />
                    </div>
                  </div>
                ))}
                <p className="pt-1 text-2xs leading-relaxed text-fg-dim">
                  Most Q beats are paced beats from records 102, 104, 107 and 217, which are excluded from training and
                  evaluation. Accuracy alone is misleading: predicting N for every beat scores about 89%.
                </p>
              </div>
            </Panel>

            <Panel
              title="Active model"
              sub={am ? am.name : "No trained CNN yet"}
              actions={am && <Link to="/evaluation" className="btn">Evaluation</Link>}
              testId="active-model"
            >
              {am ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="label">Validation macro F1</div>
                      <div className="num mt-1 text-xl">{num(am.validation?.macro_f1 ?? null)}</div>
                    </div>
                    <div>
                      <div className="label">Held-out macro F1</div>
                      <div className="num mt-1 text-xl text-teal">{am.test ? num(am.test.macro_f1) : "—"}</div>
                    </div>
                  </div>
                  {am.test && (
                    <div>
                      <div className="label mb-2">Held-out F1 by class</div>
                      <div className="grid grid-cols-4 gap-2">
                        {am.classes.map((c) => (
                          <div key={c} className="rounded-md border border-line-soft bg-ink-850 px-2 py-1.5 text-center">
                            <div className="font-mono text-2xs" style={{ color: CLASS_COLORS[c] }}>
                              {c}
                            </div>
                            <div className="num text-sm">{num(am.test!.per_class[c]?.f1 ?? null, 2)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="text-2xs text-fg-dim">
                    {int(am.n_params)} parameters · trained {relative(am.finished_at)} · classes {am.classes.join(" ")}
                  </div>
                </div>
              ) : (
                <Empty title="Train the 1D CNN">
                  <Link to="/lab" className="text-teal">Open the Model Laboratory</Link> to train the network on real
                  training-subject beats.
                </Empty>
              )}
            </Panel>

            <Panel title="Latest experiment" testId="latest-experiment">
              {d.latest_experiment ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="truncate text-sm text-fg">{d.latest_experiment.name}</div>
                    <StatusDot status={d.latest_experiment.status} />
                  </div>
                  <div className="text-2xs text-fg-dim">created {relative(d.latest_experiment.created_at)}</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="label">Val macro F1</div>
                      <div className="num mt-1 text-lg">{num(d.latest_experiment.summary?.val_macro_f1 ?? null)}</div>
                    </div>
                    <div>
                      <div className="label">Train macro F1</div>
                      <div className="num mt-1 text-lg text-fg-muted">{num(d.latest_experiment.summary?.train_macro_f1 ?? null)}</div>
                    </div>
                  </div>
                  <div className="flex gap-2 pt-1">
                    <Link to="/experiments" className="btn">
                      <FlaskConical className="h-3.5 w-3.5" /> History
                    </Link>
                    <Link to="/dataset" className="btn">
                      <Database className="h-3.5 w-3.5" /> Dataset
                    </Link>
                  </div>
                </div>
              ) : (
                <Empty title="No experiments yet" />
              )}
            </Panel>
          </div>

          <Panel title="Recent classification activity" sub="Logged inference requests from this installation" testId="recent-activity">
            {d.recent_activity.length ? (
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Type</th>
                      <th>Record</th>
                      <th>Beat</th>
                      <th>Reference</th>
                      <th>CNN</th>
                      <th className="text-right">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.recent_activity.map((a, i) => (
                      <tr key={i}>
                        <td className="num">{relative(a.timestamp)}</td>
                        <td>{a.kind === "beat" ? "Single beat" : `Full record (${String(a.summary?.mode ?? "")})`}</td>
                        <td className="num text-fg">{a.record}</td>
                        <td className="num">{a.beat_index ?? `${String(a.summary?.n_beats ?? "")} beats`}</td>
                        <td>{a.reference ? <ClassChip cls={a.reference} /> : "—"}</td>
                        <td>{a.predicted ? <ClassChip cls={a.predicted} /> : a.summary?.agreement_accuracy != null ? `agreement ${pct(a.summary.agreement_accuracy as number)}` : "—"}</td>
                        <td className="num text-right">{a.confidence != null ? pct(a.confidence) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty title="No classifications yet">
                Classify a beat or analyse a recording; each request is logged here.
              </Empty>
            )}
          </Panel>
        </>
      )}
      {!d.dataset.present && (
        <Link to="/dataset" className="btn-primary">
          <Download className="h-3.5 w-3.5" /> Download dataset
        </Link>
      )}
    </div>
  );
}
