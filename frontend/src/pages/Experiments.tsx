import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, GitCompare, Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Empty, ErrorState, Loading, PageHeader, Panel, StatusDot } from "../components/ui";
import { CLASS_COLORS } from "../lib/classes";
import { dateTime, duration, int, num } from "../lib/format";

const CFG_KEYS = ["classes", "epochs", "batch_size", "lr", "weight_decay", "dropout", "loss", "sampler", "use_rr", "augment", "window_pre", "window_post", "val_fold", "seed", "model"];

export default function Experiments() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["experiments"], queryFn: api.experiments, refetchInterval: 8000 });
  const [selected, setSelected] = useState<string[]>([]);
  const [detail, setDetail] = useState<string | null>(null);
  const cmp = useQuery({ queryKey: ["compare", selected], queryFn: () => api.compare(selected), enabled: selected.length >= 2 });
  const det = useQuery({ queryKey: ["experiment", detail], queryFn: () => api.experiment(detail!), enabled: !!detail });
  const activate = useMutation({ mutationFn: api.activate, onSuccess: () => qc.invalidateQueries() });
  const remove = useMutation({ mutationFn: api.deleteExperiment, onSuccess: () => qc.invalidateQueries() });

  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorState error={q.error} />;
  const rows = q.data!;
  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-4)));

  return (
    <div className="space-y-5">
      <PageHeader title="Experiment History" sub="Every training run is stored with its configuration, preprocessing, patient partitions, environment, metrics and checkpoint location. Select up to four runs to compare." />
      <Panel title={`${rows.length} experiments`} bodyClass="p-0" testId="experiment-table">
        {rows.length === 0 ? (
          <div className="p-4">
            <Empty title="No experiments yet" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th />
                  <th>Experiment</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th className="text-right">Duration</th>
                  <th className="text-right">Val macro F1</th>
                  <th className="text-right">Train macro F1</th>
                  <th>Test</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((e) => (
                  <tr key={e.id} className={detail === e.id ? "bg-ink-750" : ""}>
                    <td>
                      <input type="checkbox" checked={selected.includes(e.id)} onChange={() => toggle(e.id)} aria-label={`select ${e.name}`} />
                    </td>
                    <td>
                      <button className="text-left" onClick={() => setDetail(e.id)}>
                        <div className="flex items-center gap-1.5 text-fg">
                          {e.active && <Star className="h-3 w-3 fill-teal text-teal" />}
                          {e.name}
                        </div>
                        <div className="num text-2xs text-fg-dim">
                          {e.id} · {e.kind === "cnn" ? "1D CNN" : e.model_name}
                        </div>
                      </button>
                    </td>
                    <td>
                      <StatusDot status={e.status} />
                    </td>
                    <td className="num">{dateTime(e.created_at)}</td>
                    <td className="num text-right">{duration(e.duration_s)}</td>
                    <td className="num text-right font-medium text-fg">{num(e.summary.val_macro_f1 ?? null)}</td>
                    <td className="num text-right">{num(e.summary.train_macro_f1 ?? null)}</td>
                    <td>{e.has_test_evaluation ? <span className="text-cls-S">evaluated</span> : <span className="text-fg-dim">locked</span>}</td>
                    <td className="text-right">
                      <div className="flex justify-end gap-1">
                        {e.kind === "cnn" && e.status === "completed" && !e.active && (
                          <button className="btn py-0.5" onClick={() => activate.mutate(e.id)} title="Use for inference">
                            <Star className="h-3 w-3" />
                          </button>
                        )}
                        <a className="btn py-0.5" href={api.exportUrl(e.id)} download={`${e.id}.json`} title="Export JSON">
                          <Download className="h-3 w-3" />
                        </a>
                        <button
                          className="btn py-0.5"
                          title="Delete"
                          onClick={() => window.confirm(`Delete experiment "${e.name}" and its checkpoint?`) && remove.mutate(e.id)}
                          disabled={e.status === "running"}
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {selected.length >= 2 && (
        <Panel title={<span className="flex items-center gap-2"><GitCompare className="h-4 w-4 text-teal" /> Comparison</span>} bodyClass="p-0" testId="comparison">
          {cmp.data ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th>Field</th>
                    {cmp.data.map((e) => (
                      <th key={e.id} className="normal-case">
                        {e.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {["validation", "test"].flatMap((part) => [
                    <tr key={`${part}-f1`}>
                      <td className="text-fg">{part} macro F1</td>
                      {cmp.data!.map((e) => (
                        <td key={e.id} className="num text-fg">
                          {num(e.evaluations[part]?.macro_f1 ?? null)}
                        </td>
                      ))}
                    </tr>,
                    ...["N", "S", "V", "F"].map((c) => (
                      <tr key={`${part}-${c}`}>
                        <td>
                          {part} F1 <span style={{ color: CLASS_COLORS[c] }}>{c}</span>
                        </td>
                        {cmp.data!.map((e) => (
                          <td key={e.id} className="num">
                            {num(e.evaluations[part]?.per_class?.[c]?.f1 ?? null, 3)}
                          </td>
                        ))}
                      </tr>
                    )),
                  ])}
                  {CFG_KEYS.map((k) => (
                    <tr key={k}>
                      <td>{k}</td>
                      {cmp.data!.map((e) => (
                        <td key={e.id} className="num">
                          {e.config[k] === undefined ? "—" : JSON.stringify(e.config[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                  <tr>
                    <td>parameters</td>
                    {cmp.data.map((e) => (
                      <td key={e.id} className="num">
                        {int(e.n_params)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td>duration</td>
                    {cmp.data.map((e) => (
                      <td key={e.id} className="num">
                        {duration(e.duration_s)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4">
              <Loading />
            </div>
          )}
        </Panel>
      )}

      {detail && det.data && (
        <Panel title={det.data.name} sub={det.data.id} actions={<Link to={`/evaluation?exp=${det.data.id}`} className="btn">Open evaluation</Link>} testId="experiment-detail">
          <div className="grid gap-5 lg:grid-cols-3 text-xs">
            <div>
              <div className="label mb-2">Hyperparameters</div>
              <pre className="max-h-72 overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-fg-muted">{JSON.stringify(det.data.config, null, 2)}</pre>
            </div>
            <div>
              <div className="label mb-2">Preprocessing & data</div>
              <pre className="max-h-72 overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-fg-muted">
                {JSON.stringify({ dataset_version: det.data.dataset_version, pipeline_hash: det.data.pipeline_hash, pipeline: det.data.pipeline }, null, 2)}
              </pre>
            </div>
            <div>
              <div className="label mb-2">Environment & checkpoint</div>
              <pre className="max-h-72 overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-fg-muted">
                {JSON.stringify({ checkpoint: det.data.checkpoint_path, started: det.data.started_at, finished: det.data.finished_at, ...det.data.environment }, null, 2)}
              </pre>
            </div>
            <div className="lg:col-span-3">
              <div className="label mb-2">Patient partitions</div>
              <pre className="overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-fg-muted">{JSON.stringify(det.data.split, null, 2)}</pre>
            </div>
            {det.data.cv_results && (
              <div className="lg:col-span-3">
                <div className="label mb-2">
                  Cross-validation: macro F1 {num(det.data.cv_results.mean_macro_f1)} ± {num(det.data.cv_results.std_macro_f1)}
                </div>
                <div className="flex flex-wrap gap-2">
                  {det.data.cv_results.folds.map((f) => (
                    <span key={f.fold} className="num rounded border border-line px-2 py-1 text-2xs">
                      fold {f.fold}: {num(f.val_macro_f1)} ({f.records.join(",")})
                    </span>
                  ))}
                </div>
              </div>
            )}
            {det.data.error && <pre className="lg:col-span-3 overflow-auto rounded-lg bg-ink-950 p-3 text-2xs text-trace">{det.data.error}</pre>}
          </div>
        </Panel>
      )}
    </div>
  );
}
