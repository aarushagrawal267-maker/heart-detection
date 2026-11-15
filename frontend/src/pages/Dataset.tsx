import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Download, RefreshCw, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import Plot from "../charts/Plot";
import { ClassChip, Empty, ErrorState, Loading, Note, PageHeader, Panel, PartitionChip, Seg, Stat } from "../components/ui";
import { CLASS_COLORS, PARTITION_COLORS } from "../lib/classes";
import { duration, int, num, pct } from "../lib/format";

export default function Dataset() {
  const qc = useQueryClient();
  const status = useQuery({ queryKey: ["dataset-status"], queryFn: api.datasetStatus, refetchInterval: (q) => (q.state.data?.task.running ? 1000 : 15000) });
  const ready = !!status.data?.present && !!status.data?.processed;
  const inventory = useQuery({ queryKey: ["inventory"], queryFn: api.inventory, enabled: ready, staleTime: Infinity });
  const mapping = useQuery({ queryKey: ["mapping"], queryFn: api.mapping, enabled: ready });
  const splits = useQuery({ queryKey: ["splits"], queryFn: api.splits, enabled: ready });
  const prov = useQuery({ queryKey: ["provenance"], queryFn: api.provenance });
  const detection = useQuery({ queryKey: ["detection"], queryFn: api.detection, enabled: ready, staleTime: Infinity });
  const download = useMutation({ mutationFn: api.downloadDataset, onSuccess: () => qc.invalidateQueries({ queryKey: ["dataset-status"] }) });
  const [sort, setSort] = useState<"record" | "partition" | "quality" | "minority">("record");
  const [scale, setScale] = useState<"linear" | "log">("log");

  const rows = useMemo(() => {
    const r = [...(inventory.data ?? [])];
    const minority = (x: (typeof r)[0]) => (x.class_counts.S ?? 0) + (x.class_counts.V ?? 0) + (x.class_counts.F ?? 0);
    if (sort === "partition") r.sort((a, b) => a.partition.localeCompare(b.partition) || a.record.localeCompare(b.record));
    if (sort === "quality") r.sort((a, b) => a.quality.mean_score - b.quality.mean_score);
    if (sort === "minority") r.sort((a, b) => minority(b) - minority(a));
    return r;
  }, [inventory.data, sort]);

  if (status.isLoading) return <Loading />;
  if (status.error) return <ErrorState error={status.error} />;
  const st = status.data!;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dataset Explorer"
        sub="The MIT-BIH Arrhythmia Database as used by this platform: inventory, subject grouping, leads, annotation and class distributions, quality, exclusions and provenance."
        actions={
          <button className="btn-primary" onClick={() => download.mutate()} disabled={st.task.running} data-testid="download-dataset">
            {st.present ? <RefreshCw className="h-3.5 w-3.5" /> : <Download className="h-3.5 w-3.5" />}
            {st.task.running ? `${st.task.stage} ${pct(st.task.progress, 0)}` : st.present ? "Re-verify / repair download" : "Download from PhysioNet"}
          </button>
        }
      />
      {st.task.error && <ErrorState error={st.task.error} />}
      {!st.present && (
        <Panel>
          <Empty title="Dataset not downloaded">
            Downloads the official 1.0.0 release (~77 MB) from PhysioNet, verifies every .dat/.hea/.atr file against the published
            SHA-256 checksums, writes provenance and builds the segmented beat set.
          </Empty>
        </Panel>
      )}

      {ready && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Stat label="Recordings" value={st.n_records_expected} sub="≈30 min each, 2 channels" />
            <Stat label="Subjects" value={st.n_subjects} sub="201 + 202 = one subject" />
            <Stat label="Sampling rate" value="360 Hz" sub="11-bit, ±5 mV" />
            <Stat label="Beat segments" value={int(st.n_segments)} sub={`${mapping.data?.n_boundary_excluded ?? "…"} excluded at boundaries`} />
            <Stat label="Files verified" value={String((prov.data?.n_files_verified as number) ?? "—")} sub="SHA-256 vs PhysioNet" accent="#2BB3A3" />
          </div>

          <Panel
            title="Recording inventory"
            sub="Click a record to open it in the Signal Explorer"
            actions={
              <Seg
                value={sort}
                onChange={setSort}
                options={[
                  { value: "record", label: "Record" },
                  { value: "partition", label: "Partition" },
                  { value: "minority", label: "Ectopic beats" },
                  { value: "quality", label: "Quality" },
                ]}
              />
            }
            bodyClass="p-0"
            testId="inventory"
          >
            {inventory.isLoading ? (
              <div className="p-4">
                <Loading label="Computing inventory and quality profiles (first time ≈ 30 s)" />
              </div>
            ) : (
              <div className="max-h-[560px] overflow-auto">
                <table className="table">
                  <thead className="sticky top-0 z-10 bg-ink-800">
                    <tr>
                      <th>Record</th>
                      <th>Subject</th>
                      <th>Partition</th>
                      <th>Leads</th>
                      <th className="text-right">Beats</th>
                      {["N", "S", "V", "F", "Q"].map((c) => (
                        <th key={c} className="text-right" style={{ color: CLASS_COLORS[c] }}>
                          {c}
                        </th>
                      ))}
                      <th className="text-right">Non-beat</th>
                      <th className="text-right">Quality</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.record}>
                        <td>
                          <Link to={`/explorer?record=${r.record}`} className="num font-medium text-fg hover:text-teal">
                            {r.record}
                          </Link>
                        </td>
                        <td className="num">{r.subject}</td>
                        <td>
                          <PartitionChip partition={r.partition} />
                          {r.partition === "train" && <span className="num ml-1.5 text-fg-dim">f{r.fold}</span>}
                        </td>
                        <td className="num">
                          <span className={r.has_mlii ? "text-fg" : "text-cls-S"}>{r.leads.join(" / ")}</span>
                        </td>
                        <td className="num text-right text-fg">{int(r.n_beats)}</td>
                        {["N", "S", "V", "F", "Q"].map((c) => (
                          <td key={c} className="num text-right" style={{ color: r.class_counts[c] ? CLASS_COLORS[c] : "#3A4555" }}>
                            {r.class_counts[c] ?? 0}
                          </td>
                        ))}
                        <td className="num text-right">{r.n_non_beat}</td>
                        <td className="num text-right">
                          <span style={{ color: r.quality.mean_score > 0.7 ? "#2BB3A3" : r.quality.mean_score > 0.45 ? "#E8A33D" : "#F2555A" }}>
                            {num(r.quality.mean_score, 2)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel
              title="Class imbalance by partition"
              sub="Segmented beats per AAMI class"
              actions={<Seg value={scale} onChange={setScale} options={[{ value: "log", label: "Log" }, { value: "linear", label: "Linear" }]} />}
            >
              {splits.data && (
                <Plot
                  testId="imbalance-plot"
                  height={280}
                  data={(["train", "test", "excluded"] as const).map((p) => ({
                    type: "bar",
                    name: p === "train" ? "Training pool" : p === "test" ? "Held-out test" : "Excluded",
                    x: ["N", "S", "V", "F", "Q"],
                    y: ["N", "S", "V", "F", "Q"].map((c) => splits.data!.class_counts[p]?.[c] ?? 0),
                    marker: { color: PARTITION_COLORS[p] },
                  }))}
                  layout={{ barmode: "group", yaxis: { type: scale, title: { text: "beats" } } }}
                />
              )}
            </Panel>
            <Panel title="Annotation symbols" sub="All annotations in the 48 .atr files and how they are used">
              {mapping.data && (
                <div className="max-h-[300px] overflow-y-auto">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Meaning</th>
                        <th>Used as</th>
                        <th className="text-right">Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(mapping.data.symbol_counts ?? {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([sym, n]) => {
                          const cls = Object.entries(mapping.data!.classes).find(([, v]) => sym in v.symbols)?.[0];
                          const meaning = cls ? mapping.data!.classes[cls].symbols[sym] : mapping.data!.non_beat_symbols[sym] ?? "—";
                          return (
                            <tr key={sym}>
                              <td className="num text-fg">{sym}</td>
                              <td>{meaning}</td>
                              <td>{cls ? <ClassChip cls={cls} /> : <span className="text-fg-dim">excluded (non-beat)</span>}</td>
                              <td className="num text-right">{int(n)}</td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
          </div>

          {splits.data && (
            <Panel title="Subject grouping & partitions" sub={splits.data.description} testId="split-panel">
              <div className="grid gap-5 lg:grid-cols-3">
                {(["train_pool", "test", "excluded"] as const).map((k) => (
                  <div key={k}>
                    <div className="label mb-2">
                      {k === "train_pool" ? `Training pool (${splits.data.n_subjects.train} subjects)` : k === "test" ? `Held-out test (${splits.data.n_subjects.test} subjects)` : "Excluded (paced)"}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {splits.data[k].map((r) => (
                        <span
                          key={r}
                          className="num rounded-md border px-1.5 py-0.5 text-2xs"
                          style={{
                            borderColor: r === "201" || r === "202" ? "#E8A33D88" : "#273241",
                            color: r === "201" || r === "202" ? "#E8A33D" : "#8B95A5",
                          }}
                          title={r === "201" || r === "202" ? "Same subject as record " + (r === "201" ? "202" : "201") : undefined}
                        >
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <Note tone="warn">{splits.data.difference_from_original}</Note>
                <Note>{splits.data.subject_note}</Note>
              </div>
              <div className="mt-4">
                <div className="label mb-2">Automated leakage checks</div>
                <div className="grid gap-1.5 sm:grid-cols-2">
                  {splits.data.leakage.checks.map((c) => (
                    <div key={c.name} className="flex items-center gap-2 text-2xs text-fg-muted">
                      {c.passed ? <CheckCircle2 className="h-3.5 w-3.5 text-teal" /> : <XCircle className="h-3.5 w-3.5 text-trace" />}
                      {c.name}
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Exclusions" sub="Nothing is removed silently">
              <ul className="space-y-2 text-xs text-fg-muted">
                <li>
                  <strong className="text-fg">Paced records {mapping.data?.paced_records.join(", ")}</strong> — {mapping.data?.paced_records_note}
                </li>
                <li>
                  <strong className="text-fg">Q class</strong> — {mapping.data?.q_class_note}
                </li>
                <li>
                  <strong className="text-fg">Non-beat annotations</strong> — rhythm changes, noise markers, flutter waves (!), non-conducted P waves
                  (x) and comments never become heartbeat segments. They are listed in the symbol table above.
                </li>
                <li>
                  <strong className="text-fg">Boundary beats</strong> — {mapping.data?.n_boundary_excluded} beats whose 0.89 s stored window would extend past the
                  start or end of a recording.
                </li>
                <li>
                  <strong className="text-fg">Lead</strong> — MLII is used everywhere it exists (identified by header name; record 114 has it on the second
                  channel). Only paced records 102 and 104 lack MLII.
                </li>
              </ul>
            </Panel>
            <Panel title="Provenance & licence" testId="provenance">
              {prov.data && (
                <dl className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-2 text-xs">
                  <dt className="text-fg-dim">Source</dt>
                  <dd>
                    <a href={String(prov.data.source_url)} className="text-teal hover:underline" target="_blank" rel="noreferrer">
                      {String(prov.data.source_url)}
                    </a>
                  </dd>
                  <dt className="text-fg-dim">DOI</dt>
                  <dd className="num">{String(prov.data.doi)}</dd>
                  <dt className="text-fg-dim">Version</dt>
                  <dd className="num">{String(prov.data.version)}</dd>
                  <dt className="text-fg-dim">Licence</dt>
                  <dd>{String(prov.data.license)}</dd>
                  <dt className="text-fg-dim">Downloaded</dt>
                  <dd className="num">{String(prov.data.downloaded_at ?? "—")}</dd>
                  <dt className="text-fg-dim">Checksums</dt>
                  <dd className="num">{String(prov.data.n_checksums ?? "—")} files, SHA-256</dd>
                  <dt className="text-fg-dim">Cite</dt>
                  <dd className="space-y-1 text-fg-muted">
                    {(prov.data.citations as string[]).map((c) => (
                      <p key={c}>{c}</p>
                    ))}
                  </dd>
                </dl>
              )}
            </Panel>
          </div>

          <Panel title="Automatic R-peak detector (separate component)" sub="Project-built detector scored against reference beat positions, 150 ms tolerance">
            {detection.isLoading ? (
              <Loading label="Running the detector on all 48 recordings (first time ≈ 30 s)" />
            ) : detection.data ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Sensitivity (non-paced)" value={pct(detection.data.overall_non_paced.sensitivity, 2)} />
                  <Stat label="+P (non-paced)" value={pct(detection.data.overall_non_paced.positive_predictivity, 2)} />
                  <Stat label="Missed beats" value={int(detection.data.overall_non_paced.fn)} />
                  <Stat label="False detections" value={int(detection.data.overall_non_paced.fp)} />
                </div>
                <Plot
                  height={220}
                  data={[
                    {
                      type: "bar",
                      name: "Sensitivity",
                      x: detection.data.records.map((r) => r.record),
                      y: detection.data.records.map((r) => r.sensitivity),
                      marker: { color: "#2BB3A3" },
                    },
                    {
                      type: "bar",
                      name: "+P",
                      x: detection.data.records.map((r) => r.record),
                      y: detection.data.records.map((r) => r.positive_predictivity),
                      marker: { color: "#E8A33D" },
                    },
                  ]}
                  layout={{ barmode: "group", yaxis: { range: [0.5, 1.01] }, xaxis: { type: "category", tickangle: -60 } }}
                />
              </div>
            ) : null}
          </Panel>
          <p className="text-2xs text-fg-dim">Pipeline hash {st.pipeline_hash} · dataset {st.name} v{st.version} · {duration(1805.55)} per record</p>
        </>
      )}
    </div>
  );
}
