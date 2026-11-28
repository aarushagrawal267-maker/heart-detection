import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Maximize2, Minus, Plus, ScanLine } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import BeatChart from "../charts/BeatChart";
import EcgChart from "../charts/EcgChart";
import Minimap from "../charts/Minimap";
import RecordPicker from "../components/RecordPicker";
import { ClassChip, ErrorState, Loading, Note, PageHeader, Panel, PartitionChip, Seg } from "../components/ui";
import { classColor, CLASS_NAMES } from "../lib/classes";
import { clock, nearestByTime, num, pct } from "../lib/format";
import { useUrlState } from "../lib/useUrlState";

const SPANS = [2.5, 5, 10, 30, 60];

export default function Explorer() {
  const [record, setRecord] = useUrlState("record", "100");
  const [channel, setChannel] = useState<string | undefined>(undefined);
  const [processed, setProcessed] = useState(false);
  const [range, setRange] = useState<[number, number]>([0, 10]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const info = useQuery({ queryKey: ["record", record], queryFn: () => api.record(record) });
  const duration = info.data?.duration_s ?? 1805.55;
  useEffect(() => {
    setRange([0, 10]);
    setSelectedIdx(null);
    setChannel(undefined);
    setProcessed(false);
  }, [record]);

  const lead = channel ?? info.data?.analysis_lead;
  const span = range[1] - range[0];
  const sig = useQuery({
    queryKey: ["signal", record, lead, processed, range[0].toFixed(3), range[1].toFixed(3)],
    queryFn: () => api.signal(record, range[0], range[1], lead, processed, 5000),
    enabled: !!lead,
    placeholderData: keepPreviousData,
  });
  const env = useQuery({ queryKey: ["envelope", record, lead], queryFn: () => api.overviewEnvelope(record, lead), enabled: !!lead });
  const ann = useQuery({ queryKey: ["annotations", record], queryFn: () => api.annotations(record) });
  const beats = useQuery({ queryKey: ["beats", record], queryFn: () => api.beats(record) });
  const quality = useQuery({ queryKey: ["quality", record, lead], queryFn: () => api.quality(record, lead), enabled: !!lead });

  const allBeats = beats.data?.beats ?? [];
  const refMarkers = useMemo(
    () =>
      (ann.data?.annotations ?? [])
        .filter((a) => a.time_s >= range[0] - 1 && a.time_s <= range[1] + 1)
        .map((a) => ({ time_s: a.time_s, label: a.is_beat ? a.aami ?? a.symbol : a.symbol, cls: a.is_beat ? a.aami : null })),
    [ann.data, range],
  );
  const shading = useMemo(() => {
    const out: { start_s: number; end_s: number; color: string }[] = [];
    for (const w of quality.data?.windows ?? []) {
      if (w.label === "poor") out.push({ start_s: w.start_s, end_s: w.start_s + quality.data!.window_s, color: "rgba(232,163,61,0.06)" });
    }
    return out;
  }, [quality.data]);

  const sel = selectedIdx !== null ? allBeats[selectedIdx] : null;
  const beat = useQuery({
    queryKey: ["beat", record, sel?.beat_index],
    queryFn: () => api.beat(record, sel!.beat_index),
    enabled: !!sel,
  });

  const clamp = (a: number, b: number): [number, number] => {
    const w = Math.min(Math.max(b - a, 0.5), duration);
    let s = Math.max(0, a);
    if (s + w > duration) s = duration - w;
    return [s, s + w];
  };
  const setSpan = (w: number) => {
    const c = (range[0] + range[1]) / 2;
    setRange(clamp(c - w / 2, c + w / 2));
  };
  const pan = (frac: number) => setRange(clamp(range[0] + span * frac, range[1] + span * frac));

  const selectAt = (t: number) => {
    const i = nearestByTime(allBeats, t);
    if (i >= 0 && Math.abs(allBeats[i].time_s - t) < 0.6) setSelectedIdx(i);
  };
  const stepBeat = (d: number) => {
    if (!allBeats.length) return;
    const i = selectedIdx === null ? nearestByTime(allBeats, (range[0] + range[1]) / 2) : Math.min(Math.max(selectedIdx + d, 0), allBeats.length - 1);
    setSelectedIdx(i);
    const t = allBeats[i].time_s;
    if (t < range[0] || t > range[1]) setRange(clamp(t - span / 2, t + span / 2));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "SELECT" || (e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "ArrowRight") pan(0.5);
      else if (e.key === "ArrowLeft") pan(-0.5);
      else if (e.key === "]") stepBeat(1);
      else if (e.key === "[") stepBeat(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const winQuality = quality.data?.windows.filter((w) => w.start_s + quality.data!.window_s > range[0] && w.start_s < range[1]) ?? [];
  const visibleRhythms = (ann.data?.rhythms ?? []).filter((r) => r.end_s > range[0] && r.start_s < range[1]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="ECG Signal Explorer"
        sub="Browse the original recordings at full resolution. Drag across the trace to zoom, click a beat to inspect it, use ← → to pan and [ ] to step between beats."
        actions={
          <>
            <RecordPicker value={record} onChange={setRecord} />
            {info.data && (
              <select className="input w-auto" value={lead} onChange={(e) => setChannel(e.target.value)} data-testid="channel-picker">
                {info.data.leads.map((l) => (
                  <option key={l} value={l}>
                    Lead {l}
                  </option>
                ))}
              </select>
            )}
            <Seg
              value={processed ? "proc" : "raw"}
              onChange={(v) => setProcessed(v === "proc")}
              options={[
                { value: "raw", label: "Original" },
                { value: "proc", label: "Preprocessed" },
              ]}
              testId="signal-mode"
            />
          </>
        }
      />
      {processed && lead !== info.data?.analysis_lead && (
        <Note tone="warn">Preprocessed signals are stored for the analysis lead ({info.data?.analysis_lead}) only.</Note>
      )}

      <Panel
        title={
          <span className="flex items-center gap-2">
            Record {record} · Lead {sig.data?.lead ?? lead}
            {info.data && <PartitionChip partition={info.data.partition} />}
          </span>
        }
        sub={
          <span className="num">
            {clock(range[0])} – {clock(range[1])} · span {span.toFixed(1)} s · {sig.data?.decimated ? "min/max decimated for display" : "full 360 Hz resolution"}
            {processed ? " · baseline-corrected, 45 Hz low-pass" : " · original signal (mV)"}
          </span>
        }
        actions={
          <div className="flex flex-wrap items-center gap-1.5">
            <button className="btn px-2" onClick={() => pan(-0.8)} aria-label="Pan left">
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <button className="btn px-2" onClick={() => pan(0.8)} aria-label="Pan right">
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
            <button className="btn px-2" onClick={() => setSpan(span / 2)} aria-label="Zoom in">
              <Plus className="h-3.5 w-3.5" />
            </button>
            <button className="btn px-2" onClick={() => setSpan(span * 2)} aria-label="Zoom out">
              <Minus className="h-3.5 w-3.5" />
            </button>
            <Seg value={SPANS.includes(span) ? span : -1} onChange={(v) => v > 0 && setSpan(v)} options={SPANS.map((s) => ({ value: s, label: `${s}s` }))} />
            <button className="btn px-2" onClick={() => setRange([0, duration])} aria-label="Whole record">
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        }
        bodyClass="p-3 sm:p-4"
      >
        {sig.error ? (
          <ErrorState error={sig.error} />
        ) : sig.data ? (
          <EcgChart
            t={sig.data.t}
            y={sig.data.y}
            range={range}
            height={340}
            reference={refMarkers}
            selected={sel?.time_s ?? null}
            shading={shading}
            onSelectTime={selectAt}
            onZoom={(a, b) => setRange(clamp(a, b))}
            testId="explorer-ecg"
          />
        ) : (
          <Loading label="Loading waveform" />
        )}
        <div className="mt-3">
          {env.data && (
            <Minimap
              t={env.data.t}
              min={env.data.min}
              max={env.data.max}
              duration={duration}
              range={range}
              onJump={(c) => setRange(clamp(c - span / 2, c + span / 2))}
              marks={allBeats.filter((b) => b.aami !== "N").map((b) => ({ time_s: b.time_s, color: classColor(b.aami) }))}
            />
          )}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-2xs text-fg-dim">
          {["N", "S", "V", "F", "Q"].map((c) => (
            <span key={c} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm" style={{ background: classColor(c) }} /> {c} · {CLASS_NAMES[c]}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-3 rounded-sm bg-cls-S/30" /> low signal-quality window
          </span>
          {visibleRhythms.length > 0 && (
            <span className="ml-auto">Rhythm: {visibleRhythms.map((r) => r.rhythm).join(" → ")}</span>
          )}
        </div>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-5">
        <Panel
          className="lg:col-span-3"
          title={sel ? `Beat #${sel.beat_index} at ${clock(sel.time_s, 2)}` : "Selected heartbeat"}
          sub={sel ? `Reference symbol "${sel.symbol}" · ${beat.data?.description ?? ""}` : "Click a beat on the trace above"}
          actions={
            sel && (
              <div className="flex items-center gap-1.5">
                <button className="btn px-2" onClick={() => stepBeat(-1)} aria-label="Previous beat">
                  <ChevronLeft className="h-3.5 w-3.5" />
                </button>
                <button className="btn px-2" onClick={() => stepBeat(1)} aria-label="Next beat">
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
                <Link className="btn-primary" to={`/classify?record=${record}&beat=${sel.beat_index}`}>
                  <ScanLine className="h-3.5 w-3.5" /> Classify with CNN
                </Link>
              </div>
            )
          }
          testId="beat-panel"
        >
          {!sel ? (
            <div className="flex h-[240px] items-center justify-center text-xs text-fg-dim">No beat selected</div>
          ) : beat.data ? (
            <div>
              <BeatChart
                timeMs={beat.data.time_ms}
                series={[
                  { values: beat.data.raw_mv, color: "#5F6A7A", width: 1, dash: [3, 3] },
                  { values: beat.data.filtered_mv, color: "#F2555A", width: 1.7 },
                ]}
                height={240}
              />
              <div className="mt-2 flex flex-wrap items-center gap-4 text-2xs text-fg-dim">
                <span className="flex items-center gap-1.5">
                  <span className="h-px w-4 border-t border-dashed border-fg-dim" /> original
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-trace" /> preprocessed (model input before z-scoring)
                </span>
                <span>Window: 90 samples before, 166 after (0.71 s)</span>
              </div>
            </div>
          ) : (
            <Loading />
          )}
        </Panel>

        <Panel className="lg:col-span-2" title="Beat & signal details" testId="beat-details">
          <div className="space-y-4">
            {sel ? (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="label">Reference class</div>
                  <div className="mt-1">
                    <ClassChip cls={sel.aami} label={`${sel.aami} · ${sel.symbol}`} size="lg" />
                  </div>
                </div>
                <div>
                  <div className="label">Sample index</div>
                  <div className="num mt-1.5 text-fg">{sel.sample}</div>
                </div>
                <div>
                  <div className="label">RR previous</div>
                  <div className="num mt-1 text-fg">
                    {num(sel.rr_prev, 3)} s <span className="text-fg-dim">({num(sel.rr_prev_ratio, 2)}× local)</span>
                  </div>
                </div>
                <div>
                  <div className="label">RR next</div>
                  <div className="num mt-1 text-fg">
                    {num(sel.rr_next, 3)} s <span className="text-fg-dim">({num(sel.rr_next_ratio, 2)}× local)</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-fg-dim">Select a beat to see its reference annotation and RR intervals.</p>
            )}
            <div className="border-t border-line-soft pt-4">
              <div className="label mb-2">Signal quality in view ({quality.data?.window_s ?? 10} s windows)</div>
              {winQuality.length ? (
                <div className="space-y-1.5">
                  {winQuality.slice(0, 6).map((w) => (
                    <div key={w.start_s} className="grid grid-cols-[64px_1fr_auto] items-center gap-2 text-2xs">
                      <span className="num text-fg-dim">{clock(w.start_s, 0)}</span>
                      <div className="h-1.5 overflow-hidden rounded-full bg-ink-600">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${w.score * 100}%`, background: w.label === "good" ? "#4CC38A" : w.label === "fair" ? "#E8A33D" : "#F2555A" }}
                        />
                      </div>
                      <span className="num text-fg-muted" title={`baseline ${pct(w.baseline_ratio)} · 60 Hz ${pct(w.powerline_ratio, 2)} · HF ${pct(w.hf_ratio)}`}>
                        {w.label} {num(w.score, 2)}
                      </span>
                    </div>
                  ))}
                  {winQuality.length > 6 && <div className="text-2xs text-fg-dim">+ {winQuality.length - 6} more windows</div>}
                </div>
              ) : (
                <Loading />
              )}
              {quality.data && (
                <p className="mt-3 text-2xs leading-relaxed text-fg-dim">
                  Whole record: mean score {num(quality.data.summary.mean_score, 2)}, {pct(quality.data.summary.poor_fraction)} poor
                  windows. Indicators: baseline-wander power (&lt;0.5 Hz), 60 Hz power, high-frequency (&gt;45 Hz) power,
                  flat-line and ADC saturation. {quality.data.noise_annotations.length} expert noise annotations. Quality is
                  descriptive only; no beats are removed because of it.
                </p>
              )}
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
