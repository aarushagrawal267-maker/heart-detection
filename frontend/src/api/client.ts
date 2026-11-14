/* Typed client for the QRS Atlas FastAPI backend (proxied at /api). */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

const get = <T,>(p: string) => request<T>(p);
const post = <T,>(p: string, body?: unknown) =>
  request<T>(p, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const del = <T,>(p: string) => request<T>(p, { method: "DELETE" });

// ---------------------------------------------------------------- types
export interface PerClass {
  support: number;
  predicted: number;
  precision: number | null;
  recall: number | null;
  sensitivity: number | null;
  specificity: number | null;
  fpr: number | null;
  f1: number | null;
}

export interface Metrics {
  classes: string[];
  n: number;
  accuracy: number | null;
  macro_f1: number | null;
  weighted_f1: number | null;
  macro_recall: number | null;
  macro_roc_auc?: number | null;
  confusion_matrix: number[][];
  per_class: Record<string, PerClass>;
  roc?: Record<string, { auc: number; points?: number[][] }>;
  pr?: Record<string, { ap: number; prevalence: number; auc?: number; points?: number[][] }>;
}

export interface EvaluationEntry {
  metrics: Metrics;
  per_subject: Record<string, number | string | null>[];
  bootstrap: Record<string, number[] | number | string> | null;
  inference: Record<string, number | string>;
  records: string[];
  created_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  kind: "cnn" | "baseline";
  model_name: string;
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  current_epoch: number;
  total_epochs: number;
  progress: number;
  message: string;
  summary: Record<string, unknown> & {
    val_macro_f1?: number;
    val_accuracy?: number;
    val_per_class_f1?: Record<string, number | null>;
    train_macro_f1?: number;
    best_epoch?: number;
    epochs_run?: number;
    cv_mean_macro_f1?: number | null;
  };
  config: Record<string, unknown>;
  n_params: number | null;
  checkpoint_path: string | null;
  dataset_version: string;
  pipeline_hash: string;
  has_test_evaluation: boolean;
  active?: boolean;
  pipeline?: Record<string, unknown>;
  split?: {
    split_name?: string;
    train_records?: string[];
    validation_records?: string[];
    test_records?: string[];
    excluded_records?: string[];
    train_subjects?: string[];
    validation_subjects?: string[];
    n_train_beats?: number;
    n_validation_beats?: number;
  };
  architecture?: { layer: string; output_shape?: number[] | null; params?: number }[];
  environment?: Record<string, string | number | null>;
  error?: string | null;
  notes?: string;
  cv_results?: {
    folds: { fold: number; best_epoch: number; val_macro_f1: number; records: string[] }[];
    mean_macro_f1: number;
    std_macro_f1: number;
  } | null;
}

export interface EpochRow {
  fold: number;
  epoch: number;
  train_loss: number;
  val_loss: number | null;
  train_macro_f1: number | null;
  val_macro_f1: number | null;
  val_accuracy: number | null;
  val_per_class_f1?: Record<string, number | null>;
  lr?: number | null;
  duration_s?: number;
}

export interface RecordSummary {
  record: string;
  leads: string[];
  fs: number;
  duration_s: number;
  subject: string;
  partition: "train" | "test" | "excluded";
  paced: boolean;
  analysis_lead: string;
}

export interface InventoryRow extends RecordSummary {
  fold: number;
  has_mlii: boolean;
  n_samples: number;
  comments: string[];
  n_annotations: number;
  n_beats: number;
  n_non_beat: number;
  n_boundary_excluded: number;
  class_counts: Record<string, number>;
  symbol_counts: Record<string, number>;
  quality: {
    mean_score: number;
    min_score: number;
    good_fraction: number;
    fair_fraction: number;
    poor_fraction: number;
    mean_baseline_ratio: number;
    mean_powerline_ratio: number;
    mean_hf_ratio: number;
  };
}

export interface SignalWindow {
  record: string;
  lead: string;
  fs: number;
  start_sample: number;
  end_sample: number;
  processed: boolean;
  decimated: boolean;
  t: number[];
  y: number[];
}

export interface Annotation {
  sample: number;
  time_s: number;
  symbol: string;
  aami: string | null;
  is_beat: boolean;
  aux: string | null;
}

export interface BeatMeta {
  beat_index: number;
  sample: number;
  time_s: number;
  symbol: string;
  aami: string;
  rr_prev: number;
  rr_next: number;
  rr_prev_ratio: number;
  rr_next_ratio: number;
}

export interface Explanation {
  method: string;
  description: string;
  attribution: number[];
  rr_attribution: number[] | null;
  rr_features?: string[];
  target_class: string;
  completeness_gap?: number;
  score_delta?: number;
  caveat?: string;
}

export interface BeatPrediction {
  record: string;
  beat_index: number;
  sample: number;
  time_s: number;
  lead: string;
  reference: { symbol: string; description: string; aami: string; in_model_classes: boolean };
  prediction: { class: string; confidence: number; scores: Record<string, number> };
  agrees_with_reference: boolean | null;
  rr: Record<string, number>;
  waveform_mv: number[];
  model_input: number[];
  time_ms: number[];
  window: { pre: number; post: number; fs: number };
  partition: string;
  partition_note: string;
  experiment_id: string;
  latency_ms: number;
  notice: string;
  explanation: Explanation | null;
}

export interface RecordBeat {
  beat_index: number | null;
  sample: number;
  time_s: number;
  reference: string | null;
  symbol: string | null;
  predicted: string;
  confidence: number;
  scores: number[];
}

export interface RecordAnalysis {
  record: string;
  mode: "reference" | "detected";
  lead: string;
  fs: number;
  duration_s: number;
  classes: string[];
  experiment_id: string;
  partition: string;
  partition_note: string;
  n_beats: number;
  beats: RecordBeat[];
  predicted_distribution: Record<string, number>;
  reference_distribution: Record<string, number>;
  agreement: Metrics | null;
  detection: null | {
    n_reference: number;
    n_detected: number;
    tp: number;
    fp: number;
    fn: number;
    sensitivity: number;
    positive_predictivity: number;
    mean_abs_error_ms: number | null;
    missed_reference_samples: number[];
  };
  inference_s: number;
  notice: string;
}

export interface QualityProfile {
  window_s: number;
  lead: string;
  windows: {
    start_s: number;
    score: number;
    label: "good" | "fair" | "poor";
    baseline_ratio: number;
    powerline_ratio: number;
    hf_ratio: number;
    flat_fraction: number;
    saturation_fraction: number;
    amplitude_range_mv: number;
  }[];
  summary: InventoryRow["quality"];
  noise_annotations: { time_s: number; aux: string }[];
}

export interface SplitInfo {
  name: string;
  version: number;
  description: string;
  subject_note: string;
  same_subject_groups: string[][];
  train_pool: string[];
  test: string[];
  excluded: string[];
  original_protocol: { DS1: string[]; DS2: string[] };
  n_folds: number;
  difference_from_original: string;
  folds: Record<string, string[]>;
  class_counts: Record<string, Record<string, number>>;
  fold_class_counts: Record<string, Record<string, number>>;
  n_subjects: Record<string, number>;
  leakage: { passed: boolean; checks: { name: string; passed: boolean; detail: string }[] };
}

export interface Overview {
  dataset: {
    present: boolean;
    processed: boolean;
    name: string;
    version: string;
    n_records: number;
    n_subjects: number;
    n_segments?: number;
    class_counts?: Record<string, number>;
    partition_counts?: Record<string, number>;
  };
  notice: string;
  experiments: { total: number; completed: number; running: number };
  latest_experiment: null | {
    id: string;
    name: string;
    status: string;
    created_at: string;
    summary: Experiment["summary"];
    progress: number;
  };
  active_model: null | {
    id: string;
    name: string;
    summary: Experiment["summary"];
    n_params: number | null;
    finished_at: string | null;
    classes: string[];
    validation: Metrics | null;
    test: null | Pick<Metrics, "accuracy" | "macro_f1" | "weighted_f1" | "per_class" | "n">;
  };
  recent_activity: {
    timestamp: string;
    kind: string;
    record: string;
    beat_index: number | null;
    predicted: string | null;
    reference: string | null;
    confidence: number | null;
    summary: Record<string, unknown>;
  }[];
}

export interface DatasetStatus {
  present: boolean;
  processed: boolean;
  task: { running: boolean; kind: string | null; stage: string; progress: number; error: string | null };
  name: string;
  version: string;
  n_records_expected: number;
  n_subjects: number;
  pipeline_hash: string;
  provenance: Record<string, unknown> | null;
  n_segments?: number;
  class_counts?: Record<string, number>;
}

export interface Mapping {
  classes: Record<string, { name: string; symbols: Record<string, string> }>;
  non_beat_symbols: Record<string, string>;
  paced_records: string[];
  paced_records_note: string;
  primary_task_classes: string[];
  q_class_note: string;
  symbol_counts?: Record<string, number>;
  n_boundary_excluded?: number;
}

export interface TrainingDefaults {
  cnn: Record<string, unknown>;
  baseline: Record<string, unknown>;
  baselines: Record<string, string>;
  options: {
    loss: string[];
    sampler: string[];
    classes: string[][];
    val_fold: number[];
    window_limits: { max_pre: number; max_post: number };
  };
  architecture: { layer: string; output_shape: number[] | null; params: number }[];
  n_params: number;
  device: string;
}

export interface DetectorEval {
  records: { record: string; paced: boolean; sensitivity: number; positive_predictivity: number; tp: number; fp: number; fn: number; mean_abs_error_ms: number | null }[];
  tolerance_ms: number;
  overall_non_paced: { tp: number; fp: number; fn: number; sensitivity: number; positive_predictivity: number };
  detector: string;
}

// ---------------------------------------------------------------- endpoints
export const api = {
  health: () => get<{ status: string; device: string }>("/api/health"),
  overview: () => get<Overview>("/api/overview"),
  datasetStatus: () => get<DatasetStatus>("/api/dataset/status"),
  downloadDataset: () => post("/api/dataset/download"),
  buildDataset: () => post("/api/dataset/build"),
  provenance: () => get<Record<string, unknown>>("/api/dataset/provenance"),
  mapping: () => get<Mapping>("/api/dataset/mapping"),
  inventory: () => get<InventoryRow[]>("/api/dataset/records"),
  splits: () => get<SplitInfo>("/api/splits"),
  detection: () => get<DetectorEval>("/api/detection/evaluation"),
  records: () => get<RecordSummary[]>("/api/records"),
  record: (r: string) => get<RecordSummary & { leads: string[]; comments: string[] }>(`/api/records/${r}`),
  signal: (r: string, start: number, end: number, channel?: string, processed = false, maxPoints = 6000) =>
    get<SignalWindow>(
      `/api/records/${r}/signal?start=${start}&end=${end}&processed=${processed}&max_points=${maxPoints}` +
        (channel ? `&channel=${encodeURIComponent(channel)}` : ""),
    ),
  overviewEnvelope: (r: string, channel?: string, points = 1500) =>
    get<{ t: number[]; min: number[]; max: number[]; duration_s: number; lead: string }>(
      `/api/records/${r}/overview?points=${points}` + (channel ? `&channel=${encodeURIComponent(channel)}` : ""),
    ),
  annotations: (r: string, start = 0, end?: number) =>
    get<{ annotations: Annotation[]; rhythms: { rhythm: string; start_s: number; end_s: number }[] }>(
      `/api/records/${r}/annotations?start=${start}` + (end !== undefined ? `&end=${end}` : ""),
    ),
  beats: (r: string) => get<{ n: number; beats: BeatMeta[] }>(`/api/records/${r}/beats`),
  beat: (r: string, i: number) =>
    get<{ time_ms: number[]; raw_mv: number[]; filtered_mv: number[]; symbol: string; aami: string; description: string }>(
      `/api/records/${r}/beats/${i}`,
    ),
  quality: (r: string, channel?: string) =>
    get<QualityProfile>(`/api/records/${r}/quality` + (channel ? `?channel=${encodeURIComponent(channel)}` : "")),
  trainingDefaults: () => get<TrainingDefaults>("/api/training/defaults"),
  currentJob: () => get<Experiment | null>("/api/training/current"),
  experiments: () => get<Experiment[]>("/api/experiments"),
  experiment: (id: string) => get<Experiment>(`/api/experiments/${id}`),
  epochs: (id: string) => get<EpochRow[]>(`/api/experiments/${id}/epochs`),
  evaluations: (id: string) => get<Record<string, EvaluationEntry>>(`/api/experiments/${id}/evaluations`),
  compare: (ids: string[]) => get<(Experiment & { evaluations: Record<string, Partial<Metrics>> })[]>(
    `/api/experiments/compare?ids=${ids.join(",")}`,
  ),
  createExperiment: (body: { name: string; kind: "cnn" | "baseline"; config: Record<string, unknown>; notes?: string }) =>
    post<Experiment>("/api/experiments", { ...body, start: true }),
  cancelExperiment: (id: string) => post(`/api/experiments/${id}/cancel`),
  activate: (id: string) => post(`/api/experiments/${id}/activate`),
  deleteExperiment: (id: string) => del(`/api/experiments/${id}`),
  evaluateTest: (id: string, purpose: string) =>
    post<{ metrics: Metrics }>(`/api/experiments/${id}/evaluate-test`, { purpose, confirm: true }),
  testAccessLog: () => get<{ experiment_id: string; timestamp: string; purpose: string }[]>("/api/test-access-log"),
  activeModel: () => get<Experiment | null>("/api/models/active"),
  classifyBeat: (record: string, beat_index: number, explain: string | null = "integrated_gradients") =>
    post<BeatPrediction>("/api/inference/beat", { record, beat_index, explain }),
  analyzeRecord: (record: string, mode: "reference" | "detected") =>
    post<RecordAnalysis>("/api/inference/record", { record, mode }),
  explain: (record: string, beat_index: number, method: string, target?: string) =>
    post<Explanation>("/api/explain", { record, beat_index, method, target }),
  sanity: (record: string, beat_index: number) =>
    get<{ correlation_trained_vs_random: Record<string, number | null>; interpretation: string }>(
      `/api/explain/sanity?record=${record}&beat_index=${beat_index}`,
    ),
  exportUrl: (id: string) => `/api/experiments/${id}/export`,
};
