# Scientific methodology

This document describes exactly what the code does. Configuration values live in
`backend/app/config.py`, `configs/aami_mapping.json`, `configs/splits.json` and
`backend/app/training/trainer.py` (`DEFAULT_CNN_CONFIG`).

## 1. Task

Heartbeat-level classification into AAMI EC57 classes, evaluated **inter-patient**: the
model is tested on subjects whose recordings were never used for training, early stopping
or model selection.

## 2. Label mapping (AAMI EC57)

| Class | MIT-BIH symbols |
|---|---|
| N | `N` normal, `L` LBBB, `R` RBBB, `e` atrial escape, `j` nodal escape |
| S | `A` atrial premature, `a` aberrated atrial premature, `J` nodal premature, `S` supraventricular premature |
| V | `V` PVC, `E` ventricular escape |
| F | `F` fusion of ventricular and normal |
| Q | `/` paced, `f` fusion of paced and normal, `Q` unclassifiable |

Excluded and never segmented: non-beat annotations (`+ ~ | x ! [ ] " ( ) p t u ` ' ^ s T * D = @`).
Two judgement calls are documented: `!` (ventricular flutter wave) is excluded rather than
mapped to V, and `x` (non-conducted P wave, no QRS) is not a heartbeat.

**Paced records 102, 104, 107 and 217** are excluded from training and evaluation, following
the AAMI EC57 recommendation and de Chazal et al. (2004). They stay browsable in the app.
**Q class**: after that exclusion only 15 Q beats remain (8 in training, 7 in testing), too few
to learn or score, so the principal experiment is 4-class (N, S, V, F). A 5-class mode is
selectable in the Model Laboratory. EC57's optional exclusion of each record's first 5 minutes
is not applied; most inter-patient studies use the whole record, and so does this one.

## 3. Preprocessing (`backend/app/signal/filters.py`)

Applied to the MLII channel of every record:

1. NaN guard.
2. **Baseline removal**: baseline = median filter of 200 ms, then a median filter of 600 ms (de Chazal 2004); subtracted from the signal.
3. **Low-pass**: 4th-order Butterworth, 45 Hz cut-off, applied forward and backward (`sosfiltfilt`, zero phase). This removes 60 Hz mains and EMG noise without shifting R peaks.

Each model input segment is z-scored on its own (mean 0, SD 1). This step is stateless, so
nothing is estimated from validation or test data. The RR-feature standardiser is the only
fitted transform, and it is fitted on training-fold subjects only.

Signal quality (`backend/app/signal/quality.py`) is profiled in 10 s windows on the raw signal:
relative power below 0.5 Hz, at 58–62 Hz and above 45 Hz (Welch PSD), the flat-line fraction and
ADC saturation, combined into a heuristic 0–1 score. It is descriptive only: no data is dropped
because of it.

Tests check that filtering keeps R-peak positions within ±1 sample and removes added drift and
mains interference (`backend/tests/test_signal.py`).

## 4. Segmentation (`backend/app/segmentation/beats.py`)

- Every beat annotation (AAMI-mappable symbol) produces one segment centred on its annotation sample.
- The stored segment runs from 128 samples before to 192 after the annotation (0.89 s). The model window is a sub-slice: **90 before + 166 after = 256 samples (0.71 s)**. It covers the P wave (~120–200 ms before R) and the T wave (ending ~400 ms after R at normal rates). The symmetric 128/128 option wastes samples on the previous beat's T wave. Both are configurable.
- The extra stored margin allows ±5-sample shift augmentation using real neighbouring signal.
- Beats whose stored segment crosses a recording boundary are excluded and counted (50 beats).
- Each segment keeps: record, subject, beat index, sample, time, symbol, AAMI class, lead, partition and CV fold.
- **RR features**: previous RR, next RR (seconds), and each divided by the median RR over ±10 surrounding beats. They use only the beat positions of the same recording, never labels.

## 5. Patient-level partitioning (`backend/app/splits/split.py`)

The base is the inter-patient protocol of de Chazal et al. (2004):
DS1 = {101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230};
DS2 = {100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234}.

**Correction**: DS1 contains 201 and DS2 contains 202, but both come from the same subject.
Record 202 is moved to the training side:

- Training pool = DS1 ∪ {202}: 23 records, 22 subjects
- Held-out test = DS2 \ {202}: 21 records, 21 subjects
- Excluded = {102, 104, 107, 217}

The test set is therefore a strict subset of the published DS2. Numbers are not directly
comparable with papers that used the uncorrected split.

**Validation**: training-pool subjects are dealt into 5 folds by a deterministic greedy
procedure that balances the share of S, V and F beats per fold (only training-pool label
counts are used). Records 201 and 202 always share a fold. Fold 2 is the default validation
fold for early stopping. Because minority beats cluster in a few subjects (for example
record 208 holds 372 of the 415 training F beats and record 209 holds 383 of the 999 S
beats), the final model is selected by the **mean macro-F1 over all 5 folds**
(`backend/scripts/select_model.py`), not by a single fold.

**Leakage checks**, run automatically by `pytest` and shown live in the app:

- no subject or recording overlap between any pair of partitions;
- records 201 and 202 share one subject id, one partition and one fold;
- every segment's subject equals the subject of its recording;
- each training subject lies in exactly one fold, and test beats carry no fold;
- training, validation and test index sets are disjoint for every fold choice.

Held-out evaluation is a separate API action that requires explicit confirmation and is logged
in `test_access_log`. Training never evaluates on test subjects automatically.

## 6. Model (`backend/app/models/cnn.py`)

ECGNet, a 1D CNN implemented and trained from scratch (PyTorch default random
initialisation, seeded; no pretrained weights, no external model or service):

| Stage | Layers | Output |
|---|---|---|
| Input | z-scored beat | 1 × 256 |
| 1 | 2 × [Conv1d(k=7) → BN → ReLU], MaxPool(2) | 32 × 128 |
| 2 | 2 × [Conv1d(k=5) → BN → ReLU], MaxPool(2) | 64 × 64 |
| 3 | 2 × [Conv1d(k=3) → BN → ReLU], MaxPool(2) | 128 × 32 |
| 4 | Conv1d(k=3) → BN → ReLU (Grad-CAM layer), global average pool | 128 |
| Head | concat 4 standardised RR features → Linear(64) → ReLU → Dropout(0.4) → Linear(classes) | 4 |

The default configuration has 170,916 parameters (170,660 without the RR input). Channel widths, dropout and the RR input are configurable.

**Baselines** (`backend/app/models/baselines.py`): majority class; logistic regression on the 4
RR features; random forest (200 trees, depth 12, balanced class weights) on RR features plus 30
hand-crafted morphology features (downsampled QRS and T-wave samples, amplitude statistics,
approximate QRS width).

## 7. Training (`backend/app/training/trainer.py`)

- AdamW (lr 1e-3, weight decay 1e-4), one-cycle schedule, batch 256, up to 20 epochs.
- Early stopping on validation macro-F1 (patience 6); the best weights are restored.
- Loss: class-weighted cross-entropy with w_c ∝ (n / (K·n_c))^p, default p = 0.5, computed from training-fold counts. Focal loss (γ = 2) and a balanced `WeightedRandomSampler` are selectable experiments. Sampling happens inside the training partition only.
- Augmentation (training only): random shift in [-5, 0] samples (bounded by the stored margin after the beat), amplitude ×U(0.85, 1.15), offset N(0, 0.05), Gaussian noise σ = 0.03 in z-units.
- Each job runs in its own process. Per-epoch metrics stream to SQLite and the UI. Cancellation is honoured even before the worker starts.
- Each experiment records its hyperparameters, preprocessing config and hash, dataset version, exact partitions, environment (package versions, device, git commit), duration, metrics and checkpoint path. It is also exported to `models/<id>/experiment.json`.

## 8. Evaluation (`backend/app/evaluation/metrics.py`)

For each partition (train / validation / test):

- confusion matrix, accuracy, macro-F1 and weighted F1 (over present classes), macro recall;
- per class: support, sensitivity (recall), positive predictivity (precision), specificity, FPR, F1;
- one-vs-rest ROC curves with AUC, and precision–recall curves with average precision;
- a per-subject breakdown;
- a **cluster bootstrap** over subjects (500 replicates) giving 95% intervals for accuracy, macro-F1 and per-class F1 and recall;
- inference cost: batched time per beat and median single-beat latency on CPU.

## 9. Explainability (`backend/app/explain/attribution.py`)

Integrated Gradients (64 steps, flat-line waveform baseline and mean-RR baseline, with a
completeness check), gradient saliency, 1D Grad-CAM on the last convolutional layer, and
occlusion (40 ms window, 10 ms stride). A model-parameter randomisation sanity check
(Adebayo et al., 2018) reports the correlation between the maps of the trained model and a
randomly initialised one. These are analytical visualisations of model sensitivity, not
clinical reasoning.

## 10. Automatic R-peak detection (separate component)

`backend/app/signal/detect.py` implements a Pan-Tompkins-style detector (5–18 Hz band-pass,
derivative, squaring, 120 ms integration, adaptive threshold, 200 ms refractory period, peak
refinement). It is scored against the reference beats with the EC57 150 ms tolerance (Dataset
Explorer and `GET /api/detection/evaluation`) and can drive the "detected" mode of the Full ECG
Analysis page. It is never used in the principal experiment.
