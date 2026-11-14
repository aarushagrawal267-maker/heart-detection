import { useState } from "react";
import { PageHeader, ResearchNotice } from "../components/ui";

const SECTIONS = [
  { id: "ecg", title: "ECG fundamentals" },
  { id: "classification", title: "Heartbeat classification" },
  { id: "dataset", title: "Dataset" },
  { id: "preprocessing", title: "Preprocessing" },
  { id: "segmentation", title: "Segmentation" },
  { id: "model", title: "1D CNN architecture" },
  { id: "training", title: "Training & imbalance" },
  { id: "evaluation", title: "Patient-independent evaluation" },
  { id: "explainability", title: "Explainability" },
  { id: "limitations", title: "Limitations" },
  { id: "reproducibility", title: "Reproducibility" },
  { id: "references", title: "References" },
];

export default function Docs() {
  const [active, setActive] = useState("ecg");
  return (
    <div className="space-y-5">
      <PageHeader title="Research & Documentation" sub="Methodology, scientific background, limitations and reproduction instructions for this platform." />
      <ResearchNotice />
      <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
        <nav className="hidden lg:block">
          <ul className="sticky top-4 space-y-0.5">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  onClick={() => setActive(s.id)}
                  className={`block rounded-md px-2.5 py-1.5 text-xs transition-colors ${active === s.id ? "bg-ink-700 text-fg" : "text-fg-muted hover:text-fg"}`}
                >
                  {s.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <article className="panel prose-doc max-w-none p-6 sm:p-8" data-testid="docs">
          <h2 id="ecg">ECG fundamentals</h2>
          <p>
            An electrocardiogram records the heart's electrical activity at the body surface. Each heartbeat produces a characteristic sequence: the
            <strong> P wave</strong> (atrial depolarisation), the <strong>QRS complex</strong> (ventricular depolarisation, normally 80–120 ms) and the
            <strong> T wave</strong> (ventricular repolarisation). The interval between successive R peaks — the <strong>RR interval</strong> — reflects heart
            rate and rhythm.
          </p>
          <p>
            Ectopic beats originate outside the sinus node. <em>Supraventricular</em> ectopic beats (S) usually have a normal-looking QRS but arrive early.
            <em> Ventricular</em> ectopic beats (V) typically have a wide, bizarre QRS and are often followed by a compensatory pause. <em>Fusion</em> beats
            (F) are a mixture of a normal and a ventricular activation.
          </p>
          <p>The ECG paper grid in the viewer follows the clinical convention: small squares 0.04 s × 0.1 mV, large squares 0.2 s × 0.5 mV.</p>

          <h2 id="classification">Heartbeat classification (AAMI EC57)</h2>
          <p>
            The ANSI/AAMI EC57 standard groups the MIT-BIH beat labels into five classes, which makes results comparable across studies. The mapping used
            here is explicit (<code>configs/aami_mapping.json</code>) and shown symbol by symbol in the Dataset Explorer:
          </p>
          <ul>
            <li><strong>N</strong> — N (normal), L (left bundle branch block), R (right bundle branch block), e (atrial escape), j (nodal escape)</li>
            <li><strong>S</strong> — A (atrial premature), a (aberrated atrial premature), J (nodal premature), S (supraventricular premature)</li>
            <li><strong>V</strong> — V (premature ventricular contraction), E (ventricular escape)</li>
            <li><strong>F</strong> — F (fusion of ventricular and normal)</li>
            <li><strong>Q</strong> — / (paced), f (fusion of paced and normal), Q (unclassifiable)</li>
          </ul>
          <p>
            Non-beat annotations (rhythm changes, noise markers, ventricular flutter waves “!”, non-conducted P waves “x”, comments) never become
            heartbeat segments. Following EC57 and de Chazal et al. (2004), the four paced records (102, 104, 107, 217) are excluded from training and
            evaluation. That leaves only about 15 Q beats, too few to learn or score, so the principal experiment uses four classes (N, S, V, F). The Q
            beats are still counted and displayed.
          </p>

          <h2 id="dataset">Dataset</h2>
          <p>
            MIT-BIH Arrhythmia Database v1.0.0 (PhysioNet): 48 half-hour, two-channel ambulatory recordings from 47 subjects, digitised at 360 Hz with 11-bit
            resolution over ±5 mV, annotated beat by beat by at least two cardiologists. Records 201 and 202 come from the same subject. The upper
            channel is modified limb lead II (MLII) in 45 records; record 114 has MLII on the second channel and records 102 and 104 have no MLII. The
            lead is identified by header name, never by position.
          </p>
          <p>
            Files are downloaded from PhysioNet and every <code>.dat</code>, <code>.hea</code> and <code>.atr</code> file is verified against PhysioNet's published
            SHA-256 checksums. The raw directory is marked read-only and provenance (source, version, DOI, licence, download time, hashes) is written
            to <code>provenance.json</code>.
          </p>

          <h2 id="preprocessing">Preprocessing</h2>
          <ol>
            <li><strong>Baseline wander</strong>: two cascaded median filters (200 ms, then 600 ms) estimate the baseline, which is subtracted. This removes slow drift without distorting the QRS or ST segment (de Chazal et al., 2004).</li>
            <li><strong>Power-line and high-frequency noise</strong>: 4th-order Butterworth low-pass at 45 Hz applied forward and backward (zero phase), so R-peak positions are not shifted. 60 Hz interference (US recordings) is removed without a ringing notch filter.</li>
            <li><strong>Normalisation</strong>: each model input segment is z-scored individually. This is stateless, so nothing is fitted on validation or test data. The only fitted transform is the RR-feature standardiser, which is fitted on training subjects only.</li>
            <li><strong>Signal quality</strong> is assessed in 10 s windows (baseline power &lt;0.5 Hz, 60 Hz power, power &gt;45 Hz, flat-line and ADC saturation) and displayed. It is descriptive only: no beats are removed because of it.</li>
          </ol>
          <p>The original recordings are never modified; preprocessed signals are regenerated deterministically from a configuration whose hash is stored with every experiment.</p>

          <h2 id="segmentation">Heartbeat segmentation</h2>
          <p>
            Segments are centred on the expert annotation sample of every beat. The stored segment is 128 samples before and 192 after the annotation. The model
            uses a 256-sample window of <strong>90 samples before and 166 after</strong> (0.25 s + 0.46 s = 0.71 s). The asymmetric window covers the P wave
            before the QRS and the T wave after it. The symmetric 128/128 alternative spends samples on the previous beat's T wave and can cut off this
            beat's T wave at normal heart rates. The window is configurable in the Model Laboratory, and the margin allows ±5-sample shift augmentation with real
            neighbouring samples.
          </p>
          <p>
            Beats whose stored window would extend past the recording boundary are excluded and counted (50 beats). Every segment keeps its record, subject,
            beat index, original symbol, AAMI class, partition and cross-validation fold.
          </p>
          <p>
            <strong>RR features</strong>: previous RR, next RR, and both divided by the local median RR over ±10 beats. These encode prematurity, which is the key
            cue for S beats whose QRS morphology is often normal. They use only the timing of the same recording, never labels. Because they are computed
            from reference positions, real-world use would add R-peak detection error; the separate detector evaluation quantifies that.
          </p>

          <h2 id="model">1D CNN architecture (ECGNet)</h2>
          <p>A compact convolutional network written for this project and trained from random initialisation. No pretrained weights or external models are used.</p>
          <ul>
            <li>Stage 1: 2 × [Conv1d 32 filters, kernel 7, BatchNorm, ReLU], MaxPool 2 → 32 × 128</li>
            <li>Stage 2: 2 × [Conv1d 64, kernel 5, BatchNorm, ReLU], MaxPool 2 → 64 × 64</li>
            <li>Stage 3: 2 × [Conv1d 128, kernel 3, BatchNorm, ReLU], MaxPool 2 → 128 × 32</li>
            <li>Stage 4: Conv1d 128, kernel 3, BatchNorm, ReLU (Grad-CAM target), global average pooling → 128</li>
            <li>Head: concatenate the 4 standardised RR features → Linear 132→64, ReLU, Dropout 0.4 → Linear 64→classes</li>
          </ul>
          <p>
            Wide early kernels (≈19 ms) capture QRS slopes at 360 Hz; deeper layers see the whole beat (receptive field ≈ 0.5 s). Global average pooling keeps
            the parameter count low (≈170k) and makes 1D class-activation maps possible. A morphology-only variant (no RR input) is trained as an ablation.
          </p>

          <h2 id="training">Training strategy & class imbalance</h2>
          <ul>
            <li>AdamW, one-cycle learning-rate schedule, early stopping on <strong>validation</strong> macro F1 (best weights restored).</li>
            <li>Class-weighted cross-entropy by default: weight<sub>c</sub> ∝ (n / K·n<sub>c</sub>)<sup>p</sup>, computed from training-subject counts only. Focal loss and a balanced sampler are available as experiments.</li>
            <li>Oversampling (balanced sampler) happens inside the training partition only, after the patient split, so no beat or patient can cross partitions.</li>
            <li>Augmentation on training beats only: ±5-sample shift, amplitude scaling (0.85–1.15), small offset, Gaussian noise.</li>
            <li>Baselines: majority class (shows the accuracy trap), logistic regression on RR features, random forest on RR + hand-crafted morphology.</li>
            <li>Seeds are fixed; environment (package versions, device, git commit) is stored with every run.</li>
          </ul>

          <h2 id="evaluation">Patient-independent evaluation</h2>
          <p>
            A random split by heartbeat lets beats from the same patient appear in training and testing. Because a patient's beats look alike, the model can
            succeed by recognising the patient rather than the arrhythmia, giving inflated results (often &gt;99% accuracy). This platform splits strictly
            <strong> by subject</strong>.
          </p>
          <p>
            The split follows the established inter-patient protocol of de Chazal et al. (2004) — DS1 for training, DS2 for testing — with one correction: the
            original protocol places record 201 in DS1 and record 202 in DS2, although both come from the same subject. Record 202 is moved to the training side,
            so the test set is DS2 minus 202 (21 subjects). Results are therefore not directly comparable to papers using the uncorrected split.
          </p>
          <p>
            Within the training pool, subjects are dealt into five folds (201 and 202 always together) so that the rare S, V and F beats are spread as evenly as
            subject boundaries allow. One fold is the validation set used for early stopping and model selection. 5-fold cross-validation is available. Held-out test
            subjects are evaluated only by an explicit, logged action.
          </p>
          <p>Automated tests verify: no subject or recording overlap between partitions or folds; 201 and 202 share a subject, partition and fold; every segment inherits the correct identifiers.</p>
          <p>
            Reported metrics: accuracy, per-class sensitivity, positive predictivity, specificity and F1, macro and weighted F1, confusion matrix, one-vs-rest ROC-AUC,
            precision–recall curves, per-patient results, inference time, and 95% confidence intervals from a bootstrap that resamples <em>patients</em>.
          </p>

          <h2 id="explainability">Explainability</h2>
          <ul>
            <li><strong>Integrated Gradients</strong> (Sundararajan et al., 2017): integrates gradients along a path from a flat-line baseline (and mean RR) to the input. Attributions sum to the change in class score, which the UI reports as a completeness check.</li>
            <li><strong>Grad-CAM</strong> (Selvaraju et al., 2017), adapted to 1D on the last convolutional layer.</li>
            <li><strong>Saliency</strong>: magnitude of the input gradient.</li>
            <li><strong>Occlusion</strong>: drop in class probability when a 40 ms window is flattened.</li>
            <li><strong>Sanity check</strong> (Adebayo et al., 2018): compares explanations with those of a randomly-initialised network; a low correlation indicates the map depends on what the model learned.</li>
          </ul>
          <p>These maps show what the model's score is sensitive to. They are analytical visualisations, not evidence that the model reasons like a cardiologist.</p>

          <h2 id="limitations">Limitations</h2>
          <ul>
            <li><strong>Not a medical device.</strong> No clinical validation; outputs must not be used for diagnosis, monitoring or treatment decisions.</li>
            <li><strong>Small, old, single-centre data</strong>: 47 subjects recorded 1975–1979 at Boston's Beth Israel Hospital on analogue Holter tapes. Performance will not transfer automatically to other devices, leads or populations.</li>
            <li><strong>Few test patients</strong>: 21 subjects. Minority classes are concentrated in a handful of patients — e.g. record 232 holds about three quarters of all test S beats — so per-class results depend heavily on individual patients and confidence intervals are wide.</li>
            <li><strong>Inter-patient generalisation is hard</strong>: training subjects are fitted almost perfectly while validation performance is much lower. S and F beats are recognised poorly; this is consistent with the literature and reported, not hidden.</li>
            <li><strong>Reference positions</strong>: the principal experiment assumes perfect beat detection; RR features come from expert positions.</li>
            <li><strong>Single lead</strong> (MLII). Q beats are not learned. VF/flutter episodes are not beat-classified.</li>
            <li>Model scores are softmax outputs, not calibrated probabilities.</li>
          </ul>

          <h2 id="reproducibility">Reproducibility</h2>
          <ol>
            <li>Run <code>setup.ps1</code> (creates the Python environment, installs dependencies, downloads and verifies MIT-BIH, builds the beat set).</li>
            <li>Run <code>start.ps1</code> and open <code>http://localhost:5173</code>; the API documentation is at <code>http://localhost:8000/docs</code>.</li>
            <li>Reproduce the reference experiments from the command line: <code>python backend/scripts/run_experiments.py</code>.</li>
            <li>Run the tests: <code>pytest</code> in <code>backend/</code>, <code>npm test</code> and <code>npm run e2e</code> in <code>frontend/</code>.</li>
          </ol>
          <p>Every experiment stores its configuration, preprocessing hash, dataset version, patient partitions, environment and checkpoint, and can be exported as JSON.</p>

          <h2 id="references">References</h2>
          <ul>
            <li>Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol. 2001;20(3):45–50.</li>
            <li>Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000;101(23):e215–e220.</li>
            <li>MIT-BIH Arrhythmia Database v1.0.0, PhysioNet. <a href="https://doi.org/10.13026/C2F305" target="_blank" rel="noreferrer">doi:10.13026/C2F305</a></li>
            <li>ANSI/AAMI EC57: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.</li>
            <li>de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. IEEE Trans Biomed Eng. 2004;51(7):1196–1206.</li>
            <li>Mondéjar-Guerra V, et al. Heartbeat classification fusing temporal and morphological information of ECGs via ensemble of classifiers. Biomed Signal Process Control. 2019;47:41–48.</li>
            <li>Luz EJS, et al. ECG-based heartbeat classification for arrhythmia detection: a survey. Comput Methods Programs Biomed. 2016;127:144–164.</li>
            <li>Kiranyaz S, Ince T, Gabbouj M. Real-time patient-specific ECG classification by 1-D convolutional neural networks. IEEE Trans Biomed Eng. 2016;63(3):664–675.</li>
            <li>Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng. 1985;32(3):230–236.</li>
            <li>Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. ICML 2017.</li>
            <li>Selvaraju RR, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV 2017.</li>
            <li>Adebayo J, et al. Sanity checks for saliency maps. NeurIPS 2018.</li>
            <li>Lin TY, et al. Focal loss for dense object detection. ICCV 2017.</li>
          </ul>
        </article>
      </div>
    </div>
  );
}
