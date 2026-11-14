# Limitations and safety

**QRS Atlas is an educational and research implementation. It is not a medical device,
it has not been clinically validated, and its outputs are not diagnoses.** Do not use it to
assess anyone's heart, to monitor patients, or to make treatment decisions.

## Clinical interpretation

- A predicted class is a research-model output for one beat. It does not establish that a person has an arrhythmia.
- Class scores are softmax outputs, not calibrated probabilities.
- Good MIT-BIH test results do not establish clinical readiness. Clinical validation needs prospective data, multiple sites and devices, and regulatory processes far beyond this project.
- Explanation maps show what changes the model's score. They do not show that the model reasons like a clinician.

## Dataset

- 47 subjects recorded 1975–1979 at a single Boston hospital, digitised from analogue Holter tapes. The population, devices and noise are not representative of modern recordings.
- Only one lead (MLII) is used, and lead placement varies between subjects.
- The expert labels are high quality but not error-free.
- Subject identity is known only from the database documentation (201 and 202 share a subject); other shared subjects cannot be ruled out independently.
- Paced records are excluded, so the system makes no meaningful statement about paced rhythms. Q beats are not learned.

## Evaluation

- Only 21 test subjects. Minority classes are concentrated in a few people: record 232 holds about 78% of the test S beats, and record 208 holds about 90% of the training F beats. Per-class results depend heavily on single patients, so confidence intervals (subject bootstrap) are wide.
- Inter-patient generalisation is hard. The CNN fits training subjects almost perfectly while performing far worse on unseen subjects. S and F beats in particular are detected poorly. This matches the published inter-patient literature and is reported rather than hidden.
- The corrected split (202 moved to training) makes results not directly comparable to studies that used the original DS1/DS2 split.
- Model selection uses validation subjects from the training pool. Numbers on validation subjects are therefore optimistic; only held-out test numbers estimate generalisation.
- RR features come from expert beat positions. A deployed system would depend on R-peak detection, whose errors are measured separately but not propagated into the principal metrics.

## Computational and deployment

- Training runs on CPU by default (minutes per run on a laptop); 5-fold cross-validation multiplies that by about six. CUDA is used if installed but is never required.
- The application binds to `127.0.0.1` only and has no authentication. It is meant for a single local user.
- The Windows scripts were written for Windows PowerShell 5.1 and PowerShell 7. They were developed in a Linux environment and verified on Windows only through the included GitHub Actions workflow (`.github/workflows/windows.yml`), when that workflow has run.
