"""Run the reference experiment set sequentially through the real worker process.

Model selection uses validation subjects only. Usage: python scripts/run_experiments.py
"""
import subprocess, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.training import jobs

PLAN = [
    ("Majority-class baseline", "baseline", {"model": "majority"}),
    ("Logistic regression (RR features)", "baseline", {"model": "logreg_rr"}),
    ("Random forest (RR + morphology)", "baseline", {"model": "random_forest"}),
    ("ECGNet - morphology + RR (weighted CE)", "cnn", {}),
    ("ECGNet - morphology only (ablation)", "cnn", {"use_rr": False}),
    ("ECGNet - morphology + RR (focal loss)", "cnn", {"loss": "focal"}),
    ("ECGNet - morphology + RR (balanced sampler)", "cnn", {"sampler": "balanced", "loss": "ce"}),
]
for name, kind, cfg in PLAN:
    exp = jobs.create(name, kind, cfg)
    print("running", exp.id, name, flush=True)
    subprocess.run([sys.executable, "-m", "app.training.worker", exp.id],
                   cwd=os.path.join(os.path.dirname(__file__), ".."), check=False)
print("all done", flush=True)
