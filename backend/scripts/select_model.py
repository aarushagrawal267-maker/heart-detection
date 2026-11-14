"""Model selection by subject-grouped 5-fold cross-validation (training pool only).

Each candidate configuration is trained once per fold (validation = that fold's
subjects) and once more with the default validation fold; the candidate with the
highest mean CV macro-F1 becomes the active model. Held-out test subjects are NOT
used here. Run the explicit, logged test evaluation afterwards.

Usage: python scripts/select_model.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import config  # noqa: E402
from app.db import models as db  # noqa: E402
from app.training import jobs  # noqa: E402

BASE = {"cross_validate": True, "epochs": 15, "early_stopping_patience": 5}
CANDIDATES = [
    ("ECGNet CV - morphology + RR (weighted CE)", {}),
    ("ECGNet CV - regularised (16/32/64, wd 1e-3, inverse-frequency weights)",
     {"channels": [16, 32, 64], "dropout": 0.5, "weight_decay": 1e-3, "lr": 5e-4,
      "class_weight_power": 1.0}),
    ("ECGNet CV - morphology only (ablation)", {"use_rr": False}),
]


def main() -> None:
    results = []
    for name, cfg in CANDIDATES:
        exp = jobs.create(name, "cnn", {**BASE, **cfg}, notes="model selection by 5-fold subject CV")
        print("running", exp.id, name, flush=True)
        subprocess.run([sys.executable, "-m", "app.training.worker", exp.id],
                       cwd=os.path.join(os.path.dirname(__file__), ".."), check=False)
        with db.session() as s:
            e = s.get(db.Experiment, exp.id)
            if e.status == "completed" and e.cv_results:
                results.append({"id": e.id, "name": e.name,
                                "cv_mean_macro_f1": e.cv_results["mean_macro_f1"],
                                "cv_std_macro_f1": e.cv_results["std_macro_f1"],
                                "val_macro_f1": e.summary.get("val_macro_f1")})
                print("  CV macro-F1", e.cv_results["mean_macro_f1"], "+/-",
                      e.cv_results["std_macro_f1"], flush=True)
    if not results:
        raise SystemExit("no candidate completed")
    best = max(results, key=lambda r: r["cv_mean_macro_f1"])
    db.set_setting("active_experiment_id", best["id"])
    out = {"criterion": "mean macro-F1 over 5 subject-grouped CV folds of the training pool",
           "test_subjects_used": False, "candidates": results, "selected": best}
    (config.MODELS_DIR / "selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("selected", best, flush=True)


if __name__ == "__main__":
    main()
