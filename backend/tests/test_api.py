"""API tests against the real dataset, including a real (tiny) training job."""
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

from .conftest import requires_dataset

pytestmark = requires_dataset


@pytest.fixture()
def client(temp_store):
    with TestClient(app) as c:
        yield c


def test_health_and_docs(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_records_and_signal(client):
    recs = client.get("/api/records").json()
    assert len(recs) == 48
    assert {r["record"]: r["analysis_lead"] for r in recs}["114"] == "MLII"
    s = client.get("/api/records/100/signal?start=0&end=2&max_points=100000").json()
    assert len(s["y"]) == 720 and not s["decimated"]
    s2 = client.get("/api/records/100/signal?start=0&end=60&max_points=2000").json()
    assert s2["decimated"] and len(s2["y"]) <= 2000
    assert client.get("/api/records/999/signal").status_code == 404


def test_annotations_and_beats(client):
    a = client.get("/api/records/100/annotations?start=0&end=10").json()
    assert all(0 <= x["time_s"] < 10 for x in a["annotations"])
    b = client.get("/api/records/100/beats").json()
    assert b["n"] == 2271
    d = client.get("/api/records/100/beats/5").json()
    assert len(d["filtered_mv"]) == 256 and d["aami"] in "NSVFQ"


def test_splits_endpoint_reports_no_leakage(client):
    sp = client.get("/api/splits").json()
    assert sp["leakage"]["passed"]
    assert "202" in sp["train_pool"] and "202" not in sp["test"]


def test_no_model_yields_409(client):
    r = client.post("/api/inference/beat", json={"record": "100", "beat_index": 3})
    assert r.status_code == 409


def test_validation_of_training_config(client):
    assert client.post("/api/experiments", json={"kind": "cnn", "config": {"epochs": 0}, "start": False}).status_code == 400
    assert client.post("/api/experiments", json={"kind": "cnn", "config": {"bogus": 1}, "start": False}).status_code == 400
    assert client.post("/api/experiments", json={"kind": "baseline", "config": {"model": "svm"}, "start": False}).status_code == 400


@pytest.mark.slow
def test_train_infer_explain_and_locked_test_eval(client):
    r = client.post("/api/experiments", json={
        "name": "pytest tiny", "kind": "cnn",
        "config": {"epochs": 1, "max_train_beats": 3000, "batch_size": 128}})
    assert r.status_code == 200, r.text
    exp_id = r.json()["id"]
    assert client.post("/api/experiments", json={"kind": "cnn", "config": {}}).status_code == 409
    for _ in range(300):
        e = client.get(f"/api/experiments/{exp_id}").json()
        if e["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    assert e["status"] == "completed", e.get("error")
    assert e["split"]["validation_records"] and not set(e["split"]["validation_records"]) & set(e["split"]["test_records"])
    assert len(client.get(f"/api/experiments/{exp_id}/epochs").json()) == 1
    evals = client.get(f"/api/experiments/{exp_id}/evaluations").json()
    assert set(evals) == {"train", "validation"}            # test is NOT evaluated automatically

    beat = client.post("/api/inference/beat", json={"record": "100", "beat_index": 3}).json()
    assert beat["prediction"]["class"] in ["N", "S", "V", "F"]
    assert abs(sum(beat["prediction"]["scores"].values()) - 1) < 1e-3
    assert len(beat["explanation"]["attribution"]) == 256
    assert beat["explanation"]["completeness_gap"] < 0.05 * max(1.0, abs(beat["explanation"]["score_delta"]))
    for method in ["saliency", "grad_cam", "occlusion"]:
        ex = client.post("/api/explain", json={"record": "100", "beat_index": 3, "method": method}).json()
        assert len(ex["attribution"]) == 256

    rec = client.post("/api/inference/record", json={"record": "100", "mode": "reference"}).json()
    assert rec["n_beats"] == 2271 and rec["agreement"]["n"] > 2000
    det = client.post("/api/inference/record", json={"record": "100", "mode": "detected"}).json()
    assert det["detection"]["sensitivity"] > 0.95

    assert client.post(f"/api/experiments/{exp_id}/evaluate-test", json={}).status_code == 400
    t = client.post(f"/api/experiments/{exp_id}/evaluate-test", json={"confirm": True, "purpose": "pytest"})
    assert t.status_code == 200 and t.json()["metrics"]["n"] > 40000
    log = client.get("/api/test-access-log").json()
    assert log[0]["experiment_id"] == exp_id
    assert client.get("/api/overview").json()["recent_activity"]


def test_cancel_before_worker_start_is_respected(temp_store):
    """Regression: a cancel issued before the worker claims the job must not be overwritten."""
    from app.db import models as db
    from app.training import jobs, trainer
    exp = jobs.create("cancel race", "baseline", {"model": "majority"})
    jobs.cancel(exp.id)
    trainer.run_experiment(exp.id)
    with db.session() as s:
        assert s.get(db.Experiment, exp.id).status == "cancelled"


def test_basic_auth_middleware_protects_everything_but_health():
    import base64
    from fastapi import FastAPI
    from app.main import BasicAuthMiddleware
    mini = FastAPI()
    mini.add_middleware(BasicAuthMiddleware, username="qrs", password="s3cret")

    @mini.get("/api/health")
    def h():
        return {"ok": True}

    @mini.get("/api/overview")
    def o():
        return {"data": 1}

    c = TestClient(mini)
    assert c.get("/api/health").status_code == 200
    r = c.get("/api/overview")
    assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    bad = base64.b64encode(b"qrs:wrong").decode()
    assert c.get("/api/overview", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    good = base64.b64encode(b"qrs:s3cret").decode()
    assert c.get("/api/overview", headers={"Authorization": f"Basic {good}"}).json() == {"data": 1}


def test_training_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("QRS_ALLOW_TRAINING", "false")
    r = client.post("/api/experiments", json={"kind": "cnn", "config": {}, "start": False})
    assert r.status_code == 403


BUNDLED = "20261114-200900-e49e"    # the CNN chosen in models/selection.json


def test_bundled_model_serves_predictions_and_bad_inputs_get_clear_errors(client, temp_store):
    """The committed checkpoint loads and predicts; bad inputs and a lost model file give 4xx, not 500."""
    import json
    import shutil

    from app import config
    from app.db import models as db
    from app.inference import service
    from app.training.registry import import_all

    src = config.ROOT / "models" / BUNDLED
    dst = config.MODELS_DIR / BUNDLED
    dst.mkdir()
    shutil.copy(src / "model.pt", dst / "model.pt")
    rec = json.loads((src / "experiment.json").read_text(encoding="utf-8"))
    rec["experiment"]["checkpoint_path"] = str(dst / "model.pt")
    (dst / "experiment.json").write_text(json.dumps(rec), encoding="utf-8")
    assert import_all() == 1
    db.set_setting("active_experiment_id", BUNDLED)

    r = client.post("/api/inference/beat", json={"record": "100", "beat_index": 3, "explain": None})
    assert r.status_code == 200, r.text
    beat = r.json()
    assert beat["experiment_id"] == BUNDLED and beat["reference"]["aami"] == "N"
    assert beat["prediction"]["class"] in ["N", "S", "V", "F"]
    assert abs(sum(beat["prediction"]["scores"].values()) - 1) < 1e-3

    assert client.post("/api/inference/beat", json={"record": "100", "beat_index": 10**6}).status_code == 404
    assert client.post("/api/inference/beat", json={"record": "nope", "beat_index": 1}).status_code == 404
    assert client.get("/api/explain/sanity?record=100&beat_index=999999").status_code == 404
    assert client.get("/api/explain/sanity?record=nope&beat_index=1").status_code == 404
    r = client.post("/api/explain", json={"record": "100", "beat_index": 3, "target": "Z"})
    assert r.status_code == 400 and "target" in r.json()["detail"]

    (dst / "model.pt").unlink()
    service.clear_cache()
    r = client.post("/api/inference/beat", json={"record": "100", "beat_index": 3,
                                                 "experiment_id": BUNDLED})
    assert r.status_code == 409 and "missing" in r.json()["detail"]
    r = client.post("/api/inference/beat", json={"record": "100", "beat_index": 3})
    assert r.status_code == 409 and "No trained CNN" in r.json()["detail"]
    assert client.get("/api/models/active").json() is None
    r = client.post(f"/api/experiments/{BUNDLED}/activate")
    assert r.status_code == 409 and "missing" in r.json()["detail"]
