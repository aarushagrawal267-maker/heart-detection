"""Patient-level partitioning and automated leakage checks."""
import numpy as np
import pandas as pd
import pytest

from app import config
from app.data import subjects
from app.splits import split as S

from .conftest import requires_dataset


def test_every_record_has_exactly_one_partition():
    part = S.record_partition()
    assert set(part) == set(config.RECORDS)
    cfg = S.split_config()
    all_listed = cfg["train_pool"] + cfg["test"] + cfg["excluded"]
    assert len(all_listed) == len(set(all_listed)) == 48


def test_201_and_202_same_subject_and_partition():
    assert subjects.subject_of("201") == subjects.subject_of("202")
    part = S.record_partition()
    assert part["201"] == part["202"] == "train"


def test_47_subjects():
    assert subjects.n_subjects() == 47
    assert len(set(subjects.subject_map().values())) == 47


def test_no_subject_overlap_between_partitions():
    part = S.record_partition()
    smap = subjects.subject_map()
    by_part = {}
    for rec, p in part.items():
        by_part.setdefault(p, set()).add(smap[rec])
    names = list(by_part)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (by_part[a] & by_part[b]), f"subject overlap between {a} and {b}"


def test_original_protocol_conflict_is_documented_and_corrected():
    cfg = S.split_config()
    assert "201" in cfg["original_protocol"]["DS1"] and "202" in cfg["original_protocol"]["DS2"]
    assert "202" not in cfg["test"]
    assert set(cfg["test"]) == set(cfg["original_protocol"]["DS2"]) - {"202"}
    assert set(cfg["train_pool"]) == set(cfg["original_protocol"]["DS1"]) | {"202"}


def test_paced_records_excluded():
    assert set(S.split_config()["excluded"]) == {"102", "104", "107", "217"}


def test_validate_partition_rejects_split_subject():
    part = S.record_partition().copy()
    part["202"] = "test"
    with pytest.raises(ValueError, match="S201_202"):
        S.validate_partition(part)


def _synthetic_meta():
    rng = np.random.default_rng(0)
    rows = []
    part = S.record_partition()
    for rec in config.RECORDS:
        for _ in range(30):
            rows.append({"record": rec, "subject": subjects.subject_of(rec), "partition": part[rec],
                         "aami": rng.choice(["N", "N", "N", "S", "V", "F"])})
    return pd.DataFrame(rows)


def test_folds_are_subject_grouped_and_deterministic():
    meta = _synthetic_meta()
    f1 = S.assign_folds(meta)
    f2 = S.assign_folds(meta)
    assert f1 == f2
    assert set(f1) == set(S.split_config()["train_pool"])
    assert f1["201"] == f1["202"]
    assert set(f1.values()) <= set(range(S.split_config()["n_folds"]))


def test_leakage_report_detects_injected_leak():
    meta = _synthetic_meta()
    folds = S.assign_folds(meta)
    meta["fold"] = meta["record"].map(folds).fillna(-1).astype(int)
    assert S.leakage_report(meta)["passed"]
    leaked = meta.copy()
    leaked.loc[leaked["record"] == "202", "partition"] = "test"
    leaked.loc[leaked["record"] == "202", "fold"] = -1
    rep = S.leakage_report(leaked)
    assert not rep["passed"]
    failed = {c["name"] for c in rep["checks"] if not c["passed"]}
    assert "Records 201 and 202 share one partition" in failed
    assert any(n.startswith("No subject overlap") for n in failed)


@requires_dataset
def test_real_segments_pass_all_leakage_checks():
    from app.segmentation import beats
    _, meta, _ = beats.load()
    rep = S.leakage_report(meta)
    assert rep["passed"], [c for c in rep["checks"] if not c["passed"]]
    # every segment inherits the right subject and partition from its recording
    smap = subjects.subject_map()
    part = S.record_partition()
    assert (meta["subject"] == meta["record"].map(smap)).all()
    assert (meta["partition"] == meta["record"].map(part)).all()
    train_subj = set(meta.loc[meta.partition == "train", "subject"])
    test_subj = set(meta.loc[meta.partition == "test", "subject"])
    assert not train_subj & test_subj


@requires_dataset
def test_training_selection_never_touches_test_subjects():
    from app.segmentation import beats
    from app.training import data as D
    _, meta, _ = beats.load()
    classes = ["N", "S", "V", "F"]
    for vf in range(5):
        tr = D.select(meta, classes, "train", [f for f in range(5) if f != vf])
        va = D.select(meta, classes, "train", [vf])
        te = D.select(meta, classes, "test")
        s_tr, s_va, s_te = (set(meta.iloc[i]["subject"]) for i in (tr, va, te))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        assert set(np.intersect1d(tr, va)) == set()
