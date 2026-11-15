"""Subject-level data partitioning and leakage checks.

Partitions are defined per RECORD in configs/splits.json and validated per SUBJECT:
all recordings of a subject must share a partition (and a cross-validation fold).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..data import subjects


@lru_cache(maxsize=1)
def split_config() -> dict:
    return config.load_json("splits.json")


def record_partition() -> dict[str, str]:
    """record -> 'train' | 'test' | 'excluded' (train = train+validation pool)."""
    cfg = split_config()
    out: dict[str, str] = {}
    for name, key in (("train", "train_pool"), ("test", "test"), ("excluded", "excluded")):
        for r in cfg[key]:
            if r in out:
                raise ValueError(f"Record {r} assigned to two partitions")
            out[r] = name
    missing = set(config.RECORDS) - set(out)
    if missing:
        raise ValueError(f"Records without a partition: {sorted(missing)}")
    validate_partition(out)
    return out


def validate_partition(partition: dict[str, str]) -> None:
    """Raise if any subject is spread over more than one partition."""
    smap = subjects.subject_map()
    seen: dict[str, str] = {}
    for rec, part in partition.items():
        sid = smap[rec]
        if seen.setdefault(sid, part) != part:
            raise ValueError(f"Subject {sid} appears in partitions {seen[sid]} and {part}")


def assign_folds(meta: pd.DataFrame) -> dict[str, int]:
    """Deterministic subject-grouped K folds over the training pool.

    Subjects are sorted by their number of minority-class (S, V, F) beats and dealt to
    the fold that currently holds the fewest minority beats, which spreads the rare
    classes across folds. Only label COUNTS of training-pool subjects are used.
    """
    cfg = split_config()
    k = cfg["n_folds"]
    smap = subjects.subject_map()
    pool = meta[meta["record"].isin(cfg["train_pool"])]
    counts = {}
    for sid, grp in pool.groupby("subject"):
        c = grp["aami"].value_counts()
        counts[sid] = np.array([c.get("S", 0), c.get("V", 0), c.get("F", 0)], dtype=float)
    total = sum(counts.values()) + 1e-9
    order = sorted(counts, key=lambda s: (-(counts[s] / total).sum(), s))
    fold_load = np.zeros((k, 3))
    fold_n = np.zeros(k)
    subj_fold: dict[str, int] = {}
    for sid in order:
        share = counts[sid] / total
        # choose the fold minimising resulting imbalance; tie-break on fold size
        scores = [(((fold_load[f] + share) ** 2).sum(), fold_n[f], f) for f in range(k)]
        f = min(scores)[2]
        subj_fold[sid] = f
        fold_load[f] += share
        fold_n[f] += 1
    return {r: int(subj_fold[smap[r]]) for r in cfg["train_pool"]}


def leakage_report(meta: pd.DataFrame) -> dict:
    """Automated leakage checks over the processed beat table."""
    smap = subjects.subject_map()
    checks = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    parts = {p: g for p, g in meta.groupby("partition")}
    subj = {p: set(g["subject"]) for p, g in parts.items()}
    recs = {p: set(g["record"]) for p, g in parts.items()}
    names = sorted(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            add(f"No subject overlap: {a} vs {b}", not (subj[a] & subj[b]),
                f"shared={sorted(subj[a] & subj[b])}")
            add(f"No recording overlap: {a} vs {b}", not (recs[a] & recs[b]),
                f"shared={sorted(recs[a] & recs[b])}")
    p201 = set(meta.loc[meta["record"] == "201", "partition"])
    p202 = set(meta.loc[meta["record"] == "202", "partition"])
    add("Records 201 and 202 share one partition", p201 == p202 and len(p201) == 1,
        f"201={sorted(p201)} 202={sorted(p202)}")
    add("Records 201 and 202 share one subject id", smap["201"] == smap["202"], smap["201"])
    inherit = bool((meta["record"].map(smap) == meta["subject"]).all())
    add("Every segment inherits its recording's subject id", inherit)
    one_part = bool((meta.groupby("record")["partition"].nunique() == 1).all())
    add("Every recording lies in exactly one partition", one_part)
    train = meta[meta["partition"] == "train"]
    fold_subj = train.groupby("subject")["fold"].nunique()
    add("Every training subject lies in exactly one CV fold", bool((fold_subj == 1).all()))
    f201 = set(train.loc[train["record"] == "201", "fold"])
    f202 = set(train.loc[train["record"] == "202", "fold"])
    add("Records 201 and 202 share one CV fold", f201 == f202 and len(f201) == 1,
        f"201={sorted(f201)} 202={sorted(f202)}")
    add("Test beats carry no CV fold", bool((meta.loc[meta["partition"] != "train", "fold"] == -1).all()))
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def describe(meta: pd.DataFrame | None = None) -> dict:
    cfg = split_config()
    out = {
        "name": cfg["name"], "version": cfg["version"], "description": cfg["description"],
        "subject_note": cfg["subject_note"], "same_subject_groups": cfg["same_subject_groups"],
        "train_pool": cfg["train_pool"], "test": cfg["test"], "excluded": cfg["excluded"],
        "original_protocol": cfg["original_protocol"], "n_folds": cfg["n_folds"],
        "difference_from_original": "Record 202 moved from DS2 (test) to the training pool so "
                                    "that it stays with record 201 (same subject).",
    }
    if meta is not None:
        train = meta[meta["partition"] == "train"]
        out["folds"] = {str(f): sorted(g["record"].unique().tolist())
                        for f, g in train.groupby("fold")}
        table = meta.groupby(["partition", "aami"]).size().unstack(fill_value=0)
        out["class_counts"] = {p: {c: int(v) for c, v in row.items()} for p, row in table.iterrows()}
        ft = train.groupby(["fold", "aami"]).size().unstack(fill_value=0)
        out["fold_class_counts"] = {str(f): {c: int(v) for c, v in row.items()} for f, row in ft.iterrows()}
        out["n_subjects"] = {p: int(g["subject"].nunique()) for p, g in meta.groupby("partition")}
        out["leakage"] = leakage_report(meta)
    return out
