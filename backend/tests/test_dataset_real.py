"""Integrity of the real MIT-BIH files and the processed beat set."""
import numpy as np

from app import config
from app.data import acquire, loader

from .conftest import requires_dataset

pytestmark = requires_dataset


def test_all_files_match_physionet_checksums():
    rep = acquire.verify()
    assert rep["ok"], rep["problems"][:5]
    assert rep["n_verified"] == 144


def test_headers():
    for r in config.RECORDS:
        h = loader.load_header(r)
        assert h.fs == 360 and h.n_samples == 650000 and len(h.leads) == 2


def test_mlii_identified_by_name():
    assert loader.primary_channel("100") == 0
    assert loader.primary_channel("114") == 1          # leads swapped in record 114
    assert loader.primary_channel("102") is None       # paced, no MLII
    assert sum(loader.primary_channel(r) is not None for r in config.RECORDS) == 46


def test_signal_in_physical_units():
    x = loader.load_signal("100")
    assert x.shape == (650000, 2) and x.dtype == np.float32
    assert -5 < x.min() < x.max() < 5


def test_record_100_annotations():
    ann = loader.load_annotations("100")
    syms = ann["symbol"]
    assert syms.count("N") == 2239 and syms.count("A") == 33 and syms.count("V") == 1


def test_processed_beat_counts():
    from app.segmentation import beats
    segs, meta, man = beats.load()
    beat_total = sum(sum(n for s, n in r["symbol_counts"].items()
                         if s in "NLRejAaJSVEF/fQ") for r in man["records"])
    assert beat_total == 109494
    assert len(meta) == segs.shape[0] == beat_total - man["n_boundary_excluded"]
    assert segs.shape[1] == config.PIPELINE["store_pre"] + config.PIPELINE["store_post"]
    counts = meta[meta.partition == "test"].aami.value_counts()
    assert counts["S"] == 1782 and counts["V"] == 3201 and counts["F"] == 387


def test_segment_centre_aligns_with_annotation():
    from app.segmentation import beats
    segs, meta, _ = beats.load()
    row = meta[(meta.record == "100")].iloc[10]
    sig = np.asarray(beats.load_filtered("100"))
    pre = config.PIPELINE["store_pre"]
    i = row.name
    assert np.allclose(segs[i], sig[row["sample"] - pre: row["sample"] + config.PIPELINE["store_post"]])
