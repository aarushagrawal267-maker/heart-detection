"""Recording -> subject mapping.

MIT-BIH contains 48 recordings from 47 subjects: records 201 and 202 come from the
same person. The mapping is derived from configs/splits.json (same_subject_groups).
"""
from __future__ import annotations

from functools import lru_cache

from .. import config


@lru_cache(maxsize=1)
def subject_map() -> dict[str, str]:
    groups = config.load_json("splits.json")["same_subject_groups"]
    mapping = {r: f"S{r}" for r in config.RECORDS}
    for group in groups:
        sid = "S" + "_".join(sorted(group))
        for r in group:
            mapping[r] = sid
    return mapping


def subject_of(record: str) -> str:
    return subject_map()[record]


def n_subjects() -> int:
    return len(set(subject_map().values()))
