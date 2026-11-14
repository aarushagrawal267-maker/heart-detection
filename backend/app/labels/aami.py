"""Explicit MIT-BIH symbol -> AAMI class mapping (loaded from configs/aami_mapping.json)."""
from __future__ import annotations

from functools import lru_cache

from .. import config

AAMI_ORDER = ["N", "S", "V", "F", "Q"]


@lru_cache(maxsize=1)
def mapping_config() -> dict:
    return config.load_json("aami_mapping.json")


@lru_cache(maxsize=1)
def symbol_to_class() -> dict[str, str]:
    out: dict[str, str] = {}
    for cls, spec in mapping_config()["classes"].items():
        for sym in spec["symbols"]:
            if sym in out:
                raise ValueError(f"Symbol {sym!r} mapped twice")
            out[sym] = cls
    return out


def non_beat_symbols() -> dict[str, str]:
    return mapping_config()["non_beat_symbols"]


def is_beat(symbol: str) -> bool:
    return symbol in symbol_to_class()


def to_aami(symbol: str) -> str | None:
    """AAMI class for a beat symbol, or None for non-beat annotations."""
    return symbol_to_class().get(symbol)


def describe(symbol: str) -> str:
    for spec in mapping_config()["classes"].values():
        if symbol in spec["symbols"]:
            return spec["symbols"][symbol]
    return non_beat_symbols().get(symbol, "Unknown annotation symbol")


def paced_records() -> list[str]:
    return list(mapping_config()["paced_records"])


def primary_classes() -> list[str]:
    return list(mapping_config()["primary_task_classes"])
