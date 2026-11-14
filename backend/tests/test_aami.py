from app.labels import aami

from .conftest import requires_dataset

EXPECTED = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V", "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def test_mapping_matches_aami_ec57():
    assert aami.symbol_to_class() == EXPECTED


def test_non_beat_symbols_are_not_beats():
    for sym in ["+", "~", "|", "x", "!", "[", "]", '"']:
        assert not aami.is_beat(sym)
        assert aami.to_aami(sym) is None


def test_no_symbol_is_both_beat_and_non_beat():
    assert not set(aami.symbol_to_class()) & set(aami.non_beat_symbols())


@requires_dataset
def test_every_symbol_in_dataset_is_documented():
    from app.segmentation import beats
    _, _, man = beats.load()
    seen = set()
    for r in man["records"]:
        seen |= set(r["symbol_counts"])
    documented = set(aami.symbol_to_class()) | set(aami.non_beat_symbols())
    assert seen <= documented, seen - documented
