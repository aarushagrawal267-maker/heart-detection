"""Download robustness: transient PhysioNet / proxy errors must be retried, not fatal."""
from app.data import acquire


def test_session_retries_transient_http_errors():
    s = acquire.make_session(retries=4, backoff=0.5)
    retry = s.get_adapter("https://physionet.org/files/mitdb/1.0.0/").max_retries
    assert retry.total == 4
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    assert "GET" in retry.allowed_methods
    assert retry.backoff_factor == 0.5


def test_required_files_cover_every_record():
    files = acquire.required_files()
    assert len(files) == 48 * 3
    assert "100.dat" in files and "234.atr" in files and "201.hea" in files
