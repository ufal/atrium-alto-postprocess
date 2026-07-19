"""
tests/test_quality_model_correct.py
===================================
Fast, network-free tests for the issue #23 correction backends
(``tools/quality_model/correct.py``) and the score-delta report
(``report_correction_delta.py``).

The two HTTP helpers are monkeypatched, so no network is touched; the local
korektor backend is exercised with ``cat`` as a stand-in identity corrector, so
no korektor install is needed. A single ``@pytest.mark.slow`` test hits the real
LINDAT REST service and self-skips when offline.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import correct as CO  # noqa: E402
import report_correction_delta as RD  # noqa: E402

# ── Backends ────────────────────────────────────────────────────────────────


def test_noop_backend_is_identity():
    assert CO.NoopBackend().correct_batch(["abc", "def"]) == ["abc", "def"]


def test_korektor_rest_parses_result(monkeypatch):
    calls = []

    def fake_post_form(url, data, timeout):
        calls.append((url, data))
        return {"model": data["model"], "result": data["data"].upper() + "\n"}

    monkeypatch.setattr(CO, "_post_form", fake_post_form)
    backend = CO.KorektorRestBackend()
    assert backend.correct("mesto") == "MESTO"
    assert calls[0][0].endswith("/correct")
    assert calls[0][1]["data"] == "mesto"


def test_korektor_rest_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def flaky_post_form(url, data, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")
        return {"result": "ok"}

    monkeypatch.setattr(CO, "_post_form", flaky_post_form)
    backend = CO.KorektorRestBackend(max_retries=4, retry_backoff=0.0)  # no real sleeping
    assert backend.correct("x") == "ok"
    assert attempts["n"] == 3


def test_korektor_rest_raises_after_exhausting_retries(monkeypatch):
    def always_fail(url, data, timeout):
        raise ConnectionError("down")

    monkeypatch.setattr(CO, "_post_form", always_fail)
    backend = CO.KorektorRestBackend(max_retries=2, retry_backoff=0.0)
    with pytest.raises(RuntimeError):
        backend.correct("x")


def test_korektor_local_via_cat_identity():
    if shutil.which("cat") is None:
        pytest.skip("cat not available")
    backend = CO.KorektorLocalBackend(model_path="unused", command=["cat"])
    assert backend.correct_batch(["radek jedna", "radek dva"]) == ["radek jedna", "radek dva"]


def test_llm_backend_parses_chat_completion(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        assert payload["temperature"] == 0.0
        return {"choices": [{"message": {"content": "  opravená věta  "}}]}

    monkeypatch.setattr(CO, "_post_json", fake_post_json)
    backend = CO.LlmBackend("http://x/v1/chat/completions", "glm-4")
    assert backend.correct("chybna veta") == "opravená věta"


# ── Cache ───────────────────────────────────────────────────────────────────


def test_disk_cache_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "cache.jsonl"
    c1 = CO.DiskCache(path)
    key = CO.DiskCache.key("backend-x", "hello")
    assert c1.get(key) is None
    c1.put(key, "corrected")
    # A fresh cache reads the persisted entry back.
    c2 = CO.DiskCache(path)
    assert c2.get(key) == "corrected"


def test_cached_corrector_only_calls_backend_on_miss(tmp_path):
    class CountingBackend(CO.CorrectionBackend):
        id = "counting"

        def __init__(self):
            self.calls = 0

        def correct_batch(self, texts):
            self.calls += 1
            return [t.upper() for t in texts]

    backend = CountingBackend()
    cache = CO.DiskCache(tmp_path / "c.jsonl")
    corrector = CO.CachedCorrector(backend, cache)

    assert corrector.correct_batch(["a", "b"]) == ["A", "B"]
    assert backend.calls == 1
    # Second run: both cached → backend not called again.
    assert corrector.correct_batch(["a", "b"]) == ["A", "B"]
    assert backend.calls == 1
    # New text only → exactly one more backend call, for the miss.
    assert corrector.correct_batch(["a", "c"]) == ["A", "C"]
    assert backend.calls == 2


# ── Language routing ─────────────────────────────────────────────────────────


def test_route_backend_by_language():
    kor, llm = CO.NoopBackend(), CO.LlmBackend("http://x", "glm-4")
    assert CO.route_backend("ces_Latn", kor, llm) is kor
    assert CO.route_backend("slk_Latn", kor, llm) is kor
    assert CO.route_backend("deu_Latn", kor, llm) is llm
    assert CO.route_backend("eng_Latn", kor, None) is None


# ── Score-delta report ───────────────────────────────────────────────────────


def test_correction_delta_report_structure_and_gate():
    pairs = [
        # correction removes garbage → score should rise
        {
            "source_text": "rnn1 ww0rd vv_~~",
            "corrected_text": "první slovo věta",
            "lang": "ces_Latn",
            "backend": "korektor-rest",
        },
        {
            "source_text": "mesto nad rekou",
            "corrected_text": "město nad řekou",
            "lang": "ces_Latn",
            "backend": "korektor-rest",
        },
        {
            "source_text": "kniha a pero",
            "corrected_text": "kniha a pero",
            "lang": "ces_Latn",
            "backend": "korektor-rest",
        },
    ]
    scorer = RD.B.make_offline_scorer()
    records = RD.score_pairs(pairs, scorer)
    report = RD.build_delta_report(records)

    assert report["n"] == 3
    assert "median_delta" in report and "mean_delta" in report
    assert set(report["gate"]) == {"median_delta", "passed"}
    assert isinstance(report["gate"]["passed"], bool)
    assert "korektor-rest" in report["per_backend"]
    # band transitions are recorded as "before->after" keys
    assert all("->" in k for k in report["band_transitions"])
    # worst/best examples are score-ordered
    deltas = [r["delta"] for r in report["worst_examples"]]
    assert deltas == sorted(deltas)


def test_correction_improves_garbage_line_score():
    scorer = RD.B.make_offline_scorer()
    records = RD.score_pairs(
        [{"source_text": "rnn1 ww0rd vv_~~ qpqb", "corrected_text": "první slovo v řádku", "lang": "ces_Latn"}],
        scorer,
    )
    assert records[0]["delta"] > 0
    assert records[0]["changed"] is True


# ── Optional live smoke (network) ────────────────────────────────────────────


@pytest.mark.slow
def test_korektor_rest_live_smoke():
    backend = CO.KorektorRestBackend()
    try:
        out = backend.correct("Ahoj svete")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"korektor REST unreachable: {exc}")
    assert isinstance(out, str) and out
