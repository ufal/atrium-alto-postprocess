"""
tests/test_service_utils.py – Unit tests for service/utils.py: the pure ALTO
parsing / box-normalisation / legacy-categoriser helpers behind the FastAPI
/process endpoint.

Thresholds and COMMON_LANGS are imported from the module itself so the tests
track config_langID.txt rather than hard-coding stale numbers.
"""

import pytest

pytest.importorskip("lxml")

from service.utils import (  # noqa: E402
    COMMON_LANGS,
    LANG_SCORE_CLEAR,
    LANG_SCORE_ROUGH,
    PERPLEXITY_THRESHOLD_MAX,
    PERPLEXITY_THRESHOLD_MIN,
    categorize_line,
    normalize_boxes,
    parse_alto_xml,
)

_ALTO_BASIC = """<?xml version="1.0" encoding="UTF-8"?>
<alto><Layout><Page WIDTH="1000" HEIGHT="2000"><PrintSpace><TextLine>
<String CONTENT="Hello" HPOS="10" VPOS="20" WIDTH="50" HEIGHT="30"/>
<String CONTENT="World" HPOS="70" VPOS="20" WIDTH="60" HEIGHT="30"/>
</TextLine></PrintSpace></Page></Layout></alto>"""

_ALTO_HYP = """<?xml version="1.0" encoding="UTF-8"?>
<alto><Layout><Page WIDTH="1000" HEIGHT="2000"><TextLine>
<String CONTENT="Hyphen" HPOS="10" VPOS="20" WIDTH="50" HEIGHT="30"/>
<HYP CONTENT="-"/>
</TextLine></Page></Layout></alto>"""

_ALTO_SUBS = """<?xml version="1.0" encoding="UTF-8"?>
<alto><Layout><Page WIDTH="1000" HEIGHT="2000"><TextLine>
<String CONTENT="be" HPOS="10" VPOS="20" WIDTH="50" HEIGHT="30" SUBS_TYPE="HypPart1" SUBS_CONTENT="beautiful"/>
</TextLine></Page></Layout></alto>"""

_ALTO_XXE_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE alto [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>
<alto><Layout><Page WIDTH="100" HEIGHT="100"><TextLine>
<String CONTENT="&xxe;" HPOS="0" VPOS="0" WIDTH="10" HEIGHT="10"/>
</TextLine></Page></Layout></alto>"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── parse_alto_xml ──────────────────────────────────────────────────────────
def test_parse_basic_words_boxes_and_dims(tmp_path):
    words, boxes, dims = parse_alto_xml(_write(tmp_path, "a.xml", _ALTO_BASIC))
    assert words == ["Hello", "World"]
    assert boxes == [[10, 20, 60, 50], [70, 20, 130, 50]]
    assert dims == (1000, 2000)


def test_parse_appends_explicit_hyphen_tag(tmp_path):
    words, _, _ = parse_alto_xml(_write(tmp_path, "h.xml", _ALTO_HYP))
    assert words == ["Hyphen-"]


def test_parse_subs_hyppart1_expands_with_full_word(tmp_path):
    words, _, _ = parse_alto_xml(_write(tmp_path, "s.xml", _ALTO_SUBS))
    assert words == ["be- {beautiful}"]


def test_parse_missing_page_returns_empty(tmp_path):
    words, boxes, dims = parse_alto_xml(_write(tmp_path, "n.xml", "<alto><Layout/></alto>"))
    assert words == [] and boxes == [] and dims == (0, 0)


def test_parse_malformed_xml_returns_empty(tmp_path):
    words, boxes, dims = parse_alto_xml(_write(tmp_path, "bad.xml", "<alto><not-closed>"))
    assert words == [] and boxes == [] and dims == (0, 0)


def test_parse_does_not_resolve_external_entities_xxe(tmp_path):
    """Hardened parser must not load an external SYSTEM entity (XXE)."""
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET_XXE", encoding="utf-8")
    doc = _ALTO_XXE_TMPL.format(secret=secret)
    words, _, _ = parse_alto_xml(_write(tmp_path, "xxe.xml", doc))
    assert "TOPSECRET_XXE" not in " ".join(words)


# ── normalize_boxes ─────────────────────────────────────────────────────────
def test_normalize_scales_to_0_1000():
    out = normalize_boxes([[0, 0, 500, 1000]], width=1000, height=2000)
    assert out == [[0, 0, 500, 500]]


def test_normalize_clamps_to_1000():
    out = normalize_boxes([[0, 0, 5000, 5000]], width=1000, height=1000)
    assert out == [[0, 0, 1000, 1000]]


def test_normalize_empty_boxes_returns_empty():
    assert normalize_boxes([], 100, 100) == []


def test_normalize_zero_dimension_returns_zero_boxes():
    assert normalize_boxes([[1, 2, 3, 4]], width=0, height=100) == [[0, 0, 0, 0]]


# ── categorize_line (legacy fallback) ───────────────────────────────────────
def test_categorize_clear_for_common_lang_high_score():
    lang = COMMON_LANGS[0]
    out = categorize_line(lang, LANG_SCORE_CLEAR + 0.05, ppl=10.0, text="this is a clear sentence")
    assert out == "Clear"


def test_categorize_trash_on_hard_override():
    out = categorize_line(COMMON_LANGS[0], 0.9, ppl=PERPLEXITY_THRESHOLD_MAX + 100.0, text="garbage", weird_ratio=0.6)
    assert out == "Trash"


def test_categorize_noisy_between_thresholds():
    mid = (LANG_SCORE_ROUGH + LANG_SCORE_CLEAR) / 2
    out = categorize_line(
        COMMON_LANGS[0], mid, ppl=PERPLEXITY_THRESHOLD_MIN - 100.0, text="this is a longer readable line"
    )
    assert out == "Noisy"


def test_categorize_trash_for_low_score():
    out = categorize_line(COMMON_LANGS[0], 0.1, ppl=10.0, text="this is a longer readable line")
    assert out == "Trash"
