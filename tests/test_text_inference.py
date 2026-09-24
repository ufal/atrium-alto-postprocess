"""
tests/test_text_inference.py – Unit tests for service/text_inference.py.

Heavy ML libraries (torch, transformers, fasttext) are imported lazily inside
TextModelManager.load_models(), so the module itself imports cleanly on CPU and
the classification helpers can be exercised with a mocked FastText model.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("lxml")

from service.text_inference import TextModelManager, _classify_line  # noqa: E402

_ALTO_TWO_LINES_WITH_HYPHEN = """<?xml version="1.0" encoding="UTF-8"?>
<alto><Layout><Page WIDTH="1000" HEIGHT="2000"><PrintSpace>
<TextLine>
<String CONTENT="be" HPOS="10" VPOS="20" WIDTH="30" HEIGHT="30" SUBS_TYPE="HypPart1" SUBS_CONTENT="beautiful"/>
</TextLine>
<TextLine>
<String CONTENT="autiful" HPOS="10" VPOS="60" WIDTH="60" HEIGHT="30"/>
<SP/>
<String CONTENT="thing" HPOS="80" VPOS="60" WIDTH="50" HEIGHT="30"/>
</TextLine>
</PrintSpace></Page></Layout></alto>"""


def _mock_ft(lang="ces", score=0.95):
    ft = MagicMock()
    ft.predict.return_value = ([[f"__label__{lang}"]], [[score]])
    return ft


def test_manager_init_defaults():
    m = TextModelManager()
    assert m.device == "cpu"
    assert m.layout_model is None
    assert m.ft_model is None
    assert m._models_loaded is False


def test_load_models_early_return_when_already_loaded():
    """The guard must short-circuit before the deferred `import torch`."""
    m = TextModelManager()
    m._models_loaded = True
    m.device = "sentinel"
    m.load_models()
    assert m.device == "sentinel"
    assert m._models_loaded is True


def test_classify_line_full_pipeline_returns_all_fields():
    out = _classify_line(
        "this is a readable line of text",
        90.0,
        ft_model=_mock_ft("ces", 0.97),
        ppl_model=None,
        tokenizer=None,
        device="cpu",
    )
    for key in ("text", "lang", "lang_score", "perplexity", "garbage_density", "quality_score", "category"):
        assert key in out
    assert out["lang"] == "ces"
    assert isinstance(out["category"], str) and out["category"]


def _manager_with_mocked_ft():
    m = TextModelManager()
    m.ft_model = _mock_ft("ces", 0.9)
    m.ppl_model = MagicMock()
    m.ppl_tokenizer = MagicMock()
    m.device = "cpu"
    return m


def _patched_ppl(monkeypatch, value=100.0):
    """calculate_perplexity_batch needs a real model/tokenizer; patch it out
    with a fixed-value stand-in so these tests don't need torch/transformers."""
    monkeypatch.setattr(
        "service.text_inference.calculate_perplexity_batch",
        lambda texts, model, tokenizer, device: [value] * len(texts),
    )


def test_process_text_file_returns_one_entry_per_nonempty_line(tmp_path, monkeypatch):
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("First line\n\nSecond line\n", encoding="utf-8")

    result = m.process_text_file(str(txt_path))

    assert result["type"] == "plain_text"
    assert [c["text"] for c in result["cleaned_lines"]] == ["First line", "Second line"]
    assert [c["line_num"] for c in result["cleaned_lines"]] == [1, 2]


def test_process_json_extracts_target_keys_only(tmp_path, monkeypatch):
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()

    json_path = tmp_path / "doc.json"
    json_path.write_text(
        json.dumps({"metadata": {"engine": "X"}, "page": {"lines": [{"textline": "Hello"}, {"textline": "World"}]}}),
        encoding="utf-8",
    )

    result = m.process_json(str(json_path))

    assert result["type"] == "json"
    assert [c["text"] for c in result["cleaned_lines"]] == ["Hello", "World"]


def test_process_document_keeps_pages_and_restarts_line_numbers(tmp_path, monkeypatch):
    """(#31) Any other text-bearing upload: pages kept, blank lines dropped, line_num per page."""
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()

    txt_path = tmp_path / "doc.md"
    txt_path.write_text("# First page\n\nSecond line\n\fOther page\n", encoding="utf-8")

    result = m.process_document(str(txt_path))

    assert result["type"] == "document"
    assert result["format"] == "md"
    assert result["origin"] == "ocr:generic"
    assert [(c["page"], c["line_num"], c["text"]) for c in result["cleaned_lines"]] == [
        ("1", 1, "First page"),
        ("1", 2, "Second line"),
        ("2", 1, "Other page"),
    ]
    assert [p["lines"] for p in result["pages"]] == [2, 1]


def test_process_document_batches_perplexity_calls(tmp_path, monkeypatch):
    """(#31) A long document must not become one perplexity batch."""
    import service.text_inference as ti

    calls = []

    def _fake_ppl(texts, *_a, **_k):
        calls.append(len(texts))
        return [50.0] * len(texts)

    monkeypatch.setattr(ti, "calculate_perplexity_batch", _fake_ppl)
    m = _manager_with_mocked_ft()
    (tmp_path / "long.txt").write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")

    result = m.process_document(str(tmp_path / "long.txt"))

    assert len(result["cleaned_lines"]) == 300
    assert max(calls) <= ti.DOCUMENT_BATCH_LINES


def test_process_alto_reorders_and_dehyphenates_across_lines(tmp_path, monkeypatch):
    """layout_model=None (the class default) forces the document-order
    fallback deterministically, regardless of whether v3.helpers happens to be
    importable in this environment — see process_alto's boxes2inputs/
    layout_model guard."""
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()
    assert m.layout_model is None

    xml_path = tmp_path / "doc.xml"
    xml_path.write_text(_ALTO_TWO_LINES_WITH_HYPHEN, encoding="utf-8")

    result = m.process_alto(str(xml_path))

    assert result["type"] == "alto_xml"
    # The word split across the line break ("be-{beautiful}" / "autiful")
    # must be reconstructed into one word, not duplicated or left broken.
    texts = [c["text"] for c in result["cleaned_lines"]]
    assert texts == ["beautiful", "thing"]


def test_process_alto_empty_document_returns_no_lines(tmp_path, monkeypatch):
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()

    xml_path = tmp_path / "empty.xml"
    xml_path.write_text('<alto><Layout><Page WIDTH="100" HEIGHT="100"/></Layout></alto>', encoding="utf-8")

    result = m.process_alto(str(xml_path))
    assert result == {"type": "alto_xml", "cleaned_lines": []}


def test_process_alto_uses_layout_reader_when_model_available(tmp_path, monkeypatch):
    """When boxes2inputs and layout_model are both present, process_alto must
    call the LayoutReader reordering path instead of the fallback."""
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()
    m.layout_model = MagicMock()

    xml_path = tmp_path / "doc.xml"
    xml_path.write_text(_ALTO_TWO_LINES_WITH_HYPHEN, encoding="utf-8")

    with (
        patch("service.text_inference.boxes2inputs", MagicMock()),
        patch(
            "service.text_inference._run_layout_reader",
            return_value=(["thing"], [[0, 0, 10, 10]]),
        ) as mock_reader,
    ):
        result = m.process_alto(str(xml_path))

    mock_reader.assert_called_once()
    assert [c["text"] for c in result["cleaned_lines"]] == ["thing"]


# ── (#31 Phase 4) the service reads uploads like the batch path ───────────────


@pytest.fixture
def ingest_config(tmp_path, monkeypatch):
    """Point LANGID_CONFIG at a scratch config and reset the cached settings around the test."""
    import service.text_inference as ti

    def write(text):
        path = tmp_path / "config.txt"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("LANGID_CONFIG", str(path))
        ti.ingest_settings.cache_clear()

    ti.ingest_settings.cache_clear()
    yield write
    ti.ingest_settings.cache_clear()


def test_process_text_file_decodes_legacy_encodings(tmp_path, monkeypatch, ingest_config):
    ingest_config("[TEXT_INGEST]\n")
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()
    path = tmp_path / "cp1250.txt"
    path.write_bytes("Zpráva o sondě\n\nčíslo tři\n".encode("cp1250"))
    result = m.process_text_file(str(path))
    assert result["type"] == "plain_text"
    assert [c["text"] for c in result["cleaned_lines"]] == ["Zpráva o sondě", "číslo tři"]


def test_process_text_file_refuses_binary(tmp_path, ingest_config):
    from text_formats import IngestError

    ingest_config("[TEXT_INGEST]\n")
    path = tmp_path / "blob.txt"
    path.write_bytes(bytes(range(256)) * 8)
    with pytest.raises(IngestError) as info:
        _manager_with_mocked_ft().process_text_file(str(path))
    assert info.value.code == "binary_content"


def test_uploads_follow_the_configured_ingest_settings(tmp_path, monkeypatch, ingest_config):
    ingest_config("[TEXT_INGEST]\nMAX_LINE_CHARS = 12\n\n[DOCUMENT]\nSOURCE_ORIGIN_BY_KIND = md = ocr:tesseract\n")
    _patched_ppl(monkeypatch)
    m = _manager_with_mocked_ft()
    (tmp_path / "d.md").write_text("jedna dva tři čtyři pět\n", encoding="utf-8")
    result = m.process_document(str(tmp_path / "d.md"))
    assert [c["text"] for c in result["cleaned_lines"]] == ["jedna dva", "tři čtyři", "pět"]
    assert result["origin"] == "ocr:tesseract"


def test_a_bad_ingest_setting_raises_naming_the_key(ingest_config):
    import service.text_inference as ti

    ingest_config("[TEXT_INGEST]\nNOTES = margin\n")
    with pytest.raises(ValueError, match="NOTES"):
        ti.ingest_settings()


def test_process_document_without_a_text_line_is_no_text(tmp_path, ingest_config):
    from text_formats import IngestError

    ingest_config("[TEXT_INGEST]\n")
    (tmp_path / "blank.md").write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(IngestError) as info:
        _manager_with_mocked_ft().process_document(str(tmp_path / "blank.md"))
    assert info.value.code == "no_text"
