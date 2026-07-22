"""
tests/test_text_inference.py – Unit tests for service/text_inference.py.

Heavy ML libraries (torch, transformers, fasttext) are imported lazily inside
TextModelManager.load_models(), so the module itself imports cleanly on CPU and
the classification helpers can be exercised with a mocked FastText model.
"""

from unittest.mock import MagicMock

from service.text_inference import TextModelManager, _classify_line


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
