"""
tests/test_extract_llm_alto.py – Unit tests for the pure image/config helpers in
extract_LLM_ALTO_2_TXT.py.

That module imports torch + transformers at module scope, so these tests only run
where those libraries are fully installed (e.g. the GPU environment). We gate on
the *actual* module import rather than importorskip-ing individual dependency
names, because a partial/namespace install (e.g. a `transformers` on the path
without AutoConfig) would slip past importorskip and then hard-error at
collection. Trying the real import and skipping keeps the rest of the suite
runnable. Only the model-free helpers are covered here — the GLM inference path
needs a live checkpoint.
"""

import pytest

try:
    from extract_LLM_ALTO_2_TXT import _load_extract_config, resize_if_huge, trim_whitespace
except Exception as exc:  # torch / transformers / tqdm / pandas missing or partially installed
    pytest.skip(
        f"extract_LLM_ALTO_2_TXT dependencies unavailable: {exc}",
        allow_module_level=True,
    )

from PIL import Image  # noqa: E402  (PIL is a hard dependency of the module imported above)


def test_resize_if_huge_downscales_longest_side():
    out = resize_if_huge(Image.new("RGB", (4000, 2000), "white"), max_dim=1000)
    assert out.size == (1000, 500)


def test_resize_if_huge_keeps_small_image():
    out = resize_if_huge(Image.new("RGB", (300, 200), "white"), max_dim=1000)
    assert out.size == (300, 200)


def test_trim_whitespace_crops_to_content():
    img = Image.new("RGB", (200, 200), "white")
    for x in range(90, 110):
        for y in range(90, 110):
            img.putpixel((x, y), (0, 0, 0))
    out = trim_whitespace(img, padding=5)
    assert out.size[0] < 200 and out.size[1] < 200


def test_trim_whitespace_blank_image_unchanged():
    out = trim_whitespace(Image.new("RGB", (120, 120), "white"))
    assert out.size == (120, 120)


def test_load_extract_config_defaults_when_missing(tmp_path):
    cfg = _load_extract_config(str(tmp_path / "nope.txt"))
    assert cfg["model_path"] == "THUDM/glm-4v-9b"
    assert cfg["max_new_tokens"] == 4096


def test_load_extract_config_reads_overrides(tmp_path):
    cfgfile = tmp_path / "config_langID.txt"
    cfgfile.write_text(
        "[EXTRACT]\nLLM_MODEL = my/model\nLLM_MAX_NEW_TOKENS = 128\nWORKERS_MAX_LLM = 3\n",
        encoding="utf-8",
    )
    cfg = _load_extract_config(str(cfgfile))
    assert cfg["model_path"] == "my/model"
    assert cfg["max_new_tokens"] == 128
    assert cfg["max_workers"] == 3
