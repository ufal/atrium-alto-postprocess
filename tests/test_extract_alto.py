"""
Tests for extract_ALTO_2_TXT.py pure-logic helpers.
"""
from extract_ALTO_2_TXT import _dehyphenate

def test_dehyphenate_standard_hyphen():
    text = "This is a split-\nword test."
    # The hyphen is removed and the fragments are joined without a space
    assert _dehyphenate(text) == "This is a splitword test.\n"

def test_dehyphenate_multiple_lines():
    text = "First line.\nSec-\nond line.\nThird."
    assert _dehyphenate(text) == "First line.\nSecond line.\nThird.\n"

def test_dehyphenate_no_hyphen():
    text = "Line one\nLine two"
    # Preserves normal line breaks
    assert _dehyphenate(text) == "Line one\nLine two\n"

def test_dehyphenate_typographical_hyphen_variants():
    # \xad (soft hyphen), \u2013 (en-dash), \u2014 (em-dash)
    text = "Soft\xadhyphen\nEn\u2013dash\nEm\u2014dash\n"
    assert _dehyphenate(text) == "Softhyphen\nEndash\nEmdash\n"