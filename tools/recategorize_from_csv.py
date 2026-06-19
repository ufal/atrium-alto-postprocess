#!/usr/bin/env python3
"""
tools/recategorize_from_csv.py
==============================
Offline re-scorer for #3 calibration — re-runs the categorisation logic over an
already-produced ``DOC_LINE_CATEG`` CSV **without** FastText or the GPU
perplexity model, by reusing the stored signals:

    * ``perplex``                     — the GPU perplexity (frozen)
    * ``original_lang`` / ``orig_lang_score`` — the raw FastText output (frozen)
    * ``text`` / ``original_text``    — the cleaned and pre-repair lines

Everything downstream of those — ``remap_lang`` (now a CAP, #3 A1), the structural
detectors, ``compute_quality_score``, the per-line ``categorize_line`` (now fed the
post-cap score + gibberish flag, #3 A2/B), and the document-level
``apply_document_postprocessing`` (now with the page-majority arm, #3 A3) — is
recomputed with the CURRENT code. Because the heavy models are frozen, this lets
you re-measure the effect of a calibration change on real output in seconds and on
CPU only, with ZERO drift from production: the same `categorize_line` and the same
`apply_document_postprocessing` helper are imported, not re-implemented.

It is a throwaway measurement aid, not part of the pipeline.

Usage
-----
    # re-score one CSV, write the new version next to it and print a diff report
    python tools/recategorize_from_csv.py data_samples/DOC_LINE_CATEG/CTX0001.csv

    # re-score a whole directory of per-document CSVs
    python tools/recategorize_from_csv.py data_samples/DOC_LINE_CATEG/ --out /tmp/rescored

    # report only (do not write re-scored CSVs)
    python tools/recategorize_from_csv.py data_samples/DOC_LINE_CATEG/ --report-only

Caveats
-------
* Fast-track rows (Empty / Non-text written by ``pre_filter_line`` with
  ``quality_score == 0`` and ``word_count == 0``) are passed through unchanged —
  they never went through the scoring path, so re-scoring them would be wrong.
* The re-scorer reflects ONLY logic reachable from the frozen signals. It cannot
  re-derive anything that depended on the live FastText label distribution beyond
  the single stored top-1 guess (which is all the production path used anyway).
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable when run as `python tools/recategorize_from_csv.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langID_classify import (  # noqa: E402
    CSV_HEADER,
    apply_document_postprocessing,
)
from text_util_langID import (  # noqa: E402
    SHORT_PPL_CAP,
    _lang_base,
    categorize_line,
    compute_garbage_density,
    compute_quality_score,
    compute_rotatable_ratio,
    compute_valid_ratio,
    compute_vowel_ratio,
    compute_word_weird_ratio,
    detect_fused_words,
    detect_gibberish_words,
    detect_letter_digit_letter,
    detect_mid_uppercase,
    detect_repeated_chars,
    detect_wx_words,
    is_all_caps_line,
    remap_lang,
    score_words_in_line,
)

_NUMERIC = {"perplex", "orig_lang_score"}


def _load_lang_config(config_path: str):
    """Resolve EXPECTED_LANGS / TRUSTED_FOREIGN_LANGS exactly as langID_classify.main."""
    config = configparser.ConfigParser()
    config.read(config_path)
    expected = [s.strip() for s in
                config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng").split(",")
                if s.strip()]
    trusted = [s.strip() for s in
               config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS",
                          fallback="deu,eng,fra,pol,ita").split(",")
               if s.strip()]
    known_bases = frozenset(_lang_base(code) for code in (trusted + expected))
    return expected, known_bases


def _is_fast_track(row) -> bool:
    """Empty / Non-text rows written by the pre-filter never carry scores."""
    try:
        wc = int(row.get("word_count", 0) or 0)
    except (ValueError, TypeError):
        wc = 0
    return row.get("categ") in ("Empty", "Non-text") and wc == 0


def _rescore_row(row: dict, expected_langs, known_bases) -> dict:
    """Recompute one previously-scored line from its frozen signals."""
    text_content = str(row.get("text", "") or "")
    original_text = str(row.get("original_text", "") or "")
    wc = len(text_content.split())
    cc = len(text_content)

    original_lang = str(row.get("original_lang", "") or "")
    try:
        original_lang_score = float(row.get("orig_lang_score", 0.0) or 0.0)
    except (ValueError, TypeError):
        original_lang_score = 0.0

    # (#3 A1) remap CAP on the frozen raw FastText guess.
    new_lang, new_score = remap_lang(
        original_lang, original_lang_score, known_bases,
        expected_langs[0] if expected_langs else "ces",
    )

    try:
        ppl_val = float(row.get("perplex", 0.0) or 0.0)
    except (ValueError, TypeError):
        ppl_val = 0.0
    if wc <= 2 and ppl_val > SHORT_PPL_CAP:
        ppl_val = SHORT_PPL_CAP

    g_density = compute_garbage_density(original_text)
    vowel_ratio = compute_vowel_ratio(original_text)
    upper_count = detect_mid_uppercase(text_content)
    rep_count = detect_repeated_chars(text_content)
    fuse_count = detect_letter_digit_letter(text_content)
    fused_words = detect_fused_words(text_content)
    gibb_count = detect_gibberish_words(text_content)
    wx_count = detect_wx_words(text_content)
    rot_ratio = compute_rotatable_ratio(text_content)
    caps_header = is_all_caps_line(text_content)
    weird_ratio = compute_word_weird_ratio(score_words_in_line(text_content))
    valid_ratio = compute_valid_ratio(text_content)

    q_score = compute_quality_score(
        valid_word_ratio=valid_ratio, perplexity=ppl_val, text_length=cc,
        weird_ratio=weird_ratio, vowel_ratio=vowel_ratio, garbage_density=g_density,
        lang_score=original_lang_score,
        gibberish_ratio=(gibb_count + wx_count) / max(wc, 1),
        fused_ratio=fused_words / max(wc, 1), rot_ratio=rot_ratio,
    )

    # (#3 A2/B) post-cap score + gibberish flag into the categoriser.
    categ, q_score, reason = categorize_line(
        q_score, text_content, wc, vowel_ratio, ppl_val,
        rot_ratio=rot_ratio, weird_ratio=weird_ratio, return_reason=True,
        valid_word_ratio=valid_ratio, lang_score=new_score,
        gibberish_present=(gibb_count + wx_count) > 0,
    )

    out = dict(row)  # keep any columns we do not recompute
    out.update({
        "categ": categ, "quality_score": f"{q_score:.4f}",
        "lang": new_lang, "lang_score": f"{new_score:.4f}",
        "original_lang": original_lang, "orig_lang_score": f"{original_lang_score:.4f}",
        "perplex": f"{ppl_val:.2f}", "word_count": wc, "char_count": cc,
        "garbage_density": f"{g_density:.4f}", "upper": upper_count, "repeated": rep_count,
        "ldl_fuses": fuse_count, "fused_words": fused_words, "gibberish": gibb_count,
        "weird_wx": wx_count, "word_weird": f"{weird_ratio:.4f}",
        "vowel_ratio": f"{vowel_ratio:.4f}", "rot_ratio": f"{rot_ratio:.4f}",
        "caps_header": caps_header,
        "allcaps_novowel": reason == "allcaps_novowel",
        "lowppl_clear": reason == "lowppl_clear",
        "cleanprose_clear": reason == "cleanprose_clear",
        "trash_threshold": reason == "trash_threshold",
        "noisy_threshold": reason == "noisy_threshold",
        "clear_threshold": reason == "clear_threshold",
        # post-pass flags are recomputed by apply_document_postprocessing below
        "pp_dedup": False, "pp_surrounded_trash": False, "pp_inverted_run": False,
    })
    return out


def rescore_csv(in_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (old_df, new_df) for one per-document CSV."""
    old = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    expected_langs, known_bases = _load_lang_config(
        os.getenv("LANGID_CONFIG", str(_ROOT / "config_langID.txt")))

    rows = []
    for _, r in old.iterrows():
        rd = r.to_dict()
        if _is_fast_track(rd):
            rows.append(rd)            # pass through untouched
        else:
            rows.append(_rescore_row(rd, expected_langs, known_bases))

    new = pd.DataFrame(rows)
    # Re-run the document-level smoothing with the CURRENT helper (#3 A3).
    if not new.empty:
        new = apply_document_postprocessing(new)
        # Keep canonical column order where possible.
        cols = [c for c in CSV_HEADER if c in new.columns]
        cols += [c for c in new.columns if c not in cols]
        new = new[cols]
    return old, new


def _category_counts(df: pd.DataFrame):
    if "categ" not in df.columns:
        return {}
    return df["categ"].value_counts().to_dict()


def _report(in_path: Path, old: pd.DataFrame, new: pd.DataFrame) -> int:
    """Print a before/after report; return the number of changed lines."""
    oc, nc = _category_counts(old), _category_counts(new)
    cats = sorted(set(oc) | set(nc))
    print(f"\n=== {in_path.name} ({len(old)} lines) ===")
    print(f"  {'category':<10} {'before':>7} {'after':>7} {'delta':>7}")
    for c in cats:
        b, a = oc.get(c, 0), nc.get(c, 0)
        print(f"  {c:<10} {b:>7} {a:>7} {a - b:>+7}")

    changed = 0
    if "categ" in old.columns and "categ" in new.columns and len(old) == len(new):
        old_cat = old["categ"].reset_index(drop=True)
        new_cat = new["categ"].reset_index(drop=True)
        diff_mask = old_cat != new_cat
        changed = int(diff_mask.sum())
        if changed:
            print(f"  --- {changed} line(s) changed category ---")
            txt = new["text"].reset_index(drop=True) if "text" in new.columns else None
            shown = 0
            for i in diff_mask[diff_mask].index:
                snippet = (str(txt.iloc[i])[:48] if txt is not None else "")
                print(f"    L{i:<4} {old_cat.iloc[i]:>8} -> {new_cat.iloc[i]:<8} | {snippet}")
                shown += 1
                if shown >= 25:
                    print(f"    … (+{changed - shown} more)")
                    break
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline re-scorer for #3 calibration.")
    ap.add_argument("path", help="DOC_LINE_CATEG CSV file or a directory of them")
    ap.add_argument("--out", help="output file/dir (default: overwrite in place)")
    ap.add_argument("--report-only", action="store_true",
                    help="print the diff report but do not write re-scored CSVs")
    args = ap.parse_args(argv)

    in_path = Path(args.path)
    if in_path.is_dir():
        csvs = sorted(in_path.glob("*.csv"))
    else:
        csvs = [in_path]
    if not csvs:
        print(f"No CSV files found at {in_path}", file=sys.stderr)
        return 1

    total_changed = 0
    grand_old: dict = {}
    grand_new: dict = {}
    for csv_path in csvs:
        old, new = rescore_csv(csv_path)
        total_changed += _report(csv_path, old, new)
        for k, v in _category_counts(old).items():
            grand_old[k] = grand_old.get(k, 0) + v
        for k, v in _category_counts(new).items():
            grand_new[k] = grand_new.get(k, 0) + v

        if not args.report_only:
            if args.out:
                out_path = Path(args.out)
                if in_path.is_dir():
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path = out_path / csv_path.name
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = csv_path
            new.to_csv(out_path, index=False, encoding="utf-8")

    if len(csvs) > 1:
        print("\n=== GRAND TOTAL ===")
        cats = sorted(set(grand_old) | set(grand_new))
        for c in cats:
            b, a = grand_old.get(c, 0), grand_new.get(c, 0)
            print(f"  {c:<10} {b:>7} {a:>7} {a - b:>+7}")
        print(f"  total lines changed category: {total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())