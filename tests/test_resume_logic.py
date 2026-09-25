"""
Tests for the concurrency and resume logic in the classification orchestrator.
"""

import queue

import pandas as pd

from classify_TEXT import process_document


def test_process_document_resume_skips_existing(tmp_path):
    """Ensure process_document skips files that already have an output CSV."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    file_id = "test_doc_001"
    out_file = out_dir / f"{file_id}.csv"
    out_file.write_text("dummy,csv,content\n")  # Mock an existing output from a previous run

    # Dummy inputs for the process_document task tuple
    group = pd.DataFrame()
    text_dir = tmp_path / "text"
    q = queue.Queue()

    task = (file_id, group, text_dir, out_dir, 128, q, {}, ["ces"], ["deu"], None)

    result = process_document(task)

    assert result["status"] == "skipped"
    assert result["reason"] == "output already exists (resume)"


# ── (#31 Phase 5) resume only while the output is current ────────────────────


def _age(path, seconds):
    import os

    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns - seconds * 10**9, st.st_mtime_ns - seconds * 10**9))


def _task(tmp_path, file_id="doc1", pages=(1,)):
    text_dir = tmp_path / "text"
    (text_dir / file_id).mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir(exist_ok=True)
    group = pd.DataFrame({"file": [file_id] * len(pages), "page": list(pages)})
    return text_dir, out_dir, (file_id, group, str(text_dir), str(out_dir), 128, queue.Queue(), {}, ["ces"], [], None)


def test_resume_skips_while_the_output_is_newer_than_every_page_text(tmp_path):
    text_dir, out_dir, task = _task(tmp_path)
    page = text_dir / "doc1" / "doc1-1.txt"
    page.write_text("12345\n", encoding="utf-8")
    _age(page, 60)
    (out_dir / "doc1.csv").write_text("old", encoding="utf-8")

    result = process_document(task)

    assert result["status"] == "skipped"
    assert (out_dir / "doc1.csv").read_text(encoding="utf-8") == "old"


def test_a_page_text_newer_than_the_output_is_classified_again(tmp_path):
    """A re-ingested document used to keep its old categories: the output existed."""
    text_dir, out_dir, task = _task(tmp_path)
    out = out_dir / "doc1.csv"
    out.write_text("stale,rows\n", encoding="utf-8")
    _age(out, 60)
    # Only fast-track lines (no perplexity model needed): digits are Non-text.
    (text_dir / "doc1" / "doc1-1.txt").write_text("12345 678\n", encoding="utf-8")

    result = process_document(task)

    assert result["status"] == "success", result
    assert result["reclassified"] is True
    df = pd.read_csv(out, dtype={"file": str})
    assert "stale" not in out.read_text(encoding="utf-8")
    assert list(df["file"]) == ["doc1"] and list(df["page_num"]) == [1]


def test_outputs_outside_the_index_are_listed_not_deleted(tmp_path):
    from classify_TEXT import outputs_outside_the_index

    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "gone.csv").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert outputs_outside_the_index(tmp_path, ["a"]) == ["gone.csv"]
    assert (tmp_path / "gone.csv").exists()
    assert outputs_outside_the_index(tmp_path / "missing", ["a"]) == []
