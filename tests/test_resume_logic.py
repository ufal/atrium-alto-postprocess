"""
Tests for the concurrency and resume logic in the classification orchestrator.
"""

import queue

import pandas as pd

from langID_classify import process_document


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
