"""
tests/test_gpu_concurrency.py
Verifies the concurrency guards and GPU crash resilience for the orchestrator.
"""

import multiprocessing as mp

import pytest

from langID_classify import process_and_write_batch_cpu, worker_models


class DummyFT:
    """Mock FastText model to satisfy the CPU worker before it hits the GPU wait loop."""

    def predict(self, texts, k=1):
        return [["__label__ces"] for _ in texts], [[0.95] for _ in texts]


@pytest.mark.slow
def test_cpu_worker_aborts_on_gpu_dead_signal():
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()
    gpu_dead = manager.Event()

    # Simulate a fatal OOM or model-load crash in the GPU worker
    gpu_dead.set()

    # Inject the dummy model to prevent KeyError on worker_models["ft"]
    worker_models["ft"] = DummyFT()

    # meta tuples must be 7-element: (file_id, page_id, line_num, text, split_ws, split_we, original_text)
    # The original_text field was added in the #3 patch and must match the unpacking in
    # process_and_write_batch_cpu:
    #   file_id, page_id, line_num, text_content, split_ws, split_we, original_text = meta[i]
    with pytest.raises(RuntimeError, match="GPU inference worker is down"):
        process_and_write_batch_cpu(
            batch_id="test_batch_1",
            lines=["Test line one"],
            meta=[("file_1", "page_1", 1, "Test line one", "", "", "Test line one")],
            out_dir=None,
            task_queue=task_queue,
            result_dict=result_dict,
            expected_langs=["ces"],
            trusted_langs=["deu"],
            gpu_dead=gpu_dead,
        )
