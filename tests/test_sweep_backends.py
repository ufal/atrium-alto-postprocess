import sys
import warnings
from unittest.mock import MagicMock

# Standard MagicMock is safe for pytest collection.
# Inject a fake class for torch.Tensor to satisfy scipy's internal isinstance() checks.
mock_torch = MagicMock()
mock_torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = mock_torch

# Mock tqdm and explicitly mock its submodule so Optuna doesn't crash on import
mock_tqdm = MagicMock()
sys.modules["tqdm"] = mock_tqdm
sys.modules["tqdm.auto"] = mock_tqdm

sys.modules["transformers"] = MagicMock()

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tools.const_importance_sweep import (  # noqa: E402
    run_morris_backend,
    run_optuna_backend,
    run_sklearn_backend,
    run_sobol_backend,
)  # noqa: E402


@pytest.fixture
def mock_sweep_env(tmp_path):
    # Added "page_num" and "line_num" so the document post-processing engine can sort the lines
    data = pd.DataFrame(
        [
            {
                "text": "A valid mostly readable line.",
                "perplex": 150.0,
                "categ": "Clear",
                "word_count": 5,
                "page_num": 1,
                "line_num": 1,
            },
            {
                "text": "XyZ123!@# Garbage",
                "perplex": 8000.0,
                "categ": "Trash",
                "word_count": 3,
                "page_num": 1,
                "line_num": 2,
            },
        ]
    )
    base_constants = {
        "CATEG_TRASH_SCORE_MAX": 0.40,
        "CATEG_NOISY_SCORE_MAX": 0.80,
        "SHORT_PPL_CAP": 500.0,
        "PERPLEXITY_THRESHOLD_MAX": 1000.0,
    }
    params = ["CATEG_TRASH_SCORE_MAX", "CATEG_NOISY_SCORE_MAX", "SHORT_PPL_CAP", "PERPLEXITY_THRESHOLD_MAX"]

    # Mock evaluate_kwargs explicitly
    eval_kwargs = {"expected_langs": ["ces"], "known_bases": frozenset(["ces", "eng"])}

    return {
        "data": data,
        "base_constants": base_constants,
        "params": params,
        "output_dir": tmp_path,
        "n_trials": 16,  # 16 is a power of 2, satisfying Sobol's requirement, and > 10 for Optuna/RF
        "seed": 42,
        "metric": "macro_f1",
        "direction": "maximize",
        "eval_kwargs": eval_kwargs,
    }


def test_sklearn_backend(mock_sweep_env):
    pytest.importorskip("sklearn")
    res = run_sklearn_backend(**mock_sweep_env)
    assert res["backend"] == "sklearn"
    assert "mdi_importance" in res


def test_optuna_backend(mock_sweep_env):
    pytest.importorskip("optuna")
    mock_sweep_env["sampler_name"] = "random"
    mock_sweep_env["storage"] = None
    mock_sweep_env["study_name"] = "test"
    res = run_optuna_backend(**mock_sweep_env)
    assert res["backend"] == "optuna"
    # Depending on variance, it either computes importance or safely skips it
    assert "fanova_importance" in res or "importance_skipped" in res


def test_morris_backend(mock_sweep_env):
    pytest.importorskip("SALib")
    res = run_morris_backend(**mock_sweep_env)
    assert res["backend"] == "morris"
    assert "morris_importance" in res


def test_sobol_backend(mock_sweep_env):
    pytest.importorskip("SALib")
    # Sobol requires at least one constraint-free param, let's add one to params
    mock_sweep_env["params"].append("LOWPPL_CLEAR_MAX")
    mock_sweep_env["base_constants"]["LOWPPL_CLEAR_MAX"] = 60.0

    with warnings.catch_warnings():
        # SALib emits several warnings on zero-variance mock data and older numpy conversions.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        res = run_sobol_backend(**mock_sweep_env)

    assert res["backend"] == "sobol"
    assert "sobol_ST" in res
    assert "sobol_S1" in res
