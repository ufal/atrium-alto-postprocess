import json

from tools.importance_consensus import calculate_consensus


def test_calculate_consensus(tmp_path):
    # Mock backend output directories
    dir1 = tmp_path / "rf_run"
    dir1.mkdir()
    (dir1 / "param_importance.json").write_text(
        json.dumps({"LOWPPL_CLEAR_MAX": 0.35, "MOSTLY_READABLE_VALID_MIN": 0.25, "CATEG_TRASH_SCORE_MAX": 0.05})
    )

    dir2 = tmp_path / "sobol_run"
    dir2.mkdir()
    (dir2 / "param_importance.json").write_text(
        json.dumps({"LOWPPL_CLEAR_MAX": 0.40, "CATEG_TRASH_SCORE_MAX": 0.02, "MOSTLY_READABLE_VALID_MIN": 0.30})
    )

    res = calculate_consensus([dir1, dir2], top_k=2)

    assert "consensus" in res
    consensus = res["consensus"]

    # LOWPPL_CLEAR_MAX and MOSTLY_READABLE_VALID_MIN should be robust
    robust_params = [c["param"] for c in consensus if c["is_robust"]]
    assert "LOWPPL_CLEAR_MAX" in robust_params
    assert "MOSTLY_READABLE_VALID_MIN" in robust_params
    assert "CATEG_TRASH_SCORE_MAX" not in robust_params  # Was not in top 2
