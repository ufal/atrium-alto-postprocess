"""Quality-score fine-tuning tooling (issue #23).

A data factory + training harness that distils the hand-crafted OCR line-quality
engine (FastText + Qwen2.5-0.5B perplexity + the 9-signal weighted score) into a
single small regression model. See ``agent_dev_logs/plans/23.plan.md`` for the
full strategy and ``tools/quality_model/README.md`` for usage.

Design invariant (from ``tools/SWEEP_NOTES.md``): there is ONE scoring engine.
Every module here reuses the production ``text_util_langID`` /
``langID_classify`` functions — it never re-implements the score.
"""
