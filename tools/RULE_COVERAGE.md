# Rule-fire coverage & retirement criterion (B5)

## What this is

`rule_coverage_report.py` instruments the categorisation engine to answer a
question the ablation study cannot: **does a rule's action branch ever actually
execute on a given corpus?**

The ablation study (`run_ablation_study.py`) measures a rule's *decisive* effect
— lines that flip category when the rule is removed (LOO). On the self-labelled
corpus this measure suffers from **survivor entanglement**: two overlapping rules
each show zero flips alone because the other catches the line first. Neither
appears decisive, yet neither is dead.

Coverage instrumentation cuts through this by recording raw execution at the
action site (`_fire(name)` immediately before each `return` / penalty application
in `determine_category` / `categorize_line`). A rule with `fire_count == 0` is
**unreachable dead code** by definition — no entanglement can hide a rule that
never runs. That is the only gold-free, config-independent retirement criterion.

## Coverage columns

| Column           | Source                                                 | Meaning                                                            |
|------------------|--------------------------------------------------------|--------------------------------------------------------------------|
| `fire_count`     | `rule_fire_capture()` over one recategorize pass       | raw execution count                                                |
| `fire_rate`      | `fire_count / n_scored_lines`                          | fraction of scored lines that triggered this rule                  |
| `decisive_count` | LOO: `evaluate_dataframe` with `DISABLED_RULES={rule}` | lines whose category changes vs. stored categ when rule is removed |
| `clear_loss`     | confusion["Clear"]["Trash"] + ["Non-text"] in LOO run  | valid text destroyed if rule removed                               |
| `class`          | derived                                                | DEAD / REDUNDANT-HERE / LOAD-BEARING                               |

## Classification logic


```
fire_count == 0                        → DEAD           (unreachable; retire candidate)
fire_count > 0 AND decisive_count == 0 → REDUNDANT-HERE (entanglement; keep)
decisive_count > 0                     → LOAD-BEARING   (always keep)
```

`REDUNDANT-HERE` means the rule fires but is currently masked by an overlapping
rule in production order. It is **not safe to delete** even with `decisive_count
== 0`: the masking relationship depends on corpus and config. The rule is a real
guard that appears redundant only on this sample.

## Retirement criterion (gold-free)

A rule may be permanently deleted **only when all of these hold**:

1. `fire_count == 0` aggregated across the **full multi-collection corpus** (not
   just the smoke fixture). Run `rule_coverage_report.py` on the cluster with the
   production `DOC_LINE_CATEG` corpus.
2. The rule is **not** one of the cheap structural guards (`rule_inverted`,
   `rule_allcaps`, `rule_garbage_density`) unless coverage-empty across a
   broad, explicitly approved collection set — these guards cost ~nothing and
   protect against failure modes absent from small samples.
3. The deletion is reviewed and merged in a **separate commit** from the
   instrumentation; the commit message cites the full-corpus `fire_count` and
   the run provenance (date, corpus version, cluster job ID).

## Findings (Issue #5 Full Corpus Run)

During the #5 configuration map, `rule_coverage_report.py` was executed against the complete corpus.

* **Result: 0 DEAD rules.**
* 11 rules were `LOAD-BEARING`.
* 3 rules (`rule_allcaps`, `rule_garbage_density`, `rule_inverted`) were `REDUNDANT-HERE`.

**Conclusion:** No rule is unreachable. The greedy ablation study mislabeled the 3 structural guards as "PRUNE (fully redundant)" because they showed 0 LOO flips, but coverage proved they *do* fire (2, 20, and 7 times respectively). They are entangled, not dead. Under the gold-free criterion, **nothing is retired**. All 14 rules remain.

## Running the tool

```bash
# Smoke fixture (fast; proves the instrument works; results not authoritative)
python tools/rule_coverage_report.py \
    --input-dir data_samples/DOC_LINE_CATEG \
    --config config_langID.txt \
    --output rule_coverage_smoke.json

# Full corpus (cluster; authoritative for retirement decisions)
python tools/rule_coverage_report.py \
    --input-dir /path/to/full/DOC_LINE_CATEG \
    --config config_langID.txt \
    --output rule_coverage_full.json

# Coverage only (no LOO; faster when you only need fire counts)
python tools/rule_coverage_report.py \
    --input-dir data_samples/DOC_LINE_CATEG \
    --skip-loo
```

## Relationship to the ablation study

| Tool                              | Question answered                                          | Needs gold?           |
|-----------------------------------|------------------------------------------------------------|-----------------------|
| `run_ablation_study.py`           | Which rules are *decisive* on the self-labelled corpus?    | No (self-referential) |
| `rule_coverage_report.py`         | Which rules *ever execute*? Is a rule reachable dead code? | No (structural)       |
| *(future)* `build_label_queue.py` | Which lines should a human label to break self-reference?  | —                     |

The three tools are complementary. A rule can be:

* **Decisive but low-coverage**: fires rarely but changes critical outcomes
when it does → keep.
* **High-coverage but non-decisive**: fires often but is always masked by an
earlier rule → REDUNDANT-HERE; keep (entanglement; not safe to delete).
* **Zero coverage**: never fires → DEAD → retirement candidate after full-corpus
confirmation.

## Hook points for B1 (gold set — deferred)

When a gold label set becomes available, `evaluate_dataframe` gains an optional
`gold_column` path that scores **only labeled rows** against the human label
instead of the self-generated `categ`. The coverage report can then be re-run
with `--gold-column` to produce `decisive_count_gold` and `clear_loss_gold`
columns, converting the retirement criterion from structural to correctness-based.
