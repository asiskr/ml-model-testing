# ml-practice

[![Tests](https://github.com/asiskr/ml-practice/actions/workflows/tests.yml/badge.svg)](https://github.com/asiskr/ml-practice/actions/workflows/tests.yml)

Titanic survival prediction with a **26-test pytest suite around it** — schema
validation, data quality gates, and model behaviour tests, running in CI.

Built by a QA engineer. The model is the easy part and it is not the point:
any tutorial gets you to 81% on this dataset. The point is everything that
tells you when that 81% has quietly stopped being true.

---

## The problem

Testing an API is straightforward: send a request, assert the response equals
what you expected. Testing a model is not, and three things break the habit:

1. **There is no single correct answer.** A model that predicts one passenger
   wrong is not broken. A model that predicts 30% of them wrong is. The
   assertion has to be a threshold, not an equality.
2. **It fails silently.** A model given a `salary` column full of nulls does
   not crash — it trains and returns confident, wrong predictions. There is no
   stack trace to catch.
3. **The defect is usually upstream.** By the time accuracy drops, the cause is
   normally in the data, not the model. So the data gets its own tests, and
   they run first.

---

## What is tested

| Layer | Tests | What it protects |
|---|---|---|
| Schema | 7 | column types, allowed values, ranges — before anything trains |
| Data quality | 8 | row count, uniqueness, duplicates, missing-value rate |
| Model | 11 | accuracy floor, overfitting gap, determinism, contract, behaviour |

### The model tests

| Test | Type | Catches |
|---|---|---|
| `test_model_accuracy_above_threshold` | performance | model degraded below the 75% gate |
| `test_model_not_overfitting` | performance | train/test gap widened past 15% |
| `test_model_deterministic` | reproducibility | same input returning different predictions |
| `test_prediction_shape` | contract | wrong row count, or a label outside {0, 1} |
| `test_single_row_prediction` | edge case | breaks when given one row instead of a batch |
| `test_model_rejects_wrong_columns` | contract | silently accepting a missing feature |
| `test_duplicate_row_same_prediction` | invariance | identical inputs scoring differently |
| `test_first_class_higher_survival` | directional | the learned relationship inverting |
| `test_random_forest_beats_decision_tree` | comparison | the chosen model losing to the simpler one |
| `test_golden_predictions` | golden | five pinned predictions changing at all |
| `test_no_regression` | regression | accuracy dropping below the recorded baseline |

---

## Project structure

```
ml-practice/
├── .github/workflows/tests.yml     CI — runs the suite on every push and PR
├── baseline.json                   recorded accuracy the regression test compares against
├── conftest.py                     shared fixtures (data, split, trained model)
├── pytest.ini                      markers and test paths
├── data/
│   └── train.csv                   Titanic dataset, 891 rows
├── src/
│   ├── config.py                   paths, features, split settings, quality gates
│   ├── load_data.py                load, explore, split features from target
│   ├── clean_data.py               fill missing values, one-hot encode
│   ├── train_model.py              split, train, evaluate
│   ├── schema.py                   pandera schema — single source of truth
│   ├── data_quality.py             null %, duplicates, ranges, zero fares
│   ├── save_baseline.py            regenerates baseline.json
│   └── main.py                     the pipeline end to end
├── demos/
│   ├── metrics_demo.py             precision, recall, F1 on a hand-made set
│   ├── imbalanced_demo.py          why 90% accuracy can mean a useless model
│   ├── overfitting_demo.py         train vs test gap, with and without a depth limit
│   └── leakage_demo.py             preprocessing before the split vs after
└── tests/
    ├── test_schema.py              schema accepts good data, rejects bad
    ├── test_data_quality.py        counts, uniqueness, ranges, missing limits
    └── test_model.py               the 11 model tests above
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
pytest
```

Markers split the suite by layer — data first, since model results are
meaningless if the data is wrong:

```bash
pytest -m schema
```

```bash
pytest -m quality
```

```bash
pytest -m model
```

Everything else runs as a module from the project root:

```bash
python -m src.main
```

```bash
python -m src.data_quality
```

```bash
python -m demos.overfitting_demo
```

Re-record the regression baseline after a deliberate model change:

```bash
python -m src.save_baseline
```

---

## Results

| Model | Test accuracy |
|---|---|
| Decision Tree (unlimited depth) | 78.77% |
| Random Forest (unlimited depth) | 81.56% |
| Random Forest (`max_depth=3`) | 79.89% — the tested configuration |

Random Forest confusion matrix: TN 91, FP 14, FN 19, TP 55 — precision 79.71%,
recall 74.32%, F1 76.92%.

Recall matters more than precision here. A false negative is a passenger the
model writes off as dead who actually survived; on the medical version of this
problem that is the patient sent home.

The depth-3 forest scores slightly lower than the unlimited one and is still
the configuration under test — see below.

---

## Design decisions

**The tested model is not the highest-scoring one.** The unlimited Random
Forest scores 81.56% but memorises the training set: 98.03% train against
81.56% test, a 16.47% gap. Capped at depth 3 it scores 79.89% with a gap small
enough to pass the overfitting gate. Giving up 1.7 points of accuracy for a
model that behaves the same on data it has not seen is the right trade, and
`test_model_not_overfitting` is what enforces it.

**Assertions carry the actual numbers.** `assert accuracy >= 0.75` prints
`assert 0.71 >= 0.75` and nothing else. Every assertion here passes a message
with the real values, so a CI failure tells you how far off it was without
re-running anything locally.

**Quality gates live in `config.py`.** `MIN_ACCURACY`, `MAX_OVERFIT_GAP` and
`REGRESSION_TOLERANCE` sit next to the split settings, so the entire contract
the model must satisfy is readable in one twelve-line file instead of being
scattered across assertions.

**The regression baseline is JSON, not a pickled model.** Only one number is
needed to detect a regression. Pickling the whole model ties the test to the
exact scikit-learn version that wrote the file, and CI installs a fresh one —
so the test would eventually fail for reasons that have nothing to do with the
model. JSON also shows up in a diff: a changed baseline is visible in review
rather than hidden inside a binary.

**A directional test is worth more than another accuracy test.** Accuracy is an
average, and averages hide inversions. `test_first_class_higher_survival` takes
one passenger, flips only `Pclass` from 1 to 3, and asserts the survival
probability drops. If a preprocessing bug reverses the encoding, overall
accuracy can barely move while the model has learned the opposite relationship.
This is the test most likely to catch a real defect.

**The golden test is deliberately brittle.** It pins five exact predictions,
so any library upgrade or preprocessing change that shifts model output makes
it fail. That is the intent — it is a tripwire, and a fixed `random_state` is
what makes it possible.

**The leakage demo reports a 0.00% difference, and that is the finding.**
Filling `Age` with the mean of the whole dataset before splitting is textbook
leakage, but on 891 rows with a tree model the effect is invisible. The useful
conclusion is the uncomfortable one: leakage does not reliably announce itself
as a suspicious accuracy jump, so it cannot be detected by watching the score.
It has to be caught by reading the order of operations in the pipeline.

---

## What I would add next

- **Fairness slices** — accuracy per `Sex` and per `Pclass`, not just overall. A model can hold 80% overall while being far worse for one group
- **Cross-validation** — one 80/20 split is a single sample; the accuracy gate would be sturdier against a k-fold mean
- **Drift detection** — PSI or a KS test between two snapshots, rather than validating one static file
- **A leakage case with a visible gap** — scaled features or duplicate rows spanning the split, so the demo shows the damage as well as explaining it
- **Metrics published from CI** — write the run's accuracy as a build artifact so the trend is visible across commits, not just pass/fail
