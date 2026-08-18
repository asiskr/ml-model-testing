# ml-practice

Titanic survival prediction, built while working through the ML section of a
37-day QA → GenAI study plan. The focus is not the model score — it is testing
the data and the pipeline the way a QA engineer would.

## Structure

```
ml-practice/
├── data/
│   └── train.csv               Titanic dataset (891 rows)
├── src/
│   ├── config.py               paths, feature list, split settings
│   ├── load_data.py            load, explore, split features/target
│   ├── clean_data.py           fill missing values, one-hot encode
│   ├── train_model.py          split, train, evaluate
│   ├── schema.py               pandera schema (single source of truth)
│   ├── data_quality.py         missing / duplicate / range / zero-fare checks
│   └── main.py                 full pipeline end to end
├── demos/
│   ├── metrics_demo.py         precision, recall, F1 on a tiny hand-made set
│   ├── imbalanced_demo.py      why 90% accuracy can mean a useless model
│   ├── overfitting_demo.py     train vs test accuracy gap
│   └── leakage_demo.py         preprocess-before-split vs after-split
└── tests/
    ├── test_schema.py          schema passes on good data, fails on bad
    └── test_data_quality.py    row counts, uniqueness, ranges, missing limits
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run everything from the project root as a module, so imports resolve:

```bash
python -m src.main
```

```bash
python -m src.data_quality
```

```bash
python -m src.schema
```

```bash
python -m demos.overfitting_demo
```

Swap in any other module under `demos/` the same way.

## Tests

```bash
pytest
```

Markers are registered for running a subset:

```bash
pytest -m quality
```

`tests/test_schema.py` checks both directions — the schema accepts the real
dataset, and it rejects an out-of-range value in each validated column plus a
dropped column. `tests/test_data_quality.py` guards row count, PassengerId
uniqueness, duplicate rows, negative fares, and the missing-value rate for Age.

## Results

| Model | Test Accuracy |
|---|---|
| Decision Tree | 78.77% |
| Random Forest | 81.56% |

Random Forest confusion matrix: TN=91, FP=14, FN=19, TP=55 —
precision 79.71%, recall 74.32%, F1 76.92%.
