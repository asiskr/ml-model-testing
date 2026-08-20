from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
BASELINE_JSON = PROJECT_ROOT / "baseline.json"

TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_DEPTH = 3

MIN_ACCURACY = 0.75
MAX_OVERFIT_GAP = 0.15
REGRESSION_TOLERANCE = 0.05
