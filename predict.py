"""
predict.py
Loads the correct stage model and predicts risk
using ONLY data available at that exam stage.
"""

import pickle
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# University max marks
UT1_MAX = 10
MID_MAX = 20
UT2_MAX = 10
END_MAX = 40


def _load_model(stage: int):
    path = MODELS_DIR / f"model_stage{stage}.pkl"
    if not path.exists():
        # Fallback to default model if stage model not found
        path = MODELS_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError("No model found. Run train_model.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_stage1(ut1: float, ut1_att: float) -> str:
    """
    Stage 1: Only UT1 has happened.
    Uses: ut1_pct, ut1_attendance
    """
    model = _load_model(1)
    ut1_pct = (ut1 / UT1_MAX) * 100
    features = np.array([[ut1_pct, ut1_att]])
    return model.predict(features)[0]


def predict_stage2(ut1: float, mid: float,
                   ut1_att: float, mid_att: float) -> str:
    """
    Stage 2: UT1 and Mid have happened.
    Uses: ut1_pct, mid_pct, ut1_attendance, mid_attendance
    """
    model = _load_model(2)
    ut1_pct = (ut1 / UT1_MAX) * 100
    mid_pct = (mid / MID_MAX) * 100
    features = np.array([[ut1_pct, mid_pct, ut1_att, mid_att]])
    return model.predict(features)[0]


def predict_stage3(ut1: float, mid: float, ut2: float,
                   ut1_att: float, mid_att: float, ut2_att: float) -> str:
    """
    Stage 3: UT1, Mid, and UT2 have happened.
    Uses: ut1_pct, mid_pct, ut2_pct, ut1_att, mid_att, ut2_att
    """
    model = _load_model(3)
    ut1_pct = (ut1 / UT1_MAX) * 100
    mid_pct = (mid / MID_MAX) * 100
    ut2_pct = (ut2 / UT2_MAX) * 100
    features = np.array([[ut1_pct, mid_pct, ut2_pct, ut1_att, mid_att, ut2_att]])
    return model.predict(features)[0]


def predict_stage4(ut1: float, mid: float, ut2: float, end: float,
                   ut1_att: float, mid_att: float,
                   ut2_att: float, end_att: float) -> str:
    """
    Stage 4: Full semester data available.
    Uses: all 8 features
    """
    model = _load_model(4)
    ut1_pct = (ut1 / UT1_MAX) * 100
    mid_pct = (mid / MID_MAX) * 100
    ut2_pct = (ut2 / UT2_MAX) * 100
    end_pct = (end / END_MAX) * 100
    features = np.array([[ut1_pct, mid_pct, ut2_pct, end_pct,
                          ut1_att, mid_att, ut2_att, end_att]])
    return model.predict(features)[0]


def predict_risk(ut1, mid, ut2, end, ut1_att, mid_att, ut2_att, end_att) -> str:
    """Default full-semester prediction (backwards compatibility)."""
    return predict_stage4(ut1, mid, ut2, end, ut1_att, mid_att, ut2_att, end_att)


def get_model_path(stage: int = 4) -> str:
    return str(MODELS_DIR / f"model_stage{stage}.pkl")


if __name__ == "__main__":
    print("Stage 1 (UT1=9/10, att=87%):", predict_stage1(9, 87))
    print("Stage 2 (UT1=9, Mid=18, att=87,85):", predict_stage2(9, 18, 87, 85))
    print("Stage 3 (UT1=9, Mid=18, UT2=9, att=87,85,83):", predict_stage3(9, 18, 9, 87, 85, 83))
    print("Stage 4 (UT1=9, Mid=18, UT2=9, End=36, att=87,85,83,81):", predict_stage4(9, 18, 9, 36, 87, 85, 83, 81))
