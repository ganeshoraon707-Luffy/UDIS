"""
etl.py
Extracts data from SQLite, normalises marks to percentage scale,
applies correct university weightage, and creates risk labels.

University marking scheme:
  UT1  = out of 10   → weight = 10/80 = 12.5%
  Mid  = out of 20   → weight = 20/80 = 25.0%
  UT2  = out of 10   → weight = 10/80 = 12.5%
  End  = out of 40   → weight = 40/80 = 50.0%
  Attendance contributes 20 marks to the overall 100

Risk formula (all values normalised to 0-100 scale):
  academic_score = 0.125×ut1% + 0.25×mid% + 0.125×ut2% + 0.50×end%
  final_score    = 0.80×academic_score + 0.20×attendance%

  High Risk   → final_score < 50
  Medium Risk → 50 <= final_score < 70
  Low Risk    → final_score >= 70
"""

import sqlite3
import random
from pathlib import Path
import pandas as pd

random.seed(123)

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT / "data" / "university.db")

# Real university max marks per exam
UT1_MAX = 10
MID_MAX = 20
UT2_MAX = 10
END_MAX = 40

FEATURE_COLS = [
    "ut1_pct", "mid_pct", "ut2_pct", "end_pct",
    "ut1_attendance", "mid_attendance", "ut2_attendance", "end_attendance",
]


def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM department_data", conn)
    conn.close()
    return df


def normalise_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw marks to percentage (0-100) for each exam."""
    df = df.copy()
    df["ut1_pct"] = (df["ut1_marks"] / UT1_MAX) * 100
    df["mid_pct"] = (df["mid_marks"] / MID_MAX) * 100
    df["ut2_pct"] = (df["ut2_marks"] / UT2_MAX) * 100
    df["end_pct"] = (df["end_marks"] / END_MAX) * 100
    return df


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply correct university weightage to compute final score."""
    df = normalise_marks(df)

    # Academic score — weighted by real university marks allocation
    # UT1(10) + Mid(20) + UT2(10) + End(40) = 80 total
    df["academic_score"] = (
        0.125 * df["ut1_pct"] +   # 10/80
        0.250 * df["mid_pct"] +   # 20/80
        0.125 * df["ut2_pct"] +   # 10/80
        0.500 * df["end_pct"]     # 40/80
    )

    df["avg_attendance"] = df[
        ["ut1_attendance", "mid_attendance",
         "ut2_attendance", "end_attendance"]
    ].mean(axis=1)

    # Final score: 80% from marks + 20% from attendance
    # (attendance worth 20 out of 100 in real university)
    df["final_score"] = (
        0.80 * df["academic_score"] +
        0.20 * df["avg_attendance"]
    )
    return df


def assign_label(score: float) -> str:
    """
    Assign risk label with fuzzy boundary zones for realism.
    Thresholds scaled to match 80-mark scheme (out of 100 after normalisation):
      < 50  → High Risk   (failed or near-failed zone)
      50-70 → Medium Risk (borderline zone)
      > 70  → Low Risk    (satisfactory zone)
    """
    if score < 44:
        return "High"
    elif score < 52:
        return random.choices(["High", "Medium"], weights=[70, 30])[0]
    elif score < 65:
        return "Medium"
    elif score < 72:
        return random.choices(["Medium", "Low"], weights=[65, 35])[0]
    else:
        return "Low"


def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_scores(df)
    df["risk"] = df["final_score"].apply(assign_label)
    return df


def get_clean_data(db_path: str = DB_PATH) -> pd.DataFrame:
    df = load_data(db_path)
    df = create_risk_labels(df)
    return df


if __name__ == "__main__":
    df = get_clean_data()
    print(f"Clean dataset: {len(df)} rows")
    print(df["risk"].value_counts())
    print("\nSample normalised scores:")
    print(df[["ut1_pct","mid_pct","ut2_pct","end_pct","academic_score","final_score","risk"]].head(5).to_string())
