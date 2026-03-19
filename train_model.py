"""
train_model.py
Trains 4 separate ML models — one per exam stage.
Each model uses ONLY the data available at that stage.

Stage 1: After UT1    → features: ut1_pct, ut1_attendance
Stage 2: After Mid    → features: ut1_pct, mid_pct, ut1_att, mid_att
Stage 3: After UT2    → features: ut1_pct, mid_pct, ut2_pct, ut1_att, mid_att, ut2_att
Stage 4: Full semester→ features: all 8 (pct + attendance)

No fake future data. Each model learns from real available signals only.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle, json, glob
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from etl import get_clean_data

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DB_PATH    = str(ROOT / "data" / "university.db")

# Features available at each stage — no future data
STAGE_FEATURES = {
    1: ["ut1_pct", "ut1_attendance"],
    2: ["ut1_pct", "mid_pct", "ut1_attendance", "mid_attendance"],
    3: ["ut1_pct", "mid_pct", "ut2_pct", "ut1_attendance", "mid_attendance", "ut2_attendance"],
    4: ["ut1_pct", "mid_pct", "ut2_pct", "end_pct", "ut1_attendance", "mid_attendance", "ut2_attendance", "end_attendance"],
}

# All 8 features (for backwards compatibility)
FEATURE_COLS = STAGE_FEATURES[4]


def get_best_model():
    """Return the best classifier pipeline — SVM works well across all stage sizes."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            probability=True,
            random_state=42
        )),
    ])


def train_stage(df, stage: int):
    """Train one model for a given stage using only available features."""
    features = STAGE_FEATURES[stage]
    X = df[features]
    y = df["risk"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Also compare LR for stage 1 and 2 (fewer features → simpler model may win)
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", C=1.0, probability=True, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_depth=5, min_samples_leaf=8, random_state=42
        ),
    }

    best_name, best_model, best_f1 = None, None, 0.0
    results = {}

    for name, model in candidates.items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        model.fit(X_train, y_train)
        preds     = model.predict(X_test)
        acc       = accuracy_score(y_test, preds)
        train_acc = model.score(X_train, y_train)
        report    = classification_report(y_test, preds, output_dict=True, zero_division=0)
        macro_f1  = report["macro avg"]["f1-score"]

        results[name] = {
            "accuracy":       round(acc, 4),
            "train_accuracy": round(train_acc, 4),
            "macro_f1":       round(macro_f1, 4),
            "cv_f1_mean":     round(cv_f1.mean(), 4),
            "cv_f1_std":      round(cv_f1.std(), 4),
            "report":         report,
        }

        if macro_f1 > best_f1:
            best_f1   = macro_f1
            best_name = name
            best_model = model

    return best_model, best_name, best_f1, results


def train_all_stages(db_path: str = DB_PATH):
    df = get_clean_data(db_path)
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Dataset: {len(df)} rows")
    print("Class distribution:")
    print(df["risk"].value_counts().to_string())
    print()

    all_meta = {}

    for stage in [1, 2, 3, 4]:
        features = STAGE_FEATURES[stage]
        print(f"{'='*55}")
        print(f"STAGE {stage} — features: {features}")
        print(f"{'='*55}")

        model, name, f1, results = train_stage(df, stage)

        # Save stage-specific model
        model_path = MODELS_DIR / f"model_stage{stage}.pkl"
        with open(model_path, "wb") as f_out:
            pickle.dump(model, f_out)

        stage_meta = {
            "stage":      stage,
            "features":   features,
            "best_model": name,
            "macro_f1":   round(f1, 4),
            "accuracy":   results[name]["accuracy"],
            "train_accuracy": results[name]["train_accuracy"],
            "cv_f1_mean": results[name]["cv_f1_mean"],
            "trained_at": datetime.now().isoformat(),
            "all_results": results,
        }
        with open(MODELS_DIR / f"metadata_stage{stage}.json", "w") as f_out:
            json.dump(stage_meta, f_out, indent=2)

        all_meta[f"stage{stage}"] = stage_meta
        acc = results[name]["accuracy"]
        tr  = results[name]["train_accuracy"]
        print(f"Winner : {name}")
        print(f"Train  : {tr:.4f}  |  Test : {acc:.4f}  |  Gap: {tr-acc:.4f}")
        print(f"Macro F1 : {f1:.4f}")
        print(f"Saved  -> {model_path}")
        print()

    # Also save stage 4 as the default model.pkl for backwards compatibility
    import shutil
    shutil.copy(MODELS_DIR / "model_stage4.pkl", MODELS_DIR / "model.pkl")

    # Save combined metadata
    with open(MODELS_DIR / "metadata_all_stages.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print("All 4 stage models saved.")
    print("Stage 4 also saved as model.pkl (default)")
    return all_meta


if __name__ == "__main__":
    train_all_stages()
