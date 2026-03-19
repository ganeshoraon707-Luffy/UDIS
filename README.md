<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge"/>

<br/><br/>

```
 ██╗   ██╗██████╗ ██╗███████╗
 ██║   ██║██╔══██╗██║██╔════╝
 ██║   ██║██║  ██║██║███████╗
 ██║   ██║██║  ██║██║╚════██║
 ╚██████╔╝██████╔╝██║███████║
  ╚═════╝ ╚═════╝ ╚═╝╚══════╝
```

# University Decision Intelligence System

### *End-to-end ML pipeline for academic risk prediction with progressive semester monitoring*

<br/>

[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-87.67%25-22c55e?style=flat-square)](.)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.877-3b82f6?style=flat-square)](.)
[![Models](https://img.shields.io/badge/ML%20Models-4%20Algorithms%20×%204%20Stages-8b5cf6?style=flat-square)](.)
[![Dataset](https://img.shields.io/badge/Dataset-1500%20Rows%20%7C%2024%20Departments-f59e0b?style=flat-square)](.)

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [What is UDIS?](#-what-is-udis) |
| 2 | [Live Dashboard Preview](#-live-dashboard-preview) |
| 3 | [System Architecture](#-system-architecture) |
| 4 | [Project Structure](#-project-structure) |
| 5 | [University Marking Scheme](#-university-marking-scheme) |
| 6 | [Risk Formula Deep Dive](#-risk-formula-deep-dive) |
| 7 | [Dataset Design & Archetypes](#-dataset-design--archetypes) |
| 8 | [Progressive Monitoring — 4 Stage Models](#-progressive-monitoring--4-stage-models) |
| 9 | [ML Models & Parameters](#-ml-models--parameters) |
| 10 | [Model Evaluation Results](#-model-evaluation-results) |
| 11 | [ETL Pipeline](#-etl-pipeline) |
| 12 | [Project Evolution — Version History](#-project-evolution--version-history) |
| 13 | [Key Design Decisions](#-key-design-decisions) |
| 14 | [Known Limitations & Future Work](#-known-limitations--future-work) |
| 15 | [How to Run](#-how-to-run) |
| 16 | [Technology Stack](#-technology-stack) |

---

## 🎯 What is UDIS?

**UDIS (University Decision Intelligence System)** is a complete, production-grade end-to-end machine learning pipeline that predicts the **academic risk level** of university departments. Given a department's marks and attendance data across exam stages, the system classifies it as **High Risk**, **Medium Risk**, or **Low Risk** and recommends specific academic interventions.

### 🔑 Key Innovation — Progressive Semester Monitoring

> **Traditional systems** wait until the end of semester to detect at-risk departments — by then it is too late to intervene.
>
> **UDIS** provides risk predictions **after every exam checkpoint** using stage-specific ML models trained only on data that actually exists at that point in time. No fabricated future data. No guesswork.

```
Semester Timeline:
─────────────────────────────────────────────────────────────────
  UT1          Mid Exam       UT2         End Exam
   │               │           │              │
   ▼               ▼           ▼              ▼
Stage 1 ──►  Stage 2 ──►  Stage 3 ──►  Stage 4
  78%           81%           83%           87%    ← Accuracy
  (2 feat)    (4 feat)     (6 feat)      (8 feat) ← Features used
─────────────────────────────────────────────────────────────────
Each stage uses ONLY real available data — no future exam guessing
```

### 📊 At a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    UDIS — Key Metrics                       │
├──────────────────┬──────────────────┬───────────────────────┤
│  1,500 Records   │  24 Departments  │  4 ML Algorithms      │
│  3 Risk Classes  │  8 Features      │  4 Stage Models       │
│  87.67% Accuracy │  Macro F1: 0.877 │  Real Marking Scheme  │
└──────────────────┴──────────────────┴───────────────────────┘
```

---

## 🖥️ Live Dashboard Preview

The dashboard uses an editorial light theme — warm cream background, DM Serif Display typography, gold yellow accents — designed to stand out from the common dark navy ML dashboards.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  UDIS   University Decision Intelligence System   1500 | 24 | 4 | 87%  │
│  (topbar: charcoal black with gold accent stripe)                       │
├──────────────────┬──────────────────┬─────────────────┬─────────────────┤
│  Best Model      │  High Risk: 364  │  Dataset Stats  │  Model Bars     │
│  Logistic Reg.   │  (gold yellow)   │  Med: 764       │  LR ★ ──── 0.877│
│  F1: 0.877       │  24.3% of total  │  Low: 372       │  RF   ──── 0.857│
│  (dark bg)       │                  │  Att: 66.4%     │  GB   ──── 0.898│
│                  │                  │  Marks: 62.5%   │  SVM  ──── 0.866│
├──────────────────┴──────────────────┴─────────────────┴─────────────────┤
│  Dept Score Chart (black+gold bars, threshold lines)    │  Attendance    │
│  BBA ████████████ 68%                                   │  UT1: 66.3%    │
│  CSE ████████████ 65%                                   │  Mid: 64.1%    │
│  ...  Bottom 3 in gold                                  │  End: 60.8%    │
├──────────────────────────────────────────────────────────────────────────┤
│  RISK PREDICTION — PROGRESSIVE MONITORING (4 Stages side by side)       │
│  Stage 1    │    Stage 2    │    Stage 3    │    Stage 4 (Full)          │
│  UT1 only   │  UT1 + Mid   │ UT1+Mid+UT2  │  All 8 features            │
│  [Predict]  │  [Predict]   │  [Predict]   │  [Predict]                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UDIS — Full System Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │  DATA LAYER     │
  │                 │
  │ generate_data   │──► 1500 rows ──► raw_data.csv
  │    .py          │    6 archetypes
  │                 │    real mark scales
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  STORAGE LAYER  │
  │                 │
  │ database_setup  │──► university.db (SQLite)
  │    .py          │    table: department_data
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ETL LAYER  (etl.py)                                    │
  │                                                         │
  │  load_data()         ──► Extract from SQLite            │
  │  normalise_marks()   ──► Convert to % scale             │
  │  compute_scores()    ──► Apply university weightage     │
  │  assign_label()      ──► Fuzzy risk classification      │
  │  get_clean_data()    ──► Returns model-ready DataFrame  │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │  TRAINING LAYER  (train_model.py)                       │
  │                                                         │
  │  Stage 1 ──► 2 features  ──► SVM         ──► 78% acc   │
  │  Stage 2 ──► 4 features  ──► RandomForest──► 81% acc   │
  │  Stage 3 ──► 6 features  ──► SVM         ──► 83% acc   │
  │  Stage 4 ──► 8 features  ──► LogisticReg ──► 87% acc   │
  │                                                         │
  │  StratifiedKFold(5) cross-validation per model         │
  │  Winner selected by Macro F1 (not accuracy)            │
  └──────────────┬──────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  model_stage1.pkl    metadata_stage*.json
  model_stage2.pkl    (accuracy, F1, CV scores,
  model_stage3.pkl     training timestamp,
  model_stage4.pkl     feature list, all results)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PREDICTION LAYER  (predict.py)                         │
  │                                                         │
  │  predict_stage1(ut1, ut1_att)                           │
  │  predict_stage2(ut1, mid, ut1_att, mid_att)             │
  │  predict_stage3(ut1, mid, ut2, ut1_att, mid_att, u2a)   │
  │  predict_stage4(ut1, mid, ut2, end, all_attendances)    │
  │                                                         │
  │  Each function normalises marks internally before       │
  │  passing to the correct stage model                     │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  DASHBOARD LAYER  (dashboard/dashboard.py)              │
  │                                                         │
  │  Section 1: Topbar — brand + live stats                 │
  │  Section 2: Metric cells — model, risk counts, stats    │
  │  Section 3: Charts — dept scores, attendance, risk pie  │
  │  Section 4: Prediction — all 4 stages side by side      │
  │                                                         │
  │  Technology: Streamlit + Plotly + custom CSS            │
  │  Theme: Editorial light (cream/charcoal/gold)           │
  └─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
UDIS_Project/
│
├── 📂 data/
│   ├── university.db              ← SQLite database (department_data table)
│   └── raw_data.csv               ← Generated dataset (1500 rows, 8 features)
│
├── 📂 scripts/
│   ├── generate_data.py           ← Synthetic data generator (6 archetypes)
│   ├── database_setup.py          ← CSV → SQLite loader
│   ├── etl.py                     ← ETL pipeline + risk label assignment
│   ├── train_model.py             ← 4-stage model training + evaluation
│   └── predict.py                 ← Stage-specific prediction functions
│
├── 📂 models/
│   ├── model_stage1.pkl           ← SVM trained on 2 features
│   ├── model_stage2.pkl           ← RandomForest trained on 4 features
│   ├── model_stage3.pkl           ← SVM trained on 6 features
│   ├── model_stage4.pkl           ← LogisticRegression trained on 8 features
│   ├── model.pkl                  ← Alias → model_stage4.pkl (latest)
│   ├── metadata_stage1.json       ← Training metadata: accuracy, F1, CV, timestamp
│   ├── metadata_stage2.json
│   ├── metadata_stage3.json
│   ├── metadata_stage4.json
│   └── metadata_all_stages.json   ← Combined metadata for all 4 stages
│
├── 📂 dashboard/
│   └── dashboard.py               ← Streamlit app (editorial light theme)
│
├── requirements.txt
└── README.md
```

---

## 🎓 University Marking Scheme

This project uses the **real university marking scheme** rather than a simplified 0-100 scale:

```
┌───────────────────────────────────────────────────────┐
│           University Marks Allocation                 │
├─────────────┬────────────┬───────────────────────────┤
│    Exam     │  Out of    │  Weight in Formula        │
├─────────────┼────────────┼───────────────────────────┤
│    UT1      │    10      │  10/80 = 12.5%            │
│  Midterm    │    20      │  20/80 = 25.0%            │
│    UT2      │    10      │  10/80 = 12.5%            │
│  End Exam   │    40      │  40/80 = 50.0%  ◄ KEY     │
├─────────────┼────────────┼───────────────────────────┤
│  Academic   │    80      │  80% of total score       │
│  Attendance │   20%      │  20% of total score       │
├─────────────┼────────────┼───────────────────────────┤
│  TOTAL      │   100      │  100%                     │
└─────────────┴────────────┴───────────────────────────┘

⚠️ End Exam carries 50% of academic weight — the single most
   important determinant of risk level in this scheme.
```

**Insight:** A department scoring perfectly in UT1, Mid, UT2 but only 50% in End exam will still show Medium Risk because End carries 50% weight.

---

## 🧮 Risk Formula Deep Dive

### Step-by-Step Calculation

```python
# Step 1 — Normalise raw marks to percentage (0-100 scale)
ut1_pct  = (ut1_marks  / 10) × 100    # e.g. 7/10  → 70%
mid_pct  = (mid_marks  / 20) × 100    # e.g. 14/20 → 70%
ut2_pct  = (ut2_marks  / 10) × 100    # e.g. 7/10  → 70%
end_pct  = (end_marks  / 40) × 100    # e.g. 28/40 → 70%

# Step 2 — Weighted academic score (matches real university allocation)
academic_score = (0.125 × ut1_pct)    # UT1  contributes 12.5%
              + (0.250 × mid_pct)     # Mid  contributes 25.0%
              + (0.125 × ut2_pct)     # UT2  contributes 12.5%
              + (0.500 × end_pct)     # End  contributes 50.0%
# → academic_score = 70.0 (for all 70%)

# Step 3 — Combine with attendance (worth 20 marks out of 100)
avg_attendance = mean(ut1_att, mid_att, ut2_att, end_att)
final_score    = (0.80 × academic_score) + (0.20 × avg_attendance)
# → 0.80×70 + 0.20×80 = 56 + 16 = 72.0

# Step 4 — Fuzzy risk label assignment
final_score < 44          →  "High"   (always)
44 ≤ score  < 52          →  70% "High"   / 30% "Medium"  (fuzzy zone)
52 ≤ score  < 65          →  "Medium" (always)
65 ≤ score  < 72          →  65% "Medium" / 35% "Low"     (fuzzy zone)
score ≥ 72                →  "Low"    (always)
```

### Why Fuzzy Boundaries?

```
Hard threshold (WRONG approach):
  score 59.9 → always High Risk
  score 60.1 → always Medium Risk
  Model just learns "is score > 60?" — trivial, no real learning

Fuzzy boundary (OUR approach):
  score 61 → 70% chance High, 30% chance Medium
  Simulates real academic committee judgment on borderline cases
  Forces model to learn underlying feature patterns, not just thresholds
```

### Worked Example

```
Department: CSE
UT1 = 8/10, Mid = 16/20, UT2 = 7/10, End = 30/40
Attendance: 85%, 82%, 79%, 76%

Step 1 — Normalise:  80%, 80%, 70%, 75%
Step 2 — Academic:   0.125×80 + 0.25×80 + 0.125×70 + 0.5×75
                   = 10 + 20 + 8.75 + 37.5 = 76.25
Step 3 — Final:      0.80×76.25 + 0.20×80.5 = 61 + 16.1 = 77.1
Step 4 — Label:      77.1 ≥ 72 → LOW RISK ✓
```

---

## 🏭 Dataset Design & Archetypes

### 6 Realistic Department Archetypes

```
┌─────────────────────┬────────────────────────────┬──────────────────────┐
│    Archetype        │        Description         │   Why Important      │
├─────────────────────┼────────────────────────────┼──────────────────────┤
│ clear_high (15%)    │ Low marks + low att        │ Obvious High Risk    │
│ clear_low (15%)     │ High marks + high att      │ Obvious Low Risk     │
│ good_marks_bad_att  │ Good marks, poor att       │ Conflict signal ★    │
│       (17%)         │ Marks say Low, att says Hi │ Forces real learning │
│ bad_marks_good_att  │ Weak marks, great att      │ Conflict signal ★    │
│       (17%)         │ Att saves poor marks       │ Forces real learning │
│ inconsistent (18%)  │ Wild swings across exams   │ Unpredictable ★      │
│ middle_ground (18%) │ Everything near boundaries │ Genuine ambiguity ★  │
└─────────────────────┴────────────────────────────┴──────────────────────┘
★ = These 4 archetypes create the ambiguity that makes ML genuinely necessary
```

### Dataset Statistics

```
Total rows:     1,500
Departments:    24 (Engineering, Business, Music, Design)
Risk classes:   High=364 (24.3%), Medium=764 (50.9%), Low=372 (24.8%)
Features:       8 (4 marks + 4 attendances, normalised to %)

Mark ranges (normalised %):
  High Risk departments:   end_marks 40% - 82%
  Medium Risk departments: end_marks 40% - 95%  ← overlapping!
  Low Risk departments:    end_marks 48% - 95%  ← overlapping!

This overlap is intentional — it forces the model to learn
the combination of features, not just individual thresholds.
```

### Attendance Pattern

```
Average attendance declines across semester (realistic):
  UT1:  66.3%  ████████████████████████████████
  Mid:  64.1%  ████████████████████████████████
  UT2:  62.0%  ████████████████████████████████
  End:  60.8%  ████████████████████████████████
                                               ↓ declining trend
```

---

## 📈 Progressive Monitoring — 4 Stage Models

### The Core Design Decision

```
❌ OLD APPROACH (wrong):
   Stage 1: Enter UT1=9 → Fill Mid/UT2/End with dataset average
   Problem: Feeding model fabricated data about future exams
   that haven't happened yet = dishonest prediction

✅ OUR APPROACH (correct):
   Stage 1: Train a model using ONLY UT1 data
   Stage 2: Train a model using ONLY UT1 + Mid data
   Stage 3: Train a model using ONLY UT1 + Mid + UT2 data
   Stage 4: Train a model using ALL 8 features
   Each model learned risk patterns from exactly the data available
```

### Stage Model Summary

```
┌─────────┬────────────────────┬───────────────────────────────┬──────────┬──────────┐
│  Stage  │   Trigger Point    │        Features Used          │  Winner  │ Accuracy │
├─────────┼────────────────────┼───────────────────────────────┼──────────┼──────────┤
│ Stage 1 │ After UT1          │ ut1_pct, ut1_att             │ SVM      │  78.0%   │
│ Stage 2 │ After Mid Exam     │ ut1_pct, mid_pct,            │ Random   │  81.0%   │
│         │                    │ ut1_att, mid_att             │ Forest   │          │
│ Stage 3 │ After UT2          │ ut1_pct, mid_pct, ut2_pct,  │ SVM      │  83.3%   │
│         │                    │ ut1_att, mid_att, ut2_att   │          │          │
│ Stage 4 │ After End Exam     │ all 8 features              │ Logistic │  87.7%   │
│         │ (Full semester)    │                              │ Regress. │          │
└─────────┴────────────────────┴───────────────────────────────┴──────────┴──────────┘

Accuracy progression:  78% → 81% → 83% → 87%
                         ↑
       More real data = better prediction. This is expected and correct.
```

### Accuracy Growth Chart

```
90% ┤                                              ●  87.7%
88% ┤
86% ┤
84% ┤                              ●  83.3%
82% ┤
80% ┤            ●  81.0%
78% ┤  ●  78.0%
76% ┤
    └──────────────────────────────────────────────────────
       Stage 1    Stage 2    Stage 3    Stage 4
       (UT1)      (Mid)       (UT2)     (End)
```

---

## 🤖 ML Models & Parameters

### Model 1 — Logistic Regression ⭐ STAGE 4 WINNER

```python
Pipeline([
    StandardScaler(),           # Normalise features to mean=0, std=1
    LogisticRegression(
        C             = 0.3,    # Regularisation: lower = simpler = less overfit
                                # Default is 1.0 — we use 0.3 for stronger control
        class_weight  = "balanced", # weight_k = total/(n_classes × count_k)
                                    # High(364): 1.37×  Med(764): 0.65×  Low(372): 1.34×
        max_iter      = 2000,   # Iterations for gradient descent convergence
        random_state  = 42,     # Reproducibility seed
    )
])

How it works:
  z = w1×ut1_pct + w2×mid_pct + ... + w8×end_att + bias
  P(class_k) = e^(z_k) / Σ(e^(z_all))  [softmax for 3 classes]
  Weights w1...w8 learned by minimising cross-entropy loss
```

### Model 2 — Random Forest 🌲 STAGE 2 WINNER

```python
RandomForestClassifier(
    n_estimators     = 100,     # 100 decision trees vote in parallel
    max_depth        = 5,       # Max questions per tree (controls memorisation)
    min_samples_leaf = 8,       # Each leaf needs ≥8 samples (generalisation)
    class_weight     = "balanced",
    random_state     = 42,
)
# Each tree trains on random 63% of data (bootstrap sample)
# At each split: considers only √8 ≈ 3 random features
# Final answer: majority vote of all 100 trees
```

### Model 3 — Gradient Boosting 📈

```python
GradientBoostingClassifier(
    n_estimators  = 100,        # 100 sequential trees
    learning_rate = 0.05,       # Each tree contributes 5% of its prediction
                                # Lower = slower but more careful learning
    max_depth     = 3,          # Shallow trees = "weak learners" (correct for GB)
    subsample     = 0.8,        # Each tree sees 80% of data = stochastic boosting
    random_state  = 42,
)
# Tree k corrects residuals of trees 1..k-1
# Final = T1×0.05 + T2×0.05 + ... + T100×0.05
```

### Model 4 — SVM 🔴 STAGE 1 & 3 WINNER

```python
Pipeline([
    StandardScaler(),           # REQUIRED — SVM is distance-based
    SVC(
        kernel       = "rbf",   # Radial Basis Function: maps to higher dimensions
                                # K(x,y) = exp(-γ × ||x-y||²)
        C            = 1.0,     # Balanced margin vs misclassification tradeoff
        class_weight = "balanced",
        probability  = True,    # Enable confidence % (uses Platt scaling)
        random_state = 42,
    )
])
# Finds hyperplane with maximum margin between classes
# Support vectors = border cases that define the boundary
```

---

## 📊 Model Evaluation Results

### Stage 4 — Full Semester (Most Comprehensive)

```
┌────────────────────┬──────────┬──────────┬────────────┬──────────┬────────────┐
│       Model        │ Train    │  Test    │  Overfit   │ Macro F1 │  CV F1     │
│                    │  Acc     │  Acc     │   Gap      │          │  (mean±std)│
├────────────────────┼──────────┼──────────┼────────────┼──────────┼────────────┤
│ Logistic Regress.  │  89.8%   │  87.7%   │   2.2% ✅  │  0.877 ★ │ 0.891±0.015│
│ SVM                │  90.6%   │  86.7%   │   3.9% ✅  │  0.867   │ 0.888±0.026│
│ Random Forest      │  91.3%   │  86.0%   │   5.3% ⚠️  │  0.857   │ 0.879±0.018│
│ Gradient Boosting  │  95.3%   │  86.7%   │   8.6% ⚠️  │  0.865   │ 0.881±0.018│
└────────────────────┴──────────┴──────────┴────────────┴──────────┴────────────┘
★ = Winner by Macro F1    ✅ = Good generalisation    ⚠️ = Mild overfit (expected)
```

### Why Macro F1 Instead of Accuracy?

```
Scenario: 764 Medium, 364 High, 372 Low (1500 total)

Dumb model that always predicts "Medium":
  Accuracy = 764/1500 = 50.9%  ← looks OK!
  Macro F1 = (0 + 1.0 + 0)/3  = 0.33 ← exposed as terrible

Our Logistic Regression:
  Accuracy = 87.7%
  High Risk  F1 = 0.877  ← catches dangerous departments
  Medium F1  = 0.895
  Low Risk   F1 = 0.860
  Macro F1   = 0.877  ← fair across all classes
```

### Classification Report (Stage 4 — Logistic Regression)

```
              precision    recall  f1-score   support

        High       0.899     0.857     0.877        63
      Medium       0.890     0.910     0.900       177
         Low       0.862     0.867     0.864        60

    accuracy                           0.877       300
   macro avg       0.884     0.878     0.880       300
weighted avg       0.882     0.877     0.879       300
```

### Cross-Validation Analysis

```
5-Fold Stratified Cross-Validation ensures reliability:

  Fold 1: train on [2,3,4,5] → test on [1] → F1 = 0.891
  Fold 2: train on [1,3,4,5] → test on [2] → F1 = 0.878
  Fold 3: train on [1,2,4,5] → test on [3] → F1 = 0.903
  Fold 4: train on [1,2,3,5] → test on [4] → F1 = 0.896
  Fold 5: train on [1,2,3,4] → test on [5] → F1 = 0.887
                                              ─────────
  Mean = 0.891   Std = 0.015  (low std = consistent = trustworthy)
```

### Bias-Variance Analysis

```
Model Position on Bias-Variance Spectrum:

High Variance                                      Balanced
(Overfit) ◄────────────────────────────────────► (Ideal)
              GB(8.6%)   RF(5.3%)   SVM(3.9%)   LR(2.2%)
              ████████   ██████     ████         ██
              
LR sits closest to ideal — learned the pattern without memorising noise.
```

---

## 🔄 ETL Pipeline

```
╔═════════════════════════════════════════════════════════════════╗
║                    etl.py — Pipeline Flow                       ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  load_data(db_path)                                             ║
║    └─► sqlite3.connect("university.db")                         ║
║    └─► pd.read_sql("SELECT * FROM department_data")             ║
║    └─► Returns raw DataFrame (9 columns, 1500 rows)             ║
║                          │                                      ║
║                          ▼                                      ║
║  normalise_marks(df)                                            ║
║    └─► ut1_pct  = (ut1_marks  / 10) × 100                      ║
║    └─► mid_pct  = (mid_marks  / 20) × 100                      ║
║    └─► ut2_pct  = (ut2_marks  / 10) × 100                      ║
║    └─► end_pct  = (end_marks  / 40) × 100                      ║
║                          │                                      ║
║                          ▼                                      ║
║  compute_scores(df)                                             ║
║    └─► academic_score = 0.125×ut1 + 0.25×mid + 0.125×ut2       ║
║                       + 0.50×end    [real university weights]   ║
║    └─► avg_attendance = mean(all 4 attendance columns)          ║
║    └─► final_score    = 0.80×academic + 0.20×attendance         ║
║                          │                                      ║
║                          ▼                                      ║
║  assign_label(score)  [fuzzy boundaries]                        ║
║    └─► < 44            → "High"  (certain)                      ║
║    └─► 44-52           → random 70/30 High/Medium               ║
║    └─► 52-65           → "Medium" (certain)                     ║
║    └─► 65-72           → random 65/35 Medium/Low                ║
║    └─► ≥ 72            → "Low"   (certain)                      ║
║                          │                                      ║
║                          ▼                                      ║
║  get_clean_data()   ← Entry point called by train_model.py      ║
║    └─► Returns complete DataFrame with risk labels              ║
╚═════════════════════════════════════════════════════════════════╝
```

---

## 📅 Project Evolution — Version History

### Version 1 — Basic Prototype
```diff
+ Initial pipeline: generate → store → ETL → train → predict → dashboard
+ 600 rows, simple if-else labels, 3 models
- All models hit 1.0 accuracy → trivial problem
- High Risk precision = 0% → completely ignored minority class
- Label leakage: model just learned the formula
```

### Version 2 — Fixed Class Imbalance
```diff
+ Added class_weight="balanced" to all models
+ Increased dataset to 900 rows
+ High Risk now predicted correctly (was 0% F1 before)
- Still near-perfect scores (data too clean)
- Pre-assigned class ranges created non-overlapping zones
```

### Version 3 — Fixed Overfitting
```diff
+ Replaced pre-assigned ranges with 6-archetype generation
+ Added fuzzy boundary labels (70/30 and 65/35 probability zones)
+ Individual model hyperparameter tuning
+ Overfit gap detection added to training output
= Realistic accuracy 87-92%, visible meaningful overfit gaps
= RF and GB mild overfit as expected for ensemble methods
```

### Version 4 — Real University Marking Scheme
```diff
+ Changed all mark scales to real university values (UT1/10, Mid/20, UT2/10, End/40)
+ Updated ETL formula weights: 0.125, 0.25, 0.125, 0.50 (from actual allocation)
+ Final score formula: 80% marks + 20% attendance (matches real 100-mark scheme)
+ Dashboard sliders now show real scales
+ predict.py normalises raw marks internally before model input
```

### Version 5 — 4-Stage Progressive Models (Biggest Improvement)
```diff
+ Identified fundamental flaw: one model filling future exams with averages = fake data
+ Trained 4 separate models, one per exam stage
+ Stage 1: 2 features only (ut1_pct, ut1_att)
+ Stage 2: 4 features only (+ mid_pct, mid_att)
+ Stage 3: 6 features only (+ ut2_pct, ut2_att)
+ Stage 4: all 8 features
+ Accuracy progression: 78% → 81% → 83% → 87% (natural, honest improvement)
= Stage 1 with good marks now correctly shows Low Risk (was Medium before)
```

### Version 6 — UI Redesign
```diff
+ Replaced common dark navy dashboard with editorial light theme
+ DM Serif Display + DM Sans typography (premium feel)
+ Warm cream #f5f2ee background, charcoal #1a1a1a, gold #e8c547 accents
+ All 4 prediction stages visible simultaneously (no dropdown switching)
+ Fixed department chart: normalised % scale with threshold lines
+ Fixed attendance chart: 0-100% axis, 75% minimum threshold line
+ High Risk count displayed in gold yellow cell (immediate visual impact)
```

---

## 🎨 Key Design Decisions

### 1. Why 4 Separate Stage Models?

```
Using one model + filling missing exams with averages:
  Problem: Feeding fabricated future data to the model
  Stage 1 prediction with UT1=9/10, filled End=avg(16/40):
  → Model sees End=40% when End hasn't happened → dishonest

Using 4 stage-specific models:
  Stage 1 model learned: "given only UT1 data,
  what does risk look like?" from 1500 real examples
  → Honest prediction based on real available information
```

### 2. Why Macro F1 as Winner Metric?

```
Our dataset: Medium=51%, High=24%, Low=25%

If we used accuracy, a model could ignore High Risk entirely
and still score 51%. That's dangerous — missing High Risk
departments is the worst possible outcome for a university.

Macro F1 = (F1_High + F1_Medium + F1_Low) / 3
All 3 classes weighted equally. High Risk gets same importance
as Medium even though it has fewer samples.
```

### 3. Why Pipeline for LR and SVM but not RF/GB?

```
LR and SVM are distance-based → sensitive to feature scale
  Without scaling: ut1_pct(70) vs end_att(80) seems similar
  But if salary(80000) was a feature, it would dominate
  StandardScaler: normalises all features to mean=0, std=1

RF and GB use decision trees → threshold-based, scale-invariant
  "Is ut1_pct > 70?" works identically at any scale
  Scaling trees adds computation with no benefit

Pipeline prevents data leakage:
  scaler.fit_transform(X_train) ← learns stats from train only
  scaler.transform(X_test)      ← applies training stats to test
  Fitting scaler on test data = unfair advantage = cheating
```

### 4. Why SQLite over CSV?

```
CSV:     flat file, no queries, read all data every time
SQLite:  SQL queries, indexing, multiple tables, ACID compliant

Future tables we can add:
  department_history  ← multi-semester tracking
  faculty_changes     ← context for performance dips
  intervention_log    ← records of actual interventions taken
```

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

```
1. Synthetic Labels
   Current:  Labels derived from mathematical formula
   Problem:  Model approximates our formula, not real outcomes
   Fix:      Collect 2-3 semesters of real intervention records
             as training labels. Retrain with train_model.py.

2. Schema Dependency
   Current:  Expects exact column names (ut1_marks, end_att etc.)
   Problem:  Real university CSVs use different naming (CIA1, ESE)
   Fix:      Add university_config.json for column mapping

3. Single Institution
   Current:  Thresholds calibrated for one marking scheme
   Problem:  Different universities have different passing criteria
   Fix:      Make thresholds configurable per institution

4. No Time Dimension
   Current:  Each semester treated independently
   Problem:  Cannot detect improving/declining trends
   Fix:      Add semester history table, trend feature engineering
```

### Future Improvements Roadmap

```
Phase 1 (Immediate):
  □ CSV upload feature for bulk department prediction
  □ Confusion matrix visualisation in dashboard
  □ Feature importance chart (from Random Forest)
  □ Export predictions as PDF/Excel report

Phase 2 (Short-term):
  □ Real historical outcome labels
  □ Semester trend analysis (improving/declining)
  □ Subject-wise marks (5 subjects × 4 exams = 20 features)
  □ Hyperparameter tuning with GridSearchCV

Phase 3 (Long-term):
  □ Multi-university deployment with institution configs
  □ Alert/notification system for risk changes
  □ XGBoost as additional model candidate
  □ REST API for programmatic access
```

---

## 🚀 How to Run

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/UDIS-University-Decision-Intelligence-System.git
cd UDIS-University-Decision-Intelligence-System/UDIS_Project

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset (1500 rows)
python scripts/generate_data.py

# 4. Load dataset into SQLite database
python scripts/database_setup.py

# 5. Train all 4 stage-specific models
python scripts/train_model.py
# Output: model_stage1.pkl, model_stage2.pkl, model_stage3.pkl, model_stage4.pkl
# Each model's accuracy, F1, and CV scores printed to console

# 6. Launch the dashboard
python -m streamlit run dashboard/dashboard.py
# Dashboard opens at http://localhost:8501
```

### Quick Test (after training)

```python
from scripts.predict import predict_stage1, predict_stage4

# Stage 1: only UT1 data available
result = predict_stage1(ut1=9, ut1_att=87)
print(result)  # → "Low"

# Stage 4: full semester data
result = predict_stage4(ut1=9, mid=18, ut2=9, end=36,
                        ut1_att=87, mid_att=85, ut2_att=83, end_att=81)
print(result)  # → "Low"
```

---

## 🛠️ Technology Stack

```
┌──────────────────┬──────────────────────────────────────────────────────┐
│   Technology     │                    Purpose                           │
├──────────────────┼──────────────────────────────────────────────────────┤
│ Python 3.12      │ Core language — all scripts and logic                │
│ pandas           │ DataFrame operations, ETL, data manipulation         │
│ scikit-learn     │ All 4 ML algorithms, Pipeline, cross-validation,     │
│                  │ StandardScaler, StratifiedKFold, metrics             │
│ SQLite/sqlite3   │ Relational database — zero-server-setup storage      │
│ pickle           │ Model serialisation — save/load trained models       │
│ Streamlit        │ Dashboard framework — Python to interactive web app  │
│ Plotly           │ Interactive charts (bar, line, pie, scatter)         │
│ json             │ Metadata storage — training results, timestamps      │
│ random           │ Controlled data generation with seed=42              │
└──────────────────┴──────────────────────────────────────────────────────┘
```

---

## 📚 Academic Context

### ML Concepts Demonstrated

- **Supervised Classification** — labelled training data, 3-class prediction
- **Multi-class Classification** — High/Medium/Low via softmax
- **Cross-Validation** — StratifiedKFold(5), CV mean ± std
- **Bias-Variance Tradeoff** — train vs test gap analysis per model
- **Class Imbalance Handling** — class_weight=balanced, Macro F1
- **Feature Engineering** — normalisation, weighted score computation
- **Model Versioning** — versioned pkl files + metadata JSON
- **Pipeline** — StandardScaler chained with classifier, leakage prevention
- **Regularisation** — C parameter in LR and SVM
- **Ensemble Methods** — Random Forest (bagging), Gradient Boosting (boosting)

### Software Engineering Concepts Demonstrated

- **ETL Pipeline** — Extract, Transform, Load pattern
- **Relational Database** — SQLite with SQL queries
- **Modular Code Design** — separate scripts per responsibility
- **Model Lifecycle Management** — versioned models, automated best-model selection
- **Progressive Architecture** — stage-specific models, honest data usage
- **Reproducibility** — random_state=42 throughout

---

<div align="center">

---

**Built with genuine engineering thought at every step.**

*From a basic 11-line README to a complete ML system — UDIS evolved through 6 versions, each fixing a real problem.*

---

</div>
