import pandas as pd
import random
"""
generate_data.py
Generates synthetic department-level academic performance data
matching a real university marking scheme:
  UT1       = out of 10
  Midterm   = out of 20
  UT2       = out of 10
  End Exam  = out of 40
  Attendance = percentage (0-100%)
  Total marks = 80 (academic) + 20 (attendance/internal)
"""

import pandas as pd
import random
import os

random.seed(42)

DEPARTMENTS = [
    "CSE", "AI", "Data Science", "Cyber Security",
    "Mechanical", "Civil", "Electrical", "Electronics", "Robotics",
    "BBA", "MBA", "Finance", "Marketing",
    "Business Analytics", "Entrepreneurship",
    "Classical Music", "Instrumental Music", "Vocal Music", "Music Production",
    "Fashion Design", "Graphic Design", "UI/UX Design",
    "Industrial Design", "Animation Design",
]

# Real university max marks
UT1_MAX  = 10
MID_MAX  = 20
UT2_MAX  = 10
END_MAX  = 40


def _clamp(val, lo, hi):
    return max(lo, min(hi, round(val, 1)))


def generate_row(dept: str) -> dict:
    """
    Generate marks on the real university scale.
    Six archetypes to create realistic overlap between risk classes.
    """
    archetype = random.choices(
        ["clear_high", "clear_low", "good_marks_bad_att",
         "bad_marks_good_att", "inconsistent", "middle_ground"],
        weights=[15, 15, 17, 17, 18, 18]
    )[0]

    if archetype == "clear_high":
        # Weak performance across all exams
        ut1 = round(random.uniform(2.0, 5.0), 1)
        mid = round(random.uniform(4.0, 10.0), 1)
        ut2 = round(random.uniform(2.0, 5.0), 1)
        end = round(random.uniform(8.0, 20.0), 1)
        ut1_att = random.randint(40, 60)
        mid_att = _clamp(ut1_att - random.randint(0, 8), 35, 100)
        ut2_att = _clamp(mid_att - random.randint(0, 8), 35, 100)
        end_att = _clamp(ut2_att - random.randint(0, 8), 35, 100)

    elif archetype == "clear_low":
        # Strong performance across all exams
        ut1 = round(random.uniform(7.5, 10.0), 1)
        mid = round(random.uniform(15.0, 20.0), 1)
        ut2 = round(random.uniform(7.5, 10.0), 1)
        end = round(random.uniform(30.0, 40.0), 1)
        ut1_att = random.randint(85, 100)
        mid_att = _clamp(ut1_att - random.randint(0, 5), 35, 100)
        ut2_att = _clamp(mid_att - random.randint(0, 5), 35, 100)
        end_att = _clamp(ut2_att - random.randint(0, 5), 35, 100)

    elif archetype == "good_marks_bad_att":
        # Good marks but poor attendance — borderline case
        ut1 = round(random.uniform(6.0, 9.0), 1)
        mid = round(random.uniform(12.0, 18.0), 1)
        ut2 = round(random.uniform(6.0, 9.0), 1)
        end = round(random.uniform(22.0, 34.0), 1)
        ut1_att = random.randint(40, 60)
        mid_att = _clamp(ut1_att - random.randint(0, 6), 35, 100)
        ut2_att = _clamp(mid_att - random.randint(0, 6), 35, 100)
        end_att = _clamp(ut2_att - random.randint(0, 8), 35, 75)

    elif archetype == "bad_marks_good_att":
        # Low marks but excellent attendance — borderline case
        ut1 = round(random.uniform(3.0, 6.5), 1)
        mid = round(random.uniform(6.0, 13.0), 1)
        ut2 = round(random.uniform(3.0, 6.5), 1)
        end = round(random.uniform(10.0, 22.0), 1)
        ut1_att = random.randint(85, 100)
        mid_att = _clamp(ut1_att - random.randint(0, 4), 35, 100)
        ut2_att = _clamp(mid_att - random.randint(0, 4), 35, 100)
        end_att = _clamp(ut2_att - random.randint(0, 4), 35, 100)

    elif archetype == "inconsistent":
        # Wild swings between exams
        ut1 = round(random.uniform(2.0, 10.0), 1)
        mid = round(random.uniform(4.0, 20.0), 1)
        ut2 = round(random.uniform(2.0, 10.0), 1)
        end = round(random.uniform(8.0, 40.0), 1)
        ut1_att = random.randint(40, 95)
        mid_att = _clamp(ut1_att + random.randint(-20, 15), 35, 100)
        ut2_att = _clamp(mid_att + random.randint(-20, 15), 35, 100)
        end_att = _clamp(ut2_att + random.randint(-20, 15), 35, 100)

    else:  # middle_ground
        # Average performance hovering near boundaries
        ut1 = round(random.uniform(4.5, 7.5), 1)
        mid = round(random.uniform(9.0, 15.0), 1)
        ut2 = round(random.uniform(4.5, 7.5), 1)
        end = round(random.uniform(18.0, 28.0), 1)
        ut1_att = random.randint(55, 80)
        mid_att = _clamp(ut1_att + random.randint(-10, 8), 35, 100)
        ut2_att = _clamp(mid_att + random.randint(-10, 8), 35, 100)
        end_att = _clamp(ut2_att + random.randint(-10, 8), 35, 100)

    return dict(
        department=dept,
        ut1_marks=ut1, mid_marks=mid, ut2_marks=ut2, end_marks=end,
        ut1_attendance=ut1_att, mid_attendance=mid_att,
        ut2_attendance=ut2_att, end_attendance=end_att,
    )


def generate_dataset(n: int = 1500) -> pd.DataFrame:
    rows = [generate_row(random.choice(DEPARTMENTS)) for _ in range(n)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(1500)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Dataset generated: {len(df)} rows -> data/raw_data.csv")
    print(f"Mark ranges:")
    print(f"  UT1  : {df['ut1_marks'].min():.1f} - {df['ut1_marks'].max():.1f}  (out of 10)")
    print(f"  Mid  : {df['mid_marks'].min():.1f} - {df['mid_marks'].max():.1f}  (out of 20)")
    print(f"  UT2  : {df['ut2_marks'].min():.1f} - {df['ut2_marks'].max():.1f}  (out of 10)")
    print(f"  End  : {df['end_marks'].min():.1f} - {df['end_marks'].max():.1f}  (out of 40)")

def generate_data(n=200):
    data = []
    for i in range(n):
        pass_percent = random.randint(40, 90)
        attendance = random.randint(50, 95)

        data.append({
            "department": random.choice(["CSE", "ECE", "ME", "CE"]),
            "pass_percent": pass_percent,
            "attendance": attendance
        })

    df = pd.DataFrame(data)
    df.to_csv("raw_data.csv", index=False)
    print("Data generated successfully.")

if __name__ == "__main__":
    generate_data()
