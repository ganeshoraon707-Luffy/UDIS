import sqlite3
import pandas as pd

def load_clean_data():
    conn = sqlite3.connect("data/university.db")
    df = pd.read_sql("SELECT * FROM academic_data", conn)
    conn.close()

    df = df.dropna()
    df["pass_percent"] = df["pass_percent"].clip(0,100)
    df["attendance"] = df["attendance"].clip(0,100)

    # Create risk label
    def assign_risk(row):
        if row["pass_percent"] >= 75 and row["attendance"] >= 75:
            return "Low"
        elif row["pass_percent"] >= 60 and row["attendance"] >= 60:
            return "Medium"
        else:
            return "High"

    df["risk"] = df.apply(assign_risk, axis=1)

    return df
