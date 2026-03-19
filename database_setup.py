"""
database_setup.py
Loads the raw CSV into an SQLite database.
"""

import sqlite3
import pandas as pd
import os


def setup_database(
    csv_path: str = "data/raw_data.csv",
    db_path: str = "data/university.db"
) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df = pd.read_csv(csv_path)
    df.to_sql("department_data", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Stored {len(df)} rows in {db_path} -> table: department_data")


if __name__ == "__main__":
    setup_database()
