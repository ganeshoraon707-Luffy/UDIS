import sqlite3
import pandas as pd

conn = sqlite3.connect("data/university.db")

df = pd.read_csv("raw_data.csv")
df.to_sql("academic_data", conn, if_exists="replace", index=False)

conn.close()
print("Data stored in SQL database.")
