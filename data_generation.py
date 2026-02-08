import pandas as pd
import random

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
