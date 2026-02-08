import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_risk(pass_percent, attendance):
    data = np.array([[pass_percent, attendance]])
    return model.predict(data)[0]

if __name__ == "__main__":
    print(predict_risk(65, 70))
