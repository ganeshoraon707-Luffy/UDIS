import streamlit as st
import pickle
import numpy as np

st.title("University Decision Intelligence System")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

pass_percent = st.slider("Pass Percentage", 40, 100, 60)
attendance = st.slider("Attendance", 40, 100, 70)

if st.button("Predict Risk"):
    prediction = model.predict([[pass_percent, attendance]])
    st.success(f"Predicted Risk Level: {prediction[0]}")
