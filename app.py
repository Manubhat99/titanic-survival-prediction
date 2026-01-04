import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
sibsp = st.number_input("Siblings/Spouse", 0, 5, 0)
parch = st.number_input("Parents/Children", 0, 5, 0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master"])

family_size = sibsp + parch + 1

input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

input_df["Age"] = age
input_df["Fare"] = fare
input_df["SibSp"] = sibsp
input_df["Parch"] = parch
input_df["FamilySize"] = family_size
input_df["Pclass"] = pclass
input_df["Sex_male"] = 1 if sex == "male" else 0

if embarked == "Q":
    input_df["Embarked_Q"] = 1
elif embarked == "S":
    input_df["Embarked_S"] = 1

title_col = f"Title_{title}"
if title_col in input_df.columns:
    input_df[title_col] = 1

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    st.success("🎉 Survived" if prediction == 1 else "❌ Not Survived")
