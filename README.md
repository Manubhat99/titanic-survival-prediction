# 🚢 Titanic Survival Prediction

## 📌 Project Overview
The **Titanic Survival Prediction** project is a machine learning application that predicts whether a passenger survived the Titanic disaster based on personal and travel-related details.  
The project demonstrates a complete **end-to-end data science workflow**, including data preprocessing, feature engineering, model training, and deployment using **Streamlit**.

This project is built to showcase practical skills in **Python, data analysis, machine learning, and web-based ML deployment**.

---

## 🎯 Objective
The main objective of this project is to:
- Analyze passenger data from the Titanic dataset
- Build a machine learning model to predict survival
- Deploy the model through an interactive web application
- Allow users to input passenger details and receive real-time predictions

---

## 📂 Dataset
- **Source:** Kaggle – Titanic Dataset  
- **Files Used:**
  - `train.csv`
  - `test.csv`

### Key Features:
- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Fare
- Port of Embarkation
- Title extracted from passenger name
- Family Size (engineered feature)

---

## 🛠️ Technologies & Tools Used
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Version Control:** Git & GitHub

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Handled missing values in Age and Embarked columns
- Removed irrelevant features
- Converted categorical variables using one-hot encoding
- Extracted passenger titles from names
- Created a new feature called `FamilySize`

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed survival trends based on:
  - Gender
  - Passenger class
  - Age groups
  - Family size
- Visualized important patterns affecting survival rates

---

### 3️⃣ Feature Engineering
- Combined SibSp and Parch to form FamilySize
- Extracted titles such as Mr, Mrs, Miss, Master
- Encoded categorical variables for model compatibility

---

### 4️⃣ Model Building
- Used **Random Forest Classifier** for prediction
- Trained the model on processed training data
- Evaluated model performance using accuracy and classification metrics
- Saved the trained model and feature list for deployment

---

### 5️⃣ Model Deployment (Streamlit)
- Built an interactive web application using Streamlit
- User inputs passenger details via sidebar
- Model predicts survival outcome in real time
- Clean and professional UI for better user experience

---
✅ How to Run the Project Locally
🔹 Step 1: Clone the Repository
git clone https://github.com/Manubhat99/titanic-survival-prediction.git
cd titanic-survival-prediction

🔹 Step 2: Install Required Libraries
pip install pandas numpy scikit-learn streamlit joblib

🔹 Step 3: Run the Streamlit Application
python -m streamlit run app.py

🔹 Step 4: Open the Application in Browser

After running the command, open your browser and go to:

http://localhost:8501

📂 Project Structure
titanic-survival-prediction/
│
├── app.py                  # Streamlit application
├── titanic.ipynb           # Data analysis & model training
├── titanic_model.pkl       # Trained ML model
├── feature_columns.pkl     # Feature list used during training
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── README.md               # Project documentation

📊 Results

The model successfully predicts survival outcomes based on user input

Demonstrates the importance of features such as gender, class, and age

Provides an intuitive interface for non-technical users

🧠 Key Learnings

End-to-end machine learning workflow

Importance of feature consistency between training and deployment

Practical experience with Streamlit deployment

Handling real-world data challenges

👤 Author

Mahabaleshwar Bhat
Aspiring Data Scientist
GitHub: https://github.com/Manubhat99

🚀 Future Enhancements

Add survival probability score

Improve model accuracy with hyperparameter tuning

Deploy application on Streamlit Cloud

Add more visual analytics to the UI
