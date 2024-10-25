import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = pickle.load(open('stroke_model', 'rb'))

# Function to handle prediction
def main():
    st.title("Stroke Prediction App")
    st.sidebar.write("Enter the required information to predict the likelihood of stroke.")

# Collect input features
    out = {}
    out['age'] = st.slider("Age", 0, 100, 50)
    out['hypertension'] = 1 if st.radio("Hypertension", ["Yes", "No"]) == "Yes" else 0
    out['heart_disease'] = 1 if st.radio("Heart Disease", ["Yes", "No"]) == "Yes" else 0
    out['gender'] = st.selectbox("Gender", ["Male", "Female"])
    out['ever_married'] = 1 if st.radio("Ever Married", ["Yes", "No"]) == "Yes" else 0
    work_type = ["children", "Self-employed", "Private", "Govt_job", "Never_worked"]
    out['work_type'] = st.selectbox("Work Type", work_type)
    out['Residence_type'] = 1 if st.radio("Residence Type", ["Urban", "Rural"]) == "Urban" else 0
    out['avg_glucose_level'] = st.slider("Avg Glucose Level", 50, 300, 150)
    out['bmi'] = st.slider("BMI", 10, 100, 25)
    smoking_status = ["formerly smoked", "never smoked", "smokes"]
    out['smoking_status'] = st.selectbox("Smoking Status", smoking_status)
    out['smoking_not_found'] = 1 if st.radio("Smoking Not Found", ["Yes", "No"]) == "Yes" else 0

    # Prepare data for prediction
    data = {
        "gender": 1 if out['gender'] == "Male" else 0,
        "age": out['age'],
        "hypertension": out['hypertension'],
        "heart_disease": out['heart_disease'],
        "ever_married": out['ever_married'],
        "work_type": work_type.index(out['work_type']),
        "Residence_type": out['Residence_type'],
        "avg_glucose_level": out['avg_glucose_level'],
        "bmi": out['bmi'],
        "smoking_status": smoking_status.index(out['smoking_status']),
        "smoking_not_found": out['smoking_not_found']
    }

    # Perform the prediction
    if st.button("Predict"):
        features = np.array(list(out.values())).reshape(1, -1)
        prediction = model.predict(features)
        st.write("The prediction: " + ("Potential Stroke" if prediction[0] == 1 else "Clear"))


def show_explore_page():
    df = pd.read_csv("stroke.csv")
    num_cols = ["bmi", "avg_glucose_level", "age"]
    target_col = "stroke"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    st.write("### Histogram for Stroke Numerical Columns")
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, ax=ax[i], hue=target_col, kde=True, bins=50)
    st.pyplot(fig)

    st.write("### Countplot for Stroke Output")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="stroke", ax=ax)
    st.pyplot(fig)

    st.write("### Countplot for Smoking Status")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="smoking_status", ax=ax)
    st.pyplot(fig)

    st.write("### Countplot for Work Type")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="work_type", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# Sidebar to choose between "Predict" or "Explore"
page = st.sidebar.selectbox("Explore Or Prediction", ("Predict", "Explore"))
if page == "Predict":
    main()
else:
    show_explore_page()
