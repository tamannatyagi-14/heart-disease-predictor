import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_files():
    model = joblib.load("LogisticRegression_heart.pkl")
    scaler = joblib.load("scaler_heart.pkl")
    columns = joblib.load("columns_heart.pkl")
    return model, scaler, columns

model, scaler, columns = load_files()

st.title("❤️ Heart Disease Predictor")
st.write("Enter the patient's details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # start with all 20 columns set to 0
    row = dict.fromkeys(columns, 0)

    # numeric inputs
    row["Age"] = age
    row["RestingBP"] = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"] = fasting_bs
    row["MaxHR"] = max_hr
    row["Oldpeak"] = oldpeak

    # one-hot inputs: set the matching column to 1
    # (ST_Slope = "Down" has no column, so all slope columns stay 0)
    for key in [
        f"Sex_{sex}",
        f"ChestPainType_{chest_pain}",
        f"RestingECG_{resting_ecg}",
        f"ExerciseAngina_{exercise_angina}",
        f"ST_Slope_{st_slope}",
    ]:
        if key in row:
            row[key] = 1

    input_df = pd.DataFrame([row], columns=columns)

    # scale and predict
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease (probability: {probability:.1%})")
    else:
        st.success(f"Low risk of heart disease (probability: {1 - probability:.1%})")
        # --- Why this result? ---
    contributions = pd.Series(
        model.coef_[0] * scaled[0],
        index=[c.strip() for c in columns],
    )
    contributions = contributions.groupby(level=0).sum()
    top = contributions.sort_values(key=abs, ascending=False).head(6)

    st.subheader("What pushed this result?")
    st.caption("Positive = pushes toward heart disease, negative = pushes away")
    st.bar_chart(top)
    
    st.caption("This is a student ML project, not a medical diagnosis.")
 