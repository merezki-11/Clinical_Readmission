import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the Saved Model and use a try/except block in case the isn't found
try:
    package = joblib.load("readmission_model.pkl")
    model = package["model"]
    scaler = package["scaler"]
    le = package["encoder"]
    model_columns = package["columns"]

except FileNotFoundError:
    st.error("Model file not found! Please run 'clinical_readmission' first.")
    st.stop()

# Page Setup
st.set_page_config(page_title="Readmission Model", page_icon="🏥")
st.title("🏥 Hospital Readmission Predictor")
st.markdown("Enter patient details below to assess 30-day readmission risk.")

# User Inputs
   # We are going to divide the hospital input layouts into two columns fpr a cleaner look
   # col1 and col2. col1 encompasses all the columns that could contain entries from a wide range
   # of values (e.g 0-100) and is not explicitly stated in the dataset, while col2 are columns
   # where the user has to choose the category they fall into based on the predefined sections in the dataset


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=110, value=45)
    hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=1.0, max_value=20.0, value=12.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    diagnosis = st.selectbox("Primary Diagnosis", ["Sepsis", "Severe Malaria", "Typhoid Fever", "Other"])
    treatment = st.selectbox("Treatment", ["Broad-Spectrum Antibiotics", "IV Artesunate", "Ceftriaxone", "Other"])
    stay_length = st.number_input("Length of Stay (Days)", min_value=1, max_value=60, value=3)

# Prediction Logic
if st.button("Asses Risk"):
    # 1. Create a DataFrame from the users input
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Primary_Diagnosis": diagnosis,
        "Treatment": treatment,
        "Hemoglobin_Level": hemoglobin,
        "Length_of_Stay": stay_length
    }])
    # 2. Apply Preprocessing (It Must match the clinical_readmission.ipynb exactly)
    # encode gender first
    # If the user selects something the model hasn't seen, the model should notify the user
    try:
        input_df["Gender"] = le.transform(input_df["Gender"])
    except ValueError:
        st.error("Error: Please enter a valid gender.")
        st.stop()

    # One-Hot Encoding for diagnosis and Treatment
    input_df = pd.get_dummies(input_df, columns=["Primary_Diagnosis", "Treatment"], drop_first=True)


       # Aligning the columns: Here is where we ensure the input has the same columns as the training data
       #  This adds 0s for missing columns (e.g., if user selected Malaria, Sepsis column becomes 0)
    

    input_df = input_df.reindex(columns=model_columns,fill_value=0)

    # Scale the numerical values
    input_df_scaled = scaler.transform(input_df)

    # 3. Make Prediction
    probability = model.predict_proba(input_df_scaled)[0, 1]

    # 4. Display Results
    st.subheader("Results")

    # We'll use a threshold of 0.4 for safety
    if probability > 0.4:
        st.error(f"⚠️ HIGH RISK: Readmission Probability is {probability:.1%}")
        st.write("Recommendation: Keep patient for observation or schedule early follow-up.")

    else:
        st.success(f"✅ LOW RISK: Readmission Probability is {probability:.1%}")
        st.write("Recommendation: Safe for standard discharge.")