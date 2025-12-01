Patient Readmission Prevention System

 Project Overview

This project is a Machine Learning solution designed to predict the likelihood of a patient being readmitted to the hospital within 30 days of discharge. By identifying high-risk patients early, healthcare providers can intervene with better post-discharge care plans.

The project consists of a Jupyter Notebook (where the model was researched and built) and a Streamlit Web App (where the model is deployed for real-world use).

File Structure

wep_app.py: The interactive web dashboard built with Streamlit. This is the interface for doctors/nurses.

clinical_readmission.ipynb: The Jupyter Notebook containing the full Data Science pipeline: EDA, Preprocessing, Hyperparameter Tuning (GridSearch), and Model Training.

readmission_model.pkl: The serialized (saved) Random Forest model used by the web app.

Clinical_Dataset.csv: The dataset used for training and testing.

requirements.txt: List of dependencies required to run the project.

Tech Stack

Python (Core Logic)

Scikit-Learn (Random Forest Classifier, GridSearch)

Pandas & NumPy (Data Manipulation)

Streamlit (Web Interface)

Matplotlib & Seaborn (Visualization)

How to Run Locally

1. Setup

Clone the repository (or download the files) and install the required libraries:

pip install -r requirements.txt


2. The Model (Optional)

The model is already trained and saved as readmission_model.pkl.
If you wish to retrain it or see the analysis, open the notebook:

Open clinical_readmission.ipynb in Jupyter Notebook or VS Code.

Run all cells to perform EDA and regenerate the model file.

3. Launch the Web App

To start the dashboard, run the following command in your terminal:

streamlit run wep_app.py


Model Performance

The model uses a Random Forest Classifier with class_weight='balanced' to handle the dataset imbalance.

Optimization Goal: High Recall (Sensitivity) to minimize missed high-risk cases.

Key Predictors: Hemoglobin Level, Primary Diagnosis (Sepsis/Malaria), and Length of Stay.

Future Improvements

Integration with live hospital databases (SQL).

Deployment to Streamlit Cloud.