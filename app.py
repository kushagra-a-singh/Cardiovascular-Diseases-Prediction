import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the trained models and preprocessors
model = joblib.load('trained_model_checkpoints/multi_output_model.pkl')
scaler = joblib.load('trained_model_checkpoints/scaler.pkl')
poly = joblib.load('trained_model_checkpoints/poly.pkl')

# Streamlit app setup
st.title('Heart Disease Prediction')

# Collecting user input
age = st.number_input("Age", min_value=29, max_value=77, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=94, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (chol)", min_value=126, max_value=564, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=202, value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Depression Induced by Exercise Relative to Rest (oldpeak)", min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["up", "flat", "down"])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

if st.button("Predict"):
    try:
        # Create initial DataFrame with binary and numerical values
        input_data = pd.DataFrame({
            'age': [age],
            'sex_Male': [1 if sex == "Male" else 0],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs_True': [1 if fbs == "True" else 0],
            'thalach': [thalach],
            'exang_Yes': [1 if exang == "Yes" else 0],
            'oldpeak': [oldpeak],
            'ca': [ca],
            'target': [0],  # Adding target column as it was present during training
        })
        
        # Add chest pain type dummy variables
        cp_dummies = {
            'cp_atypical_angina': [1 if cp == "atypical angina" else 0],
            'cp_non-anginal_pain': [1 if cp == "non-anginal pain" else 0],
            'cp_typical_angina': [1 if cp == "typical angina" else 0]
        }
        input_data = pd.concat([input_data, pd.DataFrame(cp_dummies)], axis=1)
        
        # Add restecg dummy variables
        restecg_dummies = {
            'restecg_ST-T_wave_abnormality': [1 if restecg == "ST-T wave abnormality" else 0],
            'restecg_left_ventricular_hypertrophy': [1 if restecg == "left ventricular hypertrophy" else 0]
        }
        input_data = pd.concat([input_data, pd.DataFrame(restecg_dummies)], axis=1)
        
        # Add slope dummy variables
        slope_dummies = {
            'slope_flat': [1 if slope == "flat" else 0],
            'slope_up': [1 if slope == "up" else 0]
        }
        input_data = pd.concat([input_data, pd.DataFrame(slope_dummies)], axis=1)
        
        # Add thal dummy variables
        thal_dummies = {
            'thal_normal': [1 if thal == "normal" else 0],
            'thal_reversible_defect': [1 if thal == "reversible defect" else 0]
        }
        input_data = pd.concat([input_data, pd.DataFrame(thal_dummies)], axis=1)
        
        # Scale numerical features
        numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
        
        # Ensure the input data has the same columns as the training data
        # This includes the interaction terms created by PolynomialFeatures
        # First, create a DataFrame with all possible features (including interaction terms)
        # Use the feature names from the PolynomialFeatures object
        feature_names = poly.get_feature_names_out(input_data.columns)
        input_data_full = pd.DataFrame(columns=feature_names)
        
        # Fill the input_data_full DataFrame with the values from input_data
        for col in input_data.columns:
            if col in input_data_full.columns:
                input_data_full[col] = input_data[col]
            else:
                input_data_full[col] = 0  # Fill missing columns with 0
        
        # Apply polynomial features transformation
        X_poly = poly.transform(input_data_full)
        
        # Ensure the transformed data has the correct feature names
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(X_poly_df)
        
        # Display results
        st.subheader("Prediction Results:")
        disease_types = ['Non-Anginal Pain', 'Stable Angina (Typical Angina)', 'Unstable Angina (Atypical Angina)']
        
        for i, disease in enumerate(disease_types):
            probability = prediction[0][i]
            st.write(f"{disease}: {'High Risk' if probability > 0.5 else 'Low Risk'} ({probability:.2%} probability)")
            st.progress(float(probability))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug - Error details:", str(e))

# Add information about the model
st.sidebar.header('Model Information')
st.sidebar.write("""
This model uses a Random Forest Classifier to predict the likelihood of different types of heart disease based on patient data.
The model considers multiple factors including:
- Demographics (age, sex)
- Clinical measurements (blood pressure, cholesterol)
- Test results (ECG, fluoroscopy)
- Symptoms (chest pain type, angina)
""")