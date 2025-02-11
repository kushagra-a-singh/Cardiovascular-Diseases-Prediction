import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained models and preprocessors
@st.cache_resource
def load_models():
    model = joblib.load('trained_model_checkpoints/multi_output_model.pkl')
    scaler = joblib.load('trained_model_checkpoints/scaler.pkl')
    return model, scaler

# Function to calculate confidence intervals
def calculate_confidence_interval(probability, n_samples=1000):
    # Simulate a binomial distribution
    simulated = np.random.binomial(n=1, p=probability, size=n_samples)
    ci_lower = np.percentile(simulated, 2.5)
    ci_upper = np.percentile(simulated, 97.5)
    return ci_lower, ci_upper

# Function to create feature columns in correct order
def create_input_features(data_dict):
    columns = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'target',
        'sex_Male', 'cp_atypical_angina', 'cp_non-anginal_pain', 'cp_typical_angina',
        'fbs_True', 'restecg_ST-T_wave_abnormality', 'restecg_left_ventricular_hypertrophy',
        'exang_Yes', 'slope_flat', 'slope_up', 'thal_normal', 'thal_reversible_defect'
    ]
    df = pd.DataFrame(0, index=[0], columns=columns)
    for col in data_dict:
        if col in df.columns:
            df[col] = data_dict[col]
    return df

# Function to validate input values
def validate_inputs(age, trestbps, chol, thalach, oldpeak):
    warnings = []
    if age < 30 or age > 77:
        warnings.append("Age is outside typical range (30-77 years)")
    if trestbps < 90 or trestbps > 200:
        warnings.append("Blood pressure is outside normal range (90-200 mmHg)")
    if chol < 120 or chol > 600:
        warnings.append("Cholesterol is outside typical range (120-600 mg/dl)")
    if thalach < 60 or thalach > 220:
        warnings.append("Maximum heart rate is outside normal range (60-220 bpm)")
    return warnings

# Emergency warning signs function
def check_emergency_signs(cp, trestbps, thalach):
    emergency_signs = []
    if cp == "typical angina":
        emergency_signs.append("⚠️ You reported typical chest pain (angina)")
    if trestbps > 180:
        emergency_signs.append("⚠️ Your blood pressure is very high")
    if thalach > 200:
        emergency_signs.append("⚠️ Your heart rate is very high")
    return emergency_signs

# Set page config
st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .emergency-warning {
        background-color: #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        color: white;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #ffd700;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title with description
st.title('Heart Disease Risk Assessment Tool')
st.markdown("""
    This tool uses machine learning to assess your risk for different types of heart disease.
    Please fill in all fields accurately for the best results.
    """)

# Important medical disclaimer
st.warning("""
    🏥 MEDICAL DISCLAIMER: This tool is for educational and screening purposes only.
    It is not a substitute for professional medical diagnosis or advice.
    If you're experiencing chest pain or other concerning symptoms, seek immediate medical attention.
    """)

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Input Data"

# Create tabs for different sections
tabs = st.tabs(["Input Data", "Results & Analysis", "Information"])

# Determine the active tab
if st.session_state.selected_tab == "Input Data":
    input_tab = tabs[0]
    results_tab = tabs[1]
    info_tab = tabs[2]
elif st.session_state.selected_tab == "Results & Analysis":
    input_tab = tabs[0]
    results_tab = tabs[1]
    info_tab = tabs[2]
    st.session_state.selected_tab = "Results & Analysis" # Keep this tab active after prediction
elif st.session_state.selected_tab == "Information":
    input_tab = tabs[0]
    results_tab = tabs[1]
    info_tab = tabs[2]
    info_tab.select()  # Keep this tab active when navigating to the "Information" tab

with input_tab:
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        # Demographic information
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=29, max_value=77, value=50, help="Patient's age in years")
            sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex")
            
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=120,
                                     help="Resting blood pressure in millimeters of mercury")
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=200,
                                 help="Serum cholesterol level in mg/dl")

        # Clinical features
        st.subheader("Clinical Information")
        col3, col4 = st.columns(2)
        with col3:
            cp = st.selectbox("Chest Pain Type", 
                            ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
                            help="Type of chest pain experienced")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                             ["False", "True"],
                             help="Whether fasting blood sugar is greater than 120 mg/dl")
            
        with col4:
            restecg = st.selectbox("Resting ECG Results",
                                 ["Normal", "ST-T wave abnormality", "left ventricular hypertrophy"],
                                 help="Results of resting electrocardiogram")
            thalach = st.number_input("Maximum Heart Rate", 
                                    min_value=71, max_value=202, value=150,
                                    help="Maximum heart rate achieved during exercise")

        # Additional clinical features
        col5, col6 = st.columns(2)
        with col5:
            exang = st.selectbox("Exercise Induced Angina",
                               ["No", "Yes"],
                               help="Whether exercise induces angina")
            oldpeak = st.number_input("ST Depression (oldpeak)",
                                    min_value=0.0, max_value=6.2, value=1.0,
                                    help="ST depression induced by exercise relative to rest")
            
        with col6:
            slope = st.selectbox("ST Slope",
                               ["up", "flat", "down"],
                               help="Slope of the peak exercise ST segment")
            ca = st.number_input("Number of Major Vessels (0-4)",
                               min_value=0, max_value=4, value=0,
                               help="Number of major vessels colored by fluoroscopy")

        thal = st.selectbox("Thalassemia",
                          ["normal", "fixed defect", "reversible defect"],
                          help="Results of thallium stress test")
        
        submitted = st.form_submit_button("Analyze Risk")

# Change the 'selected_tab' value after the form submission to ensure the Results & Analysis tab is shown
if submitted:
    # Load the model and scaler
    model, scaler = load_models()
    
    # Validate inputs
    warnings = validate_inputs(age, trestbps, chol, thalach, oldpeak)
    if warnings:
        st.warning("⚠️ Potential input concerns:")
        for warning in warnings:
            st.write(f"- {warning}")

    # Check for emergency signs
    emergency_signs = check_emergency_signs(cp, trestbps, thalach)
    if emergency_signs:
        st.markdown('<div class="emergency-warning">', unsafe_allow_html=True)
        st.write("🚨 SEEK IMMEDIATE MEDICAL ATTENTION:")
        for sign in emergency_signs:
            st.write(sign)
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Create input dictionary and features
        input_dict = {
            'age': [age],
            'trestbps': [trestbps],
            'chol': [chol],
            'thalach': [thalach],
            'oldpeak': [oldpeak],
            'ca': [ca],
            'target': [0],
            'sex_Male': [1 if sex == "Male" else 0],
            'cp_atypical_angina': [1 if cp == "atypical angina" else 0],
            'cp_non-anginal_pain': [1 if cp == "non-anginal pain" else 0],
            'cp_typical_angina': [1 if cp == "typical angina" else 0],
            'fbs_True': [1 if fbs == "True" else 0],
            'restecg_ST-T_wave_abnormality': [1 if restecg == "ST-T wave abnormality" else 0],
            'restecg_left_ventricular_hypertrophy': [1 if restecg == "left ventricular hypertrophy" else 0],
            'exang_Yes': [1 if exang == "Yes" else 0],
            'slope_flat': [1 if slope == "flat" else 0],
            'slope_up': [1 if slope == "up" else 0],
            'thal_normal': [1 if thal == "normal" else 0],
            'thal_reversible_defect': [1 if thal == "reversible defect" else 0]
        }

        input_data = create_input_features(input_dict)
        
        # Scale numerical features
        numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Switch to results tab
        st.session_state.selected_tab = "Results & Analysis"  # Update tab selection

        with results_tab:
            st.subheader("Risk Assessment Results")
            
            disease_types = [
                'Non-Anginal Pain',
                'Stable Angina (Typical Angina)',
                'Unstable Angina (Atypical Angina)'
            ]

            # Create three columns for the results
            result_cols = st.columns(len(disease_types))
            
            for idx, (disease, col) in enumerate(zip(disease_types, result_cols)):
                with col:
                    probability = prediction_proba[idx][0][1]
                    ci_lower, ci_upper = calculate_confidence_interval(probability)
                    risk_level = "High Risk" if probability > 0.5 else "Low Risk"
                    
                    st.metric(
                        label=disease,
                        value=f"{probability:.1%}",
                        delta=risk_level,
                        delta_color="inverse"
                    )
                    st.progress(probability)
                    st.write(f"95% Confidence Interval: {ci_lower:.1%} - {ci_upper:.1%}")

            # Feature importance analysis
            st.subheader("Key Factors Influencing the Prediction")
            
            # Create SHAP values for feature importance
            explainer = shap.TreeExplainer(model.estimators_[0])
            shap_values = explainer.shap_values(input_data)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()

            # Detailed interpretation
            st.subheader("Result Interpretation")
            st.write("""
            #### Risk Levels Explained:
            - **High Risk (>50%)**: Further medical evaluation is recommended
            - **Low Risk (<50%)**: Regular health monitoring advised
            
            #### Important Notes:
            - These results are based on statistical models and should be interpreted by a healthcare professional
            - Risk levels can change over time and with lifestyle modifications
            - Regular check-ups are important regardless of risk level
            """)

            # Recommendations based on risk levels
            st.subheader("Recommendations")
            if any(prediction_proba[i][0][1] > 0.5 for i in range(len(disease_types))):
                st.error("""
                Based on the high-risk assessment:
                1. Schedule an appointment with your healthcare provider
                2. Bring these results to your appointment
                3. Monitor your symptoms closely
                4. Keep a log of any chest pain or discomfort
                """)
            else:
                st.success("""
                Based on the low-risk assessment:
                1. Continue regular health check-ups
                2. Maintain a heart-healthy lifestyle
                3. Monitor any changes in symptoms
                4. Follow your doctor's preventive care recommendations
                """)

    except Exception as e:
        st.error(f"An error occurred: {e}")

with info_tab:
    st.subheader("About This Tool")
    st.write("""
    #### How It Works
    This tool uses machine learning algorithms trained on clinical data to assess heart disease risk.
    The model considers multiple factors including:
    - Demographic information
    - Clinical measurements
    - Test results
    - Symptoms
    
    #### Limitations
    - This tool is not a diagnostic device
    - Results are based on statistical models
    - Individual cases may vary
    - Medical history and other factors may affect actual risk
    
    #### When to Seek Emergency Care
    Call emergency services immediately if you experience:
    - Severe chest pain or pressure
    - Difficulty breathing
    - Sudden weakness or dizziness
    - Pain spreading to arms, neck, or jaw
    """)

    st.write("""
    #### Data Privacy Notice
    - All data is processed locally
    - No personal information is stored
    - Input data is cleared after each session
    """)

# Sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This tool uses advanced machine learning to assess heart disease risk based on clinical data.

Key Features:
- Multi-disease risk assessment
- Confidence intervals
- Feature importance analysis
- Emergency warning system
- Detailed recommendations

Remember: This is a screening tool only and should not replace professional medical advice.
""")