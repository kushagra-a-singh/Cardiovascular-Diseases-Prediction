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
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
        'sex_Male', 'cp_atypical_angina', 'cp_non_anginal_pain', 'cp_typical_angina',
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
def check_emergency_signs(cp_typical_angina, cp_non_anginal_pain, cp_atypical_angina, trestbps, thalach):
    emergency_signs = []
    if cp_typical_angina == 1:
        emergency_signs.append("⚠️ You reported typical chest pain (angina)")
    if cp_non_anginal_pain == 1:
        emergency_signs.append("⚠️ You reported non-anginal pain")
    if cp_atypical_angina == 1:
        emergency_signs.append("⚠️ You reported atypical chest pain (angina)")
    if trestbps < 120:
        emergency_signs.append("⚠️ Your blood pressure is low")
    if trestbps > 180:
        emergency_signs.append("⚠️ Your blood pressure is very high")
    if thalach > 200:
        emergency_signs.append("⚠️ Your heart rate is very high")
    return emergency_signs

# Function to get disease-specific information
def get_disease_info(disease_name):
    info = {
        'Non-Anginal Pain': {
            'description': 'Chest pain that is not related to heart disease but may be caused by other conditions such as digestive issues, muscle strain, or anxiety.',
            'symptoms': ['Chest discomfort not triggered by physical activity', 'Pain that may be sharp and localized', 'Symptoms that last for hours or days', 'Pain that may be reproduced by pressing on the chest'],
            'causes': ['Muscle strain', 'Anxiety or panic attacks', 'Acid reflux or GERD', 'Costochondritis (inflammation of the chest wall)'],
            'prevention': ['Stress management techniques', 'Treatment of underlying conditions like acid reflux', 'Regular exercise', 'Maintaining good posture']
        },
        'Stable Angina (Typical Angina)': {
            'description': 'Chest pain that occurs when the heart muscle doesn\'t get enough oxygen-rich blood, usually during physical exertion.',
            'symptoms': ['Chest pain/pressure during physical activity', 'Pain that goes away with rest', 'Discomfort that lasts 3-5 minutes', 'Pain that may radiate to arms, neck, or jaw'],
            'causes': ['Coronary artery disease', 'Atherosclerosis (plaque buildup)', 'Blood clots', 'Coronary artery spasm'],
            'prevention': ['Regular exercise', 'Heart-healthy diet', 'Quitting smoking', 'Controlling blood pressure and cholesterol', 'Maintaining healthy weight']
        },
        'Unstable Angina (Atypical Angina)': {
            'description': 'A more serious condition where chest pain occurs unexpectedly, even at rest, and indicates a higher risk of heart attack.',
            'symptoms': ['Chest pain at rest', 'New onset severe angina', 'Increasing frequency and intensity of pain', 'Pain lasting longer than stable angina', 'Poor response to rest or medication'],
            'causes': ['Atherosclerosis', 'Blood clot partially blocking an artery', 'Coronary artery spasm', 'Inflammation of the arteries'],
            'prevention': ['Emergency medical evaluation', 'Medication adherence', 'Cardiac rehabilitation', 'Risk factor modification', 'Possible surgical intervention']
        },
        'Asymptomatic Heart Disease': {
            'description': 'Heart disease that exists without obvious symptoms, making it particularly dangerous if left undetected.',
            'symptoms': ['Often no noticeable symptoms', 'May be discovered during routine tests', 'Sometimes mild fatigue or shortness of breath with exertion'],
            'causes': ['Same as symptomatic heart disease', 'Atherosclerosis', 'High blood pressure', 'Diabetes', 'Family history'],
            'prevention': ['Regular health check-ups', 'Cardiac screening if at high risk', 'Healthy lifestyle', 'Management of risk factors']
        },
        'Left Ventricular Hypertrophy': {
            'description': 'Thickening of the heart\'s main pumping chamber, which can develop in response to conditions like high blood pressure.',
            'symptoms': ['Often asymptomatic in early stages', 'Shortness of breath', 'Chest pain', 'Palpitations or irregular heartbeat', 'Dizziness or fainting'],
            'causes': ['High blood pressure', 'Heart valve disease', 'Athletic training (athlete\'s heart)', 'Genetic disorders', 'Obesity'],
            'prevention': ['Blood pressure control', 'Regular medical check-ups', 'Limiting intense exercise if diagnosed', 'Medication adherence', 'Healthy lifestyle']
        },
        'Myocardial Infarction (Heart Attack)': {
            'description': 'Occurs when blood flow to part of the heart is blocked, causing damage to the heart muscle.',
            'symptoms': ['Severe chest pain or pressure', 'Pain radiating to arm, jaw, or back', 'Shortness of breath', 'Cold sweat', 'Nausea or vomiting', 'Extreme fatigue'],
            'causes': ['Coronary artery disease', 'Blood clot', 'Coronary artery spasm', 'Severe physical stress'],
            'prevention': ['Emergency treatment if suspected', 'Heart-healthy lifestyle', 'Medication as prescribed', 'Cardiac rehabilitation', 'Regular follow-up care']
        },
        'Coronary Artery Disease (CAD)': {
            'description': 'Narrowing or blockage of coronary arteries, usually caused by atherosclerosis.',
            'symptoms': ['Chest pain (angina)', 'Shortness of breath', 'Heart attack', 'Heart failure', 'Arrhythmias'],
            'causes': ['Atherosclerosis', 'Inflammation', 'High cholesterol', 'Smoking', 'Diabetes', 'Hypertension'],
            'prevention': ['Healthy diet low in saturated fat', 'Regular exercise', 'Not smoking', 'Limiting alcohol', 'Stress management', 'Medication if prescribed']
        },
        'Hypertensive Heart Disease': {
            'description': 'Heart condition caused by high blood pressure, which can lead to heart failure if untreated.',
            'symptoms': ['Often asymptomatic until advanced', 'Shortness of breath', 'Chest pain', 'Fatigue', 'Swelling in legs or abdomen', 'Irregular heartbeat'],
            'causes': ['Chronic high blood pressure', 'Obesity', 'High salt intake', 'Sedentary lifestyle', 'Genetic factors', 'Kidney disease'],
            'prevention': ['Blood pressure monitoring', 'Low-sodium diet', 'Regular exercise', 'Weight management', 'Limiting alcohol', 'Medication adherence']
        }
    }
    return info.get(disease_name, {})

# Set page config
st.set_page_config(page_title="Cardiovascular Diseases Risk Assessment", layout="wide")

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
st.title('Cardiovascular Diseases Risk Assessment Tool')
st.markdown("""
    This tool uses machine learning to assess your risk for different types of cardiovascular diseases.
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

with tabs[0]:  # Input Data tab
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

# Handle form submission
if submitted:
    st.session_state.analysis_completed = True
    # Load the model and scaler
    model, scaler = load_models()
    
    # Validate inputs
    warnings = validate_inputs(age, trestbps, chol, thalach, oldpeak)
    if warnings:
        st.warning("⚠️ Potential input concerns:")
        for warning in warnings:
            st.write(f"- {warning}")

    try:
        # Create input dictionary and features
        input_dict = {
            'age': [age],
            'trestbps': [trestbps],
            'chol': [chol],
            'thalach': [thalach],
            'oldpeak': [oldpeak],
            'ca': [ca],
            'sex_Male': [1 if sex == "Male" else 0],
            'cp_atypical_angina': [1 if cp == "atypical angina" else 0],
            'cp_non_anginal_pain': [1 if cp == "non-anginal pain" else 0],
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
        prediction_proba = model.predict_proba(input_data)  # Get probabilities for all diseases

        # Store results in session state
        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba  # Store probabilities
        st.session_state.input_data = input_data
        st.session_state.model = model
        st.session_state.selected_tab = "Results & Analysis"
    except Exception as e:
        st.error(f"An error occurred: {e}")
if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
    with tabs[0]:
        st.success("Risk assessment complete, please checkout **RESULTS & ANALYSIS** tab for results")

with tabs[1]:  # Results & Analysis tab
    if st.session_state.selected_tab == "Results & Analysis" and 'prediction' in st.session_state:
        # Check for emergency signs
        emergency_signs = check_emergency_signs(
            st.session_state.input_data['cp_typical_angina'][0],
            st.session_state.input_data['cp_atypical_angina'][0],
            st.session_state.input_data['cp_non_anginal_pain'][0],
            st.session_state.input_data['trestbps'][0],
            st.session_state.input_data['thalach'][0]
        )

        if emergency_signs:
            st.markdown('<div class="emergency-warning">', unsafe_allow_html=True)
            st.write("🚨 SEEK IMMEDIATE MEDICAL ATTENTION:")
            for sign in emergency_signs:
                st.write(sign)
            st.markdown('</div>', unsafe_allow_html=True)

        # Define disease types in the same order as they were created during training
        disease_types = [
            'Non-Anginal Pain',
            'Stable Angina (Typical Angina)',
            'Unstable Angina (Atypical Angina)',
            'Asymptomatic Heart Disease',
            'Left Ventricular Hypertrophy',
            'Myocardial Infarction (Heart Attack)',
            'Coronary Artery Disease (CAD)',
            'Hypertensive Heart Disease'
        ]
        
        st.subheader("Risk Assessment Results")
        prediction = st.session_state.prediction
        prediction_proba = st.session_state.prediction_proba
        
        # Create a grid layout for the results (2 rows, 4 columns)
        num_diseases = len(disease_types)
        rows = (num_diseases // 4) + (1 if num_diseases % 4 > 0 else 0)

        for row in range(rows):
            result_cols = st.columns(4)  # Always 4 columns
            start_idx = row * 4
            end_idx = min(start_idx + 4, num_diseases)
            
            for i in range(start_idx, end_idx):
                col_idx = i % 4
                disease = disease_types[i]
                
                with result_cols[col_idx]:
                    probability = prediction_proba[i][0][1]  # Get probability of class 1
                    ci_lower, ci_upper = calculate_confidence_interval(probability)
                    risk_level = "High Risk" if probability > 0.5 else "Low Risk"
                    
                    st.metric(
                        label=disease,
                        value=f"{probability:.1%}",
                        delta=risk_level,
                        delta_color="inverse"
                    )
                    st.progress(probability)
                    st.write(f"95% CI: {ci_lower:.1%} - {ci_upper:.1%}")

                    # Add expander for detailed disease information
                    with st.expander(f"Learn more about {disease}"):
                        disease_info = get_disease_info(disease)
                        
                        if disease_info:
                            st.markdown(f"### About {disease}")
                            st.write(disease_info.get('description', 'No description available.'))
                            
                            st.markdown("### Common Symptoms")
                            symptoms = disease_info.get('symptoms', [])
                            if symptoms:
                                for symptom in symptoms:
                                    st.write(f"• {symptom}")
                            else:
                                st.write("No specific symptoms information available.")
                            
                            st.markdown("### Common Causes")
                            causes = disease_info.get('causes', [])
                            if causes:
                                for cause in causes:
                                    st.write(f"• {cause}")
                            else:
                                st.write("No specific causes information available.")
                            
                            st.markdown("### Prevention & Management")
                            prevention = disease_info.get('prevention', [])
                            if prevention:
                                for tip in prevention:
                                    st.write(f"• {tip}")
                            else:
                                st.write("No specific prevention information available.")
                        else:
                            st.write("Detailed information not available for this condition.")

        # Feature importance analysis
        st.subheader("Key Factors Influencing the Prediction")

        try:
            # Get model and input data from session state
            model = st.session_state.model
            input_data = st.session_state.input_data
            
            # Find first disease with high risk if any
            high_risk_idx = None
            for i, proba in enumerate(st.session_state.prediction_proba):
                if proba[0][1] > 0.5:
                    high_risk_idx = i
                    break
            
            # If no high risk, use first disease
            if high_risk_idx is None:
                disease_idx = 0
            else:
                disease_idx = high_risk_idx
            
            # Display which disease we're showing factors for
            st.write(f"Showing factors for: **{disease_types[disease_idx]}**")
            
            # Get the estimator for the selected disease
            if disease_idx < len(model.estimators_):
                estimator = model.estimators_[disease_idx]
                
                # Get feature importances directly from the random forest
                importances = estimator.feature_importances_
                feature_names = list(input_data.columns)
                
                # Create a dataframe for the feature importances
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                
                # Create barplot
                ax = sns.barplot(x='Importance', y='Feature', data=importance_df)
                
                # Add labels and title
                plt.title(f'Feature Importance for {disease_types[disease_idx]}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                
                # Display the plot
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(plt.gcf())
                plt.close()
                
                st.markdown("""
                **Understanding the Feature Importance Plot:**
                - Each bar shows how much a feature influences the prediction
                - Longer bars indicate stronger impact on the risk assessment
                - Features are ordered by their importance (most important at top)
                - This analysis helps identify which factors are most relevant for your assessment
                """)
            
        except Exception as e:
            st.error(f"Error generating feature importance plot: {e}")
            st.write("We're unable to show the detailed feature analysis at this time.")

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
        if any(proba[0][1] > 0.5 for proba in st.session_state.prediction_proba):
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
        
        # Lifestyle recommendations
        st.subheader("Lifestyle Recommendations")

        # Get the highest risk disease
        max_risk_idx = 0
        max_risk_prob = 0
        for i, proba in enumerate(st.session_state.prediction_proba):
            if proba[0][1] > max_risk_prob:
                max_risk_prob = proba[0][1]
                max_risk_idx = i

        high_risk_disease = disease_types[max_risk_idx]
        disease_info = get_disease_info(high_risk_disease)

        # General recommendations
        st.write("#### General Heart Health Recommendations:")
        st.markdown("""
        - **Physical Activity**: Aim for at least 150 minutes of moderate exercise per week
        - **Healthy Diet**: Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats
        - **Blood Pressure Control**: Monitor regularly and maintain below 120/80 mm Hg
        - **Cholesterol Management**: Keep LDL low and HDL high through diet and medication if prescribed
        - **Stress Management**: Practice relaxation techniques like meditation or deep breathing
        - **Sleep**: Aim for 7-8 hours of quality sleep per night
        - **Regular Check-ups**: Schedule routine visits with your healthcare provider
        """)

        # Specific recommendations based on highest risk
        if disease_info and disease_info.get('prevention'):
            st.write(f"#### Specific Recommendations for {high_risk_disease}:")
            for tip in disease_info.get('prevention'):
                st.write(f"• {tip}")

with tabs[2]:  # Information tab
    st.subheader("About This Tool")
    st.write("""
    #### How It Works
    This tool uses machine learning algorithms trained on clinical data to assess cardiovascular diseases risk.
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

    st.subheader("Understanding Heart Disease Risk Factors")
    st.write("""
    #### Modifiable Risk Factors
    These are factors you can control or improve:
    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Obesity
    - Physical inactivity
    - Unhealthy diet
    - Excessive alcohol consumption
    - Stress

    #### Non-Modifiable Risk Factors
    These are factors you cannot change:
    - Age (risk increases with age)
    - Gender (males generally at higher risk)
    - Family history of heart disease
    - Ethnicity (some groups have higher risk)

    #### Warning Signs of Heart Problems
    - Chest pain or discomfort
    - Shortness of breath
    - Pain or discomfort in arms, back, neck, jaw, or stomach
    - Nausea, lightheadedness, or cold sweats
    - Fatigue during normal activities
    """)

    st.subheader("Next Steps After Your Assessment")
    st.write("""
    1. **Share results with your doctor**: Bring a copy of these results to your next appointment
    2. **Follow up on high-risk areas**: If any risk was identified as high, discuss specific tests or evaluations
    3. **Develop an action plan**: Work with your healthcare provider on specific steps to reduce risk
    4. **Schedule regular screenings**: Even with low risk, maintain regular health check-ups
    5. **Reassess periodically**: Consider retaking this assessment every 6-12 months
    """)

# Sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This tool uses advanced machine learning to assess cardiovascular diseases risk based on clinical data.

Key Features:
- Multi-disease risk assessment
- Confidence intervals
- Feature importance analysis
- Emergency warning system
- Detailed recommendations

Remember: This is a screening tool only and should not replace professional medical advice.
""")

# Add health resources to sidebar
st.sidebar.title("Health Resources")
st.sidebar.markdown("""
#### Emergency Resources
- If experiencing chest pain: Call emergency services immediately
- [American Heart Association](https://www.heart.org)
- [Heart Disease Information (CDC)](https://www.cdc.gov/heartdisease/)

#### Tools for Heart Health
- [Heart Risk Calculator (ACC/AHA)](https://www.cvriskcalculator.com/)
- [Blood Pressure Chart](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings)
- [BMI Calculator](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Created by:
[Kushagra Singh](https://github.com/kushagra-a-singh/Cardiovascular-Diseases-Prediction)

© 2025 All Rights Reserved
""")