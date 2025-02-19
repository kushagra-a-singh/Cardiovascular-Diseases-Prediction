import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

#load the trained models and preprocessors from the trained notebook
@st.cache_resource
def load_models():
    model = joblib.load('trained_model_checkpoints/multi_output_model.pkl')
    scaler = joblib.load('trained_model_checkpoints/scaler.pkl')
    return model, scaler

#function to calculate confidence intervals
def calculate_confidence_interval(probability, n_samples=1000):
    simulated = np.random.binomial(n=1, p=probability, size=n_samples)
    ci_lower = np.percentile(simulated, 2.5)
    ci_upper = np.percentile(simulated, 97.5)
    return ci_lower, ci_upper

#function to create feature columns in correct order
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

#function to validate input values
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

#funtion for emergency warning signs
def check_emergency_signs(cp_typical_angina, cp_non_anginal_pain, cp_atypical_angina, trestbps, thalach):
    emergency_signs = []
    if cp_typical_angina == 1:
        emergency_signs.append("⚠️ You reported typical chest pain (angina)")
    if cp_non_anginal_pain == 1:
        emergency_signs.append("⚠️ You reported atypical chest pain (angina)")
    if cp_atypical_angina == 1:
        emergency_signs.append("⚠️ You reported non-anginal pain")
    if trestbps > 180:
        emergency_signs.append("⚠️ Your blood pressure is very high")
    if thalach > 200:
        emergency_signs.append("⚠️ Your heart rate is very high")
    return emergency_signs

#function to get disease-specific information
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

st.set_page_config(page_title="Cardiovascular Diseases Risk Assessment", layout="wide")

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

st.title('Cardiovascular Diseases Risk Assessment Tool')
st.markdown("""
    This tool uses machine learning to assess your risk for different types of cardiovascular diseases.
    Please fill in all fields accurately for the best results.
    """)

st.warning("""
    🏥 MEDICAL DISCLAIMER: This tool is for educational and screening purposes only.
    It is not a substitute for professional medical diagnosis or advice.
    If you're experiencing chest pain or other concerning symptoms, seek immediate medical attention.
    """)

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Input Data"

tabs = st.tabs(["Input Data", "Results & Analysis", "Information"])

with tabs[0]:  
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=29, max_value=77, value=50, 
                                help="Enter your current age. This is crucial for accurate risk assessment.")
            sex = st.selectbox("Biological Sex", ["Female", "Male"], 
                             help="Select your biological sex at birth. This affects how risk factors are evaluated.")
            
        with col2:
            trestbps = st.number_input("Resting Blood Pressure(mm Hg)", min_value=94, max_value=200, value=120,
                                     help="Your blood pressure when at rest, measured in mm Hg. For accurate results, take after sitting quietly for 5 minutes.")
            chol = st.number_input("Total Cholesterol Level(mg/dl)", min_value=126, max_value=564, value=200,
                                 help="Your total cholesterol level from your most recent blood test (in mg/dl). If unsure, consult your latest lab results.")

        st.subheader("Clinical Information")
        col3, col4 = st.columns(2)
        with col3:
            cp = st.selectbox("Type of Chest Pain", 
                            ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
                            help="Select the type of chest pain you experience:\n• Typical angina: Triggered by activity, relieved by rest\n• Atypical angina: Not consistently related to activity\n• Non-anginal pain: Not heart-related\n• Asymptomatic: No chest pain")
            fbs = st.selectbox("High Fasting Blood Sugar", 
                             ["False", "True"],
                             help="Select 'True' if your fasting blood sugar is above 120 mg/dl. This is typically measured after not eating for 8 hours.")
            
        with col4:
            restecg = st.selectbox("Resting ECG Results",
                                 ["Normal", "ST-T wave abnormality", "left ventricular hypertrophy"],
                                 help="Results of resting electrocardiogram")
            thalach = st.number_input("Maximum Heart Rate", 
                                    min_value=71, max_value=202, value=150,
                                    help="Maximum heart rate achieved during exercise")

        col5, col6 = st.columns(2)
        with col5:
            exang = st.selectbox("Chest Pain During Exercise",
                               ["No", "Yes"],
                               help="Select 'Yes' if you experience chest pain or discomfort during physical activity.")
            oldpeak = st.number_input("ST Depression",
                                    min_value=0.0, max_value=6.2, value=1.0,
                                    help="ST depression value from your exercise stress test. If you haven't had a stress test, leave at default value.")
            
        with col6:
            slope = st.selectbox("ST Segment Slope",
                               ["up", "flat", "down"],
                               help="The slope of the ST segment on your exercise ECG:\n• Up: Upsloping\n• Flat: Horizontal\n• Down: Downsloping\nIf unsure, check your latest stress test report.")
            ca = st.number_input("Number of Vessels",
                               min_value=0, max_value=4, value=0,
                               help="Number of major heart blood vessels showing calcium buildup in fluoroscopy. This comes from a cardiac catheterization or calcium scoring test.")

        thal = st.selectbox("Thallium Test Result",
                          ["normal", "fixed defect", "reversible defect"],
                          help="Result from your thallium stress test:\n• Normal: No defects\n• Fixed defect: Permanent abnormality\n• Reversible defect: Temporary abnormality\nIf you haven't had this test, select 'normal'.")
        
        submitted = st.form_submit_button("Analyze Risk")

#handling form submission
if submitted:
    st.session_state.analysis_completed = True
    model, scaler = load_models()
    
    warnings = validate_inputs(age, trestbps, chol, thalach, oldpeak)
    if warnings:
        st.warning("⚠️ Potential input concerns:")
        for warning in warnings:
            st.write(f"- {warning}")

    try:
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
        
        numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        st.session_state.input_data = input_data
        st.session_state.model = model
        st.session_state.selected_tab = "Results & Analysis"
    except Exception as e:
        st.error(f"An error occurred: {e}")
if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
    with tabs[0]:
        st.success("Risk assessment complete, please checkout **RESULTS & ANALYSIS** tab for results")

with tabs[1]:
    if st.session_state.selected_tab == "Results & Analysis" and 'prediction' in st.session_state:
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
        
        num_diseases = len(disease_types)
        rows = (num_diseases // 4) + (1 if num_diseases % 4 > 0 else 0)

        for row in range(rows):
            result_cols = st.columns(4)
            start_idx = row * 4
            end_idx = min(start_idx + 4, num_diseases)
            
            for i in range(start_idx, end_idx):
                col_idx = i % 4
                disease = disease_types[i]
                
                with result_cols[col_idx]:
                    probability = prediction_proba[i][0][1]
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

        st.subheader("Key Factors Influencing the Prediction")

        try:
            model = st.session_state.model
            input_data = st.session_state.input_data
            
            #finding first disease with high risk if any
            high_risk_idx = None
            for i, proba in enumerate(st.session_state.prediction_proba):
                if proba[0][1] > 0.5:
                    high_risk_idx = i
                    break
            
            #if no high risk, use another disease
            if high_risk_idx is None:
                disease_idx = 0
            else:
                disease_idx = high_risk_idx
            #getting the feature importance plot
            st.write(f"Showing factors for: **{disease_types[disease_idx]}**")
            
            if disease_idx < len(model.estimators_):
                estimator = model.estimators_[disease_idx]
                importances = estimator.feature_importances_
                feature_names = list(input_data.columns)
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                importance_df = importance_df[importance_df['Feature'] != 'cp_non_anginal_pain']
                
                #sorted and get top 10
                importance_df = importance_df.sort_values('Importance', ascending=True).tail(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title=f'Top 10 Important Features for {disease_types[disease_idx]}',
                    xaxis_title='Feature Importance',
                    yaxis_title='Feature',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis={'categoryorder':'total ascending'}
                )
                
                fig.update_traces(
                    hovertemplate="<b>%{y}</b><br>" +
                    "Importance: %{x:.3f}<br>" +
                    "<extra></extra>"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **Understanding the Feature Importance Bar Chart:**
                - Longer bars indicate stronger influence on predictions
                - Color intensity corresponds to importance level
                - Hover over bars for exact values
                """)
            
            #generating synthetic data based on input features
            sample_data = pd.DataFrame()
            n_samples = 100
            for col in input_data.columns:
                if col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
                    #for numerical columns
                    mean_val = float(input_data[col].iloc[0])
                    std_val = max(mean_val * 0.1, 1)
                    sample_data[col] = np.random.normal(mean_val, std_val, n_samples)
                else:
                    #for binary columns
                    prob = float(input_data[col].iloc[0])
                    sample_data[col] = np.random.binomial(1, prob, n_samples)
            
            correlation_matrix = sample_data.corr()

            #filter correlation matrix to only include numerical columns
            numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            filtered_correlation_matrix = correlation_matrix.loc[numerical_cols, numerical_cols]

            fig = go.Figure(data=go.Heatmap(
                z=filtered_correlation_matrix.values,
                x=filtered_correlation_matrix.columns,
                y=filtered_correlation_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(filtered_correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False,
            ))

            fig.update_layout(
                title={
                    'text': 'Feature Correlation Matrix',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20}
                },
                width=800,
                height=800,
                xaxis={
                    'tickangle': 45,
                    'side': 'bottom',
                    'tickfont': {'size': 12}
                },
                yaxis={
                    'tickfont': {'size': 12}
                },
                margin=dict(t=100, l=100, r=100, b=100)
            )

            fig.update_traces(
                colorbar={
                    'title': 'Correlation',
                    'titleside': 'right',
                    'thickness': 20,
                    'len': 0.8,
                    'tickformat': '.2f',
                    'tickfont': {'size': 12},
                    'titlefont': {'size': 14}
                },
                showscale=True
            )

            st.plotly_chart(fig, use_container_width=False)
            st.markdown("""
            **Understanding the Correlation Matrix:**
            - Strong Positive Correlation (Dark Blue): Values close to +1
            - Strong Negative Correlation (Dark Red): Values close to -1
            - No Correlation (White): Values close to 0
            - Diagonal Values: Always 1.0 (perfect self-correlation)

            **Note**: Hover over cells to see exact correlation values. This matrix shows correlations only between numerical features.
            """)

        except Exception as e:
            st.error(f"Error generating correlation matrix: {str(e)}")
            st.write("Debug info:")
            st.write("Input data shape:", input_data.shape)
            st.write("Input data columns:", input_data.columns.tolist())

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

        #recommendations based on risk levels
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
        
        st.subheader("Lifestyle Recommendations")

        max_risk_idx = 0
        max_risk_prob = 0
        for i, proba in enumerate(st.session_state.prediction_proba):
            if proba[0][1] > max_risk_prob:
                max_risk_prob = proba[0][1]
                max_risk_idx = i

        high_risk_disease = disease_types[max_risk_idx]
        disease_info = get_disease_info(high_risk_disease)

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

        if disease_info and disease_info.get('prevention'):
            st.write(f"#### Specific Recommendations for {high_risk_disease}:")
            for tip in disease_info.get('prevention'):
                st.write(f"• {tip}")

with tabs[2]:
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

st.sidebar.title("Health Resources")
st.sidebar.markdown("""
#### Emergency Resources
- If experiencing chest pain: Call emergency services immediately
- [American Heart Association](https://www.heart.org)
- [Heart Disease Information (CDC)](https://www.cdc.gov/heartdisease/)

#### Tools for Heart Health
- [Blood Pressure Chart](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings)
- [BMI Calculator](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Created by:
[Kushagra Singh](https://www.linkedin.com/in/kushagra-anit-singh/)

© 2025 All Rights Reserved
""")