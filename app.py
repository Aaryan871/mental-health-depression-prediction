import streamlit as st
import joblib
import pandas as pd
        
# Load model and encoder
@st.cache_resource
def load_assets():
    model = joblib.load('catboost_depression.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    return model, le_dict

model, le_dict   = load_assets()

st.title("🧠 Mental Health Predictor")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    # Yahan 1, 2, 3 ko 1.0, 2.0, 3.0 kar diya gaya hai
    work_pressure = st.selectbox("Work Pressure", [1.0, 2.0, 3.0, 4.0, 5.0])
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "More than 8 hours"])

with col2:
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    # Yahan bhi options decimals mein kar diye gaye hain
    financial_stress = st.selectbox("Financial Stress", [1.0, 2.0, 3.0, 4.0, 5.0])
    job_satisfaction = st.selectbox("Job Satisfaction", [1.0, 2.0, 3.0, 4.0, 5.0])

col3, col4 = st.columns(2)

with col3:
    academic_pressure = st.selectbox("Academic Pressure", [1.0, 2.0, 3.0, 4.0, 5.0])
    cgpa = st.slider("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)

with col4:
    study_satisfaction = st.selectbox("Study Satisfaction", [1.0, 2.0, 3.0, 4.0, 5.0])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

# Prediction Logic
if st.button("Predict Status"):
    # Create a dictionary for input data with all required columns
    input_data = {
        'id': 0,
        'Name': 'Unknown',
        'Gender': gender,
        'Age': float(age),
        'City': 'Unknown',
        'Working Professional or Student': 'Working Professional',
        'Profession': 'Unknown',
        'Academic Pressure': str(academic_pressure),
        'Work Pressure': str(work_pressure),
        'CGPA': float(cgpa),
        'Study Satisfaction': str(study_satisfaction),
        'Job Satisfaction': str(job_satisfaction),
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Degree': 'Unknown',
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': 8.0,
        'Financial Stress': str(financial_stress),
        'Family History of Mental Illness': family_history
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode input using the saved LabelEncoder
    # Note: Ensure all columns are treated as strings as you did in training
    # Har column par uska sahi encoder lagayein
    for col in input_df.columns:
        if col in le_dict:
            specific_encoder = le_dict[col]
            try:
                input_df[col] = specific_encoder.transform(input_df[col].astype(str).values)
            except ValueError:
                input_df[col] = 0
            
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    st.divider()
    
    # Show input summary
    st.subheader("📋 Your Input Summary:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Work Pressure:** {work_pressure}")
        st.write(f"**Sleep Duration:** {sleep_duration}")
    with col2:
        st.write(f"**Suicidal Thoughts:** {suicidal_thoughts}")
        st.write(f"**Family History:** {family_history}")
        st.write(f"**Financial Stress:** {financial_stress}")
        st.write(f"**Job Satisfaction:** {job_satisfaction}")
    
    st.divider()
    
    # Show prediction result
    st.subheader("🔍 Prediction Result:")
    
    if prediction[0] == 1:
        st.error("### ⚠️ Signs of depression DETECTED")
        st.warning("**Recommendation:** Please consult a mental health professional as soon as possible.")
        st.info(f"Confidence Score: {probabilities[0][1]:.2%}")
    else:
        st.success("### ✅ No significant signs of depression detected")
        st.info(f"Confidence Score: {probabilities[0][0]:.2%}")
