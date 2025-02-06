import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

std_scaler = StandardScaler()
label_encoder = LabelEncoder()

model = pd.read_pickle('model')
data = pd.read_pickle('data')

st.title("💼 Employee Salary Prediction App")
st.markdown("With this application, you can generate **precise salary estimations** for company employees based on relevant data and analytics.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("📅 Age", value=22, step=1, min_value=18, max_value=70)

    gender = st.selectbox("⚤ Gender", ["Choose...", "Male", "Female"])
    if gender == "Choose...":
        st.warning("Please select a gender.")

    education = st.selectbox("🎓 Education Level", ["Choose...", "Bachelor's", "Master's", "PhD"])
    if education == "Choose...":
        st.warning("Please select an education level.")

with col2:
    job_titles = sorted(data['Job Title'].unique())  
    job_title = st.selectbox("💼 Job Title", ["Choose..."] + job_titles)
    if job_title == "Choose...":
        st.warning("Please select a job title.")

    years_of_experience = st.number_input("⏳ Years of Experience", step=1, min_value=0, max_value=50)

st.divider()

if st.button("📊 Predict Salary"):
    
    if gender == "Choose..." or education == "Choose..." or job_title == "Choose...":
        st.error("Please fill in all required fields!")
    else:
       
        user_input = pd.DataFrame([[age, gender, education, job_title, years_of_experience]], 
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])
        
        std_scaler.fit(data[['Age']])
        user_input["Age Scaled"] = std_scaler.transform(user_input[['Age']])
        
        label_encoder.fit(data[['Gender']])
        user_input["Gender Encode"] = label_encoder.transform(user_input[['Gender']])
        
        label_encoder.fit(data[['Education Level']])
        user_input["Education Level Encode"] = label_encoder.transform(user_input[['Education Level']])
        
        label_encoder.fit(data[['Job Title']])
        user_input["Job Title Encode"] = label_encoder.transform(user_input[['Job Title']])
        
        std_scaler.fit(data[['Years of Experience']])
        user_input["Years of Experience Scaled"] = std_scaler.transform(user_input[['Years of Experience']])
        
        input_for_model = user_input[["Age Scaled", "Gender Encode", "Education Level Encode", "Job Title Encode", "Years of Experience Scaled"]]
        
        predicted_salary = model.predict(input_for_model)[0]
        
        st.success(f"💰 Estimated Salary: **${predicted_salary:,.2f}**")
