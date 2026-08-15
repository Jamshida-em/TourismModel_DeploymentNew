import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the data below to get a prediction.
""")

AgeGroup = st.selectbox("Age Group", ["18-25","26-41","42-57","58-76+"])
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry","Company Invited"])
CityTier = st.selectbox("City Tier", [1,2,3])
DurationOfPitch = st.number_input("Duration Of Pitch", 1, 150, 15, 1)
Occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", 1, 10, 2, 1)
NumberOfFollowups = st.number_input("Number Of Follow ups", 1, 10, 2, 1)
ProductPitched = st.selectbox("Product Pitched", ["Basic","Standard","Deluxe","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3, 1)
MaritalStatus = st.selectbox("Marital Status", [ "Single","Married","Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", 1, 20, 2, 1)
Passport = st.selectbox("Passport", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
OwnCar = st.selectbox("Own Car", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", 0, 10, 0, 1)
Designation = st.selectbox("Designation", ["Manager","Senior Manager","Executive","VP","AVP"])
MonthlyIncome = st.number_input("Monthly Income", 100, 100000, 5000, 100)



input_data = pd.DataFrame([{
    "AgeGroup" : AgeGroup,
    "TypeofContact" : TypeofContact,
    "CityTier" : CityTier,
    "DurationOfPitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "Gender" : Gender,
    "NumberOfPersonVisiting" : NumberOfPersonVisiting,
    "NumberOfFollowups" : NumberOfFollowups,
    "ProductPitched" : ProductPitched,
    "PreferredPropertyStar" : PreferredPropertyStar,
    "MaritalStatus" : MaritalStatus,
    "NumberOfTrips" : NumberOfTrips,
    "Passport" : Passport,
    "PitchSatisfactionScore" : PitchSatisfactionScore,
    "OwnCar" : OwnCar,
    "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "MonthlyIncome" : MonthlyIncome,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
