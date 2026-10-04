import streamlit as st
import pandas as pd
import pickle


# Page title
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦"
)

# Title
st.title("🏦 Bank Marketing Campaign Prediction")

st.write("Enter customer details to predict the campaign result.")


# Load trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    st.success("Model loaded successfully!")

except Exception as e:
    st.error("Could not load model.pkl")
    st.write(e)
    st.stop()


# Customer inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

job = st.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.selectbox(
    "Credit Default?",
    ["no", "yes"]
)

balance = st.number_input(
    "Balance",
    value=1000
)

housing = st.selectbox(
    "Housing Loan?",
    ["no", "yes"]
)

loan = st.selectbox(
    "Personal Loan?",
    ["no", "yes"]
)

contact = st.selectbox(
    "Contact",
    ["cellular", "telephone", "unknown"]
)

day = st.number_input(
    "Contact Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.selectbox(
    "Month",
    [
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ]
)

duration = st.number_input(
    "Call Duration",
    min_value=0,
    value=300
)

campaign = st.number_input(
    "Campaign Contacts",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"]
)


# Prediction button
if st.button("🔮 Predict"):

    # Create one-row DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome]
    })

    # Make prediction
    try:
        prediction = model.predict(input_data)

        result = prediction[0]

        if str(result).lower() == "yes":
            st.success("🎉 Customer is likely to subscribe!")

        else:
            st.warning("❌ Customer is unlikely to subscribe!")

        st.write("Prediction:", result)

    except Exception as e:
        st.error("Prediction error:")
        st.write(e)