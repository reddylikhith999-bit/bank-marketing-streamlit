# ============================================
# BANK MARKETING PREDICTION - STREAMLIT APP
# ============================================

import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier


# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================
# TITLE
# ============================================

st.title("🏦 Bank Marketing Campaign Prediction")

st.write(
    "Enter the customer details below and click "
    "**Predict** to get the result."
)

st.divider()


# ============================================
# LOAD OR AUTO-TRAIN MODEL
# ============================================

@st.cache_resource
def load_or_train_model():
    model_path = "model.pkl"
    
    # If model doesn't exist, train it automatically using train.csv
    if not os.path.exists(model_path):
        try:
            df = pd.read_csv("train.csv")
            target_col = 'y' if 'y' in df.columns else df.columns[-1]
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Preprocess categorical features
            X_encoded = pd.get_dummies(X)
            
            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_encoded, y)
            
            # Save model and columns together
            with open(model_path, "wb") as file:
                pickle.dump((model, X_encoded.columns.tolist()), file)
        except Exception as e:
            st.error(f"Error training model automatically: {e}")
            st.stop()

    # Load the model and columns
    with open(model_path, "rb") as file:
        data = pickle.load(file)
        if isinstance(data, tuple):
            return data
        else:
            return data, None

try:
    model, model_columns = load_or_train_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load or train model.")
    st.write("Error:", e)
    st.stop()


# ============================================
# CUSTOMER INFORMATION
# ============================================

st.header("Customer Information")

col1, col2 = st.columns(2)

# ============================================
# COLUMN 1
# ============================================

with col1:

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
        [
            "married",
            "single",
            "divorced"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )

    default = st.selectbox(
        "Credit in Default?",
        [
            "no",
            "yes"
        ]
    )

    balance = st.number_input(
        "Account Balance",
        value=1000
    )

    housing = st.selectbox(
        "Housing Loan?",
        [
            "no",
            "yes"
        ]
    )

    loan = st.selectbox(
        "Personal Loan?",
        [
            "no",
            "yes"
        ]
    )


# ============================================
# COLUMN 2
# ============================================

with col2:

    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "unknown"
        ]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=300
    )

    campaign = st.number_input(
        "Number of Contacts",
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
        [
            "unknown",
            "failure",
            "other",
            "success"
        ]
    )


# ============================================
# PREDICT BUTTON
# ============================================

st.divider()

if st.button(
    "🔮 Predict Customer Response",
    use_container_width=True
):

    # ========================================
    # CREATE INPUT DATAFRAME
    # ========================================

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

    input_encoded = pd.get_dummies(input_data)
    if model_columns is not None:
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)


    # ========================================
    # MAKE PREDICTION
    # ========================================

    try:

        prediction = model.predict(input_encoded)
        result = prediction[0]


        # ====================================
        # DISPLAY RESULT
        # ====================================

        st.subheader("Prediction Result")


        if str(result).lower() == "yes":

            st.success(
                "🎉 Customer is likely to subscribe!"
            )

        else:

            st.warning(
                "❌ Customer is unlikely to subscribe."
            )


        st.write(
            "**Model Prediction:**",
            result
        )


    except Exception as e:

        st.error("❌ Prediction error")

        st.write("Error:", e)

        st.write("Input data used by the model:")
        st.dataframe(input_data)