import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title(
    "📊 Customer Churn Intelligence Dashboard"
)

model = joblib.load(
    MODELS_DIR / "churn_model.joblib"
)

kmeans = joblib.load(
    MODELS_DIR / "kmeans.joblib"
)

seg_scaler = joblib.load(
    MODELS_DIR / "preprocessor.joblib"
)

df = pd.read_csv(
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

def category_options(column):
    return sorted(df[column].dropna().unique().tolist())


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Prediction",
        "Segmentation"
    ]
)

if page == "Overview":

    st.subheader(
        "Business KPIs"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    c2.metric(
        "Churn Rate",
        f"{df['Churn'].mean()*100:.2f}%"
    )

    c3.metric(
        "Avg Tenure",
        f"{df['tenure'].mean():.1f}"
    )

    c4.metric(
        "Avg Monthly Charge",
        f"${df['MonthlyCharges'].mean():.2f}"
    )
    
    fig,ax = plt.subplots()

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)
    
elif page == "Prediction":

    st.header(
        "Predict Customer Churn"
    )
    
    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.slider(
        "Monthly Charges",
        18,
        120,
        50
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(tenure * monthly)
    )

    phone_service = st.selectbox(
        "Phone Service",
        category_options("PhoneService")
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        category_options("MultipleLines")
    )

    internet_service = st.selectbox(
        "Internet Service",
        category_options("InternetService")
    )

    online_security = st.selectbox(
        "Online Security",
        category_options("OnlineSecurity")
    )

    online_backup = st.selectbox(
        "Online Backup",
        category_options("OnlineBackup")
    )

    device_protection = st.selectbox(
        "Device Protection",
        category_options("DeviceProtection")
    )

    tech_support = st.selectbox(
        "Tech Support",
        category_options("TechSupport")
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        category_options("StreamingTV")
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        category_options("StreamingMovies")
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        category_options("PaymentMethod")
    )

    service_count = [
        phone_service,
        online_security,
        online_backup,
        tech_support,
        streaming_tv,
        streaming_movies
    ].count("Yes")
    
    sample = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService":[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless_billing],
        "PaymentMethod":[payment_method],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
        "service_count":[service_count]
    })
    
    if st.button(
        "Predict"
    ):

        prob = model.predict_proba(
            sample
        )[0][1]

        st.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )
        if prob < 0.3:
            st.success(
                "Low Risk"
            )

        elif prob < 0.7:
            st.warning(
                "Medium Risk"
            )

        else:
            st.error(
                "High Risk"
            )
elif page == "Segmentation":
    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.slider(
        "Monthly Charges",
        18,
        120,
        50
    )

    total = st.number_input(
        "Total Charges",
        value=1000.0
    )

    service_count = st.slider(
        "Services Used",
        0,
        6,
        2
    )
    seg = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
        "service_count":[service_count]
    })

    scaled = seg_scaler.transform(
        seg
    )

    cluster = kmeans.predict(
        scaled
    )[0]
    
    segment_names = {
        0:"New Customers",
        1:"Loyal Customers",
        2:"High Value Customers",
        3:"At Risk Customers"
    }

    st.metric(
        "Customer Segment",
        segment_names[cluster]
    )
