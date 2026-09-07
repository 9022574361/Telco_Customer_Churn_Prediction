import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — the visual layer
# ============================================================
st.markdown("""
<style>
    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* ---- Hero header ---- */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 2.4rem 2.2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.92;
        margin: 0;
    }

    /* ---- Section cards ---- */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid #eef0f4;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Result cards ---- */
    .result-risk {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #fca5a5;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #86efac;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .result-sub {
        font-size: 0.92rem;
        opacity: 0.85;
    }

    .prob-box {
        background: white;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid #eef0f4;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }
    .prob-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: #4f46e5;
    }
    .prob-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* ---- Buttons ---- */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(79, 70, 229, 0.45);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #1e1b4b;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* ---- Metric chips inside sidebar ---- */
    .chip {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('svm_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, scaler, features


model, scaler, feature_columns = load_artifacts()

# ============================================================
# SIDEBAR — project info
# ============================================================
with st.sidebar:
    st.markdown("## 📡 Telco Churn AI")
    st.markdown("A machine learning app that predicts whether a telecom customer is likely to churn, based on account, billing and service details.")
    st.markdown("---")
    st.markdown("### ⚙️ Model")
    st.markdown('<div class="chip">🧠 Algorithm: Support Vector Machine</div>', unsafe_allow_html=True)
    st.markdown('<div class="chip">📊 Preprocessing: StandardScaler + One-Hot Encoding</div>', unsafe_allow_html=True)
    st.markdown('<div class="chip">🎯 Task: Binary Classification (Churn / No Churn)</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👤 Built by")
    st.markdown("Add your name, GitHub and LinkedIn links here.")

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📞 Telco Customer Churn Predictor</h1>
    <p>Fill in the customer's account, billing and service details below to estimate their likelihood of churning.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💳 Account &amp; Billing</div>', unsafe_allow_html=True)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=9000.0,
        value=float(tenure_months * monthly_charges), step=10.0
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ Services &amp; Demographics</div>', unsafe_allow_html=True)

    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ============================================================
# PREDICTION + RESULTS
# ============================================================
if predict_clicked:
    with st.spinner("Analyzing customer profile..."):
        raw_data = {
            'Gender': 1,  # Neutral default (Male: 1, Female: 0)
            'Senior Citizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'Tenure Months': tenure_months,
            'Phone Service': 1,
            'Paperless Billing': 1 if paperless_billing == "Yes" else 0,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges,
            'Multiple Lines': 'No',
            'Internet Service': internet_service,
            'Online Security': online_security,
            'Online Backup': online_backup,
            'Device Protection': device_protection,
            'Tech Support': tech_support,
            'Streaming TV': 'No',
            'Streaming Movies': 'No',
            'Contract': contract,
            'Payment Method': payment_method
        }

        df_input = pd.DataFrame([raw_data])

        # One-hot encoding
        cat_cols = df_input.select_dtypes(include=['object', 'string']).columns.tolist()
        if cat_cols:
            df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True, dtype=int)

        # Reindex columns to match model training signature
        df_aligned = df_input.reindex(columns=feature_columns, fill_value=0)

        # Standard scale
        scaled_input = scaler.transform(df_aligned)

        # Inference
        churn_pred = model.predict(scaled_input)[0]
        churn_prob = model.predict_proba(scaled_input)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction Results")

    res_col1, res_col2 = st.columns(2, gap="medium")

    with res_col1:
        if churn_pred == 1:
            st.markdown(f"""
            <div class="result-risk">
                <div style="font-size:2.4rem;">🚨</div>
                <div class="result-title" style="color:#b91c1c;">High Risk of Churn</div>
                <div class="result-sub" style="color:#7f1d1d;">This customer shows strong signs of leaving. Consider proactive retention outreach.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:2.4rem;">✅</div>
                <div class="result-title" style="color:#15803d;">Customer Likely to Retain</div>
                <div class="result-sub" style="color:#14532d;">This customer profile looks stable, with a low likelihood of churning soon.</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="prob-box">
            <div class="prob-label">Churn Probability</div>
            <div class="prob-value">{churn_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(churn_prob))

    # Quick summary of the inputs used
    with st.expander("📋 View input summary"):
        summary_df = pd.DataFrame({
            "Feature": ["Contract", "Tenure (months)", "Monthly Charges", "Total Charges",
                        "Payment Method", "Paperless Billing", "Internet Service",
                        "Tech Support", "Online Security", "Online Backup",
                        "Device Protection", "Dependents", "Senior Citizen"],
            "Value": [contract, tenure_months, f"${monthly_charges:.2f}", f"${total_charges:.2f}",
                      payment_method, paperless_billing, internet_service,
                      tech_support, online_security, online_backup,
                      device_protection, dependents, senior_citizen]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Fill in the customer details above and click **Predict Churn Risk** to see the result.")