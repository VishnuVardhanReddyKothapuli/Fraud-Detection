import os
import json
import joblib
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# 1. Load the trained model artifact
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "fraud_detection_model.joblib")
    return joblib.load(model_path)

@st.cache_data
def load_presets():
    preset_path = os.path.join("data", "sample_presets.json")
    if os.path.exists(preset_path):
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

try:
    model = load_model()
    presets = load_presets()
except Exception as e:
    st.error(f"Error loading model or preset artifacts: {e}")
    st.stop()

# All 29 feature names expected by the model
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Key most influential PCA indicators in the Kaggle dataset
KEY_FEATURES = ["V14", "V10", "V12", "V17", "V4", "V11"]

# 2. Simple clean interface
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown(
    "Trained on the benchmark **Kaggle Credit Card Fraud Detection dataset** (*ULB / Worldline*). "
    "Enter transaction details or select a verified test sample to predict whether the transaction is fraudulent."
)

# Quick Presets Section
st.subheader("1. Quick Presets from Test Set")
col_preset1, col_preset2, col_preset3 = st.columns(3)

if "current_features" not in st.session_state:
    if "normal" in presets:
        st.session_state.current_features = presets["normal"].copy()
    else:
        st.session_state.current_features = {col: 0.0 for col in FEATURE_COLUMNS}
        st.session_state.current_features["Amount"] = 45.0

with col_preset1:
    if st.button("🟢 Load Verified Legitimate Sample", use_container_width=True):
        if "normal" in presets:
            st.session_state.current_features = presets["normal"].copy()
            st.rerun()

with col_preset2:
    if st.button("🔴 Load Verified Fraudulent Sample", use_container_width=True):
        if "fraud" in presets:
            st.session_state.current_features = presets["fraud"].copy()
            st.rerun()

with col_preset3:
    if st.button("🔄 Reset to Neutral Zeros", use_container_width=True):
        st.session_state.current_features = {col: 0.0 for col in FEATURE_COLUMNS}
        st.session_state.current_features["Amount"] = 50.0
        st.rerun()

st.markdown("---")

# Input Fields
st.subheader("2. Transaction Inputs")

# Primary Section: Amount & Top Predictive Features
col_amt, col_info = st.columns([1, 2])
with col_amt:
    amount = st.number_input(
        "Transaction Amount ($ / €)",
        min_value=0.0,
        max_value=100000.0,
        value=float(st.session_state.current_features.get("Amount", 50.0)),
        step=5.0,
        help="Monetary amount of the transaction."
    )
    st.session_state.current_features["Amount"] = amount

with col_info:
    st.info(
        "ℹ️ **Dataset Note:** Due to privacy regulations, features V1–V28 are PCA components. "
        "Research shows **V14, V10, V12, V17, V4, and V11** have the highest predictive correlation with fraudulent activity. "
        "Large negative values in V14, V10, and V12 strongly indicate unauthorized account takeovers."
    )

st.markdown("##### Key Anomaly Indicators (Top Features)")
kcols = st.columns(6)
for idx, feat in enumerate(KEY_FEATURES):
    with kcols[idx]:
        val = st.number_input(
            f"{feat}",
            value=float(st.session_state.current_features.get(feat, 0.0)),
            format="%.4f",
            step=0.5,
            key=f"input_{feat}"
        )
        st.session_state.current_features[feat] = val

# Advanced Section: All 28 PCA Features
with st.expander("🛠️ Advanced: View & Edit All 28 PCA Features (V1 to V28)"):
    st.caption("Expand to manually inspect or fine-tune any specific PCA component.")
    grid_cols = st.columns(4)
    for i in range(1, 29):
        feat = f"V{i}"
        if feat not in KEY_FEATURES:
            col_target = grid_cols[(i - 1) % 4]
            with col_target:
                v_val = st.number_input(
                    f"{feat}",
                    value=float(st.session_state.current_features.get(feat, 0.0)),
                    format="%.4f",
                    step=0.2,
                    key=f"input_adv_{feat}"
                )
                st.session_state.current_features[feat] = v_val

st.markdown("---")

# 3. Prediction Button
if st.button("🔍 Predict Transaction Status", type="primary", use_container_width=True):
    # Construct DataFrame with exact column names in exact order
    input_data = pd.DataFrame([[st.session_state.current_features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Inference
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    fraud_prob = probabilities[1] * 100
    legit_prob = probabilities[0] * 100

    # 4. Display Result
    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ Fraudulent Transaction Detected")
            st.markdown(f"**Fraud Probability:** `{fraud_prob:.2f}%`")
            st.warning("Action Recommended: Trigger automated security hold, reject payment, and alert cardholder.")
        else:
            st.success("### ✅ Transaction Appears Legitimate")
            st.markdown(f"**Fraud Probability:** `{fraud_prob:.4f}%` *(Legitimate Probability: `{legit_prob:.4f}%`)*")
            st.info("Action Recommended: Authorize payment. No abnormal behavioral patterns detected.")

    with res_col2:
        st.metric(
            label="Estimated Fraud Risk",
            value=f"{fraud_prob:.2f}%",
            delta="High Risk" if prediction == 1 else "Normal",
            delta_color="inverse" if prediction == 1 else "normal"
        )

    # Transaction details summary
    with st.expander("View Full Submitted Feature Vector"):
        st.dataframe(input_data.T.rename(columns={0: "Submitted Value"}))
