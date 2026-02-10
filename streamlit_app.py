import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("risk_model.pkl")
    with open("model_meta.json", "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
FEATURES = meta["features"]
THRESHOLD = float(meta["threshold"])

LABELS = {
    "age": "Age (years)",
    "stab.glu": "Glucose (mg/dL)",
    "chol": "Total Cholesterol (mg/dL)",
    "hdl": "HDL Cholesterol (mg/dL)",
    "bp.1s": "Systolic BP (mmHg)",
    "bp.1d": "Diastolic BP (mmHg)",
    "weight": "Weight (lb)",
    "waist": "Waist (inches)",
    "hip": "Hip (inches)",
}

RANGES = {
    "age": (18, 100, 1),
    "stab.glu": (50, 450, 1),
    "chol": (75, 450, 1),
    "hdl": (10, 130, 1),
    "bp.1s": (80, 260, 1),
    "bp.1d": (40, 160, 1),
    "weight": (70, 400, 1),
    "waist": (20, 70, 0.5),
    "hip": (25, 80, 0.5),
}

DEFAULTS = {
    "age": 45.0,
    "stab.glu": 95.0,
    "chol": 200.0,
    "hdl": 50.0,
    "bp.1s": 130.0,
    "bp.1d": 80.0,
    "weight": 175.0,
    "waist": 38.0,
    "hip": 43.0,
}

st.title("Health Risk Predictor")
st.caption("Enter patient measurements to estimate high metabolic risk.")

with st.expander("How to get the most accurate result"):
    st.markdown(
        """
- Enter **recent and accurate measurements** (ideally taken the same day).
- Use the **correct units** shown beside each input.
- Avoid guessing; if a value is unknown, measure it first.
- This model is most reliable when inputs fall within typical physiological ranges.
        """
    )

with st.expander("Model info"):
    st.write(f"**Model:** {meta.get('model_type','(not provided)')}")
    st.write(f"**Decision threshold:** {THRESHOLD}")
    st.write(f"**Target rule used:** {meta.get('rule','(not provided)')}")
    st.write("**Features expected:**", FEATURES)

st.subheader("Patient Inputs")

user_input = {}

left, right = st.columns(2)

for i, f in enumerate(FEATURES):
    label = LABELS.get(f, f)
    vmin, vmax, step = RANGES.get(f, (None, None, 1.0))
    default = float(DEFAULTS.get(f, 0.0))

    col = left if i % 2 == 0 else right
    with col:
        if vmin is not None:
            user_input[f] = st.number_input(
                label,
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(min(max(default, vmin), vmax)),
                step=float(step),
            )
        else:
            user_input[f] = st.number_input(label, value=default, step=1.0)

warnings = []
for f, val in user_input.items():
    if f in RANGES:
        vmin, vmax, _ = RANGES[f]
        if val <= vmin or val >= vmax:
            warnings.append(f"- {LABELS.get(f,f)} is at an extreme value ({val}). Double-check units/measurement.")

if warnings:
    st.warning("Input check:\n" + "\n".join(warnings))

st.divider()

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("Predict risk", type="primary")
with col2:
    show_details = st.checkbox("Show probability details", value=True)

if predict_btn:
    X_one = pd.DataFrame([user_input], columns=FEATURES)

    prob = float(model.predict_proba(X_one)[0, 1])
    pred = int(prob >= THRESHOLD)

    if pred == 1:
        st.error(f"High risk predicted  (P = {prob:.1%})")
        st.write("Suggested action: consider follow-up screening and lifestyle interventions.")
    else:
        st.success(f"Lower risk predicted  (P = {prob:.1%})")
        st.write("Suggested action: continue healthy habits and regular monitoring.")

    if show_details:
        st.write(f"**Probability (risk=1):** {prob:.4f}")
        st.write(f"**Threshold used:** {THRESHOLD}")


