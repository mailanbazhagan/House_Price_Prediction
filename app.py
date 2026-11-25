from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
METRICS_PATH = HERE / "metrics.txt"

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 House Price Predictor")
st.write("Enter property details to predict an estimated house price.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Property details")
left, right = st.columns(2)
with left:
    area = st.number_input("Area (square feet)", 500, 3000, 1200, step=50)
    bedrooms = st.selectbox("Number of bedrooms", [1, 2, 3, 4, 5], index=2)
with right:
    age = st.slider("Age of house (years)", 0, 40, 5)

if st.button("Predict house price", type="primary", use_container_width=True):
    sample = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "age_years": age
    }])
    result = float(model.predict(sample)[0])
    st.success(f"Estimated price: ₹ {result:,.2f} lakh")
    st.caption("Classroom demonstration using synthetic data; not a real-estate valuation.")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    st.code(METRICS_PATH.read_text(encoding="utf-8"), language="text")
else:
    st.info("Metrics will appear here after training.")
