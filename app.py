import streamlit as st
import pandas as pd
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Tea Garden Climate Impact Analyzer",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Climate Impact Analyzer for Tea Gardens")
st.caption("Early-warning system using climate data & ML")

# ===============================
# Load Model & Encoder
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("models/tea_stress_model.pkl")
    le = joblib.load("models/label_encoder.pkl")

    return model, le

model, le = load_model()

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🌦️ Climate Inputs")

avg_temperature = st.sidebar.slider("Avg Temperature (°C)", 10.0, 45.0, 28.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 80.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 70.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 40.0, 10.0)
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 50.0)
aqi = st.sidebar.slider("AQI", 0, 500, 120)

month = st.sidebar.selectbox(
    "Month",
    list(range(1, 13)),
    format_func=lambda x: [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][x-1]
)

week = st.sidebar.slider("Week of Year", 1, 52, 20)

# ===============================
# Prepare Input
# ===============================
features = [
    "avg_temperature",
    "rainfall",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "aqi",
    "month",
    "week"
]

input_df = pd.DataFrame([[
    avg_temperature,
    rainfall,
    humidity,
    wind_speed,
    cloud_cover,
    aqi,
    month,
    week
]], columns=features)

# ===============================
# Prediction
# ===============================
pred_class = model.predict(input_df)[0]
pred_label = le.inverse_transform([pred_class])[0]
confidence = model.predict_proba(input_df).max()

# ===============================
# Output
# ===============================
st.subheader("📊 Prediction Result")

if pred_label == "Severe Stress":
    st.error(f"🚨 Severe Plant Stress Detected\n\nConfidence: {confidence:.2f}")
elif pred_label == "Mild Stress":
    st.warning(f"⚠️ Mild Plant Stress Detected\n\nConfidence: {confidence:.2f}")
else:
    st.success(f"✅ Healthy Conditions\n\nConfidence: {confidence:.2f}")

# ===============================
# Recommendations
# ===============================
st.subheader("📝 Recommended Actions")

if pred_label == "Severe Stress":
    st.markdown("""
    • Increase irrigation immediately  
    • Avoid pruning operations  
    • Apply mulch to retain soil moisture  
    • Monitor disease and pest outbreaks  
    """)
elif pred_label == "Mild Stress":
    st.markdown("""
    • Monitor weather conditions closely  
    • Light irrigation if rainfall decreases  
    • Preventive disease management  
    """)
else:
    st.markdown("""
    • Climate conditions are favorable  
    • Continue regular plantation practices  
    • Weekly monitoring recommended  
    """)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("ML model trained offline • Loaded using Pickle • Streamlit UI")
