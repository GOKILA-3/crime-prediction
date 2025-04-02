import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set Streamlit Page Config
st.set_page_config(
    page_title="🚔 Crime Prediction App",
    page_icon="⚠️",
    layout="wide"
)

# Load the trained ML model
@st.cache_resource
def load_model():
    with open(r"C:\\Users\\gokil\\Downloads\\pavi's project\\crime_model.pkl", "rb") as f:
        return pickle.load(f)

# Load label encoders
@st.cache_resource
def load_encoders():
    with open(r"C:\\Users\\gokil\\Downloads\\pavi's project\\label_encoders.pkl", "rb") as f:
        return pickle.load(f)

# Load model & encoders
model = load_model()
encoders = load_encoders()

# Load crime data
crime_data = pd.read_csv(r"C:\\Users\\gokil\\Downloads\\pavi's project\\Indian-Crime-Data-Analysis-Forecasting-main\\crime.csv")

# Extract unique locations
locations = crime_data["STATE/UT"].unique()

# Create a dictionary mapping states to their districts
state_district_map = crime_data.groupby("STATE/UT")["DISTRICT"].unique().to_dict()

# Custom CSS for better UI
st.markdown("""
    <style>
    body { background: linear-gradient(to right, #ff7e5f, #feb47b); }
    .main { background-color: white; padding: 20px; border-radius: 12px; }
    h1 { color: #e63946; font-weight: bold; text-align: center; }
    .stButton>button { border-radius: 8px; padding: 10px 20px; background-color: #007BFF; color: white; }
    .stButton>button:hover { background-color: #0056b3; }
    .alert { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; }
    .alert-danger { background-color: #ff4b4b; color: white; }
    .alert-success { background-color: #28a745; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1>🚨 Crime Prediction App</h1>", unsafe_allow_html=True)
st.markdown("## 🔍 Check the crime risk in your area!")

# User Input Section
st.markdown("### 📌 Select Details Below")

col1, col2, col3 = st.columns(3)
with col1:
    location = st.selectbox("📍 **Choose State/UT**", locations)

# Filter districts based on selected state
available_districts = state_district_map.get(location, [])

with col2:
    district = st.selectbox("🏙️ **Select District**", available_districts)

with col3:
    year = st.number_input("📅 **Enter Year**", min_value=2000, max_value=2025, value=2023, step=1)

# Predict Crime Risk
if st.button("🔮 **Predict Crime Risk**"):
    try:
        with st.spinner("🔎 Analyzing crime data... Please wait."):
            # Encode inputs using the saved LabelEncoders
            location_encoded = encoders["STATE/UT"].transform([location])[0]
            district_encoded = encoders["DISTRICT"].transform([district])[0]

            # Prepare input data
            input_data = pd.DataFrame([[location_encoded, district_encoded, year]], columns=["STATE/UT", "DISTRICT", "YEAR"])

            # Make prediction
            prediction = model.predict(input_data)[0]

        # Display Result
        if prediction == 1:
            st.markdown('<div class="alert alert-danger">⚠️ <b>High Crime Risk! Be Cautious.</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-success">✅ <b>Low Crime Risk! Area is relatively safe.</b></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

# Footer Section
st.markdown("---")
st.markdown("""
    <div style="text-align:center;">
        🔹 <b>Developed by Pavithra S</b> | AI-Powered Crime Analysis 🔹<br>
        📧 Contact: <a href="mailto:pavithra.contact@gmail.com">Vithrap0@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
