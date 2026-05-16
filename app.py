import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import time

# Configure the Streamlit page
st.set_page_config(page_title="Sonar Classification App", page_icon="🌊", layout="centered")

# --- CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextInput>div>div>input {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    h1 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
    .prediction-box-mine {
        padding: 20px;
        background: linear-gradient(135deg, #490202 0%, #290000 100%);
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.2);
        animation: fadeIn 0.5s;
    }
    .prediction-box-rock {
        padding: 20px;
        background: linear-gradient(135deg, #023618 0%, #001a0a 100%);
        border-radius: 10px;
        border-left: 5px solid #00C851;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.2);
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🌊 Sonar Submarine Classification")
st.markdown("This app uses a Deep Learning Neural Network with **Dropout Regularization** to classify sonar signals as either a **Mine** or a **Rock**.")

st.markdown("---")

# Load dependencies safely
@st.cache_resource
def load_assets():
    try:
        model = load_model('sonar_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("⚠️ Model or Scaler not found! Please run `python train_and_save.py` first to generate `sonar_model.keras` and `scaler.pkl`.")
    st.stop()

st.subheader("📡 Input Sonar Data")
st.markdown("Enter 60 comma-separated numerical values corresponding to the sonar signal frequencies:")

# Default example (a Rock example from the dataset)
example_data = "0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032"

user_input = st.text_area("Sonar Frequencies Array", value=example_data, height=150)

if st.button("🚀 Analyze Signal"):
    if user_input:
        try:
            # Parse input
            input_list = [float(x.strip()) for x in user_input.split(',')]
            
            if len(input_list) != 60:
                st.error(f"⚠️ Expected exactly 60 values, but got {len(input_list)}. Please check your input.")
            else:
                with st.spinner("Analyzing frequencies..."):
                    time.sleep(1) # Small delay for UX effect
                    
                    # Convert to numpy array and reshape
                    input_array = np.array(input_list).reshape(1, -1)
                    
                    # Scale data
                    input_scaled = scaler.transform(input_array)
                    
                    # Predict
                    prediction_prob = model.predict(input_scaled)[0][0]
                    
                    # Output styling
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    if prediction_prob >= 0.5:
                        # Depending on label encoder, usually M is 0 and R is 1 if M comes first, 
                        # but M=1, R=0 or vice versa. Let's assume standard behavior (R=1 usually, M=0 because alphabetically M comes first, but wait, LabelEncoder sorts alphabetically, so M -> 0, R -> 1).
                        # Let's map dynamically based on standard LabelEncoder output: M=0, R=1.
                        # Wait, if y_pred >= 0.5, it is class 1 (Rock). 
                        st.markdown('<div class="prediction-box-rock">🪨 The signal indicates a ROCK.</div>', unsafe_allow_html=True)
                        st.caption(f"Confidence (Class 1): {prediction_prob*100:.2f}%")
                    else:
                        st.markdown('<div class="prediction-box-mine">💣 The signal indicates a MINE.</div>', unsafe_allow_html=True)
                        st.caption(f"Confidence (Class 0): {(1 - prediction_prob)*100:.2f}%")
                        
        except ValueError:
            st.error("⚠️ Invalid input! Please make sure all 60 values are valid numbers separated by commas.")
    else:
        st.warning("Please enter some data to analyze.")
