
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# --- Configuration ---
st.set_page_config(
    page_title="BMI Predictor Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = '../models/models_exported/bmi_model.pkl'
    # Use absolute path relative to this script if needed, but ../ works from 'dashboard' dir
    # Streamlit runs from the root usually, but let's handle paths safely
    # If run as `streamlit run dashboard/app.py`, cwd is project root.
    # If run as `cd dashboard && streamlit run app.py`, cwd is dashboard.
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    elif os.path.exists('models/models_exported/bmi_model.pkl'): # fallback if run from root
        with open('models/models_exported/bmi_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Model file not found. Expected at {model_path} or models/models_exported/bmi_model.pkl")
        return None

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- Mappings ---
# Gender: Male=0, Female=1
GENDER_MAP = {"Male": 0, "Female": 1}

# BMI Categories (Index 0-5)
# Using standard dataset labels commonly associated with this specific Kaggle dataset
BMI_CATEGORIES = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

BMI_COLORS = {
    0: "#3498db",  # Blue (Underweight)
    1: "#6dd5fa",  # Light Blue
    2: "#2ecc71",  # Green (Normal)
    3: "#f1c40f",  # Yellow (Overweight)
    4: "#e67e22",  # Orange (Obesity)
    5: "#e74c3c"   # Red (Extreme Obesity)
}

# --- Sidebar ---
st.sidebar.header("User Input 📝")

name = st.sidebar.text_input("Name", "User")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.info("Adjust the sliders in the main panel to calculate BMI.")

# --- Main Panel ---
st.title("⚖️ BMI Health Dashboard")
st.markdown("### Monitor your health metrics instantly with AI")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Measurements")
    weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.slider("Height (cm)", min_value=120, max_value=220, value=170)

    calculate_btn = st.button("Calculate BMI 🚀", use_container_width=True, type="primary")

# --- Calculation & Prediction ---
if calculate_btn and model:
    # 1. Calculate BMI Value
    # formula: weight (kg) / [height (m)]^2
    height_m = height / 100
    bmi_value = weight / (height_m ** 2)
    
    # 2. Prepare Input for Model
    # Features expected: ['Height', 'Weight', 'BMI_Value', 'Gender_Encoded']
    # *Note: Check feature order from training logic. Usually scikit-learn models expect array or dataframe with same cols.
    # Looking at notebook 03-Training.ipynb:
    # numeric_features = ['Height', 'Weight', 'BMI_Value']
    # preprocessor applies scaling to numeric_features, pass through Gender_Encoded.
    # The dataframe column order was: Height, Weight, BMI_Value, Gender_Encoded.
    
    gender_encoded = GENDER_MAP[gender]
    
    input_data = pd.DataFrame({
        'Height': [height],
        'Weight': [weight],
        'BMI_Value': [bmi_value],
        'Gender_Encoded': [gender_encoded]
    })
    
    # 3. Predict
    try:
        prediction_index = model.predict(input_data)[0]
        prediction_label = BMI_CATEGORIES.get(prediction_index, "Unknown")
        color = BMI_COLORS.get(prediction_index, "#grey")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        prediction_index = -1
        prediction_label = "Error"
        color = "#e74c3c"

    # --- Display Results ---
    with col2:
        st.subheader("Your Results")
        
        # Color-coded card
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h2 style="margin:0;">{prediction_label}</h2>
            <p style="margin:5px 0 0 0; font-size: 1.2rem;">BMI Index: {prediction_index}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Calculated BMI Value:** `{bmi_value:.2f}`")
        if prediction_index == 2:
            st.success("You are in a healthy range! 🎉")
        elif prediction_index in [0, 1]:
            st.warning("You are underweight. Consider consulting a nutritionist. 🍎")
        else:
            st.warning("You are above normal weight. Adopt a healthier lifestyle! 🏃")

    # --- Visualization ---
    st.markdown("---")
    st.subheader("Health Category Gauge")
    
    # Plotly Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi_value,
        title = {'text': f"BMI Score for {name}"},
        gauge = {
            'axis': {'range': [10, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [10, 18.5], 'color': '#6dd5fa'},
                {'range': [18.5, 25], 'color': '#2ecc71'},
                {'range': [25, 30], 'color': '#f1c40f'},
                {'range': [30, 40], 'color': '#e67e22'},
                {'range': [40, 50], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': bmi_value
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

else:
    with col2:
        st.info("👈 Adjust inputs and click Calculate to see results.")
