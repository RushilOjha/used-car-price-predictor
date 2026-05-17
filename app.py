import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource(show_spinner="Training model... Please wait.")
def load_data_and_train_model():
    try:
        # Load data
        df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    except FileNotFoundError:
        return None, None, None, None, None

    # Feature Engineering
    df['Brand'] = df['name'].apply(lambda x: x.split()[0])
    df['Age'] = 2026 - df['year'] # Using 2026 as the current context year

    # Define columns
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'Brand']
    numerical_cols = ['Age', 'km_driven']

    # Keep a copy of original categories for the UI dropdowns
    ui_options = {col: list(df[col].unique()) for col in categorical_cols}

    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Prepare X and y
    X = df[numerical_cols + categorical_cols]
    y = df['selling_price']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    return model, encoders, scaler, ui_options, df['selling_price'].mean()

# Load backend
model, encoders, scaler, ui_options, avg_price = load_data_and_train_model()

# ==========================================
# SIDEBAR UI: USER INPUTS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3202/3202061.png", width=100)
st.sidebar.title("Car Specifications")
st.sidebar.write("Please select the details of the car to predict its selling price.")

if model is None:
    st.error("Dataset 'CAR DETAILS FROM CAR DEKHO.csv' not found. Please place it in the same directory.")
    st.stop()

# Input fields
with st.sidebar.form("input_form"):
    brand = st.selectbox("Car Brand", sorted(ui_options['Brand']))
    year = st.slider("Manufacturing Year", min_value=1995, max_value=2024, value=2015, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000)
    fuel = st.selectbox("Fuel Type", sorted(ui_options['fuel']))
    transmission = st.selectbox("Transmission", sorted(ui_options['transmission']))
    seller_type = st.selectbox("Seller Type", sorted(ui_options['seller_type']))
    owner = st.selectbox("Owner History", sorted(ui_options['owner']))
    
    submit_button = st.form_submit_button(label="Predict Price 🚀")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Authors:**\n* Rushil Ojha\n* Honey Kalaria")
st.sidebar.markdown("**Institution:**\n* Chang Jung Christian University")

# ==========================================
# MAIN PAGE UI
# ==========================================
st.title("🚗 Used Car Market Value Predictor")
st.markdown("""
Welcome to the Used Car Price Prediction App! This tool utilizes a **Random Forest Machine Learning Algorithm** trained on over 4,000 historical transactions from Car Dekho to provide fair market valuations for second-hand vehicles.
""")

st.write("---")

# Layout columns for visual balance
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Your Car's Profile")
    st.write(f"**{brand}** manufactured in **{year}** with **{km_driven:,} km** on the odometer.")
    st.write(f"**Specs:** {fuel} | {transmission} | {owner} | Sold by: {seller_type}")

# ==========================================
# PREDICTION LOGIC
# ==========================================
if submit_button:
    # 1. Prepare input array
    age = 2026 - year
    
    # Create DataFrame for the input
    input_data = pd.DataFrame([[age, km_driven, fuel, seller_type, transmission, owner, brand]], 
                              columns=['Age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'Brand'])
    
    # 2. Apply Encoders
    for col in encoders.keys():
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except ValueError:
            # Handle unseen categories gracefully by assigning to the most frequent/first label
            input_data[col] = 0 
            
    # 3. Apply Scaler
    input_data[['Age', 'km_driven']] = scaler.transform(input_data[['Age', 'km_driven']])
    
    # 4. Predict
    prediction = model.predict(input_data)[0]
    
    # Display Result
    with col2:
        st.subheader("Estimated Fair Market Price")
        st.metric(label="Selling Price (INR)", value=f"₹ {int(prediction):,}")
        
        # Add context delta
        delta = prediction - avg_price
        delta_color = "normal" if delta > 0 else "inverse"
        st.metric(label="Compared to Market Average (₹5.04L)", 
                  value=f"{'Above' if delta > 0 else 'Below'} Average", 
                  delta=f"₹ {int(abs(delta)):,}", 
                  delta_color=delta_color)

    st.success(f"✅ Prediction Successful! The fair market value for this {year} {brand} is approximately **₹ {int(prediction):,}**.")

else:
    with col2:
        st.info("👈 Enter the vehicle specifications in the sidebar and click **Predict Price** to see the valuation.")

# ==========================================
# EXTRA INFORMATION / DATA INSIGHTS
# ==========================================
with st.expander("📊 How does this model work?"):
    st.write("""
    - **Algorithm:** The backend uses a Random Forest Regressor, an ensemble learning method that operates by constructing a multitude of decision trees at training time.
    - **Preprocessing:** Numerical features (Age, Km Driven) are standardized using `StandardScaler`. Categorical features (Fuel, Transmission, etc.) are converted to numerical format using `LabelEncoder`.
    - **Feature Importance:** Age (Year) and Transmission type heavily influence the final valuation, followed by Kilometers Driven.
    """)
