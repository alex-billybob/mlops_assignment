import streamlit as st
import pandas as pd
from datetime import datetime
from pycaret.regression import load_model, predict_model

# Load the saved PyCaret model
model = load_model('models/catboost_used_car_model')  # ensure correct path

st.title("🚗 Used Car Price Predictor (India)")
st.write("Predict car prices either one at a time or in batch mode.")

# Create Tabs
tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

with tab_single:
    st.subheader("Single Car Prediction")
    
    # User Inputs
    brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'BMW', 'Audi', 'Mercedes-Benz'])
    car_model = st.text_input("Model", 'XUV500 W8 2WD')
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner Type", ['First', 'Second', 'Third', 'Fourth & Above'])
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=18.0)
    engine = st.number_input("Engine (CC)", min_value=600, max_value=5000, value=1200)
    power = st.number_input("Power (bhp)", min_value=30, max_value=400, value=100)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
    location = st.selectbox("Location", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'])
    
    # Prepare input as DataFrame
    input_data = pd.DataFrame([{
        'Brand': brand,
        'Model': car_model,
        'Year': year,
        'Kilometers_Driven': km_driven,
        'Fuel_Type': fuel,
        'Transmission': transmission,
        'Owner_Type': owner,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Seats': seats,
        'Location': location
    }])
    
    current_year = datetime.now().year
    input_data['Car_Age'] = current_year - input_data['Year']
    input_data['km/year'] = input_data['Kilometers_Driven'] / input_data['Car_Age']
    
    # Prediction
    if st.button("Predict Price"):
        prediction = predict_model(model, data=input_data)
        price_rounded = int(round(prediction['prediction_label'][0], -3))
        st.success(f"Estimated Price: ₹ {price_rounded:,} INR")

with tab_batch:
    st.subheader("Batch Car Prediction")

    uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if uploaded_file is not None:
        # Load the uploaded CSV
        batch_data = pd.read_csv(uploaded_file)
        
        # Preview the uploaded data
        st.write("Preview of uploaded data:")
        st.dataframe(batch_data.head())
    
        # Feature Engineering
        current_year = datetime.now().year
        # Ensure these columns exist in uploaded CSV
        batch_data['Car_Age'] = current_year - batch_data['Year']
        batch_data['km/year'] = round(batch_data['Kilometers_Driven'] / batch_data['Car_Age'])
    
    
        # Run predictions
        predictions = predict_model(model, data=batch_data)
    
        # Round predicted price to nearest thousand and convert to integer
        predictions['prediction_label'] = (predictions['prediction_label'].round(-3)).astype(int)
        predictions.rename(columns={'prediction_label':'Predicted_Price'}, inplace=True)
    
        # Display results
        st.write("Predictions:")
        st.dataframe(predictions.head())
    
        # Allow user to download the predictions
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='batch_predictions.csv',
            mime='text/csv'
        )

