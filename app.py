import streamlit as st
import joblib
import bz2
from constants import (
    fuel_types, 
    conditions, 
    drive_types, 
    years,
    title_statuses,
    transmissions
)
from utils import preprocess_data, predict_prices

@st.cache_data
def load_model():
    with bz2.BZ2File('car_price_model_complete.joblib.bz2', 'rb') as file:
        return joblib.load(file)

def main():
    """Main function to run the Streamlit app"""
    st.title("🚗 Car Price Predictor")
    st.write("Get an estimated price for your used car!")

    model_package = load_model()

    st.header("Car Details")
    col1, col2 = st.columns(2)
    with col1:
        car_name = st.selectbox("Car Model:", model_package['car_cylinders_mapping'].keys(), index=list(model_package['car_cylinders_mapping'].keys()).index('toyota rav4'))
        odometer = st.number_input("Mileage:", min_value=5000, max_value=300000, value=50000)
        condition = st.selectbox("Condition:", conditions, index=conditions.index('good'))
        transmission = st.selectbox("Transmission:", transmissions)

    with col2:
        year = st.selectbox("Year", years, index=years.index(2015))
        drive_type = st.selectbox("Drive Type", drive_types)
        title_status = st.selectbox("Title Status", title_statuses)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        
    if st.button("Predict Price"):
        inputs = {
            'car_name': car_name,
            'odometer': odometer,
            'condition': condition,
            'transmission': transmission,
            'year': year,
            'drive': drive_type,
            'title_status': title_status,
            'fuel': fuel_type
        }
        
        processed_data = preprocess_data(inputs, model_package)
        point_estimate, lower_bound, upper_bound = predict_prices(processed_data, model_package)
        
        st.subheader("Estimated Price")
        st.write(f"**Predicted Price:** ${point_estimate:,.0f}")
        st.write(f"**Reasonable Range:** \\${lower_bound:,.0f} -\\${upper_bound:,.0f}")

if __name__ == "__main__":
    main()