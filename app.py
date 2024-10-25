import streamlit as st
import pandas as pd
from datetime import datetime

# Import model training and plotting functions from survival_model.py
from models.survival_model import load_and_prepare_data, train_rsf_model, plot_survival_curve

# Load initial dataset
data = load_and_prepare_data('data/ship_fuel_analysis.csv')

# Train the Random Survival Forest model
rsf, X, y = train_rsf_model(data)

# Streamlit UI
st.title("Ship Fuel Analysis and Survival Prediction with Random Survival Forests")

# Sidebar - Data Input
st.sidebar.header("Add New Failure Report")
ship = st.sidebar.text_input("Ship")
engine = st.sidebar.text_input("Engine")
engine_id = st.sidebar.text_input("Engine ID")
fuel_pump_id = st.sidebar.text_input("Fuel Pump ID")
time_until_failure = st.sidebar.number_input("Time Til Failure (hours)", min_value=1)
fuel_tank_feed = st.sidebar.text_input("Fuel Tank Feed")
date = st.sidebar.date_input("Date", datetime.today())

if st.sidebar.button("Add Failure Report"):
    new_data = pd.DataFrame({
        'Ship': [ship],
        'Engine': [engine],
        'Engine ID': [engine_id],
        'Fuel Pump ID': [fuel_pump_id],
        'Time Til Failure (hours)': [time_until_failure],
        'Fuel Tank Feed': [fuel_tank_feed],
        'Date': [date],
        # Add default values for fuel analysis columns
        'Density (kg/m3)': [None],
        'Water Reaction Vol Change (ml)': [None],
        'Flash Point (celsius)': [None],
        'Filter Blocking Tendency': [None],
        'Cloud Point (celsius)': [None],
        'Sulphur (%)': [None],
        'Colony Forming Units (CFU/ml)': [None],
        'Water content (mg/kg)': [None]
    })
    data = pd.concat([data, new_data], ignore_index=True)
    rsf, X, y = train_rsf_model(data)  # Re-train the model with the new data
    st.sidebar.success("New failure report added!")

# Sidebar - Fuel Analysis Report Input
st.sidebar.header("Add Fuel Analysis Report")
density = st.sidebar.number_input("Density (kg/m3)")
water_reaction_vol_change = st.sidebar.number_input("Water Reaction Vol Change (ml)")
flash_point = st.sidebar.number_input("Flash Point (celsius)")
filter_blocking_tendency = st.sidebar.number_input("Filter Blocking Tendency")
cloud_point = st.sidebar.number_input("Cloud Point (celsius)")
sulphur = st.sidebar.number_input("Sulphur (%)")
colony_forming_units = st.sidebar.number_input("Colony Forming Units (CFU/ml)")
water_content = st.sidebar.number_input("Water content (mg/kg)")
fuel_analysis_date = st.sidebar.date_input("Fuel Analysis Date", datetime.today())
fuel_analysis_ship = st.sidebar.text_input("Fuel Analysis Ship")
fuel_tank_feed_analysis = st.sidebar.text_input("Fuel Tank Feed Analysis")

if st.sidebar.button("Add Fuel Analysis Report"):
    new_analysis_data = pd.DataFrame({
        'Ship': [fuel_analysis_ship],
        'Fuel Tank Feed': [fuel_tank_feed_analysis],
        'Density (kg/m3)': [density],
        'Water Reaction Vol Change (ml)': [water_reaction_vol_change],
        'Flash Point (celsius)': [flash_point],
        'Filter Blocking Tendency': [filter_blocking_tendency],
        'Cloud Point (celsius)': [cloud_point],
        'Sulphur (%)': [sulphur],
        'Colony Forming Units (CFU/ml)': [colony_forming_units],
        'Water content (mg/kg)': [water_content],
        'Date': [fuel_analysis_date]
    })
    st.sidebar.success("New fuel analysis report added!")

# Display Random Survival Forest curve for an instance
st.header("Random Survival Forest Survival Curve")
instance_idx = st.slider("Select instance index to plot survival curve", 0, len(X)-1)
rsf_fig = plot_survival_curve(rsf, X, instance_idx)
st.pyplot(rsf_fig)

# Prediction Section
st.header("Predict Time Until Failure")
predict_density = st.number_input("Density for Prediction (kg/m3)")
predict_water_reaction_vol_change = st.number_input("Water Reaction Vol Change for Prediction (ml)")
predict_flash_point = st.number_input("Flash Point for Prediction (celsius)")
predict_filter_blocking_tendency = st.number_input("Filter Blocking Tendency for Prediction")
predict_cloud_point = st.number_input("Cloud Point for Prediction (celsius)")
predict_sulphur = st.number_input("Sulphur for Prediction (%)")
predict_colony_forming_units = st.number_input("Colony Forming Units for Prediction (CFU/ml)")
predict_water_content = st.number_input("Water Content for Prediction (mg/kg)")

if st.button("Predict Time Until Failure"):
    new_instance = pd.DataFrame({
        'Density (kg/m3)': [predict_density],
        'Water Reaction Vol Change (ml)': [predict_water_reaction_vol_change],
        'Flash Point (celsius)': [predict_flash_point],
        'Filter Blocking Tendency': [predict_filter_blocking_tendency],
        'Cloud Point (celsius)': [predict_cloud_point],
        'Sulphur (%)': [predict_sulphur],
        'Colony Forming Units (CFU/ml)': [predict_colony_forming_units],
        'Water content (mg/kg)': [predict_water_content]
    })
    # Predict survival function for the new instance
    new_instance_encoded = encode_categorical(new_instance, columns=['Ship', 'Fuel Tank Feed'])
    prediction = rsf.predict_survival_function(new_instance_encoded, return_array=True)
    st.write(f"Predicted survival function: {prediction}")
