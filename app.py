import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from models.survival_model import load_and_prepare_data, train_rsf_model
import matplotlib.pyplot as plt

# Load initial dataset
data, X = load_and_prepare_data('data/ship_fuel_analysis.csv')

# Train the Random Survival Forest model
rsf, X, y = train_rsf_model(data, X)

# Streamlit UI
st.title("Fuel Analysis and Pump Failure Prediction")

# Sidebar - Filters
st.sidebar.header("Filter Data")
selected_ship = st.sidebar.selectbox("Select Ship", options=data["Ship"].unique())
selected_engine = st.sidebar.selectbox("Select Engine", options=data["Engine"].unique())

# Filter data based on selected ship and engine
filtered_data = data[(data["Ship"] == selected_ship) & (data["Engine"] == selected_engine)]
filtered_X = X[(data["Ship"] == selected_ship) & (data["Engine"] == selected_engine)]

# Sidebar - Add New Reports
st.sidebar.header("Add New Report")
report_type = st.sidebar.radio("Report Type", ["Failure Report", "Fuel Analysis Report"])

with st.sidebar.form(key="report_form"):
    ship = st.text_input("Ship")
    engine = st.text_input("Engine")
    fuel_tank_feed = st.text_input("Fuel Tank Feed")
    date = st.date_input("Date", datetime.today())
    
    if report_type == "Failure Report":
        engine_id = st.text_input("Engine ID")
        fuel_pump_id = st.text_input("Fuel Pump ID")
        time_until_failure = st.number_input("Time Til Failure (hours)", min_value=1)
        new_report = pd.DataFrame({
            'Ship': [ship], 'Engine': [engine], 'Engine ID': [engine_id],
            'Fuel Pump ID': [fuel_pump_id], 'Time Til Failure (hours)': [time_until_failure],
            'Fuel Tank Feed': [fuel_tank_feed], 'Date': [date], 'Density (kg/m3)': [None],
            'Water Reaction Vol Change (ml)': [None], 'Flash Point (celsius)': [None],
            'Filter Blocking Tendency': [None], 'Cloud Point (celsius)': [None],
            'Sulphur (%)': [None], 'Colony Forming Units (CFU/ml)': [None], 'Water content (mg/kg)': [None]
        })

    elif report_type == "Fuel Analysis Report":
        density = st.number_input("Density (kg/m3)")
        water_reaction_vol_change = st.number_input("Water Reaction Vol Change (ml)")
        flash_point = st.number_input("Flash Point (celsius)")
        filter_blocking_tendency = st.number_input("Filter Blocking Tendency")
        cloud_point = st.number_input("Cloud Point (celsius)")
        sulphur = st.number_input("Sulphur (%)")
        colony_forming_units = st.number_input("Colony Forming Units (CFU/ml)")
        water_content = st.number_input("Water content (mg/kg)")
        new_report = pd.DataFrame({
            'Ship': [ship], 'Engine': [engine], 'Fuel Tank Feed': [fuel_tank_feed], 'Date': [date],
            'Density (kg/m3)': [density], 'Water Reaction Vol Change (ml)': [water_reaction_vol_change],
            'Flash Point (celsius)': [flash_point], 'Filter Blocking Tendency': [filter_blocking_tendency],
            'Cloud Point (celsius)': [cloud_point], 'Sulphur (%)': [sulphur],
            'Colony Forming Units (CFU/ml)': [colony_forming_units], 'Water content (mg/kg)': [water_content],
            'Engine ID': [None], 'Fuel Pump ID': [None], 'Time Til Failure (hours)': [None]
        })

    if st.form_submit_button("Add Report"):
        data = pd.concat([data, new_report], ignore_index=True)
        rsf, X, y = train_rsf_model(data, X)
        st.sidebar.success("New report added!")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Survival Curve", "Dataset", "Predict Time Until Failure"])

# Tab 1: Average Survival Curve for Filtered Dataset
with tab1:
    st.header("Average Survival Curve for Filtered Dataset")
    
    if not filtered_X.empty:
        # Predict survival function for each instance in the filtered dataset
        survival_functions = rsf.predict_survival_function(filtered_X, return_array=False)
        
        # Aggregate time points and calculate the average survival probabilities
        time_points = np.unique(np.concatenate([sf.x for sf in survival_functions]))
        avg_survival_probs = np.zeros_like(time_points)
        
        for sf in survival_functions:
            avg_survival_probs += np.interp(time_points, sf.x, sf.y, left=1, right=0)
        
        avg_survival_probs /= len(survival_functions)
        
        # Plot the average survival curve
        fig, ax = plt.subplots()
        ax.step(time_points, avg_survival_probs, where="post")
        ax.set_xlabel("Time Until Failure (hours)")
        ax.set_ylabel("Average Survival Probability")
        ax.set_title(f"Average Survival Curve for {selected_ship} - {selected_engine}")
        st.pyplot(fig)
    else:
        st.write("No data available for the selected ship and engine.")

# Tab 2: Dataset Display
with tab2:
    st.header("Dataset Preview")
    st.dataframe(data, use_container_width=True)

# Tab 3: Predict Time Til Failure
with tab3:
    st.header("Time Til Failure based on Fuel Analysis")
    with st.form(key="prediction_form"):
        pred_density = st.number_input("Density (kg/m3)", key="pred_density")
        pred_water_reaction_vol_change = st.number_input("Water Reaction Vol Change (ml)", key="pred_water_reaction")
        pred_flash_point = st.number_input("Flash Point (celsius)", key="pred_flash_point")
        pred_filter_blocking_tendency = st.number_input("Filter Blocking Tendency", key="pred_filter_blocking")
        pred_cloud_point = st.number_input("Cloud Point (celsius)", key="pred_cloud_point")
        pred_sulphur = st.number_input("Sulphur (%)", key="pred_sulphur")
        pred_colony_forming_units = st.number_input("Colony Forming Units (CFU/ml)", key="pred_colony_forming_units")
        pred_water_content = st.number_input("Water content (mg/kg)", key="pred_water_content")
        submit_prediction = st.form_submit_button("Predict Time Until Failure")

    if submit_prediction:
        new_instance = pd.DataFrame({
            'Density (kg/m3)': [pred_density], 'Water Reaction Vol Change (ml)': [pred_water_reaction_vol_change],
            'Flash Point (celsius)': [pred_flash_point], 'Filter Blocking Tendency': [pred_filter_blocking_tendency],
            'Cloud Point (celsius)': [pred_cloud_point], 'Sulphur (%)': [pred_sulphur],
            'Colony Forming Units (CFU/ml)': [pred_colony_forming_units], 'Water content (mg/kg)': [pred_water_content]
        })

        # Predict survival function for the new instance
        survival_function = rsf.predict_survival_function(new_instance, return_array=False)[0]

        # Extract time and survival probability
        times = survival_function.x  # Time points
        survival_probs = survival_function.y  # Corresponding survival probabilities

        # Estimate the time til failure (when survival probability drops below 0.5)
        threshold = 0.5
        time_until_failure = next((t for t, s in zip(times, survival_probs) if s < threshold), None)

        if time_until_failure is not None:
            st.write(f"Predicted time until failure: {time_until_failure:.2f} hours")
        else:
            st.write("The survival probability did not drop below the threshold within the observed time range.")
