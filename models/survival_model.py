import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import encode_categorical
from sksurv.util import Surv
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Convert 'Date' columns to datetime if necessary
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Encode categorical variables (like 'Ship' and 'Fuel Tank Feed')
    data_encoded = encode_categorical(data, columns=['Ship', 'Fuel Tank Feed'])

    return data_encoded

def train_rsf_model(data):
    # Prepare the survival dataset
    X = data.drop(columns=['Time Til Failure (hours)', 'Date'])
    
    # Create the structured array for survival (time and event status)
    y = Surv.from_arrays(event=np.ones(len(data)), time=data['Time Til Failure (hours)'])

    # Train Random Survival Forest
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=42)
    rsf.fit(X, y)
    
    return rsf, X, y

def plot_survival_curve(rsf, X, instance_idx=0):
    # Get the survival function for a specific instance
    survival_fn = rsf.predict_survival_function(X.iloc[[instance_idx]], return_array=True)

    fig, ax = plt.subplots()
    time_points = rsf.event_times_
    ax.step(time_points, survival_fn[0], where="post")
    ax.set_title(f"Survival Function for Instance {instance_idx}")
    ax.set_xlabel("Time Until Failure (hours)")
    ax.set_ylabel("Survival Probability")
    return fig

