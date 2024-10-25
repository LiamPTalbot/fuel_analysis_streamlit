import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    # Load my .csv datafile
    data = pd.read_csv(file_path)

    # Convert Date column to datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Exclude identifier columns (Ship, Engine, etc.) from the features dataset
    features = [
        'Density (kg/m3)', 'Water Reaction Vol Change (ml)', 'Flash Point (celsius)',
        'Filter Blocking Tendency', 'Cloud Point (celsius)', 'Sulphur (%)',
        'Colony Forming Units (CFU/ml)', 'Water content (mg/kg)'
    ]
    
    # Prep dataset with relevant features
    X = data[features].copy()

    # Ensure no missing values in features
    X = X.fillna(X.mean())

    return data, X

def train_rsf_model(data, X):
    # All rows represent failures, so event indicator is 1 
    event = np.ones(len(data), dtype=bool)  # True indicates failure events

    # Create structured array for survival (time and event status)
    y = Surv.from_arrays(event=event, time=data['Time Til Failure (hours)'])

    # Train Random Survival Forest
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=42)
    rsf.fit(X, y)
    
    return rsf, X, y

def plot_survival_curve(rsf, X, instance_idx=0):
    # Get the survival function for a specific instance
    survival_fn = rsf.predict_survival_function(X.iloc[[instance_idx]])

    fig, ax = plt.subplots()

    # Each survival function corresponds to a different set of time points. Take first one
    for fn in survival_fn:
        ax.step(fn.x, fn.y, where="post")
    
    ax.set_title(f"Survival Function for Instance {instance_idx}")
    ax.set_xlabel("Time Until Failure (hours)")
    ax.set_ylabel("Survival Probability")
    return fig
