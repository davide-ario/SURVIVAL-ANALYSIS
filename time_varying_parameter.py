# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:15:53 2025

@author: arioldid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxTimeVaryingFitter
from lifelines.datasets import load_panel_test

# Set random seed for reproducibility
np.random.seed(1234)

# For this example, we'll first create our own synthetic dataset
# Let's simulate 200 patients followed for up to 10 years

def generate_survival_data_with_tvc(n_subjects=200, max_time=10):
    """
    Generate a synthetic dataset with time-varying covariates for survival analysis.
    
    - Each subject has a baseline age
    - Blood pressure is measured at multiple timepoints
    - Treatment status can change during follow-up
    """
    # Generate subject IDs
    ids = range(1, n_subjects + 1)
    
    records = []
    
    for i in ids:
        # Baseline characteristics
        age = np.random.normal(65, 10)  # Age at baseline, mean 65, sd 10
        sex = np.random.binomial(1, 0.5)  # 0=female, 1=male
        
        # Initial record (start of follow-up)
        start_time = 0
        
        # Initial blood pressure and treatment status
        initial_bp = np.random.normal(130, 20)
        initial_treatment = np.random.binomial(1, 0.3)  # 30% initially on treatment
        
        # Generate number of follow-up visits (1 to 5)
        n_measurements = np.random.randint(1, 6)
        
        # End time will be determined by event or censoring
        event_time = np.random.exponential(scale=5)  # Generate potential event time
        if event_time > max_time:
            event_time = max_time
            event = 0  # Censored
        else:
            event = 1  # Event occurred
        
        # Create follow-up measurements at various times
        measurement_times = np.sort(np.random.uniform(0, event_time, n_measurements))
        
        # Always include baseline measurement
        measurement_times = np.insert(measurement_times, 0, 0)
        
        # Add the final event/censoring time if it's not already included
        if measurement_times[-1] != event_time:
            measurement_times = np.append(measurement_times, event_time)
        
        # Create records for each interval
        bp = initial_bp
        treatment = initial_treatment
        
        for j in range(len(measurement_times) - 1):
            start = measurement_times[j]
            stop = measurement_times[j + 1]
            
            # Record the current interval
            event_in_interval = 1 if (stop == event_time and event == 1) else 0
            
            records.append({
                'id': i,
                'start': start,
                'stop': stop,
                'event': event_in_interval,
                'age': age,
                'sex': sex,
                'bp': bp,
                'treatment': treatment
            })
            
            # Update time-varying covariates for next interval
            # Blood pressure can change between visits (with correlation to previous value)
            bp = bp * 0.8 + np.random.normal(130, 10) * 0.2
            
            # Treatment status can change (with higher probability if BP is high)
            if treatment == 0 and bp > 140:
                treatment = np.random.binomial(1, 0.4)  # 40% chance to start treatment if BP is high
            elif treatment == 1:
                treatment = 1 - np.random.binomial(1, 0.1)  # 10% chance to stop treatment
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df

# Generate the dataset
df_tvc = generate_survival_data_with_tvc()

# Display the first few rows
print("Sample of generated time-varying dataset:")
print(df_tvc.head(10))

# Summary statistics
print("\nDataset summary:")
print(f"Number of subjects: {df_tvc['id'].nunique()}")
print(f"Total number of records: {len(df_tvc)}")
print(f"Average records per subject: {len(df_tvc) / df_tvc['id'].nunique():.2f}")
print(f"Number of events: {df_tvc['event'].sum()}")
print(f"Event rate: {df_tvc['event'].sum() / df_tvc['id'].nunique():.2%}")

# Fit the Cox model with time-varying covariates
ctv = CoxTimeVaryingFitter()
ctv.fit(df_tvc, id_col='id', event_col='event', start_col='start', stop_col='stop', 
        show_progress=True)

# Print model summary
print("\nCox model with time-varying covariates results:")
ctv.print_summary()

# Visualize the effect of treatment on survival
fig, ax = plt.subplots(figsize=(10, 6))

# For time-varying covariates, we need a different approach to visualize treatment effect
# We'll use the hazard ratio from the model to calculate an approximate effect

# Get the hazard ratio for treatment from the model
treatment_hr = np.exp(ctv.params_["treatment"])
print(f"\nHazard Ratio for treatment: {treatment_hr:.3f}")

# Create a simple survival function for visualization
from lifelines import KaplanMeierFitter

# First, get data at the subject level (last record per subject)
df_subject = df_tvc.loc[df_tvc.groupby('id')['stop'].idxmax()]
kmf = KaplanMeierFitter()
kmf.fit(df_subject['stop'], df_subject['event'])

# Plot overall survival curve
time_points = kmf.survival_function_.index.values
baseline_survival = kmf.survival_function_['KM_estimate'].values
plt.step(time_points, baseline_survival, where="post", label="Overall Survival", color="gray")

# Plot approximate curves with and without treatment effect
# Using the formula S(t)^exp(β) where β is the coefficient
treated_survival = baseline_survival ** treatment_hr
untreated_survival = baseline_survival ** (1/treatment_hr)

plt.step(time_points, treated_survival, where="post", label="With Treatment (approximation)", color="blue")
plt.step(time_points, untreated_survival, where="post", label="Without Treatment (approximation)", color="red")

plt.title("Average Predicted Survival With and Without Treatment")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# Show how to use real-world scenario with treatments that change over time
print("\nReal-world clinical interpretation example:")
print("In this model, treatment status can change during follow-up, allowing us")
print("to estimate the actual effect of treatment accounting for when patients")
print("start or stop medication, rather than just using baseline treatment status.")
print("The hazard ratio for treatment represents the instantaneous risk reduction")
print("when a patient is on treatment versus off treatment.")
