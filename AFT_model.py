# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:35:23 2025

@author: arioldid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter
from lifelines.datasets import load_rossi

# Load the Rossi recidivism dataset (commonly used in survival analysis)
rossi = load_rossi()

# Print the first few rows to understand the data
print(rossi.head())

# Create an AFT model with Weibull distribution
aft = WeibullAFTFitter()

# Fit the model
# 'week' is the duration (time to event)
# 'arrest' is the event indicator (1 if arrested, 0 if censored)
# The other columns are covariates
aft.fit(rossi, duration_col='week', event_col='arrest')

# Print the model summary
print(aft.summary)

# Plot the survival function for specific individuals
# First, create example subjects
example1 = rossi.loc[[0]].copy()  # Low-risk individual
example2 = rossi.loc[[10]].copy()  # Medium-risk individual
example3 = rossi.copy().iloc[[20]]  # High-risk individual

# Define a time grid for visualization
time_points = np.linspace(0, 52, 100)  # Weekly points for one year

# Get survival function predictions
surv_func1 = aft.predict_survival_function(example1, times=time_points)
surv_func2 = aft.predict_survival_function(example2, times=time_points)
surv_func3 = aft.predict_survival_function(example3, times=time_points)

# Plot survival functions - fixing the shape issue
plt.figure(figsize=(10, 6))
plt.plot(surv_func1.index.values, surv_func1[0].values, label='Low risk', linewidth=2)
plt.plot(surv_func2.index.values, surv_func2[10].values, label='Medium risk', linewidth=2)
plt.plot(surv_func3.index.values, surv_func3[20].values, label='High risk', linewidth=2)
plt.ylabel("Survival probability")
plt.xlabel("Weeks")
plt.title("AFT Model: Survival Function by Risk Level")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Calculate hazard rates by using the relationship between PDF and survival function
# Hazard function h(t) = f(t)/S(t), where f(t) is the PDF
# Calculate hazard using a simpler approach - directly from cumulative hazard function
def calculate_hazard_from_chf(chf_values, times):
    """Calculate hazard rate from cumulative hazard function using finite differences"""
    hazard = np.zeros_like(times, dtype=float)
    for i in range(len(times)-1):
        dt = times[i+1] - times[i]
        dH = chf_values[i+1] - chf_values[i]
        hazard[i] = dH/dt if dt > 0 else 0
    
    # Set the last point equal to the previous point
    if len(hazard) > 1:
        hazard[-1] = hazard[-2]
    
    return hazard

# Get cumulative hazard functions
chf1 = aft.predict_cumulative_hazard(example1, times=time_points)
chf2 = aft.predict_cumulative_hazard(example2, times=time_points)
chf3 = aft.predict_cumulative_hazard(example3, times=time_points)

# Calculate hazard rates from cumulative hazard
hazard1 = calculate_hazard_from_chf(chf1[0].values, time_points)
hazard2 = calculate_hazard_from_chf(chf2[10].values, time_points)
hazard3 = calculate_hazard_from_chf(chf3[20].values, time_points)


# Plot the hazard functions
plt.figure(figsize=(10, 6))
plt.plot(time_points, hazard1, label='Low risk', linewidth=2)
plt.plot(time_points, hazard2, label='Medium risk', linewidth=2)
plt.plot(time_points, hazard3, label='Higher risk', linewidth=2)
plt.xlabel('Time (weeks)')
plt.ylabel('Hazard Rate (risk of event per unit time)')
plt.title('Weibull AFT Hazard Functions by Risk Level')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot the cumulative hazard functions
plt.figure(figsize=(10, 6))
plt.plot(chf1.index.values, chf1[0].values, label='Low risk', linewidth=2)
plt.plot(chf2.index.values, chf2[10].values, label='Medium risk', linewidth=2)
plt.plot(chf3.index.values, chf3[20].values, label='Higher risk', linewidth=2)
plt.xlabel('Time (weeks)')
plt.ylabel('Cumulative Hazard')
plt.title('Weibull AFT Cumulative Hazard Functions by Risk Level')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# Predict median survival time for new data
median_survival = aft.predict_median(rossi)
print("Predicted median survival times:")
print(median_survival.head())

# We can also use other distributions beyond Weibull
from lifelines import LogNormalAFTFitter, LogLogisticAFTFitter

# Log-normal AFT model
ln_aft = LogNormalAFTFitter()
ln_aft.fit(rossi, duration_col='week', event_col='arrest')
print(ln_aft.summary)

# Log-logistic AFT model
ll_aft = LogLogisticAFTFitter()
ll_aft.fit(rossi, duration_col='week', event_col='arrest')
print(ll_aft.summary)

# Compare AIC of different models to see which fits better
print(f"Weibull AIC: {aft.AIC_}")
print(f"Log-normal AIC: {ln_aft.AIC_}")
print(f"Log-logistic AIC: {ll_aft.AIC_}")
