# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:54:14 2025

@author: arioldid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, LogLogisticFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split
from lifelines.statistics import proportional_hazard_test

# Set the style for the plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Function to generate synthetic health data
def generate_health_data(n=1000, random_seed=42):
    """
    Generate synthetic health dataset for survival analysis
    
    Parameters:
    -----------
    n : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic health data
    """
    np.random.seed(random_seed)
    
    # Create empty dataframe
    data = pd.DataFrame()
    
    # Generate demographic and health behavior variables
    data['age'] = np.random.randint(18, 102, size=n)  # Age between 20-80
    data['sex'] = np.random.choice(['Male', 'Female'], size=n)
    data['smoker'] = np.random.choice(['Yes', 'No'], size=n, p=[0.3, 0.7])  # 30% are smokers
    
    # Physical activity level (1-5 scale, where 5 is very active)
    data['physical_activity'] = np.random.randint(1, 6, size=n)
    
    # Calculate survival time based on risk factors
    # Base survival time between 1-20 years
    survival_time = np.random.uniform(1, 20, size=n)
    
    # Adjust survival time based on risk factors
    # Smoking reduces survival time
    survival_time[data['smoker'] == 'Yes'] *= 0.7
    
    # Physical activity increases survival time
    for i in range(n):
        survival_time[i] *= (0.8 + data['physical_activity'][i] * 0.1)
    
    # Age affects survival time
    for i in range(n):
        survival_time[i] *= max(0.1, 1 - (data['age'][i] - 20) / 100)
    
    # Sex affects survival time (slight difference)
    survival_time[data['sex'] == 'Female'] *= 1.1  # Females tend to have higher life expectancy
    
    # Add some random noise to survival times
    survival_time = np.maximum(0.1, survival_time + np.random.normal(0, 1, size=n))
    
    # Determine if event occurred (death) or was censored
    data['event'] = np.random.choice([1, 0], size=n, p=[0.7, 0.3])  # 70% experienced the event, 30% censored
    
    # For censored observations, we need to adjust the time
    # In a real study, these would be patients who dropped out or study ended before they experienced the event
    for i in range(n):
        if data['event'][i] == 0:  # Censored
            # Censoring happens before the actual (unknown) survival time
            survival_time[i] = min(survival_time[i] * np.random.uniform(0.1, 0.9), survival_time[i])
    
    data['time'] = np.round(survival_time, 2)
    
    # Create categorical versions for plotting
    data['age_group'] = pd.cut(data['age'], 
                              bins=[19, 40, 60, 81], 
                              labels=['Young', 'Middle', 'Older'])
    
    data['pa_group'] = pd.cut(data['physical_activity'], 
                             bins=[0, 2, 3, 5], 
                             labels=['Low', 'Medium', 'High'])
    
    return data

# Generate the data
np.random.seed(42)  # For reproducibility
health_df = generate_health_data(1000)

# Display the first few rows of the dataset
print("Sample of the health dataset:")
print(health_df.head())

# Basic statistics of the dataset
print("\nSummary statistics:")
print(health_df.describe())

# Count the number of events vs censored
print("\nEvent counts:")
print(health_df['event'].value_counts())

# Distribution of risk factors
print("\nSmoking status distribution:")
print(health_df['smoker'].value_counts())
print("\nPhysical activity distribution:")
print(health_df['physical_activity'].value_counts())
print("\nSex distribution:")
print(health_df['sex'].value_counts())
print("\nAge group distribution:")
print(health_df['age_group'].value_counts())

# Plot the distribution of survival times
plt.figure(figsize=(10, 6))
# Alternative approach to avoid the "mode.use_inf_as_null" error
plt.hist([
    health_df[health_df['event'] == 0]['time'],  # Censored
    health_df[health_df['event'] == 1]['time']   # Event occurred
], bins=20, stacked=True, label=['Censored', 'Event Occurred'])
plt.title('Distribution of Survival Times')
plt.xlabel('Survival Time (years)')
plt.ylabel('Count')
plt.legend(title='Event')
plt.savefig('survival_time_distribution.png')


# Perform Kaplan-Meier analysis for overall survival
kmf = KaplanMeierFitter()
kmf.fit(health_df['time'], health_df['event'], label='Overall Survival')

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Overall Kaplan-Meier Survival Curve')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('overall_survival_curve.png')

# Kaplan-Meier by smoking status
plt.figure(figsize=(10, 6))
kmf_smoker = KaplanMeierFitter()
kmf_nonsmoker = KaplanMeierFitter()

smoker_mask = (health_df['smoker'] == 'Yes')
kmf_smoker.fit(health_df[smoker_mask]['time'], 
               health_df[smoker_mask]['event'], 
               label='Smoker')

kmf_nonsmoker.fit(health_df[~smoker_mask]['time'], 
                  health_df[~smoker_mask]['event'], 
                  label='Non-Smoker')

kmf_smoker.plot_survival_function(color='red')
kmf_nonsmoker.plot_survival_function(color='blue')
plt.title('Kaplan-Meier Curves by Smoking Status')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('survival_by_smoking.png')

# Log-rank test for smoking
results = logrank_test(health_df[smoker_mask]['time'], 
                      health_df[~smoker_mask]['time'],
                      health_df[smoker_mask]['event'], 
                      health_df[~smoker_mask]['event'])
print("\nLog-rank test results for smoking status:")
print(f"p-value: {results.p_value:.4f}")
print(f"Test statistic: {results.test_statistic:.4f}")

# Kaplan-Meier by sex
plt.figure(figsize=(10, 6))
kmf_male = KaplanMeierFitter()
kmf_female = KaplanMeierFitter()

male_mask = (health_df['sex'] == 'Male')
kmf_male.fit(health_df[male_mask]['time'], 
             health_df[male_mask]['event'], 
             label='Male')

kmf_female.fit(health_df[~male_mask]['time'], 
               health_df[~male_mask]['event'], 
               label='Female')

kmf_male.plot_survival_function(color='blue')
kmf_female.plot_survival_function(color='pink')
plt.title('Kaplan-Meier Curves by Sex')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('survival_by_sex.png')

# Log-rank test for sex
results = logrank_test(health_df[male_mask]['time'], 
                      health_df[~male_mask]['time'],
                      health_df[male_mask]['event'], 
                      health_df[~male_mask]['event'])
print("\nLog-rank test results for sex:")
print(f"p-value: {results.p_value:.4f}")
print(f"Test statistic: {results.test_statistic:.4f}")

# Kaplan-Meier by physical activity level
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

colors = ['red', 'orange', 'green']
for i, pa_level in enumerate(['Low', 'Medium', 'High']):
    mask = (health_df['pa_group'] == pa_level)
    
    if sum(mask) > 0:  # Make sure there are observations in this group
        kmf_pa = KaplanMeierFitter()
        kmf_pa.fit(health_df[mask]['time'], 
                  health_df[mask]['event'], 
                  label=f'PA: {pa_level}')
        kmf_pa.plot_survival_function(ax=ax, color=colors[i])

plt.title('Kaplan-Meier Curves by Physical Activity Level')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('survival_by_physical_activity.png')

# Kaplan-Meier by age group
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

colors = ['purple', 'blue', 'red']
for i, age_level in enumerate(['Young', 'Middle', 'Older']):
    mask = (health_df['age_group'] == age_level)
    
    if sum(mask) > 0:  # Make sure there are observations in this group
        kmf_age = KaplanMeierFitter()
        kmf_age.fit(health_df[mask]['time'], 
                   health_df[mask]['event'], 
                   label=f'Age: {age_level}')
        kmf_age.plot_survival_function(ax=ax, color=colors[i])

plt.title('Kaplan-Meier Curves by Age Group')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('survival_by_age.png')

# Prepare data for Cox Proportional Hazards model
# Convert categorical variables to dummy variables
df_cox = health_df.copy()
df_cox['smoker_yes'] = (df_cox['smoker'] == 'Yes').astype(int)
df_cox['male'] = (df_cox['sex'] == 'Male').astype(int)

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox[['time', 'event', 'age', 'smoker_yes', 'male', 'physical_activity']], 
        duration_col='time', 
        event_col='event')

# Print the Cox model summary
print("\nCox Proportional Hazards Model Summary:")
print(cph.summary)

# Plot the model coefficients
plt.figure(figsize=(10, 6))
cph.plot()
plt.title('Cox Model Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('cox_coefficients.png')

# Check the proportional hazards assumption

cph.check_assumptions(df_cox[['time', 'event', 'age', 'smoker_yes', 'male', 'physical_activity']], 
                    p_value_threshold=0.05,
                    show_plots=True)


#The violations of the proportional hazards assumption is a common issue in survival analysis.
#When it fails, it means that the effect of a variable changes over time, which cannot be captured by a standard Cox model.


model_cols = ['time', 'event', 'age', 'smoker_yes', 'male', 'physical_activity']
# Check for non-numeric columns in model data
for col in model_cols:
    print(f"Column {col} dtype: {df_cox[col].dtype}")

cph = CoxPHFitter()
try:
    cph.fit(df_cox[model_cols], duration_col='time', event_col='event')
    print("Cox model summary:")
    print(cph.summary)
except Exception as e:
    print(f"Error fitting Cox model: {e}")
    # Try to identify problematic columns
    for col in model_cols:
        try:
            print(f"Testing column {col}: {df_cox[col].head()}")
            # Try to cast to ensure it's numeric
            test = np.array(df_cox[col]).astype(float)
        except Exception as col_e:
            print(f"  - Error with column {col}: {col_e}")


#A ALTERNATIVE APPROACH: Simplified proportional hazards test
print("\n--- PROPORTIONAL HAZARDS TEST ---")
try:
    # Create a test dataframe with explicit numeric types
    test_df = pd.DataFrame()
    for col in model_cols:
        test_df[col] = df_cox[col].astype(float)
    
    # Try the manual test using lifelines.statistics
    from lifelines.statistics import proportional_hazard_test
    results = proportional_hazard_test(cph, test_df, time_transform='rank')
    print("\nProportional hazards test results:")
    print(results.summary)
    
    # Store the test results for each variable
    ph_test_results = {}
    for var in ['age', 'smoker_yes', 'male', 'physical_activity']:
        p_value = results.summary.loc[var, 'p']
        ph_test_results[var] = p_value
        
        # Report the results
        print(f"{var}: p-value = {p_value:.4f}, {'VIOLATED' if p_value < 0.05 else 'OK'}")
    
    # Check global test result
    global_p = results.summary.loc['global', 'p']
    print(f"Global test: p-value = {global_p:.4f}, {'VIOLATED' if global_p < 0.05 else 'OK'}")
except Exception as e:
    print(f"Error in proportional hazards test: {e}")
    ph_test_results = {}



# B. TIME-DEPENDENT COVARIATES TEST - More robust approach
print("\n--- TIME-DEPENDENT COVARIATES TEST ---")
try:
    # Create interactions with time
    df_time_int = df_cox.copy()
    df_time_int['log_time'] = np.log(df_time_int['time'] + 0.1)  # Add 0.1 to avoid log(0)
    
    # Create time interactions
    df_time_int['age_time'] = df_time_int['age'] * df_time_int['log_time']
    df_time_int['smoker_time'] = df_time_int['smoker_yes'] * df_time_int['log_time']
    df_time_int['male_time'] = df_time_int['male'] * df_time_int['log_time']
    df_time_int['pa_time'] = df_time_int['physical_activity'] * df_time_int['log_time']
    
    # # Convert all columns to float to be safe
    # for col in df_time_int.columns:
    #     df_time_int[col] = df_time_int[col].astype(float)
    
    # Fit model with time interactions
    cph_tv = CoxPHFitter()
    tv_cols = ['time', 'event', 'age', 'smoker_yes', 'male', 'physical_activity',
               'age_time', 'smoker_time', 'male_time', 'pa_time']
    
    cph_tv.fit(df_time_int[tv_cols], duration_col='time', event_col='event')
    
    print("Time-varying coefficients model:")
    print(cph_tv.summary)
    
    # Check if time interaction terms are significant
    significant_vars = []
    for var in ['age_time', 'smoker_time', 'male_time', 'pa_time']:
        p_value = cph_tv.summary.loc[var, 'p']
        base_var = var.replace('_time', '')
        if p_value < 0.05:
            significant_vars.append(base_var)
            print(f"* {base_var}: Time-varying effect detected (p={p_value:.4f})")
        else:
            print(f"  {base_var}: No time-varying effect (p={p_value:.4f})")
    
    if significant_vars:
        print(f"\nThe following variables violate the proportional hazards assumption: {', '.join(significant_vars)}")
    else:
        print("\nAll variables appear to satisfy the proportional hazards assumption based on time interactions.")
except Exception as e:
    print(f"Error in time-dependent model: {e}")


#C. LOG-LOG PLOTS: Robust implementation
from scipy.stats import linregress
print("\n--- LOG-LOG PLOT ANALYSIS ---")
try:
    from lifelines import KaplanMeierFitter
    
    # Variables to stratify by
    binary_vars = [
        ('smoker', 'Yes', 'No', 'Smoking Status'),
        ('sex', 'Male', 'Female', 'Sex')
    ]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    for var, val1, val2, title in binary_vars:
        plt.subplot(2, 2, plot_idx)
        
        # Filter data for each group
        group1 = health_df[health_df[var] == val1].copy()
        group2 = health_df[health_df[var] == val2].copy()
        
        if len(group1) == 0 or len(group2) == 0:
            print(f"Warning: One of the groups for {var} is empty")
            continue
            
        # Make sure time and event are numeric
        for df in [group1, group2]:
            df['time'] = df['time'].astype(float)
            df['event'] = df['event'].astype(int)
        
        # Fit KM for both groups
        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        
        kmf1.fit(group1['time'], group1['event'], label=val1)
        kmf2.fit(group2['time'], group2['event'], label=val2)
        
        # Helper function to create log-log coordinates safely
        def get_log_log_coords(kmf):
            sf = kmf.survival_function_.iloc[:, 0].values
            times = kmf.timeline
            
            # Only use valid points for log-log transform
            valid = (sf > 0) & (sf < 1)
            log_times = np.log(times[valid])
            log_log_sf = np.log(-np.log(sf[valid]))
            
            return log_times, log_log_sf
        
        # Get coordinates and plot
        x1, y1 = get_log_log_coords(kmf1)
        x2, y2 = get_log_log_coords(kmf2)
        
        plt.plot(x1, y1, 'o-', label=val1, alpha=0.7)
        plt.plot(x2, y2, 'x-', label=val2, alpha=0.7)
        
        # Add trendlines to assess parallelism
        slope1, intercept1, _, p1, _ = linregress(x1, y1)
        slope2, intercept2, _, p2, _ = linregress(x2, y2)
        
        x_range = np.linspace(min(np.min(x1), np.min(x2)), max(np.max(x1), np.max(x2)), 100)
        plt.plot(x_range, intercept1 + slope1 * x_range, 'r--', 
                 label=f"{val1} slope: {slope1:.2f}")
        plt.plot(x_range, intercept2 + slope2 * x_range, 'b--',
                 label=f"{val2} slope: {slope2:.2f}")
        
        # Calculate the difference in slopes
        slope_diff = abs(slope1 - slope2)
        slope_verdict = "PARALLEL" if slope_diff < 0.15 else "NOT PARALLEL"
        print(f"{title}: Slopes {slope1:.2f} vs {slope2:.2f} - {slope_verdict}")
        
        plt.title(f'Log-Log Plot for {title}\nSlope Difference: {slope_diff:.2f} ({slope_verdict})')
        plt.xlabel('log(Time)')
        plt.ylabel('log(-log(Survival))')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Also check physical activity - high vs low
    plt.subplot(2, 2, 3)
    
    # Create PA groups
    health_df['pa_group'] = 'Medium'
    health_df.loc[health_df['physical_activity'] <= 2, 'pa_group'] = 'Low' 
    health_df.loc[health_df['physical_activity'] >= 4, 'pa_group'] = 'High'
    
    # Plot just Low vs High to assess PH
    for pa_level in ['Low', 'High']:
        group = health_df[health_df['pa_group'] == pa_level].copy()
        
        if len(group) == 0:
            continue
            
        group['time'] = group['time'].astype(float)
        group['event'] = group['event'].astype(int)
        
        kmf = KaplanMeierFitter()
        kmf.fit(group['time'], group['event'], label=pa_level)
        
        # Get coordinates and plot
        x, y = get_log_log_coords(kmf)
        plt.plot(x, y, 'o-', label=pa_level, alpha=0.7)
        
        # Add trendline
        slope, intercept, _, p, _ = linregress(x, y)
        x_range = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(x_range, intercept + slope * x_range, '--', 
                 label=f"{pa_level} slope: {slope:.2f}")
                 
    # Calculate the difference in slopes for PA
    print(f"Physical Activity: PH assessment based on log-log plots")
    
    plt.title('Log-Log Plot for Physical Activity Level')
    plt.xlabel('log(Time)')
    plt.ylabel('log(-log(Survival))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add age groups
    plt.subplot(2, 2, 4)
    # Create age groups
    health_df['age_group'] = pd.cut(health_df['age'], 
                                 bins=[19, 40, 60, 81], 
                                 labels=['Young', 'Middle', 'Older'])
    
    # Plot age groups
    for age_group in ['Young', 'Older']:  # Just compare youngest vs oldest
        group = health_df[health_df['age_group'] == age_group].copy()
        
        if len(group) == 0:
            continue
            
        group['time'] = group['time'].astype(float)
        group['event'] = group['event'].astype(int)
        
        kmf = KaplanMeierFitter()
        kmf.fit(group['time'], group['event'], label=age_group)
        
        # Get coordinates and plot
        x, y = get_log_log_coords(kmf)
        plt.plot(x, y, 'o-', label=age_group, alpha=0.7)
        
        # Add trendline
        slope, intercept, _, p, _ = linregress(x, y)
        x_range = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(x_range, intercept + slope * x_range, '--', 
                 label=f"{age_group} slope: {slope:.2f}")
    
    print(f"Age: PH assessment based on log-log plots")
    
    plt.title('Log-Log Plot for Age Groups')
    plt.xlabel('log(Time)')
    plt.ylabel('log(-log(Survival))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('log_log_plots_robust.png')
except Exception as e:
    print(f"Error in log-log plot analysis: {e}")


#D. SUMMARY OF FINDINGS AND RECOMMENDATIONS
print("\n=====================================")
print("SUMMARY OF PROPORTIONAL HAZARDS CHECK")
print("=====================================")

# Collect evidence from all tests
all_violations = []

# From proportional hazards test
if ph_test_results:
    print("\nMethod 1: Statistical Test Results")
    for var, p_value in ph_test_results.items():
        verdict = "VIOLATED" if p_value < 0.05 else "OK"
        print(f"  {var}: p-value = {p_value:.4f} - {verdict}")
        if p_value < 0.05 and var not in all_violations:
            all_violations.append(var)

# From time interactions
if 'significant_vars' in locals():
    print("\nMethod 2: Time Interaction Results")
    for var in ['age', 'smoker_yes', 'male', 'physical_activity']:
        base_var = var
        tv_var = f"{var}_time"
        
        # Check if we have results from the time-varying test
        if hasattr(cph_tv, 'summary') and tv_var in cph_tv.summary.index:
            p_value = cph_tv.summary.loc[tv_var, 'p']
            verdict = "VIOLATED" if p_value < 0.05 else "OK"
            print(f"  {var}: p-value = {p_value:.4f} - {verdict}")
            
            if p_value < 0.05 and var not in all_violations:
                all_violations.append(var)

# Final recommendations based on all tests
print("\nFINAL ASSESSMENT:")
if all_violations:
    vars_map = {
        'age': 'Age',
        'smoker_yes': 'Smoking status',
        'male': 'Sex', 
        'physical_activity': 'Physical activity'
    }
    
    print("The proportional hazards assumption is violated for:")
    for var in all_violations:
        print(f"  - {vars_map.get(var, var)}")
    
    print("\nRECOMMENDED APPROACHES:")
    categorical_vars = [v for v in all_violations if v in ['smoker_yes', 'male']]
    continuous_vars = [v for v in all_violations if v in ['age', 'physical_activity']]
    
    if categorical_vars:
        print(f"1. Stratify the model by: {', '.join([vars_map.get(v, v) for v in categorical_vars])}")
        print("   This creates separate baseline hazards for each stratum.")
    
    if continuous_vars:
        print(f"2. Add time-interaction terms for: {', '.join([vars_map.get(v, v) for v in continuous_vars])}")
        print("   This allows the effect of these variables to change over time.")
    
    print("\n3. Consider alternative modeling approaches:")
    print("   - Accelerated failure time models")
    print("   - Flexible parametric survival models")
    print("   - Time-varying coefficient models")
else:
    print("The proportional hazards assumption appears to be satisfied for all variables.")
    print("The standard Cox model is appropriate for this data.")



#Four different approaches can be created to handle these violations:

# 1 Stratified Cox Model
# Uses separate baseline hazards for different strata of categorical variables
# Stratifies by smoking status, sex, and age groups
# This approach is ideal for categorical variables with few unique values (like smoker_yes and male)


# 2 Time-Varying Effects Model
# Adds interaction terms between variables and time
# Allows the effect of variables to change over time
# Particularly useful for physical activity, where the effect might strengthen or weaken over time


# 3 Penalized Splines Model
# Uses regularization (penalizer=0.1) to handle overfitting
# Includes non-linear transformations of continuous variables (age_squared)
# Helps address non-linearity in the relationship between age and hazard


# 4 Hybrid Approach
# Combines stratification for binary variables with time-varying effects
# Uses the strengths of both approaches
# Often provides the best balance between model complexity and interpretability



#1 FIRST MODEL - Stratification Model

# Create age groups for stratification
df_cox['age_group_cat'] = pd.cut(df_cox['age'], 
                               bins=[19, 35, 50, 65, 81], 
                               labels=['20-35', '36-50', '51-65', '66-80'])

# Create age-squared term to handle non-linearity
df_cox['age_squared'] = df_cox['age'] ** 2

# Create interactions with time to handle non-proportionality
# First, we need a scaled version of time for numerical stability
median_time = df_cox['time'].median()
df_cox['time_scaled'] = df_cox['time'] / median_time

# Create time interactions (in lifelines, this is handled differently)
df_cox['physical_activity_tv'] = df_cox['physical_activity'] * np.log(df_cox['time_scaled'] + 0.1)

# First model: Using stratification for categorical variables that violated PH assumption
print("\n--- Model 1: Using Stratification ---")
cph_stratified = CoxPHFitter()
# Include the stratification variables in the dataframe passed to fit
cph_stratified.fit(df_cox[['time', 'event', 'physical_activity', 'age_squared', 
                           'smoker_yes', 'male', 'age_group_cat']], 
                  duration_col='time', 
                  event_col='event',
                  strata=['smoker_yes', 'male', 'age_group_cat'])  # Stratify by variables violating PH

# Print the stratified model summary
print("\nCox Proportional Hazards Model with Stratification:")
print(cph_stratified.summary)

# Plot the stratified model coefficients
plt.figure(figsize=(10, 6))
cph_stratified.plot()
plt.title('Stratified Cox Model Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('stratified_cox_coefficients.png')



#2 - SECOND MODEL: Using time-varying coefficients
print("\n--- Model 2: Using Time-Varying Effects ---")
cph_tv = CoxPHFitter()
cph_tv.fit(df_cox[['time', 'event', 'age', 'age_squared', 'smoker_yes', 
                  'male', 'physical_activity', 'physical_activity_tv']], 
          duration_col='time', 
          event_col='event')

# Print the time-varying model summary
print("\nCox Proportional Hazards Model with Time-Varying Effects:")
print(cph_tv.summary)

# Plot the time-varying model coefficients
plt.figure(figsize=(10, 6))
cph_tv.plot()
plt.title('Time-Varying Cox Model Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('timevarying_cox_coefficients.png')





#3 Model SPLINE
from lifelines import CoxPHFitter
print("\n--- Model 3: Using Penalized Splines ---")

# Try fitting with penalized splines for age
cph_spline = CoxPHFitter(penalizer=0.1)
formula = "age + age_squared + smoker_yes + male + physical_activity"
cph_spline.fit(df_cox[['time', 'event', 'age', 'age_squared', 'smoker_yes', 'male', 'physical_activity']], 
              duration_col='time', 
              event_col='event',
              formula=formula)

# Print the spline model summary
print("\nCox Proportional Hazards Model with Penalized Splines:")
print(cph_spline.summary)

# Plot the spline model coefficients
plt.figure(figsize=(10, 6))
cph_spline.plot()
plt.title('Cox Model with Penalized Splines - Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('spline_cox_coefficients.png')




#4 MODEL - Using stratification for some variables and time-varying for others
print("\n--- Model 4: Hybrid Approach ---")
cph_hybrid = CoxPHFitter()
# Include the stratification variables in the dataframe passed to fit
cph_hybrid.fit(df_cox[['time', 'event', 'age_squared', 'physical_activity', 
                       'physical_activity_tv', 'smoker_yes', 'male']], 
              duration_col='time', 
              event_col='event',
              strata=['smoker_yes', 'male'])  # Stratify by binary variables

# Print the hybrid model summary
print("\nCox Proportional Hazards Hybrid Model:")
print(cph_hybrid.summary)

# Plot the hybrid model coefficients
plt.figure(figsize=(10, 6))
cph_hybrid.plot()
plt.title('Hybrid Cox Model Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('hybrid_cox_coefficients.png')





# Compare AIC of the different models to select the best one
print("\nModel Comparison:")
models = [cph_stratified, cph_tv, cph_spline, cph_hybrid]
model_names = ["Stratified", "Time-Varying", "Penalized Splines", "Hybrid"]

for name, model in zip(model_names, models):
    try:
        print(f"{name} model AIC: {model.AIC_partial_}")
    except:
        print(f"{name} model AIC: Not available")

#The stratified model has the lowest AIC



# For final analysis, use the model with the best AIC
# For demonstration, we'll choose the hybrid model
cph = cph_stratified  # This will be used for later analysis

# Plot the model coefficients of the selected "best" model
plt.figure(figsize=(10, 6))
cph.plot()
plt.title('Selected Cox Model Coefficients with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('best_cox_coefficients.png')

# Calculate hazard ratios and confidence intervals for reporting
hazard_ratios = np.exp(cph.params_)
confidence_intervals = np.exp(cph.confidence_intervals_)

# Print the hazard ratios with confidence intervals
print("\nHazard Ratios with 95% Confidence Intervals:")
for i, var in enumerate(cph.params_.index):
    hr = hazard_ratios[i]
    ci_lower = confidence_intervals.iloc[i, 0]
    ci_upper = confidence_intervals.iloc[i, 1]
    
    print(f"{var}: HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
    
    # Interpretation
    if var == 'physical_activity_tv':
        print(f"  • Time-varying effect of physical activity")
        continue
    
    if hr > 1:
        print(f"  • Increases risk by {(hr-1)*100:.0f}%")
    else:
        print(f"  • Decreases risk by {(1-hr)*100:.0f}%")
        

# Survival function prediction for specific profiles with the new model
# Define some example profiles for prediction that match our model structure
# Note: We need to adjust these profiles based on which model was selected as best

# For stratified model : profiles need to match the structure
if isinstance(cph, CoxPHFitter) and hasattr(cph, 'strata'):
    # For stratified model, we need to predict within strata
    print("\nPredicting survival using stratified model for different profiles:")
    
    strata_combos = []
    for smoker in [0, 1]:
        for male in [0, 1]:
            strata_combos.append((smoker, male))
    
    profile_descs = [
        'Non-Smoker Female', 
        'Non-Smoker Male',
        'Smoker Female',
        'Smoker Male'
    ]
    
    plt.figure(figsize=(12, 8))
    for i, (smoker, male) in enumerate(strata_combos):
        # Create a profile with different physical activity levels but same strata
        profiles = pd.DataFrame({
            'physical_activity': [1, 5],  # Low vs High activity
            'age_squared': [40**2, 40**2]  # Keep age constant at 40
        })
        
        # Get the strata values
        strata_values = {'smoker_yes': smoker, 'male': male}
        
        # Create the specific labels
        activity_labels = ['Low Activity', 'High Activity']
        
        # Try to predict survival - this requires careful handling with strata
        try:
            for j, pa in enumerate([1, 5]):
                # We need to manually create the survival function since stratified prediction is more complex
                baseline_survival = cph.baseline_survival_
                
                # Find the right stratum
                stratum_name = "smoker_yes={}".format(smoker) + "_male={}".format(male)
                if hasattr(baseline_survival, 'loc') and stratum_name in baseline_survival.columns:
                    times = baseline_survival.index.values
                    baseline = baseline_survival[stratum_name].values
                    
                    # Calculate linear predictor for this profile
                    profile = {
                        'physical_activity': pa,
                        'age_squared': 40**2
                    }
                    
                    lp = sum(cph.params_[var] * profile[var] for var in cph.params_.index if var in profile)
                    survival = baseline ** np.exp(lp)
                    
                    # Plot the survival curve
                    plt.plot(times, survival, 
                             label=f"{profile_descs[i]}, {activity_labels[j]}")
                else:
                    print(f"Warning: Could not find stratum {stratum_name} in baseline survival")
        except Exception as e:
            print(f"Error predicting for stratified model: {e}")
            
    plt.title('Predicted Survival by Strata and Physical Activity')
    plt.xlabel('Time (years)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig('stratified_predicted_survival.png')
else:
    # For non-stratified models, we can use the standard approach
    # Prepare the profiles based on which variables our model has
    profiles = pd.DataFrame()
    
    if 'age_squared' in cph.params_.index:
        # Set age and age_squared
        ages = [30, 60]
        profiles['age_squared'] = [age**2 for age in ages * 2]
        if 'age' in cph.params_.index:
            profiles['age'] = ages * 2
    
    if 'physical_activity' in cph.params_.index:
        # Set physical activity
        profiles['physical_activity'] = [4, 4, 1, 1]  # High, High, Low, Low activity
    
    if 'physical_activity_tv' in cph.params_.index:
        # We need a time interaction term
        median_time = df_cox['time'].median()
        # Use a value of 1 for the scaled time for initial prediction
        profiles['physical_activity_tv'] = profiles['physical_activity'] * np.log(1 + 0.1)
    
    if 'smoker_yes' in cph.params_.index:
        profiles['smoker_yes'] = [0, 0, 1, 1]  # No, No, Yes, Yes
    
    if 'male' in cph.params_.index:
        profiles['male'] = [0, 1, 0, 1]  # Female, Male, Female, Male
    
    # Create labels based on what variables we have
    profile_labels = []
    for i in range(profiles.shape[0]):
        label_parts = []
        
        if 'age' in profiles.columns:
            label_parts.append(f"{'Young' if profiles.iloc[i]['age'] < 45 else 'Older'}")
        
        if 'male' in profiles.columns:
            label_parts.append(f"{'Male' if profiles.iloc[i]['male'] == 1 else 'Female'}")
        
        if 'smoker_yes' in profiles.columns:
            label_parts.append(f"{'Smoker' if profiles.iloc[i]['smoker_yes'] == 1 else 'Non-Smoker'}")
        
        if 'physical_activity' in profiles.columns:
            label_parts.append(f"{'High' if profiles.iloc[i]['physical_activity'] >= 3 else 'Low'} Activity")
        
        profile_labels.append(", ".join(label_parts))
    
    # Only predict if we have all the necessary columns
    if not profiles.empty and all(var in profiles.columns for var in cph.params_.index):
        # Predict survival for these profiles
        plt.figure(figsize=(10, 6))
        for i, profile in enumerate(profile_labels):
            try:
                surv_func = cph.predict_survival_function(profiles.iloc[i:i+1])
                plt.plot(surv_func.index, surv_func.values.flatten(), label=profile)
            except Exception as e:
                print(f"Error predicting survival for profile {profile}: {e}")
        
        plt.title('Predicted Survival Functions for Different Profiles')
        plt.xlabel('Time (years)')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.legend()
        plt.savefig('predicted_survival_functions.png')
    else:
        print("Warning: Could not generate prediction profiles that match the model variables")

# Additional analysis: Median survival time for different groups
print("\nMedian Survival Times:")

# Overall
median_survival = kmf.median_survival_time_
print(f"Overall: {median_survival:.2f} years")

# By smoking status
smoker_median = kmf_smoker.median_survival_time_
nonsmoker_median = kmf_nonsmoker.median_survival_time_
print(f"Smokers: {smoker_median:.2f} years")
print(f"Non-smokers: {nonsmoker_median:.2f} years")
print(f"Difference: {nonsmoker_median - smoker_median:.2f} years")

# By sex
male_median = kmf_male.median_survival_time_
female_median = kmf_female.median_survival_time_
print(f"Males: {male_median:.2f} years")
print(f"Females: {female_median:.2f} years")
print(f"Difference: {female_median - male_median:.2f} years")

# Calculate restricted mean survival time (RMST) - area under the survival curve
# RMST represents the life expectancy up to a specific time point
print("\nRestricted Mean Survival Time (up to 10 years):")

# Overall RMST
overall_rmst = kmf.survival_function_at_times(times=10)


# RMST by smoking status
smoker_rmst = kmf_smoker.survival_function_at_times(times=10)[10]
nonsmoker_rmst = kmf_nonsmoker.survival_function_at_times(times=10)[10]
print(f"Smokers: {smoker_rmst:.2f} years")
print(f"Non-smokers: {nonsmoker_rmst:.2f} years")
print(f"Difference: {nonsmoker_rmst - smoker_rmst:.2f} years")

smoker_rmst = kmf_smoker.median_survival_time_
nonsmoker_rmst = kmf_nonsmoker.median_survival_time_
print(f"Smokers: {smoker_rmst:.2f} years")
print(f"Non-smokers: {nonsmoker_rmst:.2f} years")
print(f"Difference: {nonsmoker_rmst - smoker_rmst:.2f} years")

