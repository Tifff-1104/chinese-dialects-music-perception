import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid
from matplotlib.gridspec import GridSpec

# Load data
file_path = '/Users/tiffany/Downloads/responses FINAL.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Define outcomes and variables
outcomes = {
    'Total_Pitch perception': 'Pitch Perception',
    'Total_Rhythm perception': 'Rhythm Perception',
    'Total_Cadence learning': 'Cadence Learning',
    'Total_Style learning': 'Style Learning'
}
covariates = ['foreign_language', 'absolute_pitch', 'music_training']
iv = 'number of tones'

# Set up plot style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 10)

# Create a comprehensive figure
fig = plt.figure(figsize=(18, 24))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1.5, 1])

# 1. Partial Regression Plots
for i, (outcome, outcome_name) in enumerate(outcomes.items()):
    # Clean data
    clean_data = data[[outcome, iv] + covariates].dropna()
    y = clean_data[outcome]
    X = clean_data[[iv] + covariates]
    X = sm.add_constant(X)  # Add intercept

    # Fit model
    model = sm.OLS(y, X).fit()

    # Create partial regression plot
    ax = fig.add_subplot(gs[i * 2])
    plot_partregress_grid(
        model,
        exog_idx=iv,  # Focus on number of tones
        grid=(1, 1),
        ax=ax
    )
    ax.set_title(f'Partial Regression: {outcome_name}', fontsize=16)
    ax.set_xlabel('Number of Tones (residualized)', fontsize=14)
    ax.set_ylabel(f'{outcome_name} (residualized)', fontsize=14)
    ax.annotate(f"β = {model.params[iv]:.2f}, p = {model.pvalues[iv]:.3f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8),
                fontsize=14)

# 2. Coefficient Plots
coef_data = []
for outcome, outcome_name in outcomes.items():
    # Clean data
    clean_data = data[[outcome, iv] + covariates].dropna()
    y = clean_data[outcome]
    X = clean_data[[iv] + covariates]
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()

    # Store coefficients
    for var in [iv] + covariates:
        coef_data.append({
            'Outcome': outcome_name,
            'Predictor': var,
            'Coefficient': model.params[var],
            'CI_low': model.conf_int().loc[var, 0],
            'CI_high': model.conf_int().loc[var, 1],
            'p_value': model.pvalues[var]
        })

# Convert to DataFrame
coef_df = pd.DataFrame(coef_data)

# Create coefficient plot
ax_coef = fig.add_subplot(gs[1])
sns.pointplot(
    data=coef_df,
    x='Coefficient',
    y='Predictor',
    hue='Outcome',
    join=False,
    dodge=0.3,
    markers='o',
    capsize=0.1,
    ax=ax_coef
)
ax_coef.set_title('Regression Coefficients Across Outcomes', fontsize=16)
ax_coef.set_xlabel('Coefficient Value', fontsize=14)
ax_coef.set_ylabel('Predictor', fontsize=14)
ax_coef.axvline(0, ls='--', color='gray')
ax_coef.legend(title='Outcome', loc='center left', bbox_to_anchor=(1, 0.5))

# 3. R² Comparison Plot
r2_data = []
for outcome, outcome_name in enumerate(outcomes.items()):
    # Clean data
    clean_data = data[[outcome, iv] + covariates].dropna()

    # Model 1: Covariates only
    X1 = sm.add_constant(clean_data[covariates])
    model1 = sm.OLS(clean_data[outcome], X1).fit()

    # Model 2: Full model
    X2 = sm.add_constant(clean_data[[iv] + covariates])
    model2 = sm.OLS(clean_data[outcome], X2).fit()

    r2_data.append({
        'Outcome': outcome_name,
        'Model': 'Covariates Only',
        'R²': model1.rsquared
    })
    r2_data.append({
        'Outcome': outcome_name,
        'Model': 'Full Model',
        'R²': model2.rsquared
    })

# Convert to DataFrame
r2_df = pd.DataFrame(r2_data)

# Create R² plot
ax_r2 = fig.add_subplot(gs[3])
sns.barplot(
    data=r2_df,
    x='Outcome',
    y='R²',
    hue='Model',
    palette={'Covariates Only': 'skyblue', 'Full Model': 'coral'},
    ax=ax_r2
)
ax_r2.set_title('Model Comparison: R² Values', fontsize=16)
ax_r2.set_xlabel('Outcome', fontsize=14)
ax_r2.set_ylabel('R²', fontsize=14)
ax_r2.legend(title='Model')

# Add delta R² annotations
for i, outcome in enumerate(outcomes.keys()):
    r2_cov = r2_df[(r2_df['Outcome'] == outcome) &
                   (r2_df['Model'] == 'Covariates Only')]['R²'].values[0]
    r2_full = r2_df[(r2_df['Outcome'] == outcome) &
                    (r2_df['Model'] == 'Full Model')]['R²'].values[0]
    delta = r2_full - r2_cov
    ax_r2.text(i, r2_full + 0.02, f"ΔR² = {delta:.3f}",
               ha='center', fontsize=12, color='black')

# Adjust layout
plt.tight_layout()
plt.savefig('hierarchical_regression_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Create individual partial regression plots with enhanced styling
for outcome, outcome_name in outcomes.items():
    plt.figure(figsize=(10, 8))

    # Clean data
    clean_data = data[[outcome, iv] + covariates].dropna()
    y = clean_data[outcome]
    X = clean_data[[iv] + covariates]
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()

    # Create partial regression data
    y_resid = model.resid + model.predict(X)  # Residualized outcome
    x_resid = clean_data[iv]  # Original IV values

    # Create enhanced scatter plot
    sns.regplot(
        x=x_resid,
        y=y_resid,
        scatter_kws={'alpha': 0.6, 'color': 'steelblue'},
        line_kws={'color': 'coral', 'linewidth': 2},
        ci=95
    )

    # Add statistics
    plt.annotate(f"β = {model.params[iv]:.2f} (p = {model.pvalues[iv]:.3f})\n"
                 f"R² = {model.rsquared:.3f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8),
                 fontsize=14)

    # Format plot
    plt.title(f'Partial Regression: {outcome_name}\n(Controlling for Foreign Language, Abs Pitch, Music Training)',
              fontsize=16, pad=20)
    plt.xlabel('Number of Tones', fontsize=14)
    plt.ylabel(f'Partial {outcome_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save
    filename = f'partial_regression_{outcome_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("All visualizations saved successfully.")