import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
file_path = '/Users/tiffany/Downloads/responses FINAL.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Define outcomes and independent variables
outcomes = {
    'Total_Pitch perception': 'Pitch Perception',
    'Total_Rhythm perception': 'Rhythm Perception',
    'Total_Cadence learning': 'Cadence Learning',
    'Total_Style learning': 'Style Learning'
}

covariates = ['foreign_language', 'absolute_pitch', 'music_training']
iv = 'number of tones'

# Prepare results storage
results_summary = []

# Perform hierarchical regression for each outcome
for outcome, outcome_name in outcomes.items():
    # Clean data for this outcome
    clean_data = data[[outcome, iv] + covariates].dropna()

    # Model 1: Covariates only
    formula1 = f'Q("{outcome}") ~ foreign_language + absolute_pitch + music_training'
    model1 = ols(formula1, data=clean_data).fit()

    # Model 2: Covariates + IV
    formula2 = f'Q("{outcome}") ~ foreign_language + absolute_pitch + music_training + Q("{iv}")'
    model2 = ols(formula2, data=clean_data).fit()

    # Compare models
    anova_results = sm.stats.anova_lm(model1, model2)
    r2_change = model2.rsquared - model1.rsquared
    f_change = anova_results['F'][1]
    p_change = anova_results['Pr(>F)'][1]

    # Store results
    results_summary.append({
        'Outcome': outcome_name,
        'Model1 R²': model1.rsquared,
        'Model2 R²': model2.rsquared,
        'R² Change': r2_change,
        'F Change': f_change,
        'p-value Change': p_change,
        'Number of Tones Beta': model2.params[f'Q("{iv}")'],
        'Number of Tones p': model2.pvalues[f'Q("{iv}")']
    })

    # Print detailed results
    print(f"\n\n===== {outcome_name} =====")
    print(f"Model 1 (Covariates only): R² = {model1.rsquared:.3f}")
    print(model1.summary())
    print(f"\nModel 2 (Covariates + Number of Tones): R² = {model2.rsquared:.3f}")
    print(f"R² Change = {r2_change:.3f}, F Change = {f_change:.2f}, p = {p_change:.4f}")
    print(model2.summary())
    print("\n" + "=" * 80)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results_summary)
results_df.to_csv('hierarchical_regression_results.csv', index=False)

# Print summary table
print("\n\nHierarchical Regression Summary:")
print(results_df[['Outcome', 'Model1 R²', 'Model2 R²', 'R² Change',
                  'F Change', 'p-value Change', 'Number of Tones Beta',
                  'Number of Tones p']])