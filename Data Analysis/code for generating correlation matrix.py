import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data
file_path = '/Users/tiffany/Downloads/responses FINAL.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Define all variables of interest
predictors = [
    'number of tones',
    'usage_frequency',
    'fluency',
    'foreign_language',
    'music_training',
    'music_freq',
    'absolute_pitch'
]

outcomes = [
    'Total_Pitch perception',
    'Total_Rhythm perception',
    'Total_Cadence learning',
    'Total_Style learning',
    'Total'
]

# Combine into one list
all_vars = predictors + outcomes

# Initialize correlation matrix
corr_matrix = pd.DataFrame(index=all_vars, columns=all_vars)
pvalue_matrix = pd.DataFrame(index=all_vars, columns=all_vars)

# Calculate correlations
for row in all_vars:
    for col in all_vars:
        # Skip diagonal (will be 1 later)
        if row == col:
            continue

        clean_data = data[[row, col]].dropna()
        if len(clean_data) > 3:  # Minimum for correlation
            corr, pval = pearsonr(clean_data[row], clean_data[col])
            corr_matrix.loc[row, col] = corr
            pvalue_matrix.loc[row, col] = pval
        else:
            corr_matrix.loc[row, col] = np.nan
            pvalue_matrix.loc[row, col] = np.nan

# Set diagonal to 1
np.fill_diagonal(corr_matrix.values, 1)
np.fill_diagonal(pvalue_matrix.values, 0)

# Save to CSV
corr_matrix.to_csv('full_correlation_matrix.csv')
pvalue_matrix.to_csv('full_pvalue_matrix.csv')

