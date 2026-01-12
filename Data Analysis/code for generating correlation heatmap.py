import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
file_path = '/Users/tiffany/Downloads/responses FINAL.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Define all variables with your specified names
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
    'Total_Style learning'
]

# Combine into one list
all_vars = predictors + outcomes

# Calculate correlation matrix
corr_matrix = data[all_vars].corr()

# Create mapping to new names
name_mapping = {
    'number of tones': 'Number of tones',
    'usage_frequency': 'Usage frequency',
    'fluency': 'Fluency',
    'foreign_language': 'Foreign language',
    'music_training': 'Music training',
    'music_freq': 'Music frequency',
    'absolute_pitch': 'Absolute pitch',
    'Total_Pitch perception': 'Melody perception',
    'Total_Rhythm perception': 'Rhythm perception',
    'Total_Cadence learning': 'Cadence learning',
    'Total_Style learning': 'Style learning'
}

# Rename rows and columns
corr_matrix_renamed = corr_matrix.rename(
    index=name_mapping,
    columns=name_mapping
)

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix_renamed, dtype=bool))

# Create heatmap showing only lower triangle
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    corr_matrix_renamed,
    mask=mask,  # Apply the mask to hide upper triangle
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient'},
    annot_kws={'size': 9}
)

# Customize plot
plt.title('Correlation Matrix (Lower Triangle)', fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add divider lines between predictors and outcomes
predictor_count = len(predictors)
ax.axhline(y=predictor_count, color='black', linewidth=1)
ax.axvline(x=predictor_count, color='black', linewidth=1)



plt.tight_layout()
plt.savefig('lower_triangle_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Save to CSV
corr_matrix_renamed.to_csv('correlation_matrix_renamed.csv')
print("Correlation matrix saved to 'correlation_matrix_renamed.csv'")