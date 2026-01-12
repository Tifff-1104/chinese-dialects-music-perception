import pandas as pd
from scipy.stats import skew, kurtosis

# Load the data
file_path = ('/Users/tiffany/Downloads/responses FINAL.csv')
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Define outcome variables
outcomes = [
    'Total_Pitch perception',
    'Total_Rhythm perception',
    'Total_Cadence learning',
    'Total_Style learning',
    'Total'  # Overall total if needed
]

# Initialize a dictionary to store results
results = []

# Calculate statistics for each outcome
for outcome in outcomes:
    # Skip if column doesn't exist
    if outcome not in data:
        continue

    # Clean data - remove missing values
    clean_data = data[outcome].dropna()

    # Only calculate if we have sufficient data
    if len(clean_data) > 1:
        stats = {
            'Variable': outcome,
            'Mean': clean_data.mean(),
            'SD': clean_data.std(),
            'Min': clean_data.min(),
            'Max': clean_data.max(),
            'Skewness': skew(clean_data),
            'Kurtosis': kurtosis(clean_data, fisher=True),  # Fisher's definition (normal = 0)
            'N': len(clean_data)
        }
        results.append(stats)

# Create DataFrame and save as CSV
results_df = pd.DataFrame(results)
output_file = 'outcome_statistics.csv'
results_df.to_csv(output_file, index=False)

print(f"Descriptive statistics saved to {output_file}")
print("\nResults:")
print(results_df)