import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "us_market_data_hourly_with_inflation_modified.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Exclude the first column
data_subset = data.iloc[:, 1:]

# Compute the correlation matrix
correlation_matrix = data_subset.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
'''plt.show()'''

# Filter columns based on their correlation with inflation_rate
target_column = "inflation_rate"  # Replace with the exact name of the column
relevant_columns = correlation_matrix.loc[
    correlation_matrix[target_column] > 0
].index.tolist()

# Filter the original data subset to retain only these relevant columns
filtered_data = data_subset[relevant_columns]

# Display the filtered dataset
print("Columns retained after filtering based on correlation with inflation_rate:")
print(filtered_data.columns)

# Save the filtered data if needed
filtered_data.to_csv("filtered_data_with_inflation_rate.csv", index=False)