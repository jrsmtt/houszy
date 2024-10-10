plt.figure(figsize=(10, 6))
pd.Series(transformed_data_flat).plot(kind='density', label='Density')
plt.hist(transformed_data_flat, bins=30, density=True, alpha=0.5, color='gray', label='Histogram')
plt.title('Density and Histogram of Yeo-Johnson Transformed Data')
plt.xlabel('Transformed Value')
plt.ylabel('Density')
plt.legend()
plt.show()





import numpy as np

# Calculate Q1 and Q3
Q1 = np.percentile(transformed_data_flat, 25)
Q3 = np.percentile(transformed_data_flat, 75)
IQR = Q3 - Q1

# Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = transformed_data_flat[(transformed_data_flat < lower_bound) | (transformed_data_flat > upper_bound)]

print("Number of outliers:", len(outliers))
print("Outliers:", outliers)




from scipy.stats import zscore

# Calculate the z-scores
z_scores = zscore(transformed_data_flat)

# Find outliers
outliers_z = transformed_data_flat[np.abs(z_scores) > 3]

print("Number of outliers using Z-Score:", len(outliers_z))
print("Outliers:", outliers_z)
