from sklearn.feature_selection import VarianceThreshold

# Set a threshold for variance. Features with variance below this threshold will be removed
threshold = 0.01
selector = VarianceThreshold(threshold)
df_reduced = selector.fit_transform(df_fof)

# Check how many features were dropped
removed_features = len(df_fof.columns) - df_reduced.shape[1]
print(f"Removed {removed_features} low-variance features")




import numpy as np

# Compute the correlation matrix
corr_matrix = df_fof.corr().abs()

# Set a threshold for highly correlated features
threshold = 0.85

# Select upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation higher than the threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop these columns
df_fof_reduced = df_fof.drop(columns=to_drop)

print(f"Removed {len(to_drop)} highly correlated features")
