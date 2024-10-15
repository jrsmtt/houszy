from sklearn.feature_selection import VarianceThreshold

# Set a threshold for variance. Features with variance below this threshold will be removed
threshold = 0.01
selector = VarianceThreshold(threshold)
df_reduced = selector.fit_transform(df_fof)

# Check how many features were dropped
removed_features = len(df_fof.columns) - df_reduced.shape[1]
print(f"Removed {removed_features} low-variance features")
