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






from sklearn.linear_model import LassoCV

# Fit LassoCV for automatic feature selection
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)

# Get the mask of selected features (non-zero coefficients)
selected_features = X_train.columns[(lasso.coef_ != 0)]

# Reduce dataset to selected features
X_train_lasso_reduced = X_train[selected_features]
X_test_lasso_reduced = X_test[selected_features]

print(f"Reduced to {len(selected_features)} features after Lasso selection")





from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Assuming 'target' is the label column
X = df_fof_reduced.drop('target', axis=1)
y = df_fof_reduced['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit RandomForest to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance scores
importances = rf.feature_importances_

# Sort the feature importance scores in descending order
indices = np.argsort(importances)[::-1]

# Keep top N important features
N = 100  # Number of top features you want to keep
important_features = X_train.columns[indices[:N]]

# Reduce dataset to these top N features
X_train_reduced = X_train[important_features]
X_test_reduced = X_test[important_features]

print(f"Reduced to {N} most important features")





