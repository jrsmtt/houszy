   from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Example dataset with categorical features
X_categorical = X.select_dtypes(include=['object', 'category'])

# Convert categorical variables to integer encoding
X_encoded = pd.get_dummies(X_categorical, drop_first=True)

# Apply the Chi-square test
selector = SelectKBest(chi2, k=20)  # Select top 20 features
X_reduced = selector.fit_transform(X_encoded, y)

# Get the selected feature names
selected_features = X_encoded.columns[selector.get_support()]
X_reduced_df = X_encoded[selected_features]





