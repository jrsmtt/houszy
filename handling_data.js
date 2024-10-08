import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce
import sweetviz as sv

# Load the dataset
# Assuming 'df' is your loaded dataframe
# df = pd.read_csv('your_dataset.csv')  # Load your dataset accordingly

# Step 1: Analyze the dataset using SweetViz
# Generate the SweetViz report
report = sv.analyze(df)
report.show_html("sweetviz_report.html")  # This will generate a detailed report

# You can inspect the SweetViz report for more details.
# SweetViz will show you the distribution, missing values, and data types.

# Step 2: Based on analysis, we preprocess the data.

# Split categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Step 3: Handling Missing Values based on distribution analysis

# SweetViz will give you distributions of columns.
# If the distribution is normal-like (bell curve), use mean.
# If skewed or has outliers, use median. For categorical, use mode.

# 3.1 Numerical Features
# Analyze distributions using SweetViz. If the column is skewed or has outliers, use median; otherwise, use mean.

def get_imputation_strategy(column):
    # Based on SweetViz, determine whether to use 'mean' or 'median'
    if df[column].skew() > 1 or df[column].skew() < -1:  # Detecting skewness for choosing median
        return 'median'
    else:
        return 'mean'

# Apply the strategy to all numerical columns
for col in numerical_cols:
    strategy = get_imputation_strategy(col)
    imputer = SimpleImputer(strategy=strategy)
    df[col] = imputer.fit_transform(df[[col]])

# 3.2 Categorical Features
# Use mode (most frequent) for categorical columns
mode_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

# Step 4: Encoding Categorical Features
# If categorical features have many categories, use Target Encoding; otherwise, use Label Encoding

for col in categorical_cols:
    if df[col].nunique() > 50:  # Arbitrary threshold for deciding when to use Target Encoding
        target_encoder = ce.TargetEncoder(cols=[col])
        df[col] = target_encoder.fit_transform(df[col], df['resp_tag'])  # Assuming 'resp_tag' is the target variable
    else:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

# Step 5: Scaling Numerical Features

# Use StandardScaler to normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Now your data is preprocessed and ready for modeling.
