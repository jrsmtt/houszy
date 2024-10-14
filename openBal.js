from sklearn.preprocessing import StandardScaler

def preprocess_opn_bal_columns(df, columns):
    """
    Preprocess open balance columns by:
    - Keeping negative balances (valid)
    - Capping large positive outliers at the 99th percentile
    - Creating binary flags for zero balances and negative balances
    - Optionally scaling the capped balance column
    
    Parameters:
    - df: DataFrame containing the columns to preprocess
    - columns: List of column names representing open balance columns
    
    Returns:
    - df: DataFrame with new processed columns
    """
    for col in columns:
        # 1. Cap large positive outliers at the 99th percentile (keep negative balances)
        cap_value = df[col].quantile(0.99)
        df[col + '_capped'] = df[col].apply(lambda x: min(x, cap_value) if x > 0 else x)
        
        # 2. Create a binary flag for zero balances
        df[col + '_zero_flag'] = df[col].apply(lambda x: 1 if x == 0 else 0)
        
        # 3. Create a binary flag for negative balances
        df[col + '_negative_flag'] = df[col].apply(lambda x: 1 if x < 0 else 0)
        
        # 4. Scale the capped balances (optional)
        scaler = StandardScaler()
        df[col + '_scaled'] = scaler.fit_transform(df[[col + '_capped']])
    
    return df




// from sklearn.preprocessing import StandardScaler

// def preprocess_open_account_count_columns(df, columns):
//     """
//     Preprocess open account count columns by:
//     - Capping large values at the 99th percentile
//     - Creating binary flags for households with any open accounts
//     - Optionally scaling the capped count column
    
//     Parameters:
//     - df: DataFrame containing the columns to preprocess
//     - columns: List of column names representing count of open accounts
    
//     Returns:
//     - df: DataFrame with new processed columns
//     """
//     for col in columns:
//         # 1. Cap large values at the 99th percentile
//         cap_value = df[col].quantile(0.99)
//         df[col + '_capped'] = df[col].apply(lambda x: min(x, cap_value))
        
//         # 2. Create a binary flag for households with any open accounts
//         df[col + '_has_open_account'] = df[col].apply(lambda x: 1 if x > 0 else 0)
        
//         # 3. Scale the capped counts (optional, useful for models like Logistic Regression)
//         scaler = StandardScaler()
//         df[col + '_scaled'] = scaler.fit_transform(df[[col + '_capped']])
    
//     return df





from sklearn.preprocessing import StandardScaler

def preprocess_open_account_columns(columns):
    """
    Preprocess open account count columns directly on the global DataFrame df_vhh by:
    - Capping large values at the 99th percentile
    - Creating binary flags for households with any open accounts
    - Scaling the capped count column
    
    Parameters:
    - columns: List of column names representing count of open accounts
    
    Returns:
    - Modifies the DataFrame df_vhh in-place
    """
    # Access global DataFrame df_vhh
    global df_vhh

    # Initialize the scaler once
    scaler = StandardScaler()

    for col in columns:
        # Cap large values at the 99th percentile
        cap_value = df_vhh[col].quantile(0.99)
        df_vhh[col + '_capped'] = df_vhh[col].clip(upper=cap_value)

        # Create binary flag for households with any open accounts
        df_vhh[col + '_has_open_account'] = (df_vhh[col] > 0).astype(int)

        # Scale the capped counts
        df_vhh[col + '_scaled'] = scaler.fit_transform(df_vhh[[col + '_capped']])





# 1. Cap outliers at the 99th percentile
cap_value = df_vhh['hh_tenure_mths'].quantile(0.99)
df_vhh['hh_tenure_mths_capped'] = df_vhh['hh_tenure_mths'].clip(upper=cap_value)

# 2. Create tenure buckets
df_vhh['tenure_category'] = pd.cut(df_vhh['hh_tenure_mths'], 
                                   bins=[0, 12, 60, np.inf], 
                                   labels=['Short', 'Medium', 'Long'])

# 3. Scale the capped tenure values (optional for scaling-sensitive models)
scaler = StandardScaler()
df_vhh['hh_tenure_mths_scaled'] = scaler.fit_transform(df_vhh[['hh_tenure_mths_capped']])

# Check the updated DataFrame
df_vhh.head()





def convert_yes_no_to_binary(columns):
    """
    Converts 'yes'/'no' values to 1/0 in the specified columns of the DataFrame df_vhh.
    
    Parameters:
    - columns: List of column names to convert
    
    Returns:
    - Modifies the DataFrame df_vhh in-place
    """
    global df_vhh
    
    for col in columns:
        # Map 'yes' to 1 and 'no' to 0
        df_vhh[col + '_binary'] = df_vhh[col].map({'yes': 1, 'no': 0})

# Example usage:
cbol_columns = ['cbol_ind_1', 'cbol_ind_2', 'cbol_ind_3', 'cbol_ind_6', 'cbol_ind_12']
convert_yes_no_to_binary(cbol_columns)

# Check the results
df_vhh.head()




# Impute missing values in cbol_2mo_cnt with the median
df_vhh['cbol_2mo_cnt'] = df_vhh['cbol_2mo_cnt'].fillna(df_vhh['cbol_2mo_cnt'].median())

# Optionally, create a flag for missing cbol_2mo_cnt before imputing
df_vhh['cbol_2mo_cnt_missing'] = df_vhh['cbol_2mo_cnt'].isna().astype(int)



# Capping outliers for cbol_2mo_cnt at the 99th percentile
cap_value = df_vhh['cbol_2mo_cnt'].quantile(0.99)
df_vhh['cbol_2mo_cnt_capped'] = df_vhh['cbol_2mo_cnt'].clip(upper=cap_value)

# Scaling the capped count (optional for models sensitive to scaling)
scaler = StandardScaler()
df_vhh['cbol_2mo_cnt_scaled'] = scaler.fit_transform(df_vhh[['cbol_2mo_cnt_capped']])




from sklearn.preprocessing import LabelEncoder

def handle_nominal_data_with_label_encoding(columns, fill_method='missing'):
    """
    Handles nominal categorical data by filling NaN values and applying Label Encoding
    directly to the global DataFrame (df_vhh).

    Parameters:
    - columns: List of nominal columns to process
    - fill_method: How to handle NaN values:
        - 'missing': Fill NaN values with a placeholder 'Missing'
        - 'mode': Fill NaN values with the most frequent value (mode)
    
    Returns:
    - None: Modifies the global DataFrame (df_vhh) in place
    """
    global df_vhh  # Refers to the global DataFrame

    # Step 1: Handle NaN values
    for col in columns:
        if fill_method == 'missing':
            # Fill NaN values with 'Missing'
            df_vhh[col] = df_vhh[col].fillna('Missing')
        elif fill_method == 'mode':
            # Fill NaN with the most frequent value (mode)
            mode_value = df_vhh[col].mode()[0]
            df_vhh[col] = df_vhh[col].fillna(mode_value)
        
        # Step 2: Apply Label Encoding
        le = LabelEncoder()
        df_vhh[col + '_encoded'] = le.fit_transform(df_vhh[col])

        # Optional: Print label mapping for reference
        print(f"Label mapping for {col}:")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

