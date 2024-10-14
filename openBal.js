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




from sklearn.preprocessing import StandardScaler

def preprocess_open_account_count_columns(df, columns):
    """
    Preprocess open account count columns by:
    - Capping large values at the 99th percentile
    - Creating binary flags for households with any open accounts
    - Optionally scaling the capped count column
    
    Parameters:
    - df: DataFrame containing the columns to preprocess
    - columns: List of column names representing count of open accounts
    
    Returns:
    - df: DataFrame with new processed columns
    """
    for col in columns:
        # 1. Cap large values at the 99th percentile
        cap_value = df[col].quantile(0.99)
        df[col + '_capped'] = df[col].apply(lambda x: min(x, cap_value))
        
        # 2. Create a binary flag for households with any open accounts
        df[col + '_has_open_account'] = df[col].apply(lambda x: 1 if x > 0 else 0)
        
        # 3. Scale the capped counts (optional, useful for models like Logistic Regression)
        scaler = StandardScaler()
        df[col + '_scaled'] = scaler.fit_transform(df[[col + '_capped']])
    
    return df

